# sweep_sine_omega.py

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from datasets import ImageINRDataset
from models import MLPINR


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def omega_to_name(x: float) -> str:
    return str(x).replace(".", "p").replace("-", "m")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def sample_batch(
    coords: torch.Tensor,
    values: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_samples = coords.shape[0]

    if batch_size >= n_samples:
        idx = torch.arange(n_samples, device=coords.device)
    else:
        idx = torch.randperm(n_samples, device=coords.device)[:batch_size]

    return coords[idx], values[idx]


def normalized_to_image01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)


def compute_psnr_from_mse(mse: float) -> float:
    if mse <= 0.0:
        return float("inf")
    return -10.0 * math.log10(mse)


def compute_basic_metrics(pred_01: torch.Tensor, target_01: torch.Tensor) -> dict[str, float]:
    err = pred_01 - target_01
    abs_err = torch.abs(err)

    mse = torch.mean(err ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(abs_err)
    max_abs = torch.max(abs_err)
    p95_abs = torch.quantile(abs_err, 0.95)
    p99_abs = torch.quantile(abs_err, 0.99)
    bias = torch.mean(err)

    mse_value = float(mse.item())

    return {
        "mse": mse_value,
        "psnr": compute_psnr_from_mse(mse_value),
        "rmse": float(rmse.item()),
        "mae": float(mae.item()),
        "max_abs": float(max_abs.item()),
        "p95_abs": float(p95_abs.item()),
        "p99_abs": float(p99_abs.item()),
        "bias": float(bias.item()),
    }


def tensor_to_uint8_image(x01: torch.Tensor, sidelength: int) -> np.ndarray:
    img = x01.view(sidelength, sidelength).detach().cpu().numpy()
    img = (img * 255.0).round().clip(0, 255).astype(np.uint8)
    return img


def save_image(path: Path, img_uint8: np.ndarray) -> None:
    Image.fromarray(img_uint8, mode="L").save(path)


def save_side_by_side(path: Path, left_uint8: np.ndarray, right_uint8: np.ndarray) -> None:
    panel = np.concatenate([left_uint8, right_uint8], axis=1)
    Image.fromarray(panel, mode="L").save(path)


@torch.no_grad()
def render_full_image_in_chunks(
    model: torch.nn.Module,
    coords: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    outputs = []
    n_samples = coords.shape[0]

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        outputs.append(model(coords[start:end]))

    return torch.cat(outputs, dim=0)


def append_csv_row(path: Path, fieldnames: list[str], row: dict) -> None:
    file_exists = path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def train_one_combo(
    args: argparse.Namespace,
    combo_id: int,
    first_omega_0: float,
    hidden_omega_0: float,
    full_coords: torch.Tensor,
    full_values: torch.Tensor,
    gt_image_01: torch.Tensor,
    gt_uint8: np.ndarray,
    device: torch.device,
    out_root: Path,
) -> dict[str, float]:
    set_seed(args.seed)

    combo_name = (
        f"combo{combo_id:03d}_"
        f"f{omega_to_name(first_omega_0)}_"
        f"h{omega_to_name(hidden_omega_0)}"
    )

    out_dir = out_root / combo_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "combo_id": combo_id,
        "first_omega_0": first_omega_0,
        "hidden_omega_0": hidden_omega_0,
        "sidelength": args.sidelength,
        "hidden_features": args.hidden_features,
        "hidden_layers": args.hidden_layers,
        "batch_size": args.batch_size,
        "num_steps": args.num_steps,
        "lr": args.lr,
        "seed": args.seed,
        "worker_id": args.worker_id,
        "num_workers": args.num_workers,
    }

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    model = MLPINR(
        in_features=2,
        out_features=1,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        activation="sine",
        first_omega_0=first_omega_0,
        hidden_omega_0=hidden_omega_0,
        outermost_linear=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history_path = out_dir / "history.csv"

    history_fields = [
        "combo_id",
        "step",
        "first_omega_0",
        "hidden_omega_0",
        "train_mse_raw_minus1_1",
        "full_mse_raw_minus1_1",
        "mse",
        "psnr",
        "rmse",
        "mae",
        "max_abs",
        "p95_abs",
        "p99_abs",
        "bias",
        "steps_since_last_eval",
        "train_interval_sec",
        "eval_interval_sec",
        "total_interval_sec",
        "train_it_per_sec",
        "effective_it_per_sec",
        "combo_elapsed_sec",
    ]

    last_metrics = None
    last_loss = None
    last_full_mse_raw = None

    print()
    print(f"=== Combo {combo_id}: first={first_omega_0}, hidden={hidden_omega_0} ===")

    sync_if_cuda(device)
    combo_t0 = time.perf_counter()
    interval_t0 = time.perf_counter()
    last_eval_step = 0

    for step in range(1, args.num_steps + 1):
        model.train()

        batch_coords, batch_values = sample_batch(
            full_coords,
            full_values,
            args.batch_size,
        )

        pred = model(batch_coords)
        loss = F.mse_loss(pred, batch_values)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        last_loss = float(loss.item())

        if step % args.log_every == 0 or step == 1:
            print(
                f"combo {combo_id:03d} | "
                f"step {step:6d}/{args.num_steps:6d} | "
                f"train_mse_raw_minus1_1 {last_loss:.8f}"
            )

        if step % args.eval_every == 0 or step == args.num_steps:
            sync_if_cuda(device)

            train_interval_sec = time.perf_counter() - interval_t0
            steps_since_last_eval = step - last_eval_step

            if train_interval_sec > 0.0:
                train_it_per_sec = steps_since_last_eval / train_interval_sec
            else:
                train_it_per_sec = float("nan")

            eval_t0 = time.perf_counter()

            model.eval()

            full_pred = render_full_image_in_chunks(
                model=model,
                coords=full_coords,
                chunk_size=args.render_chunk_size,
            )

            full_mse_raw_minus1_1 = F.mse_loss(full_pred, full_values).item()
            last_full_mse_raw = float(full_mse_raw_minus1_1)

            pred_image_01 = normalized_to_image01(full_pred)
            metrics = compute_basic_metrics(pred_image_01, gt_image_01)
            last_metrics = metrics

            sync_if_cuda(device)

            eval_interval_sec = time.perf_counter() - eval_t0
            total_interval_sec = train_interval_sec + eval_interval_sec

            if total_interval_sec > 0.0:
                effective_it_per_sec = steps_since_last_eval / total_interval_sec
            else:
                effective_it_per_sec = float("nan")

            combo_elapsed_sec = time.perf_counter() - combo_t0

            row = {
                "combo_id": combo_id,
                "step": step,
                "first_omega_0": first_omega_0,
                "hidden_omega_0": hidden_omega_0,
                "train_mse_raw_minus1_1": last_loss,
                "full_mse_raw_minus1_1": last_full_mse_raw,
                **metrics,
                "steps_since_last_eval": steps_since_last_eval,
                "train_interval_sec": train_interval_sec,
                "eval_interval_sec": eval_interval_sec,
                "total_interval_sec": total_interval_sec,
                "train_it_per_sec": train_it_per_sec,
                "effective_it_per_sec": effective_it_per_sec,
                "combo_elapsed_sec": combo_elapsed_sec,
            }

            append_csv_row(history_path, history_fields, row)

            print(
                f"combo {combo_id:03d} | "
                f"step {step:6d} | "
                f"psnr {metrics['psnr']:.3f} dB | "
                f"mse {metrics['mse']:.8f} | "
                f"mae {metrics['mae']:.8f} | "
                f"p99 {metrics['p99_abs']:.8f} | "
                f"bias {metrics['bias']:.8f} | "
                f"train {train_it_per_sec:.2f} it/s | "
                f"effective {effective_it_per_sec:.2f} it/s | "
                f"eval {eval_interval_sec:.2f}s"
            )

            sync_if_cuda(device)
            interval_t0 = time.perf_counter()
            last_eval_step = step

    if last_metrics is None:
        raise RuntimeError("No metrics were computed. Check eval_every and num_steps.")

    if args.save_model:
        torch.save(model.state_dict(), out_dir / "model_final.pt")

    if args.save_images:
        model.eval()

        full_pred = render_full_image_in_chunks(
            model=model,
            coords=full_coords,
            chunk_size=args.render_chunk_size,
        )

        pred_image_01 = normalized_to_image01(full_pred)
        pred_uint8 = tensor_to_uint8_image(pred_image_01, args.sidelength)

        save_image(out_dir / "prediction_final.png", pred_uint8)
        save_side_by_side(out_dir / "compare_final.png", gt_uint8, pred_uint8)

    summary_row = {
        "combo_id": combo_id,
        "first_omega_0": first_omega_0,
        "hidden_omega_0": hidden_omega_0,
        "final_step": args.num_steps,
        "train_mse_raw_minus1_1": last_loss,
        "full_mse_raw_minus1_1": last_full_mse_raw,
        **last_metrics,
        "output_dir": str(out_dir),
    }

    del model
    del optimizer

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary_row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Sweep definition
    parser.add_argument(
        "--first_omegas",
        type=str,
        default="30",
        help="Comma-separated first-layer omega values.",
    )
    parser.add_argument(
        "--hidden_omegas",
        type=str,
        default="5,10,15,20,30,45,60,90",
        help="Comma-separated hidden-layer omega values.",
    )

    # Split sweep across manually launched workers
    parser.add_argument(
        "--worker_id",
        type=int,
        default=0,
        help="Worker index. For 3 workers, use 0, 1, or 2.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Total number of manually launched workers.",
    )

    # Data
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--sidelength", type=int, default=512)

    # Model
    parser.add_argument("--hidden_features", type=int, default=256)
    parser.add_argument("--hidden_layers", type=int, default=3)

    # Training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=999999999,
        help="Default is effectively full batch.",
    )
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)

    # Logging and evaluation
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--render_chunk_size", type=int, default=32768)
    parser.add_argument("--output_dir", type=str, default="omega_sweep_outputs")

    # Save options
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--save_images", action="store_true")

    # Device
    parser.add_argument("--cpu", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1")

    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        raise ValueError("--worker_id must satisfy 0 <= worker_id < num_workers")

    first_omegas = parse_float_list(args.first_omegas)
    hidden_omegas = parse_float_list(args.hidden_omegas)

    combos = []
    combo_id = 0

    for f0 in first_omegas:
        for h0 in hidden_omegas:
            combos.append((combo_id, f0, h0))
            combo_id += 1

    selected_combos = [
        combo for i, combo in enumerate(combos)
        if i % args.num_workers == args.worker_id
    ]

    if not selected_combos:
        raise ValueError("This worker has no combinations to run.")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    print(f"Using device: {device}")
    print(f"Total combos: {len(combos)}")
    print(f"This worker combos: {len(selected_combos)}")
    print(f"Worker {args.worker_id + 1}/{args.num_workers} (worker_id={args.worker_id})")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load dataset once per worker
    dataset = ImageINRDataset(
        sidelength=args.sidelength,
        image_path=args.image_path,
    )

    sample = dataset[0]

    full_coords = sample["coords"].to(device)
    full_values = sample["values"].to(device)

    gt_image_01 = normalized_to_image01(full_values)
    gt_uint8 = tensor_to_uint8_image(gt_image_01, args.sidelength)

    save_image(out_root / "target.png", gt_uint8)

    n_total = full_coords.shape[0]
    print(f"Total coordinate-value samples: {n_total}")
    print(f"Requested batch size: {args.batch_size}")

    if args.batch_size >= n_total:
        print("Training mode: full batch")
    else:
        frac = 100.0 * args.batch_size / n_total
        print(f"Training mode: minibatch ({frac:.3f}% of domain per step)")

    summary_path = out_root / f"summary_worker_{args.worker_id}.csv"

    summary_fields = [
        "combo_id",
        "first_omega_0",
        "hidden_omega_0",
        "final_step",
        "train_mse_raw_minus1_1",
        "full_mse_raw_minus1_1",
        "mse",
        "psnr",
        "rmse",
        "mae",
        "max_abs",
        "p95_abs",
        "p99_abs",
        "bias",
        "output_dir",
    ]

    worker_t0 = time.perf_counter()

    for combo_id, f0, h0 in selected_combos:
        summary_row = train_one_combo(
            args=args,
            combo_id=combo_id,
            first_omega_0=f0,
            hidden_omega_0=h0,
            full_coords=full_coords,
            full_values=full_values,
            gt_image_01=gt_image_01,
            gt_uint8=gt_uint8,
            device=device,
            out_root=out_root,
        )

        append_csv_row(summary_path, summary_fields, summary_row)

    worker_elapsed = time.perf_counter() - worker_t0

    print()
    print(f"Wrote summary to: {summary_path}")
    print(f"Worker elapsed time: {worker_elapsed:.2f} s")


if __name__ == "__main__":
    main()