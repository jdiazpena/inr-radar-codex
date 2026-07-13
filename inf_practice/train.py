## train.py

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from datasets import ImageINRDataset
from models import MLPINR


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


def sample_batch(
    coords: torch.Tensor,
    values: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly sample a minibatch of coordinate-value pairs.

    coords: [N, 2]
    values: [N, 1]
    """
    n_samples = coords.shape[0]

    if batch_size >= n_samples:
        idx = torch.arange(n_samples, device=coords.device)
    else:
        idx = torch.randperm(n_samples, device=coords.device)[:batch_size]

    batch_coords = coords[idx]
    batch_values = values[idx]
    return batch_coords, batch_values


def normalized_to_image01(x: torch.Tensor) -> torch.Tensor:
    """
    Convert values from [-1, 1] to [0, 1].
    """
    return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)


def tensor_to_uint8_image(x01: torch.Tensor, sidelength: int) -> np.ndarray:
    """
    Convert a flattened [N, 1] tensor in [0, 1] into a uint8 image [H, W].
    """
    img = x01.view(sidelength, sidelength).detach().cpu().numpy()
    img = (img * 255.0).round().clip(0, 255).astype(np.uint8)
    return img


def compute_psnr_from_mse(mse: float) -> float:
    """
    Compute PSNR assuming image values are in [0, 1].
    """
    if mse <= 0.0:
        return float("inf")
    return -10.0 * math.log10(mse)


def compute_basic_metrics(pred_01: torch.Tensor, target_01: torch.Tensor) -> dict[str, float]:
    """
    Compute reconstruction metrics on [0, 1] values.

    pred_01:   [N, 1]
    target_01: [N, 1]
    """
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
        "rmse": float(rmse.item()),
        "mae": float(mae.item()),
        "max_abs": float(max_abs.item()),
        "p95_abs": float(p95_abs.item()),
        "p99_abs": float(p99_abs.item()),
        "bias": float(bias.item()),
        "psnr": compute_psnr_from_mse(mse_value),
    }


@torch.no_grad()
def render_full_image_in_chunks(
    model: torch.nn.Module,
    coords: torch.Tensor,
    sidelength: int,
    chunk_size: int,
) -> torch.Tensor:
    """
    Render the full image by evaluating the model on all coordinates in chunks.

    Returns:
        pred_values: [N, 1] in the model's output range, typically around [-1, 1]
    """
    outputs = []
    n_samples = coords.shape[0]

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk_coords = coords[start:end]
        chunk_pred = model(chunk_coords)
        outputs.append(chunk_pred)

    pred_values = torch.cat(outputs, dim=0)
    return pred_values


def save_image(path: Path, img_uint8: np.ndarray) -> None:
    """
    Save a single grayscale image.
    """
    Image.fromarray(img_uint8, mode="L").save(path)


def save_side_by_side(
    path: Path,
    gt_uint8: np.ndarray,
    pred_uint8: np.ndarray,
) -> None:
    """
    Save a side-by-side comparison image: [ground truth | prediction]
    """
    panel = np.concatenate([gt_uint8, pred_uint8], axis=1)
    Image.fromarray(panel, mode="L").save(path)


def append_csv_row(path: Path, fieldnames: list[str], row: dict) -> None:
    """
    Append one row to a CSV file.
    If the file does not exist, write the header first.
    """
    file_exists = path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def load_model_if_requested(
    model: torch.nn.Module,
    resume_path: str | None,
    device: torch.device,
) -> None:
    """
    Load saved model weights if --resume is provided.

    Supports:
    - plain state_dict files
    - checkpoint dictionaries containing "model_state_dict"
    """
    if resume_path is None:
        return

    path = Path(resume_path)
    obj = torch.load(path, map_location=device)

    if isinstance(obj, dict) and "model_state_dict" in obj:
        model.load_state_dict(obj["model_state_dict"])
        print(f"Loaded checkpoint model_state_dict from: {path}")
    else:
        model.load_state_dict(obj)
        print(f"Loaded plain model weights from: {path}")


def train(args: argparse.Namespace) -> None:
    # ------------------------------------------------------------------
    # 1) Setup
    # ------------------------------------------------------------------
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = out_dir / "history.csv"
    history_fields = [
        "step",
        "activation",
        "sidelength",
        "hidden_features",
        "hidden_layers",
        "first_omega_0",
        "hidden_omega_0",
        "lr",
        "batch_size",
        "num_steps",
        "seed",
        "batch_mse_raw_minus1_1",
        "full_mse_raw_minus1_1",
        "full_mse_01",
        "psnr",
        "rmse",
        "mae",
        "max_abs",
        "p95_abs",
        "p99_abs",
        "bias",
    ]

    print(
        "Run config: "
        f"activation={args.activation}, "
        f"sidelength={args.sidelength}, "
        f"hidden_features={args.hidden_features}, "
        f"hidden_layers={args.hidden_layers}, "
        f"first_omega_0={args.first_omega_0}, "
        f"hidden_omega_0={args.hidden_omega_0}, "
        f"lr={args.lr}, "
        f"batch_size={args.batch_size}, "
        f"num_steps={args.num_steps}, "
        f"seed={args.seed}"
    )

    # ------------------------------------------------------------------
    # 2) Load dataset
    # ------------------------------------------------------------------
    dataset = ImageINRDataset(
        sidelength=args.sidelength,
        image_path=args.image_path,
    )

    sample = dataset[0]
    full_coords = sample["coords"].to(device)   # [N, 2]
    full_values = sample["values"].to(device)   # [N, 1], values in [-1, 1]

    n_total = full_coords.shape[0]
    print(f"Total coordinate-value samples: {n_total}")

    if args.batch_size >= n_total:
        print("Training mode: full batch")
    else:
        frac = 100.0 * args.batch_size / n_total
        print(f"Training mode: minibatch ({frac:.3f}% of domain per step)")

    # Save target image once
    gt_image_01 = normalized_to_image01(full_values)
    gt_uint8 = tensor_to_uint8_image(gt_image_01, args.sidelength)
    save_image(out_dir / "target.png", gt_uint8)

    # ------------------------------------------------------------------
    # 3) Build model
    # ------------------------------------------------------------------
    model = MLPINR(
        in_features=2,
        out_features=1,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        activation=args.activation,
        first_omega_0=args.first_omega_0,
        hidden_omega_0=args.hidden_omega_0,
        outermost_linear=True,
    ).to(device)

    load_model_if_requested(model, args.resume, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------------
    # 4) Training loop
    # ------------------------------------------------------------------
    for step in range(1, args.num_steps + 1):
        model.train()

        batch_coords, batch_values = sample_batch(
            full_coords,
            full_values,
            args.batch_size,
        )

        pred = model(batch_coords)                          # [B, 1]
        loss = F.mse_loss(pred, batch_values)               # MSE on [-1, 1]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Basic loss logging
        if step % args.log_every == 0 or step == 1:
            print(f"step {step:6d} | batch_mse_raw_minus1_1 {loss.item():.8f}")

        # Full-image evaluation, saving, and CSV logging
        if step % args.summary_every == 0 or step == args.num_steps:
            model.eval()

            full_pred = render_full_image_in_chunks(
                model=model,
                coords=full_coords,
                sidelength=args.sidelength,
                chunk_size=args.render_chunk_size,
            )  # [N, 1], usually around [-1, 1]

            # Raw full-image MSE on the same scale as training, [-1, 1]
            full_mse_raw_minus1_1 = F.mse_loss(full_pred, full_values).item()

            # Metrics on [0, 1], same scale used for image PSNR
            pred_image_01 = normalized_to_image01(full_pred)
            metrics = compute_basic_metrics(pred_image_01, gt_image_01)

            full_mse_01 = metrics["mse"]
            full_psnr = metrics["psnr"]

            pred_uint8 = tensor_to_uint8_image(pred_image_01, args.sidelength)
            save_image(out_dir / f"recon_step_{step:06d}.png", pred_uint8)
            save_side_by_side(
                out_dir / f"compare_step_{step:06d}.png",
                gt_uint8,
                pred_uint8,
            )

            row = {
                "step": step,
                "activation": args.activation,
                "sidelength": args.sidelength,
                "hidden_features": args.hidden_features,
                "hidden_layers": args.hidden_layers,
                "first_omega_0": args.first_omega_0,
                "hidden_omega_0": args.hidden_omega_0,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "num_steps": args.num_steps,
                "seed": args.seed,
                "batch_mse_raw_minus1_1": float(loss.item()),
                "full_mse_raw_minus1_1": float(full_mse_raw_minus1_1),
                "full_mse_01": float(full_mse_01),
                "psnr": float(full_psnr),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "max_abs": metrics["max_abs"],
                "p95_abs": metrics["p95_abs"],
                "p99_abs": metrics["p99_abs"],
                "bias": metrics["bias"],
            }
            append_csv_row(history_path, history_fields, row)

            print(
                f"step {step:6d} | "
                f"batch_mse_raw_minus1_1 {loss.item():.8f} | "
                f"full_mse_raw_minus1_1 {full_mse_raw_minus1_1:.8f} | "
                f"full_mse_01 {full_mse_01:.8f} | "
                f"rmse {metrics['rmse']:.8f} | "
                f"mae {metrics['mae']:.8f} | "
                f"psnr {full_psnr:.3f} dB | "
                f"max_abs {metrics['max_abs']:.8f} | "
                f"p95_abs {metrics['p95_abs']:.8f} | "
                f"p99_abs {metrics['p99_abs']:.8f} | "
                f"bias {metrics['bias']:.8f}"
            )

    # ------------------------------------------------------------------
    # 5) Final save
    # ------------------------------------------------------------------
    final_model_path = out_dir / "model_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to: {final_model_path}")
    print(f"Saved metric history to: {history_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--image_path", type=str, default=None,
                        help="Path to a grayscale image. If omitted, use built-in camera image.")
    parser.add_argument("--sidelength", type=int, default=256,
                        help="Resize image to sidelength x sidelength.")

    # Model
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "tanh", "softplus", "sine"])
    parser.add_argument("--hidden_features", type=int, default=256)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--first_omega_0", type=float, default=30.0,
                        help="Used only for sine/SIREN-style networks.")
    parser.add_argument("--hidden_omega_0", type=float, default=30.0,
                        help="Used only for sine/SIREN-style networks.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a saved model .pt file to continue training from.")

    # Training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--num_steps", type=int, default=5000)

    # Logging / rendering
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--summary_every", type=int, default=500,
                        help="Evaluate full image, save images, and append metrics to history.csv every N steps.")
    parser.add_argument("--render_chunk_size", type=int, default=32768)
    parser.add_argument("--output_dir", type=str, default="outputs")

    # Reproducibility / device
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if CUDA is available.")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)