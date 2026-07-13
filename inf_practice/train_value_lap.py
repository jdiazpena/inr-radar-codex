# train_value_lap.py

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from datasets import ImageINRDataset
from models import MLPINR


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def sample_batch_indices(
    n_samples: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample random row indices for minibatch training.
    """
    if batch_size >= n_samples:
        return torch.arange(n_samples, device=device)
    return torch.randperm(n_samples, device=device)[:batch_size]


def normalized_to_image01(x: torch.Tensor) -> torch.Tensor:
    """
    Convert values from [-1, 1] to [0, 1].
    """
    return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)


def tensor_to_uint8_image(x01: torch.Tensor, sidelength: int) -> np.ndarray:
    """
    Convert flattened [N, 1] image tensor in [0, 1] to uint8 [H, W].
    """
    img = x01.view(sidelength, sidelength).detach().cpu().numpy()
    img = (img * 255.0).round().clip(0, 255).astype(np.uint8)
    return img


def compute_psnr_from_mse(mse: float) -> float:
    """
    PSNR assuming pixel values are in [0, 1].
    """
    if mse <= 0.0:
        return float("inf")
    return -10.0 * math.log10(mse)


def compute_target_laplacian(
    values: torch.Tensor,
    sidelength: int,
) -> torch.Tensor:
    """
    Compute finite-difference Laplacian on the resized grayscale image.

    Input:
        values: [N, 1], image values in [-1, 1]

    Output:
        lap: [N, 1]

    Coordinates live in [-1, 1], so spacing is:
        dy = dx = 2 / (sidelength - 1)
    """
    img = values.view(sidelength, sidelength)

    dy = 2.0 / (sidelength - 1)
    dx = 2.0 / (sidelength - 1)

    d2y = torch.zeros_like(img)
    d2x = torch.zeros_like(img)

    # Interior: central second differences
    d2y[1:-1, :] = (img[2:, :] - 2.0 * img[1:-1, :] + img[:-2, :]) / (dy * dy)
    d2x[:, 1:-1] = (img[:, 2:] - 2.0 * img[:, 1:-1] + img[:, :-2]) / (dx * dx)

    # Borders: second-order one-sided second differences
    # Need at least 4 points, which is fine for any reasonable sidelength here.
    d2y[0, :] = (2.0 * img[0, :] - 5.0 * img[1, :] + 4.0 * img[2, :] - img[3, :]) / (dy * dy)
    d2y[-1, :] = (2.0 * img[-1, :] - 5.0 * img[-2, :] + 4.0 * img[-3, :] - img[-4, :]) / (dy * dy)

    d2x[:, 0] = (2.0 * img[:, 0] - 5.0 * img[:, 1] + 4.0 * img[:, 2] - img[:, 3]) / (dx * dx)
    d2x[:, -1] = (2.0 * img[:, -1] - 5.0 * img[:, -2] + 4.0 * img[:, -3] - img[:, -4]) / (dx * dx)

    lap = d2y + d2x
    return lap.reshape(-1, 1)


def signed_field_to_uint8(
    field: torch.Tensor,
    sidelength: int,
    vmax: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    Visualize a signed scalar field as grayscale with zero mapped to mid-gray.

    Output mapping:
        -vmax -> 0
         0    -> 127
        +vmax -> 255

    If vmax is None, use max(abs(field)).
    """
    img = field.view(sidelength, sidelength).detach().cpu()

    if vmax is None:
        vmax = float(img.abs().max().item())
        if vmax <= 0.0:
            vmax = 1.0

    scaled = 0.5 + 0.5 * (img / vmax)
    scaled = torch.clamp(scaled, 0.0, 1.0)

    img_uint8 = (scaled.numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return img_uint8, vmax


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
    """
    Evaluate the model on the full coordinate grid in chunks.

    Returns:
        pred_values: [N, 1]
    """
    outputs = []
    n_samples = coords.shape[0]

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk_coords = coords[start:end]
        chunk_pred = model(chunk_coords)
        outputs.append(chunk_pred)

    return torch.cat(outputs, dim=0)


def predict_laplacian_in_chunks(
    model: torch.nn.Module,
    coords: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """
    Evaluate the Laplacian of the model output with respect to input coords.

    coords are ordered as [y, x].

    Returns:
        pred_lap: [N, 1]
    """
    laps = []
    n_samples = coords.shape[0]

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)

        chunk_coords = coords[start:end].clone().detach().requires_grad_(True)
        chunk_pred = model(chunk_coords)  # [M, 1]

        first_grad = torch.autograd.grad(
            outputs=chunk_pred,
            inputs=chunk_coords,
            grad_outputs=torch.ones_like(chunk_pred),
            create_graph=True,
            retain_graph=True,
        )[0]  # [M, 2]

        d2y = torch.autograd.grad(
            outputs=first_grad[:, 0],
            inputs=chunk_coords,
            grad_outputs=torch.ones_like(first_grad[:, 0]),
            create_graph=False,
            retain_graph=True,
        )[0][:, 0:1]

        d2x = torch.autograd.grad(
            outputs=first_grad[:, 1],
            inputs=chunk_coords,
            grad_outputs=torch.ones_like(first_grad[:, 1]),
            create_graph=False,
            retain_graph=False,
        )[0][:, 1:2]

        chunk_lap = d2y + d2x
        laps.append(chunk_lap.detach())

    return torch.cat(laps, dim=0)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    step: int,
    args: argparse.Namespace,
) -> None:
    """
    Save a checkpoint with model weights and metadata.
    """
    ckpt = {
        "model_state_dict": model.state_dict(),
        "step": step,
        "args": vars(args),
    }
    torch.save(ckpt, path)


def load_model_weights_or_checkpoint(
    model: torch.nn.Module,
    resume_path: Path,
    device: torch.device,
) -> int:
    """
    Load either:
    - a plain state_dict, or
    - a checkpoint dict containing model_state_dict and step

    Returns:
        start_step (0 if unknown)
    """
    obj = torch.load(resume_path, map_location=device)

    if isinstance(obj, dict) and "model_state_dict" in obj:
        model.load_state_dict(obj["model_state_dict"])
        start_step = int(obj.get("step", 0))
        print(f"Loaded checkpoint from: {resume_path} (saved at step {start_step})")
        return start_step

    model.load_state_dict(obj)
    print(f"Loaded plain model weights from: {resume_path}")
    return 0


def train(args: argparse.Namespace) -> None:
    # ------------------------------------------------------------------
    # 1) Setup
    # ------------------------------------------------------------------
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        _ = torch.empty(1, device=device)
        torch.cuda.synchronize()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2) Load dataset and precompute target Laplacian
    # ------------------------------------------------------------------
    dataset = ImageINRDataset(
        sidelength=args.sidelength,
        image_path=args.image_path,
    )

    sample = dataset[0]
    full_coords = sample["coords"].to(device)   # [N, 2]
    full_values = sample["values"].to(device)   # [N, 1]

    full_target_lap = compute_target_laplacian(
        values=full_values,
        sidelength=args.sidelength,
    ).to(device)  # [N, 1]

    n_total = full_coords.shape[0]
    print(f"Total coordinate-value samples: {n_total}")

    # Save target image and target Laplacian visualization once
    gt_image_01 = normalized_to_image01(full_values)
    gt_uint8 = tensor_to_uint8_image(gt_image_01, args.sidelength)
    save_image(out_dir / "target.png", gt_uint8)

    gt_lap_uint8, lap_vmax = signed_field_to_uint8(full_target_lap, args.sidelength)
    save_image(out_dir / "target_laplacian.png", gt_lap_uint8)

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

    start_step = 0
    if args.resume is not None:
        resume_path = Path(args.resume)
        start_step = load_model_weights_or_checkpoint(model, resume_path, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------------
    # 4) Training loop
    # ------------------------------------------------------------------
    for local_step in range(1, args.num_steps + 1):
        step = start_step + local_step
        model.train()

        idx = sample_batch_indices(
            n_samples=n_total,
            batch_size=args.batch_size,
            device=device,
        )

        batch_coords = full_coords[idx].clone().detach().requires_grad_(True)  # [B, 2]
        batch_target_values = full_values[idx]                                  # [B, 1]
        batch_target_lap = full_target_lap[idx]                                 # [B, 1]

        pred = model(batch_coords)                                              # [B, 1]

        first_grad = torch.autograd.grad(
            outputs=pred,
            inputs=batch_coords,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
            retain_graph=True,
        )[0]  # [B, 2]

        d2y = torch.autograd.grad(
            outputs=first_grad[:, 0],
            inputs=batch_coords,
            grad_outputs=torch.ones_like(first_grad[:, 0]),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]

        d2x = torch.autograd.grad(
            outputs=first_grad[:, 1],
            inputs=batch_coords,
            grad_outputs=torch.ones_like(first_grad[:, 1]),
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]

        pred_lap = d2y + d2x  # [B, 1]

        value_loss = F.mse_loss(pred, batch_target_values)
        lap_loss = F.mse_loss(pred_lap, batch_target_lap)

        total_loss = value_loss + args.lap_weight * lap_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            pct = 100.0 * local_step / args.num_steps
            msg = (
                f"step {step:6d} "
                f"(this run {local_step:6d}/{args.num_steps:6d}, {pct:6.2f}%) | "
                f"value_mse {value_loss.item():.8f} | "
                f"lap_mse {lap_loss.item():.8f} | "
                f"total {total_loss.item():.8f}"
            )
            print(f"\r{msg}", end="", flush=True)

        # Periodic full-image evaluation
        if step % args.summary_every == 0 or local_step == args.num_steps:
            model.eval()

            full_pred = render_full_image_in_chunks(
                model=model,
                coords=full_coords,
                chunk_size=args.render_chunk_size,
            )  # [N, 1]

            full_pred_lap = predict_laplacian_in_chunks(
                model=model,
                coords=full_coords,
                chunk_size=args.render_chunk_size,
            )  # [N, 1]

            pred_image_01 = normalized_to_image01(full_pred)

            full_image_mse = F.mse_loss(pred_image_01, gt_image_01).item()
            full_psnr = compute_psnr_from_mse(full_image_mse)
            full_lap_mse = F.mse_loss(full_pred_lap, full_target_lap).item()

            pred_uint8 = tensor_to_uint8_image(pred_image_01, args.sidelength)
            save_image(out_dir / f"recon_step_{step:06d}.png", pred_uint8)
            save_side_by_side(
                out_dir / f"compare_step_{step:06d}.png",
                gt_uint8,
                pred_uint8,
            )

            pred_lap_uint8, _ = signed_field_to_uint8(
                full_pred_lap,
                args.sidelength,
                vmax=lap_vmax,
            )
            save_image(out_dir / f"laplacian_step_{step:06d}.png", pred_lap_uint8)
            save_side_by_side(
                out_dir / f"compare_laplacian_step_{step:06d}.png",
                gt_lap_uint8,
                pred_lap_uint8,
            )

            ckpt_path = out_dir / f"checkpoint_step_{step:06d}.pt"
            save_checkpoint(ckpt_path, model, step, args)

            print()
            print(
                f"step {step:6d} | "
                f"value_mse {value_loss.item():.8f} | "
                f"lap_mse {lap_loss.item():.8f} | "
                f"full_lap_mse {full_lap_mse:.8f} | "
                f"full_image_mse {full_image_mse:.8f} | "
                f"psnr {full_psnr:.3f} dB | "
                f"saved {ckpt_path.name}"
            )

    print()
    final_model_path = out_dir / "model_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to: {final_model_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--image_path", type=str, default=None,
                        help="Path to grayscale image. If omitted, use built-in camera image.")
    parser.add_argument("--sidelength", type=int, default=256)

    # Model
    parser.add_argument("--activation", type=str, default="sine",
                        choices=["relu", "tanh", "softplus", "sine"])
    parser.add_argument("--hidden_features", type=int, default=256)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--first_omega_0", type=float, default=30.0)
    parser.add_argument("--hidden_omega_0", type=float, default=1.0)

    # Resume / fine-tuning
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to saved model or checkpoint to continue from.")

    # Training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="Value+Laplacian training is heavier than value+gradient.")
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--lap_weight", type=float, default=1e-6,
                        help="Weight multiplying the Laplacian loss.")

    # Logging / rendering
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--summary_every", type=int, default=500)
    parser.add_argument("--render_chunk_size", type=int, default=8192)
    parser.add_argument("--output_dir", type=str, default="outputs_value_lap")

    # Reproducibility / device
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)