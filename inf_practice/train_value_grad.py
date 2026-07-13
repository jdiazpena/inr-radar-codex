# train_value_grad.py

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


def compute_target_gradients(
    values: torch.Tensor,
    sidelength: int,
) -> torch.Tensor:
    """
    Compute finite-difference target gradients on the resized grayscale image.

    Input:
        values: [N, 1], image values in [-1, 1]
    Output:
        grads: [N, 2], ordered as [dI/dy, dI/dx]

    Coordinates live in [-1, 1], so spacing is:
        dy = dx = 2 / (sidelength - 1)
    """
    img = values.view(sidelength, sidelength)

    dy = 2.0 / (sidelength - 1)
    dx = 2.0 / (sidelength - 1)

    grad_y = torch.zeros_like(img)
    grad_x = torch.zeros_like(img)

    # Central differences for interior
    grad_y[1:-1, :] = (img[2:, :] - img[:-2, :]) / (2.0 * dy)
    grad_x[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / (2.0 * dx)

    # Forward/backward differences at borders
    grad_y[0, :] = (img[1, :] - img[0, :]) / dy
    grad_y[-1, :] = (img[-1, :] - img[-2, :]) / dy

    grad_x[:, 0] = (img[:, 1] - img[:, 0]) / dx
    grad_x[:, -1] = (img[:, -1] - img[:, -2]) / dx

    grads = torch.stack([grad_y.reshape(-1), grad_x.reshape(-1)], dim=-1)
    return grads


def gradient_magnitude_to_uint8(
    grads: torch.Tensor,
    sidelength: int,
) -> np.ndarray:
    """
    Convert flattened [N, 2] gradients into a gradient-magnitude visualization.
    """
    mag = torch.sqrt(grads[:, 0] ** 2 + grads[:, 1] ** 2)
    mag = mag.view(sidelength, sidelength)

    mag_min = mag.min()
    mag_max = mag.max()

    if mag_max > mag_min:
        mag = (mag - mag_min) / (mag_max - mag_min)
    else:
        mag = torch.zeros_like(mag)

    mag_uint8 = (mag.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return mag_uint8


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


def predict_gradients_in_chunks(
    model: torch.nn.Module,
    coords: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """
    Evaluate gradients of the model output with respect to input coords.

    Returns:
        pred_grads: [N, 2], ordered as [dI/dy, dI/dx]
    """
    grads = []
    n_samples = coords.shape[0]

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)

        chunk_coords = coords[start:end].clone().detach().requires_grad_(True)
        chunk_pred = model(chunk_coords)

        chunk_grads = torch.autograd.grad(
            outputs=chunk_pred,
            inputs=chunk_coords,
            grad_outputs=torch.ones_like(chunk_pred),
            create_graph=False,
            retain_graph=False,
        )[0]

        grads.append(chunk_grads.detach())

    return torch.cat(grads, dim=0)


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
    # 2) Load dataset and precompute target gradients
    # ------------------------------------------------------------------
    dataset = ImageINRDataset(
        sidelength=args.sidelength,
        image_path=args.image_path,
    )

    sample = dataset[0]
    full_coords = sample["coords"].to(device)   # [N, 2]
    full_values = sample["values"].to(device)   # [N, 1]

    full_target_grads = compute_target_gradients(
        values=full_values,
        sidelength=args.sidelength,
    ).to(device)  # [N, 2]

    n_total = full_coords.shape[0]
    print(f"Total coordinate-value samples: {n_total}")

    # Save target image and target gradient magnitude once
    gt_image_01 = normalized_to_image01(full_values)
    gt_uint8 = tensor_to_uint8_image(gt_image_01, args.sidelength)
    save_image(out_dir / "target.png", gt_uint8)

    gt_gradmag_uint8 = gradient_magnitude_to_uint8(full_target_grads, args.sidelength)
    save_image(out_dir / "target_gradmag.png", gt_gradmag_uint8)

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

        batch_coords = full_coords[idx].clone().detach().requires_grad_(True)   # [B, 2]
        batch_target_values = full_values[idx]                                   # [B, 1]
        batch_target_grads = full_target_grads[idx]                              # [B, 2]

        pred = model(batch_coords)                                               # [B, 1]

        pred_grads = torch.autograd.grad(
            outputs=pred,
            inputs=batch_coords,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
            retain_graph=True,
        )[0]  # [B, 2]

        value_loss = F.mse_loss(pred, batch_target_values)
        grad_loss = F.mse_loss(pred_grads, batch_target_grads)

        total_loss = value_loss + args.grad_weight * grad_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            pct = 100.0 * local_step / args.num_steps
            msg = (
                f"step {step:6d} "
                f"(this run {local_step:6d}/{args.num_steps:6d}, {pct:6.2f}%) | "
                f"value_mse {value_loss.item():.8f} | "
                f"grad_mse {grad_loss.item():.8f} | "
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

            full_pred_grads = predict_gradients_in_chunks(
                model=model,
                coords=full_coords,
                chunk_size=args.render_chunk_size,
            )  # [N, 2]

            pred_image_01 = normalized_to_image01(full_pred)

            full_image_mse = F.mse_loss(pred_image_01, gt_image_01).item()
            full_psnr = compute_psnr_from_mse(full_image_mse)
            full_grad_mse = F.mse_loss(full_pred_grads, full_target_grads).item()

            pred_uint8 = tensor_to_uint8_image(pred_image_01, args.sidelength)
            save_image(out_dir / f"recon_step_{step:06d}.png", pred_uint8)
            save_side_by_side(
                out_dir / f"compare_step_{step:06d}.png",
                gt_uint8,
                pred_uint8,
            )

            pred_gradmag_uint8 = gradient_magnitude_to_uint8(full_pred_grads, args.sidelength)
            save_image(out_dir / f"gradmag_step_{step:06d}.png", pred_gradmag_uint8)
            save_side_by_side(
                out_dir / f"compare_gradmag_step_{step:06d}.png",
                gt_gradmag_uint8,
                pred_gradmag_uint8,
            )

            ckpt_path = out_dir / f"checkpoint_step_{step:06d}.pt"
            save_checkpoint(ckpt_path, model, step, args)

            print()
            print(
                f"step {step:6d} | "
                f"value_mse {value_loss.item():.8f} | "
                f"grad_mse {grad_loss.item():.8f} | "
                f"full_grad_mse {full_grad_mse:.8f} | "
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
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="Combined value+gradient training is heavier than value-only training.")
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--grad_weight", type=float, default=0.1,
                        help="Weight multiplying the gradient loss.")

    # Logging / rendering
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--summary_every", type=int, default=500)
    parser.add_argument("--render_chunk_size", type=int, default=16384)
    parser.add_argument("--output_dir", type=str, default="outputs_value_grad")

    # Reproducibility / device
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)