# eval_inr_512.py

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


def normalized_to_image01(x: torch.Tensor) -> torch.Tensor:
    """
    Convert values from [-1, 1] to [0, 1].
    """
    return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)


def tensor_to_uint8_image(x01: torch.Tensor, sidelength: int) -> np.ndarray:
    """
    Convert flattened [N, 1] tensor in [0, 1] to uint8 grayscale image [H, W].
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


@torch.no_grad()
def render_full_image_in_chunks(
    model: torch.nn.Module,
    coords: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """
    Evaluate model on full coordinate grid in chunks.

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


def save_image(path: Path, img_uint8: np.ndarray) -> None:
    Image.fromarray(img_uint8, mode="L").save(path)


def save_side_by_side(path: Path, left_uint8: np.ndarray, right_uint8: np.ndarray) -> None:
    panel = np.concatenate([left_uint8, right_uint8], axis=1)
    Image.fromarray(panel, mode="L").save(path)


def save_error_map(path: Path, abs_err_01: torch.Tensor, sidelength: int) -> None:
    """
    Save absolute error map in grayscale.
    It is min-max normalized for visualization only.
    """
    err = abs_err_01.view(sidelength, sidelength).detach().cpu()

    err_min = err.min()
    err_max = err.max()

    if err_max > err_min:
        err_vis = (err - err_min) / (err_max - err_min)
    else:
        err_vis = torch.zeros_like(err)

    err_uint8 = (err_vis.numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(err_uint8, mode="L").save(path)


def load_model_weights_or_checkpoint(
    model: torch.nn.Module,
    resume_path: Path,
    device: torch.device,
) -> None:
    """
    Load either:
    - a plain state_dict, or
    - a checkpoint dict containing model_state_dict
    """
    obj = torch.load(resume_path, map_location=device)

    if isinstance(obj, dict) and "model_state_dict" in obj:
        model.load_state_dict(obj["model_state_dict"])
        print(f"Loaded checkpoint from: {resume_path}")
        return

    model.load_state_dict(obj)
    print(f"Loaded plain model weights from: {resume_path}")


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the evaluation dataset at 512x512 (or whatever eval_sidelength is)
    dataset = ImageINRDataset(
        sidelength=args.eval_sidelength,
        image_path=args.image_path,
    )

    sample = dataset[0]
    full_coords = sample["coords"].to(device)   # [N, 2]
    full_values = sample["values"].to(device)   # [N, 1], the true target at eval resolution

    # Build the same architecture used in training
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

    load_model_weights_or_checkpoint(model, Path(args.resume), device)
    model.eval()

    # Render prediction on the denser grid
    pred = render_full_image_in_chunks(
        model=model,
        coords=full_coords,
        chunk_size=args.render_chunk_size,
    )  # [N, 1]

    # Convert to [0,1] for metrics and saving
    pred_01 = normalized_to_image01(pred)
    target_01 = normalized_to_image01(full_values)

    mse = F.mse_loss(pred_01, target_01).item()
    mae = F.l1_loss(pred_01, target_01).item()
    psnr = compute_psnr_from_mse(mse)
    max_abs = torch.max(torch.abs(pred_01 - target_01)).item()

    print(f"Evaluation resolution: {args.eval_sidelength}x{args.eval_sidelength}")
    print(f"MSE     : {mse:.8f}")
    print(f"MAE     : {mae:.8f}")
    print(f"PSNR    : {psnr:.3f} dB")
    print(f"Max abs : {max_abs:.8f}")

    # Save images
    pred_uint8 = tensor_to_uint8_image(pred_01, args.eval_sidelength)
    target_uint8 = tensor_to_uint8_image(target_01, args.eval_sidelength)

    save_image(out_dir / "prediction.png", pred_uint8)
    save_image(out_dir / "target.png", target_uint8)
    save_side_by_side(out_dir / "compare.png", target_uint8, pred_uint8)

    abs_err = torch.abs(pred_01 - target_01)
    save_error_map(out_dir / "abs_error_map.png", abs_err, args.eval_sidelength)

    print(f"Saved outputs to: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--resume", type=str, required=True,
                        help="Path to trained model .pt or checkpoint .pt")

    parser.add_argument("--image_path", type=str, default=None,
                        help="Optional grayscale image path. If omitted, use built-in camera image.")
    parser.add_argument("--eval_sidelength", type=int, default=512,
                        help="Resolution at which to evaluate the INR.")

    parser.add_argument("--activation", type=str, default="sine",
                        choices=["relu", "tanh", "softplus", "sine"])
    parser.add_argument("--hidden_features", type=int, default=256)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--first_omega_0", type=float, default=30.0)
    parser.add_argument("--hidden_omega_0", type=float, default=1.0)

    parser.add_argument("--render_chunk_size", type=int, default=32768)
    parser.add_argument("--output_dir", type=str, default="eval_outputs_512")

    parser.add_argument("--cpu", action="store_true")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    evaluate(args)