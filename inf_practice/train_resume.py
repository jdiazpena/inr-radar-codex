# train.py

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


def train(args: argparse.Namespace) -> None:
    # ------------------------------------------------------------------
    # 1) Setup
    # ------------------------------------------------------------------
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2) Load dataset
    # ------------------------------------------------------------------
    dataset = ImageINRDataset(
        sidelength=args.sidelength,
        image_path=args.image_path,
    )

    sample = dataset[0]
    full_coords = sample["coords"].to(device)   # [N, 2]
    full_values = sample["values"].to(device)   # [N, 1]

    n_total = full_coords.shape[0]
    print(f"Total coordinate-value samples: {n_total}")

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

    if args.resume is not None:
        resume_path = Path(args.resume)
        state_dict = torch.load(resume_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from: {resume_path}")

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
        loss = F.mse_loss(pred, batch_values)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Basic loss logging
        if step % args.log_every == 0 or step == 1:
            print(f"step {step:6d} | batch_mse {loss.item():.8f}")

        # Full-image evaluation and saving
        if step % args.summary_every == 0 or step == args.num_steps:
            model.eval()

            full_pred = render_full_image_in_chunks(
                model=model,
                coords=full_coords,
                sidelength=args.sidelength,
                chunk_size=args.render_chunk_size,
            )  # [N, 1]

            pred_image_01 = normalized_to_image01(full_pred)

            full_mse = F.mse_loss(pred_image_01, gt_image_01).item()
            full_psnr = compute_psnr_from_mse(full_mse)

            pred_uint8 = tensor_to_uint8_image(pred_image_01, args.sidelength)
            save_image(out_dir / f"recon_step_{step:06d}.png", pred_uint8)
            save_side_by_side(
                out_dir / f"compare_step_{step:06d}.png",
                gt_uint8,
                pred_uint8,
            )

            print(
                f"step {step:6d} | "
                f"batch_mse {loss.item():.8f} | "
                f"full_mse {full_mse:.8f} | "
                f"psnr {full_psnr:.3f} dB"
            )

    # ------------------------------------------------------------------
    # 5) Final save
    # ------------------------------------------------------------------
    final_model_path = out_dir / "model_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to: {final_model_path}")


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
    parser.add_argument("--hidden_omega_0", type=float, default=1.0,
                        help="Used only for sine/SIREN-style networks.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a saved model .pt file to continue training from.")


    # Training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--num_steps", type=int, default=5000)

    # Logging / rendering
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--summary_every", type=int, default=500)
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