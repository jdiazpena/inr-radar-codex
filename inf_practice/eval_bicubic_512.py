# eval_bicubic_512.py

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

from datasets import ImageINRDataset


def normalized_to_image01(x):
    """
    Convert values from [-1, 1] to [0, 1].
    """
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def tensor_to_uint8_image(x01, sidelength: int) -> np.ndarray:
    """
    Convert flattened [N, 1] tensor in [0, 1] to uint8 grayscale image [H, W].
    """
    img = x01.view(sidelength, sidelength).detach().cpu().numpy()
    img = (img * 255.0).round().clip(0, 255).astype(np.uint8)
    return img


def compute_psnr_from_mse(mse: float) -> float:
    if mse <= 0.0:
        return float("inf")
    return -10.0 * math.log10(mse)


def save_image(path: Path, img_uint8: np.ndarray) -> None:
    Image.fromarray(img_uint8, mode="L").save(path)


def save_side_by_side(path: Path, left_uint8: np.ndarray, right_uint8: np.ndarray) -> None:
    panel = np.concatenate([left_uint8, right_uint8], axis=1)
    Image.fromarray(panel, mode="L").save(path)


def save_error_map(path: Path, abs_err_01: np.ndarray) -> None:
    """
    Save absolute error map in grayscale, min-max normalized for visualization.
    """
    err_min = abs_err_01.min()
    err_max = abs_err_01.max()

    if err_max > err_min:
        err_vis = (abs_err_01 - err_min) / (err_max - err_min)
    else:
        err_vis = np.zeros_like(abs_err_01)

    err_uint8 = (err_vis * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(err_uint8, mode="L").save(path)


def evaluate(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # This is the exact training-resolution target the INR saw
    ds_train = ImageINRDataset(
        sidelength=args.train_sidelength,
        image_path=args.image_path,
    )
    train_sample = ds_train[0]
    train_values_01 = normalized_to_image01(train_sample["values"])
    train_uint8 = tensor_to_uint8_image(train_values_01, args.train_sidelength)

    # This is the true evaluation-resolution target
    ds_eval = ImageINRDataset(
        sidelength=args.eval_sidelength,
        image_path=args.image_path,
    )
    eval_sample = ds_eval[0]
    eval_values_01 = normalized_to_image01(eval_sample["values"])
    eval_uint8 = tensor_to_uint8_image(eval_values_01, args.eval_sidelength)

    # Bicubic upsample: 256 -> 512
    train_img = Image.fromarray(train_uint8, mode="L")
    bicubic_img = train_img.resize(
        (args.eval_sidelength, args.eval_sidelength),
        resample=Image.BICUBIC,
    )
    bicubic_uint8 = np.array(bicubic_img, dtype=np.uint8)

    # Metrics in [0,1]
    target_01 = eval_uint8.astype(np.float32) / 255.0
    bicubic_01 = bicubic_uint8.astype(np.float32) / 255.0

    diff = bicubic_01 - target_01
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    psnr = compute_psnr_from_mse(mse)
    max_abs = float(np.max(np.abs(diff)))

    print(f"Train resolution : {args.train_sidelength}x{args.train_sidelength}")
    print(f"Eval resolution  : {args.eval_sidelength}x{args.eval_sidelength}")
    print(f"MSE     : {mse:.8f}")
    print(f"MAE     : {mae:.8f}")
    print(f"PSNR    : {psnr:.3f} dB")
    print(f"Max abs : {max_abs:.8f}")

    save_image(out_dir / "train_target.png", train_uint8)
    save_image(out_dir / "eval_target.png", eval_uint8)
    save_image(out_dir / "bicubic_prediction.png", bicubic_uint8)
    save_side_by_side(out_dir / "compare.png", eval_uint8, bicubic_uint8)

    abs_err = np.abs(diff)
    save_error_map(out_dir / "abs_error_map.png", abs_err)

    print(f"Saved outputs to: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, default=None,
                        help="Optional grayscale image path. If omitted, use built-in camera image.")
    parser.add_argument("--train_sidelength", type=int, default=256)
    parser.add_argument("--eval_sidelength", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="eval_bicubic_512")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    evaluate(args)