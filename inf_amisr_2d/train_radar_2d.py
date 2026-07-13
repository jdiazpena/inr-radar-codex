# -*- coding: utf-8 -*-
"""
train_radar_2d.py

Train a 2D INR on one prepared AMISR slice.

Current experiment:
    f_theta(x_norm, y_norm) -> normalized log10(Ne)

Important separation:
    measured radar points:
        used for the data loss

    dense x-y grid:
        used only for visualization

    valid grid mask:
        nearest-distance mask only
        no convex hull
        no polygon
        no matplotlib Path

This script uses command-line arguments because it is meant for many runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets import RadarSliceDataset
from models import MLPINR


# ============================================================
# GENERAL HELPERS
# ============================================================

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def append_csv_row(path: Path, fieldnames: list[str], row: dict) -> None:
    file_exists = path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def sample_batch(
    coords: torch.Tensor,
    values: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_samples = coords.shape[0]

    if batch_size <= 0 or batch_size >= n_samples:
        idx = torch.arange(n_samples, device=coords.device)
    else:
        idx = torch.randperm(n_samples, device=coords.device)[:batch_size]

    return coords[idx], values[idx]


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)

    residual = pred - target
    abs_residual = np.abs(residual)

    mse = float(np.mean(residual ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_residual))
    bias = float(np.mean(residual))
    max_abs = float(np.max(abs_residual))
    p95_abs = float(np.quantile(abs_residual, 0.95))
    p99_abs = float(np.quantile(abs_residual, 0.99))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "max_abs": max_abs,
        "p95_abs": p95_abs,
        "p99_abs": p99_abs,
    }


# ============================================================
# GRID AND MASK HELPERS
# ============================================================

def make_query_grid_from_points(
    x_km: np.ndarray,
    y_km: np.ndarray,
    nx: int,
    ny: int,
    padding_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Make a rectangular x-y query grid.

    This grid intentionally includes points outside the radar footprint.
    Those points are masked later with nearest-distance masking.
    """

    x_min = float(np.min(x_km))
    x_max = float(np.max(x_km))
    y_min = float(np.min(y_km))
    y_max = float(np.max(y_km))

    dx = x_max - x_min
    dy = y_max - y_min

    x_min -= padding_frac * dx
    x_max += padding_frac * dx
    y_min -= padding_frac * dy
    y_max += padding_frac * dy

    x_grid = np.linspace(x_min, x_max, nx)
    y_grid = np.linspace(y_min, y_max, ny)

    X, Y = np.meshgrid(x_grid, y_grid)

    return X, Y


def estimate_nearest_radius(
    measured_xy: np.ndarray,
    factor: float,
) -> float:
    """
    Estimate a mask radius from the spacing between measured radar points.

    The radius is:
        factor * median nearest-neighbor distance
    """

    measured_xy = np.asarray(measured_xy, dtype=float)

    diff = measured_xy[:, None, :] - measured_xy[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    np.fill_diagonal(dist, np.inf)

    nearest = np.min(dist, axis=1)
    nearest = nearest[np.isfinite(nearest)]

    if nearest.size == 0:
        raise ValueError("Could not estimate nearest-neighbor spacing.")

    median_nearest = float(np.median(nearest))

    return factor * median_nearest


def nearest_distance_mask(
    X: np.ndarray,
    Y: np.ndarray,
    measured_xy: np.ndarray,
    radius_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Valid grid points are those close enough to at least one measured radar point.

    This is intentionally simple:
        no hull
        no polygon
        no matplotlib Path

    Returns:
        mask:
            True where grid point is valid
        nearest_dist_grid:
            distance from each grid point to nearest measured radar point
    """

    grid_xy = np.column_stack([X.ravel(), Y.ravel()])
    measured_xy = np.asarray(measured_xy, dtype=float)

    diff = grid_xy[:, None, :] - measured_xy[None, :, :]
    dist2 = np.sum(diff ** 2, axis=2)

    nearest_dist = np.sqrt(np.min(dist2, axis=1))
    nearest_dist_grid = nearest_dist.reshape(X.shape)

    mask = nearest_dist_grid <= radius_km

    return mask, nearest_dist_grid


def normalize_grid_with_dataset(
    dataset: RadarSliceDataset,
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """
    Normalize physical x/y grid in km using the dataset scalers.
    """

    x_min = dataset.coord_scalers["x_km"]["min"]
    x_max = dataset.coord_scalers["x_km"]["max"]

    y_min = dataset.coord_scalers["y_km"]["min"]
    y_max = dataset.coord_scalers["y_km"]["max"]

    Xn = 2.0 * (X - x_min) / (x_max - x_min) - 1.0
    Yn = 2.0 * (Y - y_min) / (y_max - y_min) - 1.0

    coords_grid = np.column_stack([Xn.ravel(), Yn.ravel()]).astype(np.float32)

    return coords_grid


@torch.no_grad()
def evaluate_model_on_grid(
    model: torch.nn.Module,
    coords_grid_np: np.ndarray,
    dataset: RadarSliceDataset,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    """
    Evaluate model on dense grid coordinates.

    Returns predictions in physical target scale, currently log10(Ne).
    """

    model.eval()

    outputs = []
    n = coords_grid_np.shape[0]

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        coords_chunk = torch.from_numpy(coords_grid_np[start:end]).to(device)
        pred_chunk = model(coords_chunk).detach().cpu().numpy()

        outputs.append(pred_chunk)

    pred_norm = np.concatenate(outputs, axis=0)
    pred_log10 = dataset.denormalize_target(pred_norm)

    return pred_log10[:, 0]


# ============================================================
# PLOTTING HELPERS
# ============================================================

def plot_measured_points(
    df: pd.DataFrame,
    out_dir: Path,
    save_plots: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    sc = ax.scatter(
        df["x_km"],
        df["y_km"],
        c=df["log10_Ne"],
        s=45,
        edgecolor="k",
        linewidth=0.4,
    )

    fig.colorbar(sc, ax=ax, label="measured log10(Ne)")

    ax.set_xlabel("x east [km]")
    ax.set_ylabel("y north [km]")
    ax.set_title("Measured AMISR slice")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        path = out_dir / "measured_points_log10Ne.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")

    plt.close(fig)


def plot_grid_prediction(
    X: np.ndarray,
    Y: np.ndarray,
    pred_grid_masked: np.ndarray,
    df: pd.DataFrame,
    out_dir: Path,
    save_plots: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    vmin = float(df["log10_Ne"].min())
    vmax = float(df["log10_Ne"].max())

    im = ax.pcolormesh(
        X,
        Y,
        pred_grid_masked,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    fig.colorbar(im, ax=ax, label="SIREN log10(Ne), masked grid")

    ax.scatter(
        df["x_km"],
        df["y_km"],
        c=df["log10_Ne"],
        s=35,
        edgecolor="k",
        linewidth=0.4,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel("x east [km]")
    ax.set_ylabel("y north [km]")
    ax.set_title("SIREN reconstruction on nearest-distance masked grid")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        path = out_dir / "siren_grid_masked_log10Ne.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")

    plt.close(fig)


def plot_residuals(
    pred_df: pd.DataFrame,
    out_dir: Path,
    save_plots: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    sc = ax.scatter(
        pred_df["x_km"],
        pred_df["y_km"],
        c=pred_df["resid_log10_Ne"],
        s=45,
        edgecolor="k",
        linewidth=0.4,
    )

    fig.colorbar(sc, ax=ax, label="prediction - measured log10(Ne)")

    ax.set_xlabel("x east [km]")
    ax.set_ylabel("y north [km]")
    ax.set_title("Residuals at measured radar points")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        path = out_dir / "measured_point_residuals_log10Ne.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")

    plt.close(fig)


def plot_history(
    history_path: Path,
    out_dir: Path,
    save_plots: bool,
) -> None:
    hist = pd.read_csv(history_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(hist["step"], hist["train_mse_norm"], marker="o", markersize=3)
    ax.set_xlabel("step")
    ax.set_ylabel("MSE on normalized log10(Ne)")
    ax.set_title("Training loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        path = out_dir / "training_history.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")

    plt.close(fig)


# ============================================================
# TRAINING
# ============================================================

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_plots = not args.no_plots

    history_path = out_dir / "history.csv"

    if history_path.exists() and not args.resume_history:
        history_path.unlink()

    config_path = out_dir / "run_config.json"

    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    print(f"Using device: {device}")

    # ------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------
    dataset = RadarSliceDataset(csv_path=args.csv_path)
    dataset.summary()

    sample = dataset[0]

    full_coords = sample["coords"].to(device)
    full_values = sample["values"].to(device)

    df = dataset.df.copy()

    n_total = full_coords.shape[0]

    print()
    print("Training data:")
    print(f"  measured points: {n_total}")
    print(f"  coords shape:    {tuple(full_coords.shape)}")
    print(f"  values shape:    {tuple(full_values.shape)}")

    if args.batch_size <= 0 or args.batch_size >= n_total:
        print("  training mode:   full batch")
    else:
        print(f"  training mode:   minibatch, batch_size={args.batch_size}")

    # ------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------
    model = MLPINR(
        in_features=dataset.in_features,
        out_features=dataset.out_features,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        activation=args.activation,
        first_omega_0=args.first_omega_0,
        hidden_omega_0=args.hidden_omega_0,
        outermost_linear=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print()
    print("Model config:")
    print(f"  activation:       {args.activation}")
    print(f"  hidden_features:  {args.hidden_features}")
    print(f"  hidden_layers:    {args.hidden_layers}")
    print(f"  first_omega_0:    {args.first_omega_0}")
    print(f"  hidden_omega_0:   {args.hidden_omega_0}")
    print(f"  lr:               {args.lr}")
    print(f"  num_steps:        {args.num_steps}")

    # ------------------------------------------------------------
    # 3. Train only on measured radar points
    # ------------------------------------------------------------
    history_fields = [
        "step",
        "train_mse_norm",
        "rmse_log10",
        "mae_log10",
        "bias_log10",
        "max_abs_log10",
        "p95_abs_log10",
        "p99_abs_log10",
    ]

    latest_metrics = {
        "rmse": np.nan,
        "mae": np.nan,
        "bias": np.nan,
        "max_abs": np.nan,
        "p95_abs": np.nan,
        "p99_abs": np.nan,
    }

    pbar = tqdm(
        range(1, args.num_steps + 1),
        disable=args.disable_tqdm,
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout,
        mininterval=0.5,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    for step in pbar:
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

        if step == 1 or step % args.summary_every == 0 or step == args.num_steps:
            model.eval()

            with torch.no_grad():
                pred_norm_np = model(full_coords).detach().cpu().numpy()

            pred_df = dataset.make_prediction_dataframe(pred_norm_np)

            metrics = compute_metrics(
                pred=pred_df["pred_log10_Ne"].to_numpy(),
                target=pred_df["log10_Ne"].to_numpy(),
            )

            latest_metrics = metrics

            row = {
                "step": step,
                "train_mse_norm": float(loss.item()),
                "rmse_log10": metrics["rmse"],
                "mae_log10": metrics["mae"],
                "bias_log10": metrics["bias"],
                "max_abs_log10": metrics["max_abs"],
                "p95_abs_log10": metrics["p95_abs"],
                "p99_abs_log10": metrics["p99_abs"],
            }

            append_csv_row(history_path, history_fields, row)

        if step == 1 or step % args.log_every == 0 or step == args.num_steps:
            pbar.set_postfix_str(
                f"mse={loss.item():.2e} "
                f"rmse={latest_metrics['rmse']:.2e} "
                f"mae={latest_metrics['mae']:.2e}"
            )

    # ------------------------------------------------------------
    # 4. Save model and measured-point predictions
    # ------------------------------------------------------------
    model.eval()

    with torch.no_grad():
        pred_norm_np = model(full_coords).detach().cpu().numpy()

    pred_df = dataset.make_prediction_dataframe(pred_norm_np)

    pred_csv = out_dir / "predictions_at_measured_points.csv"
    pred_df.to_csv(pred_csv, index=False)

    final_metrics = compute_metrics(
        pred=pred_df["pred_log10_Ne"].to_numpy(),
        target=pred_df["log10_Ne"].to_numpy(),
    )

    model_path = out_dir / "model_final.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "coord_scalers": dataset.coord_scalers,
            "target_scaler": dataset.target_scaler,
            "final_metrics": final_metrics,
        },
        model_path,
    )

    print()
    print(f"Saved model: {model_path}")
    print(f"Saved history: {history_path}")
    print(f"Saved measured-point predictions: {pred_csv}")

    print()
    print("Final measured-point metrics in log10(Ne):")
    for key, value in final_metrics.items():
        print(f"  {key:12s}: {value:.8e}")

    # ------------------------------------------------------------
    # 5. Dense query grid for visualization only
    # ------------------------------------------------------------
    measured_xy = df[["x_km", "y_km"]].to_numpy(dtype=float)

    X, Y = make_query_grid_from_points(
        x_km=df["x_km"].to_numpy(dtype=float),
        y_km=df["y_km"].to_numpy(dtype=float),
        nx=args.grid_nx,
        ny=args.grid_ny,
        padding_frac=args.grid_padding_frac,
    )

    coords_grid_np = normalize_grid_with_dataset(dataset, X, Y)

    pred_grid_log10_flat = evaluate_model_on_grid(
        model=model,
        coords_grid_np=coords_grid_np,
        dataset=dataset,
        device=device,
        chunk_size=args.grid_chunk_size,
    )

    pred_grid_log10 = pred_grid_log10_flat.reshape(X.shape)

    # ------------------------------------------------------------
    # 6. Valid footprint mask, simple nearest-distance only
    # ------------------------------------------------------------
    nearest_radius_km = estimate_nearest_radius(
        measured_xy,
        factor=args.nearest_radius_factor,
    )

    valid_mask, nearest_dist_grid = nearest_distance_mask(
        X,
        Y,
        measured_xy,
        radius_km=nearest_radius_km,
    )

    pred_grid_masked = pred_grid_log10.copy()
    pred_grid_masked[~valid_mask] = np.nan

    print()
    print("Query grid:")
    print(f"  X/Y shape:             {X.shape}")
    print("  grid use:              visualization only")
    print("  mask type:             nearest real radar point")
    print(f"  nearest radius [km]:   {nearest_radius_km:.3f}")
    print(f"  valid grid fraction:   {valid_mask.mean():.3f}")

    if args.save_grid_csv:
        grid_df = pd.DataFrame(
            {
                "x_km": X.ravel(),
                "y_km": Y.ravel(),
                "pred_log10_Ne": pred_grid_log10.ravel(),
                "nearest_dist_km": nearest_dist_grid.ravel(),
                "valid_mask": valid_mask.ravel(),
            }
        )

        grid_csv = out_dir / "grid_predictions.csv"
        grid_df.to_csv(grid_csv, index=False)
        print(f"Saved grid predictions: {grid_csv}")

    # ------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------
    if save_plots:
        plot_measured_points(df, out_dir, save_plots=save_plots)
        plot_grid_prediction(X, Y, pred_grid_masked, df, out_dir, save_plots=save_plots)
        plot_residuals(pred_df, out_dir, save_plots=save_plots)
        plot_history(history_path, out_dir, save_plots=save_plots)

        print(f"Saved plots in: {out_dir}")

    print()
    print("DONE")


# ============================================================
# ARGUMENTS
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/slice111748_h330_best_slice.csv",
        help="Prepared AMISR 2D slice CSV.",
    )

    # Model
    parser.add_argument(
        "--activation",
        type=str,
        default="sine",
        choices=["relu", "tanh", "softplus", "sine"],
    )
    parser.add_argument("--hidden_features", type=int, default=256)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--first_omega_0", type=float, default=30.0)
    parser.add_argument("--hidden_omega_0", type=float, default=30.0)

    # Training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="0 or >= N means full batch.",
    )
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")

    # Logging
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--summary_every", type=int, default=250)
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--resume_history", action="store_true")

    # Grid visualization
    parser.add_argument("--grid_nx", type=int, default=250)
    parser.add_argument("--grid_ny", type=int, default=250)
    parser.add_argument("--grid_padding_frac", type=float, default=0.05)
    parser.add_argument("--grid_chunk_size", type=int, default=65536)

    # Simple nearest-distance mask
    parser.add_argument("--nearest_radius_factor", type=float, default=2.5)

    # Outputs
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/radar_2d_overfit",
    )
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--save_grid_csv", action="store_true")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)