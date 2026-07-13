# -*- coding: utf-8 -*-
"""
train_radar_3d.py

Train a 3D INR on AMISR data.

3D here means:
    x, y, time

not:
    x, y, z

Current experiment:
    f_theta(x_norm, y_norm, t_norm) -> normalized log10(Ne)

Loss:
    data_loss only

No curvature yet.
No gradient-cap yet.
No time-continuity loss yet.

This file is the first sanity test for:
    HDF5 reader -> RadarTimeH5Dataset -> MLPINR(in_features=3) -> training -> x-y plots at selected times
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

from datasets import RadarTimeH5Dataset
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
# GRID HELPERS
# ============================================================

def make_query_grid_from_points(
    x_km: np.ndarray,
    y_km: np.ndarray,
    nx: int,
    ny: int,
    padding_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
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
    grid_xy = np.column_stack([X.ravel(), Y.ravel()])
    measured_xy = np.asarray(measured_xy, dtype=float)

    diff = grid_xy[:, None, :] - measured_xy[None, :, :]
    dist2 = np.sum(diff ** 2, axis=2)

    nearest_dist = np.sqrt(np.min(dist2, axis=1))
    nearest_dist_grid = nearest_dist.reshape(X.shape)

    mask = nearest_dist_grid <= radius_km

    return mask, nearest_dist_grid


def normalize_xy_t_grid_with_dataset(
    dataset: RadarTimeH5Dataset,
    X: np.ndarray,
    Y: np.ndarray,
    t_sec: float,
) -> np.ndarray:
    """
    Normalize an x-y grid at one fixed physical time.

    Input:
        X, Y in km
        t_sec in seconds relative to the dataset t0

    Output:
        coords_grid: [Ngrid, 3] = [x_norm, y_norm, t_norm]
    """

    x_min = dataset.coord_scalers["x_km"]["min"]
    x_max = dataset.coord_scalers["x_km"]["max"]

    y_min = dataset.coord_scalers["y_km"]["min"]
    y_max = dataset.coord_scalers["y_km"]["max"]

    t_min = dataset.coord_scalers["t_sec"]["min"]
    t_max = dataset.coord_scalers["t_sec"]["max"]

    Xn = 2.0 * (X - x_min) / (x_max - x_min) - 1.0
    Yn = 2.0 * (Y - y_min) / (y_max - y_min) - 1.0
    Tn = 2.0 * (float(t_sec) - t_min) / (t_max - t_min) - 1.0

    coords_grid = np.column_stack(
        [
            Xn.ravel(),
            Yn.ravel(),
            np.full(X.size, Tn, dtype=np.float64),
        ]
    ).astype(np.float32)

    return coords_grid


@torch.no_grad()
def evaluate_model_on_coords(
    model: torch.nn.Module,
    coords_np: np.ndarray,
    dataset: RadarTimeH5Dataset,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    """
    Evaluate model on normalized coordinates.

    Returns predictions in physical target scale, currently log10(Ne).
    """

    model.eval()

    outputs = []
    n = coords_np.shape[0]

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        coords_chunk = torch.from_numpy(coords_np[start:end]).to(device)
        pred_chunk = model(coords_chunk).detach().cpu().numpy()

        outputs.append(pred_chunk)

    pred_norm = np.concatenate(outputs, axis=0)
    pred_log10 = dataset.denormalize_target(pred_norm)

    return pred_log10[:, 0]


def select_plot_time_indices(
    df: pd.DataFrame,
    num_plot_times: int,
) -> list[int]:
    """
    Select a few existing AMISR time records for x-y plotting.

    For example:
        num_plot_times = 3 -> first, middle, last
    """

    unique_times = np.sort(df["time_index"].unique())

    if num_plot_times <= 0:
        return []

    if num_plot_times >= unique_times.size:
        return [int(x) for x in unique_times]

    picks = np.linspace(0, unique_times.size - 1, num_plot_times)
    picks = np.round(picks).astype(int)

    return [int(unique_times[i]) for i in picks]


# ============================================================
# PLOTTING
# ============================================================

def plot_history(
    history_path: Path,
    out_dir: Path,
) -> None:
    hist = pd.read_csv(history_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(hist["step"], hist["train_mse_norm"], marker="o", markersize=3, label="train MSE")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE on normalized log10(Ne)")
    ax.set_title("3D INR training loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = out_dir / "training_history.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"Saved history plot: {path}")


def plot_xy_prediction_at_time(
    model: torch.nn.Module,
    dataset: RadarTimeH5Dataset,
    df: pd.DataFrame,
    time_index: int,
    out_dir: Path,
    device: torch.device,
    grid_nx: int,
    grid_ny: int,
    grid_padding_frac: float,
    nearest_radius_factor: float,
    grid_chunk_size: int,
    save_grid_csv: bool,
) -> None:
    """
    Plot model prediction on an x-y grid at one existing AMISR time.
    """

    df_time = df[df["time_index"] == time_index].copy()

    if len(df_time) == 0:
        raise ValueError(f"No dataframe rows for time_index={time_index}")

    t_sec = float(df_time["t_sec"].median())
    t_hours = float(df_time["t_hours"].median())
    unix_mid = float(df_time["unix_mid"].median()) if "unix_mid" in df_time.columns else np.nan

    X, Y = make_query_grid_from_points(
        x_km=df["x_km"].to_numpy(dtype=float),
        y_km=df["y_km"].to_numpy(dtype=float),
        nx=grid_nx,
        ny=grid_ny,
        padding_frac=grid_padding_frac,
    )

    measured_xy = df_time[["x_km", "y_km"]].to_numpy(dtype=float)

    nearest_radius_km = estimate_nearest_radius(
        measured_xy,
        factor=nearest_radius_factor,
    )

    valid_mask, nearest_dist_grid = nearest_distance_mask(
        X,
        Y,
        measured_xy,
        radius_km=nearest_radius_km,
    )

    coords_grid_np = normalize_xy_t_grid_with_dataset(
        dataset=dataset,
        X=X,
        Y=Y,
        t_sec=t_sec,
    )

    pred_flat = evaluate_model_on_coords(
        model=model,
        coords_np=coords_grid_np,
        dataset=dataset,
        device=device,
        chunk_size=grid_chunk_size,
    )

    pred_grid = pred_flat.reshape(X.shape)

    pred_masked = pred_grid.copy()
    pred_masked[~valid_mask] = np.nan

    vmin = float(9)
    vmax = float(12)

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.pcolormesh(
        X,
        Y,
        pred_masked,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    fig.colorbar(im, ax=ax, label="predicted log10(Ne)")

    ax.scatter(
        df_time["x_km"],
        df_time["y_km"],
        c=df_time["log10_Ne"],
        s=35,
        edgecolor="k",
        linewidth=0.4,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel("x east [km]")
    ax.set_ylabel("y north [km]")
    ax.set_title(
        f"3D INR x-y prediction | time_index={time_index} | t={t_hours:.2f} h"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    fig_path = out_dir / f"xy_time_index_{time_index:04d}.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(f"Saved x-y time plot: {fig_path}")

    if save_grid_csv:
        grid_df = pd.DataFrame(
            {
                "time_index": np.full(X.size, int(time_index), dtype=int),
                "unix_mid": np.full(X.size, unix_mid, dtype=float),
                "t_sec": np.full(X.size, t_sec, dtype=float),
                "t_hours": np.full(X.size, t_hours, dtype=float),
                "x_km": X.ravel(),
                "y_km": Y.ravel(),
                "pred_log10_Ne": pred_grid.ravel(),
                "nearest_dist_km": nearest_dist_grid.ravel(),
                "valid_mask": valid_mask.ravel(),
            }
        )

        csv_path = out_dir / f"grid_prediction_time_index_{time_index:04d}.csv"
        grid_df.to_csv(csv_path, index=False)

        print(f"Saved grid CSV: {csv_path}")


# ============================================================
# TRAINING
# ============================================================

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = out_dir / "history.csv"

    if history_path.exists() and not args.resume_history:
        history_path.unlink()

    max_records = None if args.max_records <= 0 else args.max_records

    config = vars(args).copy()
    config["max_records_effective"] = max_records

    config_path = out_dir / "run_config.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    print(f"Using device: {device}")

    # ------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------
    dataset = RadarTimeH5Dataset(
        h5_path=args.h5_path,
        h0_km=args.h0_km,
        half_width_km=args.half_width_km,
        time_start_utc=args.time_start_utc,
        time_end_utc=args.time_end_utc,
        record_stride=args.record_stride,
        max_records=max_records,
        verbose=True,
    )

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
    print(f"  time records:    {df['time_index'].nunique()}")

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
    print(f"  in_features:      {dataset.in_features}")
    print(f"  out_features:     {dataset.out_features}")
    print(f"  activation:       {args.activation}")
    print(f"  hidden_features:  {args.hidden_features}")
    print(f"  hidden_layers:    {args.hidden_layers}")
    print(f"  first_omega_0:    {args.first_omega_0}")
    print(f"  hidden_omega_0:   {args.hidden_omega_0}")
    print(f"  lr:               {args.lr}")
    print(f"  num_steps:        {args.num_steps}")

    # ------------------------------------------------------------
    # 3. Train data-only
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
            "config": config,
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
    # 5. Plots
    # ------------------------------------------------------------
    if not args.no_plots:
        plot_history(history_path, out_dir)

        plot_time_indices = select_plot_time_indices(
            df=df,
            num_plot_times=args.num_plot_times,
        )

        print()
        print("Plot time indices:")
        print(plot_time_indices)

        for time_index in plot_time_indices:
            plot_xy_prediction_at_time(
                model=model,
                dataset=dataset,
                df=df,
                time_index=time_index,
                out_dir=out_dir,
                device=device,
                grid_nx=args.grid_nx,
                grid_ny=args.grid_ny,
                grid_padding_frac=args.grid_padding_frac,
                nearest_radius_factor=args.nearest_radius_factor,
                grid_chunk_size=args.grid_chunk_size,
                save_grid_csv=args.save_grid_csv,
            )

    print()
    print("DONE")


# ============================================================
# ARGUMENTS
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument(
        "--h5_path",
        type=str,
        default="../data/20120122.001_lp_5min.h5",
        help="AMISR HDF5 file.",
    )
    parser.add_argument("--h0_km", type=float, default=330.0)
    parser.add_argument("--half_width_km", type=float, default=15.0)
    parser.add_argument("--time_start_utc", type=str, default=None)
    parser.add_argument("--time_end_utc", type=str, default=None)
    parser.add_argument("--record_stride", type=int, default=1)
    parser.add_argument(
        "--max_records",
        type=int,
        default=5,
        help="Maximum number of time records. Use 0 for all selected records.",
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
    parser.add_argument("--first_omega_0", type=float, default=5.0)
    parser.add_argument("--hidden_omega_0", type=float, default=5.0)

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
    parser.add_argument("--nearest_radius_factor", type=float, default=2.5)
    parser.add_argument(
        "--num_plot_times",
        type=int,
        default=3,
        help="Number of existing time records to plot. 3 means first/middle/last.",
    )

    # Outputs
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/radar_3d_data_only",
    )
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--save_grid_csv", action="store_true")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)