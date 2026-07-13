# -*- coding: utf-8 -*-
"""
plot_all_times_for_windows.py

Plot every measured time step for every trained window folder.

This does NOT merge windows.
This does NOT make videos.
This only does:

    one trained window folder -> all real time steps in that window -> PNGs

Run from inside inf_amisr_3d.

Example:

    python3 plot_all_times_for_windows.py \
      --windows_root outputs/reference_ratio_2min_win31_stride15 \
      --checkpoint_name model_final.pt \
      --skip_existing

Quick test:

    python3 plot_all_times_for_windows.py \
      --windows_root outputs/reference_ratio_2min_win31_stride15 \
      --max_windows 2 \
      --max_times_per_window 3 \
      --skip_existing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from datasets import RadarTimeH5Dataset
from models import MLPINR


# ============================================================
# Grid helpers
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
    x_min = dataset.coord_scalers["x_km"]["min"]
    x_max = dataset.coord_scalers["x_km"]["max"]

    y_min = dataset.coord_scalers["y_km"]["min"]
    y_max = dataset.coord_scalers["y_km"]["max"]

    t_min = dataset.coord_scalers["t_sec"]["min"]
    t_max = dataset.coord_scalers["t_sec"]["max"]

    Xn = 2.0 * (X - x_min) / (x_max - x_min) - 1.0
    Yn = 2.0 * (Y - y_min) / (y_max - y_min) - 1.0

    if t_max == t_min:
        Tn = 0.0
    else:
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


# ============================================================
# Window loading
# ============================================================

def parse_start_index_from_name(path: Path) -> int:
    match = re.search(r"win_start_(\d+)", path.name)

    if match is None:
        return 10**12

    return int(match.group(1))


def find_window_dirs(windows_root: Path) -> list[Path]:
    dirs = [
        p for p in windows_root.iterdir()
        if p.is_dir() and p.name.startswith("win_start_")
    ]

    dirs = sorted(dirs, key=parse_start_index_from_name)

    return dirs


def load_config(window_dir: Path) -> dict:
    config_path = window_dir / "run_config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing run_config.json in {window_dir}")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def build_dataset_from_config(
    config: dict,
    h5_path_override: str | None,
    verbose: bool,
) -> RadarTimeH5Dataset:
    h5_path = h5_path_override if h5_path_override is not None else config["h5_path"]

    dataset = RadarTimeH5Dataset(
        h5_path=h5_path,
        h0_km=float(config.get("h0_km", 330.0)),
        half_width_km=float(config.get("half_width_km", 15.0)),
        time_start_utc=config.get("time_start_utc", None),
        time_end_utc=config.get("time_end_utc", None),
        record_stride=int(config.get("record_stride", 1)),
        max_records=None,
        window_start_index=int(config["window_start_index"]),
        window_size_records=int(config["window_size_records"]),
        verbose=verbose,
    )

    return dataset


def build_model_from_config(
    config: dict,
    dataset: RadarTimeH5Dataset,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    model = MLPINR(
        in_features=dataset.in_features,
        out_features=dataset.out_features,
        hidden_features=int(config.get("hidden_features", 256)),
        hidden_layers=int(config.get("hidden_layers", 3)),
        activation=str(config.get("activation", "sine")),
        first_omega_0=float(config.get("first_omega_0", 5.0)),
        hidden_omega_0=float(config.get("hidden_omega_0", 5.0)),
        outermost_linear=True,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


# ============================================================
# Plotting
# ============================================================

def plot_one_time(
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
    vmin: float,
    vmax: float,
    save_grid_csv: bool,
) -> Path:
    df_time = df[df["time_index"] == time_index].copy()

    if len(df_time) == 0:
        raise ValueError(f"No dataframe rows for time_index={time_index}")

    t_sec = float(df_time["t_sec"].median())
    t_min = t_sec / 60.0
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

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.pcolormesh(
        X,
        Y,
        pred_masked,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        cmap="inferno",
    )

    fig.colorbar(im, ax=ax, label="predicted log10(Ne)")

    sc = ax.scatter(
        df_time["x_km"],
        df_time["y_km"],
        c=df_time["log10_Ne"],
        s=35,
        edgecolor="k",
        linewidth=0.4,
        vmin=vmin,
        vmax=vmax,
        cmap="inferno",
    )

    ax.set_xlabel("x east [km]")
    ax.set_ylabel("y north [km]")
    ax.set_title(
        f"Window start={int(df['time_index'].min())} to {int(df['time_index'].max())} | "
        f"time_index={time_index} | t={t_min:.1f} min"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    fig_path = out_dir / f"xy_time_index_{time_index:04d}.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    if save_grid_csv:
        grid_df = pd.DataFrame(
            {
                "time_index": np.full(X.size, int(time_index), dtype=int),
                "unix_mid": np.full(X.size, unix_mid, dtype=float),
                "t_sec": np.full(X.size, t_sec, dtype=float),
                "t_min": np.full(X.size, t_min, dtype=float),
                "x_km": X.ravel(),
                "y_km": Y.ravel(),
                "pred_log10_Ne": pred_grid.ravel(),
                "nearest_dist_km": nearest_dist_grid.ravel(),
                "valid_mask": valid_mask.ravel(),
            }
        )

        csv_path = out_dir / f"grid_prediction_time_index_{time_index:04d}.csv"
        grid_df.to_csv(csv_path, index=False)

    return fig_path


def plot_all_times_for_window(
    window_dir: Path,
    checkpoint_name: str,
    plot_subdir: str,
    h5_path_override: str | None,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    config = load_config(window_dir)

    checkpoint_path = window_dir / checkpoint_name

    if not checkpoint_path.exists():
        return {
            "window_dir": str(window_dir),
            "status": f"missing_checkpoint:{checkpoint_name}",
            "n_times": 0,
            "n_plotted": 0,
        }

    dataset = build_dataset_from_config(
        config=config,
        h5_path_override=h5_path_override,
        verbose=args.verbose_dataset,
    )

    df = dataset.df.copy()

    model = build_model_from_config(
        config=config,
        dataset=dataset,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    out_dir = window_dir / plot_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    time_indices = [int(x) for x in np.sort(df["time_index"].unique())]

    if args.max_times_per_window is not None and args.max_times_per_window > 0:
        time_indices = time_indices[:args.max_times_per_window]

    if args.color_scale == "window":
        vmin = float(df["log10_Ne"].min())
        vmax = float(df["log10_Ne"].max())
    elif args.color_scale == "fixed":
        if args.vmin is None or args.vmax is None:
            raise ValueError("--color_scale fixed requires --vmin and --vmax")
        vmin = float(args.vmin)
        vmax = float(args.vmax)
    else:
        raise ValueError(f"Unknown color_scale: {args.color_scale}")

    n_plotted = 0

    for time_index in tqdm(
        time_indices,
        desc=f"{window_dir.name}",
        leave=False,
        disable=args.disable_tqdm,
    ):
        fig_path = out_dir / f"xy_time_index_{time_index:04d}.png"

        if args.skip_existing and fig_path.exists():
            continue

        plot_one_time(
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
            vmin=vmin,
            vmax=vmax,
            save_grid_csv=args.save_grid_csv,
        )

        n_plotted += 1

    return {
        "window_dir": str(window_dir),
        "status": "ok",
        "checkpoint_name": checkpoint_name,
        "plot_dir": str(out_dir),
        "window_start_index": int(config["window_start_index"]),
        "window_size_records": int(config["window_size_records"]),
        "time_index_min": int(np.min(time_indices)) if len(time_indices) else "",
        "time_index_max": int(np.max(time_indices)) if len(time_indices) else "",
        "n_times": len(time_indices),
        "n_plotted": n_plotted,
        "vmin": vmin,
        "vmax": vmax,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--windows_root", type=str, required=True)
    parser.add_argument("--checkpoint_name", type=str, default="model_final.pt")
    parser.add_argument("--plot_subdir", type=str, default="plots_all_times")

    parser.add_argument(
        "--h5_path_override",
        type=str,
        default=None,
        help="Optional override if run_config.json points to the wrong HDF5 file.",
    )

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--verbose_dataset", action="store_true")

    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--max_times_per_window", type=int, default=None)

    parser.add_argument("--grid_nx", type=int, default=250)
    parser.add_argument("--grid_ny", type=int, default=250)
    parser.add_argument("--grid_padding_frac", type=float, default=0.05)
    parser.add_argument("--grid_chunk_size", type=int, default=65536)
    parser.add_argument("--nearest_radius_factor", type=float, default=2.5)

    parser.add_argument(
        "--color_scale",
        type=str,
        default="window",
        choices=["window", "fixed"],
        help="window = each window uses its own data min/max. fixed = use --vmin/--vmax.",
    )
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)

    parser.add_argument("--save_grid_csv", action="store_true")

    args = parser.parse_args()

    windows_root = Path(args.windows_root)

    if not windows_root.exists():
        raise FileNotFoundError(f"Missing windows_root: {windows_root}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    window_dirs = find_window_dirs(windows_root)

    if args.max_windows is not None and args.max_windows > 0:
        window_dirs = window_dirs[:args.max_windows]

    print("Plot all times for trained windows")
    print(f"  windows_root:       {windows_root}")
    print(f"  n windows:          {len(window_dirs)}")
    print(f"  checkpoint_name:    {args.checkpoint_name}")
    print(f"  plot_subdir:        {args.plot_subdir}")
    print(f"  h5_path_override:   {args.h5_path_override}")
    print(f"  device:             {device}")
    print(f"  grid:               {args.grid_nx} x {args.grid_ny}")
    print(f"  color_scale:        {args.color_scale}")
    print()

    rows = []

    for window_dir in tqdm(
        window_dirs,
        desc="windows",
        disable=args.disable_tqdm,
    ):
        try:
            row = plot_all_times_for_window(
                window_dir=window_dir,
                checkpoint_name=args.checkpoint_name,
                plot_subdir=args.plot_subdir,
                h5_path_override=args.h5_path_override,
                device=device,
                args=args,
            )
        except Exception as e:
            row = {
                "window_dir": str(window_dir),
                "status": f"error:{type(e).__name__}:{e}",
                "n_times": 0,
                "n_plotted": 0,
            }

            print()
            print(f"ERROR in {window_dir}: {type(e).__name__}: {e}", file=sys.stderr)

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_path = windows_root / f"{args.plot_subdir}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print()
    print("DONE")
    print(f"Saved summary: {summary_path}")
    print(summary_df["status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()