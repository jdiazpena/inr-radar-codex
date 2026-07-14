# -*- coding: utf-8 -*-
"""Analyze the four-run synthetic regularization sweep."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import synthetic_analyze_reconsturction_linear_errors as single_run

RUNS = [
    ("data_only", Path("outputs/codex_sweep_high_amp_data_only_15000")),
    ("xy030_t030", Path("outputs/codex_sweep_high_amp_xy030_t030_15000")),
    ("xy070_t030", Path("outputs/codex_sweep_high_amp_xy070_t030_15000")),
    ("xy070_t070", Path("outputs/codex_sweep_high_amp_xy070_t070_15000")),
]

SYNTHETIC_CSV = Path("outputs/synthetic_high_amp_left_right/synthetic_observations.csv")
SYNTHETIC_CONFIG = Path("outputs/synthetic_high_amp_left_right/synthetic_config.json")
COMPARISON_DIR = Path("outputs/comparison")
ANALYSIS_SUBDIR = "error_analysis"
CHECKPOINT_NAME = "model_final.pt"

GRID_NX = 250
GRID_NY = 250
DOMAIN_MODE = "full_domain"
TIME_SELECTION: str | list[int] = "first_middle_last"
COMPUTE_GRADIENT_ERRORS = True
SAVE_DENSE_CSV = True

INTERIOR_BOUNDARY_FRACTION = 0.10
NEAR_OBSERVATION_DISTANCE_KM = 50.0
HIGH_GRADIENT_PERCENTILE = 75.0


def compute_error_metrics(error: np.ndarray, prefix: str) -> dict[str, float]:
    error = np.asarray(error, dtype=np.float64)
    error = error[np.isfinite(error)]
    if error.size == 0:
        return {f"{prefix}{k}": np.nan for k in ["rmse", "mae", "bias", "p95_abs", "max_abs"]}
    abs_error = np.abs(error)
    return {
        f"{prefix}rmse": float(np.sqrt(np.mean(error ** 2))),
        f"{prefix}mae": float(np.mean(abs_error)),
        f"{prefix}bias": float(np.mean(error)),
        f"{prefix}p95_abs": float(np.quantile(abs_error, 0.95)),
        f"{prefix}max_abs": float(np.max(abs_error)),
    }


def nearest_observation_mask(df_time: pd.DataFrame, obs_time: pd.DataFrame, distance_km: float) -> np.ndarray:
    grid_xy = df_time[["x_km", "y_km"]].to_numpy(dtype=np.float64)
    obs_xy = obs_time[["x_km", "y_km"]].to_numpy(dtype=np.float64)
    diff = grid_xy[:, None, :] - obs_xy[None, :, :]
    nearest = np.sqrt(np.min(np.sum(diff ** 2, axis=2), axis=1))
    return nearest <= float(distance_km)


def masks_for_time(df_time: pd.DataFrame, obs_time: pd.DataFrame) -> dict[str, np.ndarray]:
    x = df_time["x_km"].to_numpy(dtype=np.float64)
    y = df_time["y_km"].to_numpy(dtype=np.float64)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    bx = INTERIOR_BOUNDARY_FRACTION * (x_max - x_min)
    by = INTERIOR_BOUNDARY_FRACTION * (y_max - y_min)

    interior = (x >= x_min + bx) & (x <= x_max - bx) & (y >= y_min + by) & (y <= y_max - by)
    near_obs = nearest_observation_mask(df_time, obs_time, NEAR_OBSERVATION_DISTANCE_KM)

    grad_mag = np.sqrt(
        df_time["true_dlog10Ne_dx_km"].to_numpy(dtype=np.float64) ** 2
        + df_time["true_dlog10Ne_dy_km"].to_numpy(dtype=np.float64) ** 2
        + df_time["true_dlog10Ne_dt_sec"].to_numpy(dtype=np.float64) ** 2
    )
    thresh = float(np.nanpercentile(grad_mag, HIGH_GRADIENT_PERCENTILE))
    high_gradient = grad_mag >= thresh

    return {
        "full_domain": np.ones(len(df_time), dtype=bool),
        "interior": interior,
        "near_observation": near_obs,
        "high_gradient": high_gradient,
    }


def regional_rows_for_run(label: str, run_dir: Path) -> list[dict[str, object]]:
    analysis_dir = run_dir / ANALYSIS_SUBDIR
    obs_df = pd.read_csv(SYNTHETIC_CSV)
    rows: list[dict[str, object]] = []

    dense_paths = sorted(analysis_dir.glob("dense_reconstruction_time_*.csv"))
    if not dense_paths:
        raise FileNotFoundError(f"No dense reconstruction CSVs found in {analysis_dir}")

    for dense_path in dense_paths:
        df_time = pd.read_csv(dense_path)
        time_index = int(df_time["time_index"].iloc[0])
        obs_time = obs_df[obs_df["time_index"] == time_index]
        masks = masks_for_time(df_time, obs_time)

        for region, mask in masks.items():
            sub = df_time.loc[mask]
            row: dict[str, object] = {
                "run": label,
                "run_dir": str(run_dir),
                "time_index": time_index,
                "t_sec": float(df_time["t_sec"].median()),
                "region": region,
                "n_points": int(len(sub)),
                "interior_boundary_fraction": INTERIOR_BOUNDARY_FRACTION,
                "near_observation_distance_km": NEAR_OBSERVATION_DISTANCE_KM,
                "high_gradient_percentile": HIGH_GRADIENT_PERCENTILE,
            }
            row.update(compute_error_metrics(sub["error_Ne"].to_numpy(dtype=np.float64), "density_"))

            if "error_dlog10Ne_dx_km" in sub.columns:
                gx = sub["error_dlog10Ne_dx_km"].to_numpy(dtype=np.float64)
                gy = sub["error_dlog10Ne_dy_km"].to_numpy(dtype=np.float64)
                gt = sub["error_dlog10Ne_dt_sec"].to_numpy(dtype=np.float64)
                gmag = np.sqrt(gx ** 2 + gy ** 2 + gt ** 2)
                row["dx_gradient_rmse"] = float(np.sqrt(np.mean(gx ** 2))) if gx.size else np.nan
                row["dy_gradient_rmse"] = float(np.sqrt(np.mean(gy ** 2))) if gy.size else np.nan
                row["dt_gradient_rmse"] = float(np.sqrt(np.mean(gt ** 2))) if gt.size else np.nan
                row["combined_gradient_magnitude_rmse"] = float(np.sqrt(np.mean(gmag ** 2))) if gmag.size else np.nan
                row["gradient_p95_abs"] = float(np.quantile(gmag, 0.95)) if gmag.size else np.nan
            rows.append(row)

    per_run = pd.DataFrame(rows)
    per_run.to_csv(analysis_dir / "regional_error_summary.csv", index=False)
    return rows


def run_single_analysis(label: str, run_dir: Path) -> None:
    if not (run_dir / CHECKPOINT_NAME).exists():
        raise FileNotFoundError(f"Missing checkpoint for {label}: {run_dir / CHECKPOINT_NAME}")

    single_run.RUN_DIR = run_dir
    single_run.SYNTHETIC_CSV = SYNTHETIC_CSV
    single_run.SYNTHETIC_CONFIG = SYNTHETIC_CONFIG
    single_run.CHECKPOINT_NAME = CHECKPOINT_NAME
    single_run.ANALYSIS_SUBDIR = ANALYSIS_SUBDIR
    single_run.GRID_NX = GRID_NX
    single_run.GRID_NY = GRID_NY
    single_run.DOMAIN_MODE = DOMAIN_MODE
    single_run.TIME_SELECTION = TIME_SELECTION
    single_run.COMPUTE_GRADIENT_ERRORS = COMPUTE_GRADIENT_ERRORS
    single_run.SAVE_DENSE_CSV = SAVE_DENSE_CSV
    single_run.main()


def main() -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []

    for label, run_dir in RUNS:
        print(f"Analyzing {label}: {run_dir}")
        run_single_analysis(label, run_dir)
        all_rows.extend(regional_rows_for_run(label, run_dir))

    regional_df = pd.DataFrame(all_rows)
    regional_df.to_csv(COMPARISON_DIR / "regional_error_summary.csv", index=False)
    print(f"Saved {COMPARISON_DIR / 'regional_error_summary.csv'}")


if __name__ == "__main__":
    main()
