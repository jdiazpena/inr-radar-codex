# -*- coding: utf-8 -*-
"""Summarize completed pilot runs with event-aware reconstruction metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ACTIVE_NE_THRESHOLD = 1.0e10
NEAR_OBSERVATION_KM = 50.0


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values ** 2))) if values.size else np.nan


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def nearest_observation_mask(dense: pd.DataFrame, obs: pd.DataFrame) -> np.ndarray:
    grid_xy = dense[["x_km", "y_km"]].to_numpy(dtype=np.float64)
    obs_xy = obs[["x_km", "y_km"]].to_numpy(dtype=np.float64)
    min_d2 = np.full(len(grid_xy), np.inf, dtype=np.float64)
    for point in obs_xy:
        min_d2 = np.minimum(min_d2, np.sum((grid_xy - point) ** 2, axis=1))
    return min_d2 <= NEAR_OBSERVATION_KM ** 2


def case_metadata(case_id: str, data_dir: Path) -> dict[str, object]:
    config = pd.read_json(data_dir / "synthetic_config.json", typ="series")
    return {
        "case_id": case_id,
        "motion": str(config["motion"]),
        "geometry": str(config["sample_geometry"]),
        "beam_count": int(pd.read_csv(data_dir / "synthetic_sample_geometry.csv")["beam_id"].nunique()),
        "speed_km_s": float(config["speed_km_s"]),
        "integration_time_sec": float(config["integration_time_sec"]),
        "n_times": int(config["n_times"]),
        "seed": int(config["seed"]),
    }


def summarize_run(root: Path, case_id: str, regularization: str) -> dict[str, object]:
    data_dir = root / "data" / case_id
    run_dir = root / "runs" / case_id / regularization
    obs = pd.read_csv(data_dir / "synthetic_observations.csv")
    event_time_index = int(obs.loc[(obs["t_sec"] - 600.0).abs().idxmin(), "time_index"])
    obs_event = obs[obs["time_index"] == event_time_index]
    dense = pd.read_csv(
        run_dir / "error_analysis" / f"dense_reconstruction_time_{event_time_index:04d}.csv"
    )

    observed_active = dense["true_Ne"].to_numpy() > ACTIVE_NE_THRESHOLD
    midpoint_active = dense["midpoint_true_Ne"].to_numpy() > ACTIVE_NE_THRESHOLD
    near_observation = nearest_observation_mask(dense, obs_event)
    grad_error = np.sqrt(
        dense["error_dlog10Ne_dx_km"].to_numpy() ** 2
        + dense["error_dlog10Ne_dy_km"].to_numpy() ** 2
        + dense["error_dlog10Ne_dt_sec"].to_numpy() ** 2
    )

    history = pd.read_csv(run_dir / "history.csv")
    tail = history.iloc[max(0, int(0.75 * len(history))):]
    final = history.iloc[-1]
    row = {
        **case_metadata(case_id, data_dir),
        "regularization": regularization,
        "event_time_index": event_time_index,
        "event_t_sec": float(dense["t_sec"].median()),
        "observed_active_fraction": float(np.mean(observed_active)),
        "midpoint_active_fraction": float(np.mean(midpoint_active)),
        "observed_full_rmse_Ne": rmse(dense["error_Ne"].to_numpy()),
        "observed_active_rmse_Ne": rmse(dense.loc[observed_active, "error_Ne"].to_numpy()),
        "midpoint_full_rmse_Ne": rmse(dense["midpoint_error_Ne"].to_numpy()),
        "midpoint_active_rmse_Ne": rmse(dense.loc[midpoint_active, "midpoint_error_Ne"].to_numpy()),
        "near_observation_rmse_Ne": rmse(dense.loc[near_observation, "error_Ne"].to_numpy()),
        "far_observation_rmse_Ne": rmse(dense.loc[~near_observation, "error_Ne"].to_numpy()),
        "observed_active_bias_Ne": float(dense.loc[observed_active, "error_Ne"].mean()),
        "observed_peak_Ne": float(dense["true_Ne"].max()),
        "midpoint_peak_Ne": float(dense["midpoint_true_Ne"].max()),
        "predicted_peak_Ne": float(dense["pred_Ne"].max()),
        "gradient_full_rmse": rmse(grad_error),
        "gradient_active_rmse": rmse(grad_error[observed_active]),
        "observed_full_correlation": correlation(dense["true_Ne"], dense["pred_Ne"]),
        "midpoint_full_correlation": correlation(dense["midpoint_true_Ne"], dense["pred_Ne"]),
        "final_training_rmse_log10": float(final["rmse_log10"]),
        "tail_median_fxx_rms": float(pd.to_numeric(tail["fxx_rms"], errors="coerce").median()),
        "tail_median_fxy_rms": float(pd.to_numeric(tail["fxy_rms"], errors="coerce").median()),
        "tail_median_fyy_rms": float(pd.to_numeric(tail["fyy_rms"], errors="coerce").median()),
        "tail_median_ftt_rms": float(pd.to_numeric(tail["ftt_rms"], errors="coerce").median()),
        "tail_median_lambda_xy": float(pd.to_numeric(tail["lambda_curv_xy_base"], errors="coerce").median()),
        "tail_median_lambda_t": float(pd.to_numeric(tail["lambda_curv_t_base"], errors="coerce").median()),
        "tail_median_xy_over_data_ref": float(pd.to_numeric(tail["xy_over_data_ref"], errors="coerce").median()),
        "tail_median_t_over_data_ref": float(pd.to_numeric(tail["t_over_data_ref"], errors="coerce").median()),
    }
    observed_signal_rms = rmse(dense.loc[observed_active, "true_Ne"].to_numpy() - 1.0e9)
    midpoint_signal_rms = rmse(dense.loc[midpoint_active, "midpoint_true_Ne"].to_numpy() - 1.0e9)
    row["observed_active_normalized_rmse"] = row["observed_active_rmse_Ne"] / observed_signal_rms
    row["midpoint_active_normalized_rmse"] = row["midpoint_active_rmse_Ne"] / midpoint_signal_rms
    row["predicted_to_observed_peak_ratio"] = row["predicted_peak_Ne"] / row["observed_peak_Ne"]
    row["predicted_to_midpoint_peak_ratio"] = row["predicted_peak_Ne"] / row["midpoint_peak_Ne"]
    return row


def pairwise_table(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "observed_full_rmse_Ne",
        "observed_active_rmse_Ne",
        "midpoint_full_rmse_Ne",
        "midpoint_active_rmse_Ne",
        "near_observation_rmse_Ne",
        "far_observation_rmse_Ne",
        "gradient_full_rmse",
        "gradient_active_rmse",
    ]
    rows = []
    for case_id, group in summary.groupby("case_id"):
        data = group.set_index("regularization")
        if not {"data_only", "xy030_t030"}.issubset(data.index):
            continue
        row: dict[str, object] = {"case_id": case_id}
        for col in ("motion", "geometry", "beam_count", "speed_km_s", "integration_time_sec", "n_times"):
            row[col] = data.loc["data_only", col]
        for metric in metrics:
            baseline = float(data.loc["data_only", metric])
            regularized = float(data.loc["xy030_t030", metric])
            row[f"data_only_{metric}"] = baseline
            row[f"xy030_t030_{metric}"] = regularized
            row[f"regularization_improvement_pct_{metric}"] = 100.0 * (baseline - regularized) / baseline
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/velocity_integration_benchmark"))
    args = parser.parse_args()
    root = args.root.resolve()

    manifest = pd.read_csv(root / "benchmark_manifest.csv")
    run_keys = manifest[["case_id", "regularization"]].drop_duplicates()
    rows = [
        summarize_run(root, str(item.case_id), str(item.regularization))
        for item in run_keys.itertuples(index=False)
    ]
    summary = pd.DataFrame(rows).sort_values(["motion", "beam_count", "speed_km_s", "integration_time_sec", "regularization"])
    pairs = pairwise_table(summary)
    summary.to_csv(root / "benchmark_event_summary.csv", index=False)
    pairs.to_csv(root / "benchmark_regularization_comparison.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved {root / 'benchmark_event_summary.csv'}")
    print(f"Saved {root / 'benchmark_regularization_comparison.csv'}")


if __name__ == "__main__":
    main()
