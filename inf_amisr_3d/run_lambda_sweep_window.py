# -*- coding: utf-8 -*-
"""
run_lambda_sweep_window.py

Sequential wrapper for lambda sweeps on one fixed AMISR time window.

Purpose:
    Test lambda_curv_xy and lambda_curv_t without changing the data window.

This does NOT train sliding windows yet.
This only sweeps regularization values for one window.

Run from inside inf_amisr_3d:

    python3 run_lambda_sweep_window.py

Optional:

    python3 run_lambda_sweep_window.py --dry_run
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# Fixed experiment settings
# ============================================================

TRAIN_SCRIPT = "train_radar_3d_window_reg.py"

BASE_OUTPUT_DIR = Path("outputs/lambda_sweep_window100_size11_focused")

WINDOW_START_INDEX = 100
WINDOW_SIZE_RECORDS = 11

NUM_STEPS = 10000
SEED = 0
REG_RAMP_FRAC = 0.2

# Keep these fixed for this sweep.
NUM_COLLOCATION = 8192
COLLOCATION_GRID_NX = 80
COLLOCATION_GRID_NY = 80

# Sweep values.
# Start modest. This is already 6 x 4 = 24 runs.
LAMBDA_CURV_XY_VALUES = [
    5e-9,
    7.5e-9,
    1e-8,
    2.5e-8,
    5e-8
]

LAMBDA_CURV_T_VALUES = [
    1e-9,
    2.5e-9,
    5e-9,
    7.5e-9,
    1e-8
]


# ============================================================
# Helpers
# ============================================================

def lambda_tag(value: float) -> str:
    """
    Make filesystem-safe lambda tags.
    """

    if value == 0.0:
        return "0"

    return f"{value:.0e}".replace("-", "m")


def read_run_summary(output_dir: Path) -> dict:
    """
    Read history.csv from one run and summarize final/best losses.
    """

    history_path = output_dir / "history.csv"

    if not history_path.exists():
        return {
            "status": "missing_history",
        }

    hist = pd.read_csv(history_path)

    if len(hist) == 0:
        return {
            "status": "empty_history",
        }

    # final = hist.iloc[-1]

    # best_total_idx = hist["total_loss"].idxmin()
    # best_data_idx = hist["data_loss"].idxmin()

    # best_total = hist.loc[best_total_idx]
    # best_data = hist.loc[best_data_idx]

    final = hist.iloc[-1]

    # Do not select best checkpoints inside the regularization ramp.
    ramp_steps = int(REG_RAMP_FRAC * NUM_STEPS)

    hist_after_ramp = hist[hist["step"] >= ramp_steps].copy()

    if len(hist_after_ramp) == 0:
        return {
            "status": "no_history_after_ramp",
        }

    best_total_idx = hist_after_ramp["total_loss"].idxmin()
    best_data_idx = hist_after_ramp["data_loss"].idxmin()

    best_total = hist_after_ramp.loc[best_total_idx]
    best_data = hist_after_ramp.loc[best_data_idx]

    return {
        "status": "ok",

        "final_step": int(final["step"]),
        "final_total_loss": float(final["total_loss"]),
        "final_data_loss": float(final["data_loss"]),
        "final_curv_xy_weighted": float(final["curv_xy_weighted"]),
        "final_curv_t_weighted": float(final["curv_t_weighted"]),
        "final_rmse_log10": float(final["rmse_log10"]),

        "best_total_step": int(best_total["step"]),
        "best_total_loss": float(best_total["total_loss"]),
        "best_total_data_loss": float(best_total["data_loss"]),
        "best_total_curv_xy_weighted": float(best_total["curv_xy_weighted"]),
        "best_total_curv_t_weighted": float(best_total["curv_t_weighted"]),
        "best_total_rmse_log10": float(best_total["rmse_log10"]),

        "best_data_step": int(best_data["step"]),
        "best_data_loss": float(best_data["data_loss"]),
        "best_data_total_loss": float(best_data["total_loss"]),
        "best_data_rmse_log10": float(best_data["rmse_log10"]),
    }


def append_summary(summary_path: Path, row: dict) -> None:
    """
    Append one row to sweep_summary.csv.
    """

    file_exists = summary_path.exists()

    fieldnames = [
        "run_name",
        "output_dir",
        "lambda_curv_xy",
        "lambda_curv_t",
        "window_start_index",
        "window_size_records",
        "num_steps",
        "seed",
        "num_collocation",
        "status",

        "final_step",
        "final_total_loss",
        "final_data_loss",
        "final_curv_xy_weighted",
        "final_curv_t_weighted",
        "final_rmse_log10",

        "best_total_step",
        "best_total_loss",
        "best_total_data_loss",
        "best_total_curv_xy_weighted",
        "best_total_curv_t_weighted",
        "best_total_rmse_log10",

        "best_data_step",
        "best_data_loss",
        "best_data_total_loss",
        "best_data_rmse_log10",
    ]

    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def build_command(
    lambda_curv_xy: float,
    lambda_curv_t: float,
    output_dir: Path,
) -> list[str]:
    """
    Build one training command.
    """

    return [
        sys.executable,
        TRAIN_SCRIPT,

        "--window_start_index", str(WINDOW_START_INDEX),
        "--window_size_records", str(WINDOW_SIZE_RECORDS),

        "--output_dir", str(output_dir),

        "--lambda_curv_xy", str(lambda_curv_xy),
        "--lambda_curv_t", str(lambda_curv_t),

        "--num_steps", str(NUM_STEPS),
        "--seed", str(SEED),

        "--num_collocation", str(NUM_COLLOCATION),
        "--collocation_grid_nx", str(COLLOCATION_GRID_NX),
        "--collocation_grid_ny", str(COLLOCATION_GRID_NY),
    ]


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = BASE_OUTPUT_DIR / "sweep_summary.csv"

    runs = []

    for lambda_xy in LAMBDA_CURV_XY_VALUES:
        for lambda_t in LAMBDA_CURV_T_VALUES:
            run_name = (
                f"win{WINDOW_START_INDEX:04d}_size{WINDOW_SIZE_RECORDS:02d}"
                f"_xy{lambda_tag(lambda_xy)}"
                f"_t{lambda_tag(lambda_t)}"
            )

            output_dir = BASE_OUTPUT_DIR / run_name

            runs.append(
                {
                    "run_name": run_name,
                    "output_dir": output_dir,
                    "lambda_curv_xy": lambda_xy,
                    "lambda_curv_t": lambda_t,
                }
            )

    print("Lambda sweep")
    print(f"  runs:              {len(runs)}")
    print(f"  base output dir:   {BASE_OUTPUT_DIR}")
    print(f"  summary path:      {summary_path}")
    print(f"  window start:      {WINDOW_START_INDEX}")
    print(f"  window size:       {WINDOW_SIZE_RECORDS}")
    print(f"  num steps:         {NUM_STEPS}")
    print(f"  seed:              {SEED}")
    print()

    for i, run in enumerate(runs, start=1):
        run_name = run["run_name"]
        output_dir = run["output_dir"]
        lambda_xy = run["lambda_curv_xy"]
        lambda_t = run["lambda_curv_t"]

        print("=" * 80)
        print(f"Run {i}/{len(runs)}")
        print(f"  name:        {run_name}")
        print(f"  lambda_xy:   {lambda_xy}")
        print(f"  lambda_t:    {lambda_t}")
        print(f"  output_dir:  {output_dir}")

        if args.skip_existing and (output_dir / "history.csv").exists():
            print("  skipping existing run")

            summary = read_run_summary(output_dir)

            row = {
                "run_name": run_name,
                "output_dir": str(output_dir),
                "lambda_curv_xy": lambda_xy,
                "lambda_curv_t": lambda_t,
                "window_start_index": WINDOW_START_INDEX,
                "window_size_records": WINDOW_SIZE_RECORDS,
                "num_steps": NUM_STEPS,
                "seed": SEED,
                "num_collocation": NUM_COLLOCATION,
                **summary,
            }

            append_summary(summary_path, row)
            continue

        cmd = build_command(
            lambda_curv_xy=lambda_xy,
            lambda_curv_t=lambda_t,
            output_dir=output_dir,
        )

        print()
        print("Command:")
        print(" ".join(cmd))
        print()

        if args.dry_run:
            continue

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"Run failed with return code {result.returncode}")
            raise SystemExit(result.returncode)

        summary = read_run_summary(output_dir)

        row = {
            "run_name": run_name,
            "output_dir": str(output_dir),
            "lambda_curv_xy": lambda_xy,
            "lambda_curv_t": lambda_t,
            "window_start_index": WINDOW_START_INDEX,
            "window_size_records": WINDOW_SIZE_RECORDS,
            "num_steps": NUM_STEPS,
            "seed": SEED,
            "num_collocation": NUM_COLLOCATION,
            **summary,
        }

        append_summary(summary_path, row)

        print()
        print("Run summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    print()
    print("DONE")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()