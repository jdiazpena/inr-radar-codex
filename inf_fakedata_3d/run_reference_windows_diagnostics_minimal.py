# -*- coding: utf-8 -*-
"""
run_reference_windows_first20.py

Sequential wrapper for train_radar_3d_window_reference_reg.py.

Purpose:
    Run the reference-ratio regularized trainer on multiple overlapping
    temporal windows.

Default:
    start_index = 0
    window_size_records = 11
    stride_records = 5
    num_windows = 20

This produces starts:
    0, 5, 10, ..., 95

Each window is trained separately and saved in its own output folder.

Run from inside inf_amisr_3d:

    python3 run_reference_windows_first20.py --dry_run

    python3 run_reference_windows_first20.py

Resume without rerunning completed windows:

    python3 run_reference_windows_first20.py --skip_existing
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


# ============================================================
# Defaults for this experiment
# ============================================================

TRAIN_SCRIPT = "train_radar_3d_window_reference_reg_diagnostics.py"

DEFAULT_H5_PATH = "../data/20120122.001_lp_2min-fitcal.h5"

DEFAULT_BASE_OUTPUT_DIR = Path("outputs/reference_ratio_windows_2minute_denser")

DEFAULT_START_INDEX = 0
# DEFAULT_WINDOW_SIZE_RECORDS = 11
DEFAULT_WINDOW_SIZE_RECORDS = 31
# DEFAULT_STRIDE_RECORDS = 5
DEFAULT_STRIDE_RECORDS = 15

DEFAULT_NUM_WINDOWS = 75

DEFAULT_TARGET_XY_RATIO = 0.30
DEFAULT_TARGET_T_RATIO = 0.30
DEFAULT_EPSILON_DATA = 1e-6

DEFAULT_NUM_STEPS = 10000
DEFAULT_SEED = 0

DEFAULT_NUM_COLLOCATION = 8192
DEFAULT_COLLOCATION_GRID_NX = 80
DEFAULT_COLLOCATION_GRID_NY = 80

DEFAULT_FREEZE_LAMBDAS_AFTER_STEP = 0

# Diagnostics passed to the diagnostic trainer.
# component_grad_every=500 means expensive parameter-gradient diagnostics
# are computed every 500 training steps.
DEFAULT_DERIV_ZERO_EPSILON = 1e-10
DEFAULT_NUM_DIAGNOSTIC_COLLOCATION = 8192
DEFAULT_COMPONENT_GRAD_EVERY = 500


# ============================================================
# Helpers
# ============================================================

def make_window_starts(
    start_index: int,
    stride_records: int,
    num_windows: int,
) -> list[int]:
    return [
        start_index + i * stride_records
        for i in range(num_windows)
    ]


def window_center_index(window_start: int, window_size_records: int) -> int:
    return window_start + window_size_records // 2


def run_name_for_window(
    window_start: int,
    window_size_records: int,
) -> str:
    center = window_center_index(window_start, window_size_records)

    return (
        f"win_start_{window_start:04d}"
        f"_center_{center:04d}"
        f"_size_{window_size_records:02d}"
    )


def build_command(
    args: argparse.Namespace,
    window_start: int,
    output_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        TRAIN_SCRIPT,

        "--h5_path", str(args.h5_path),

        "--window_start_index", str(window_start),
        "--window_size_records", str(args.window_size_records),

        "--output_dir", str(output_dir),

        "--reference_loss_weights",
        "--target_xy_ratio", str(args.target_xy_ratio),
        "--target_t_ratio", str(args.target_t_ratio),
        "--epsilon_data", str(args.epsilon_data),

        "--num_steps", str(args.num_steps),
        "--seed", str(args.seed),

        "--num_collocation", str(args.num_collocation),
        "--collocation_grid_nx", str(args.collocation_grid_nx),
        "--collocation_grid_ny", str(args.collocation_grid_ny),

        "--deriv_zero_epsilon", str(args.deriv_zero_epsilon),
        "--num_diagnostic_collocation", str(args.num_diagnostic_collocation),
        "--component_grad_every", str(args.component_grad_every),
    ]

    if args.freeze_lambdas_after_step > 0:
        cmd += [
            "--freeze_lambdas_after_step",
            str(args.freeze_lambdas_after_step),
        ]

    if args.no_plots:
        cmd += ["--no_plots"]

    return cmd


def read_final_summary(output_dir: Path) -> dict:
    """
    Read a compact summary from one finished window.

    Uses history.csv and predictions_at_measured_points.csv if available.
    """

    out = {
        "status": "ok",
        "final_step": "",
        "final_total_loss": "",
        "final_data_loss": "",
        "final_curv_xy_weighted": "",
        "final_curv_t_weighted": "",
        "final_lambda_curv_xy_eff": "",
        "final_lambda_curv_t_eff": "",
        "final_data_reference": "",
        "final_xy_over_data_reference": "",
        "final_t_over_data_reference": "",
        "final_rmse_log10": "",
        "best_total_step": "",
        "best_total_loss": "",
        "best_data_loss_at_best_total": "",
        "best_rmse_log10_at_best_total": "",
    }

    history_path = output_dir / "history.csv"

    if not history_path.exists():
        out["status"] = "missing_history"
        return out

    hist = pd.read_csv(history_path)

    if len(hist) == 0:
        out["status"] = "empty_history"
        return out

    final = hist.iloc[-1]

    def maybe_get(row, key, default=""):
        if key in row.index:
            return row[key]
        return default

    out["final_step"] = int(maybe_get(final, "step", -1))
    out["final_total_loss"] = float(maybe_get(final, "total_loss", "nan"))
    out["final_data_loss"] = float(maybe_get(final, "data_loss", "nan"))
    out["final_curv_xy_weighted"] = float(maybe_get(final, "curv_xy_weighted", "nan"))
    out["final_curv_t_weighted"] = float(maybe_get(final, "curv_t_weighted", "nan"))
    out["final_lambda_curv_xy_eff"] = float(maybe_get(final, "lambda_curv_xy_eff", "nan"))
    out["final_lambda_curv_t_eff"] = float(maybe_get(final, "lambda_curv_t_eff", "nan"))
    out["final_data_reference"] = float(maybe_get(final, "data_reference", "nan"))
    out["final_xy_over_data_reference"] = float(maybe_get(final, "xy_over_data_reference", "nan"))
    out["final_t_over_data_reference"] = float(maybe_get(final, "t_over_data_reference", "nan"))
    out["final_rmse_log10"] = float(maybe_get(final, "rmse_log10", "nan"))

    if "total_loss" in hist.columns:
        best_idx = hist["total_loss"].idxmin()
        best = hist.loc[best_idx]

        out["best_total_step"] = int(maybe_get(best, "step", -1))
        out["best_total_loss"] = float(maybe_get(best, "total_loss", "nan"))
        out["best_data_loss_at_best_total"] = float(maybe_get(best, "data_loss", "nan"))
        out["best_rmse_log10_at_best_total"] = float(maybe_get(best, "rmse_log10", "nan"))

    return out


def append_or_update_summary(summary_path: Path, row: dict) -> None:
    """
    Upsert one row into the summary file.

    This avoids duplicated rows if the wrapper is restarted.
    """

    fieldnames = [
        "run_name",
        "output_dir",
        "window_start_index",
        "window_center_index",
        "window_size_records",
        "stride_records",
        "target_xy_ratio",
        "target_t_ratio",
        "epsilon_data",
        "num_steps",
        "seed",
        "num_collocation",
        "deriv_zero_epsilon",
        "num_diagnostic_collocation",
        "component_grad_every",
        "runtime_sec",
        "status",

        "final_step",
        "final_total_loss",
        "final_data_loss",
        "final_curv_xy_weighted",
        "final_curv_t_weighted",
        "final_lambda_curv_xy_eff",
        "final_lambda_curv_t_eff",
        "final_data_reference",
        "final_xy_over_data_reference",
        "final_t_over_data_reference",
        "final_rmse_log10",

        "best_total_step",
        "best_total_loss",
        "best_data_loss_at_best_total",
        "best_rmse_log10_at_best_total",
    ]

    row_df = pd.DataFrame([row], columns=fieldnames)

    if summary_path.exists():
        old_df = pd.read_csv(summary_path)

        if "run_name" in old_df.columns:
            old_df = old_df[old_df["run_name"] != row["run_name"]].copy()

        out_df = pd.concat([old_df, row_df], ignore_index=True)
    else:
        out_df = row_df

    out_df.to_csv(summary_path, index=False)


def write_coverage_report(
    base_output_dir: Path,
    starts: list[int],
    window_size_records: int,
) -> None:
    """
    Save a small CSV showing how many windows cover each time record.
    """

    coverage = {}

    for start in starts:
        for idx in range(start, start + window_size_records):
            coverage[idx] = coverage.get(idx, 0) + 1

    rows = [
        {
            "time_index": idx,
            "n_windows_covering": coverage[idx],
        }
        for idx in sorted(coverage)
    ]

    df = pd.DataFrame(rows)

    path = base_output_dir / "coverage_report.csv"
    df.to_csv(path, index=False)

    print(f"Saved coverage report: {path}")
    print()
    print("Coverage summary:")
    print(df["n_windows_covering"].value_counts().sort_index())


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_output_dir",
        type=str,
        default=str(DEFAULT_BASE_OUTPUT_DIR),
    )

    parser.add_argument(
        "--h5_path",
        type=str,
        default=DEFAULT_H5_PATH,
        help="AMISR HDF5 file passed to the trainer.",
    )

    parser.add_argument("--start_index", type=int, default=DEFAULT_START_INDEX)
    parser.add_argument("--window_size_records", type=int, default=DEFAULT_WINDOW_SIZE_RECORDS)
    parser.add_argument("--stride_records", type=int, default=DEFAULT_STRIDE_RECORDS)
    parser.add_argument("--num_windows", type=int, default=DEFAULT_NUM_WINDOWS)

    parser.add_argument("--target_xy_ratio", type=float, default=DEFAULT_TARGET_XY_RATIO)
    parser.add_argument("--target_t_ratio", type=float, default=DEFAULT_TARGET_T_RATIO)
    parser.add_argument("--epsilon_data", type=float, default=DEFAULT_EPSILON_DATA)

    parser.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--num_collocation", type=int, default=DEFAULT_NUM_COLLOCATION)
    parser.add_argument("--collocation_grid_nx", type=int, default=DEFAULT_COLLOCATION_GRID_NX)
    parser.add_argument("--collocation_grid_ny", type=int, default=DEFAULT_COLLOCATION_GRID_NY)

    parser.add_argument(
        "--freeze_lambdas_after_step",
        type=int,
        default=DEFAULT_FREEZE_LAMBDAS_AFTER_STEP,
        help="0 means never freeze lambdas.",
    )

    parser.add_argument("--deriv_zero_epsilon", type=float, default=DEFAULT_DERIV_ZERO_EPSILON)
    parser.add_argument("--num_diagnostic_collocation", type=int, default=DEFAULT_NUM_DIAGNOSTIC_COLLOCATION)
    parser.add_argument("--component_grad_every", type=int, default=DEFAULT_COMPONENT_GRAD_EVERY)

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--no_plots", action="store_true")

    args = parser.parse_args()

    base_output_dir = Path(args.base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(TRAIN_SCRIPT).exists():
        raise FileNotFoundError(
            f"Could not find {TRAIN_SCRIPT}. "
            f"Run this wrapper from inside inf_amisr_3d."
        )

    starts = make_window_starts(
        start_index=args.start_index,
        stride_records=args.stride_records,
        num_windows=args.num_windows,
    )

    summary_path = base_output_dir / "window_run_summary.csv"

    print("Reference-ratio window wrapper")
    print(f"  train script:          {TRAIN_SCRIPT}")
    print(f"  base output dir:       {base_output_dir}")
    print(f"  summary path:          {summary_path}")
    print(f"  h5 path:               {args.h5_path}")
    print(f"  start index:           {args.start_index}")
    print(f"  window size records:   {args.window_size_records}")
    print(f"  stride records:        {args.stride_records}")
    print(f"  num windows:           {args.num_windows}")
    print(f"  first start:           {starts[0]}")
    print(f"  last start:            {starts[-1]}")
    print(f"  target xy ratio:       {args.target_xy_ratio}")
    print(f"  target t ratio:        {args.target_t_ratio}")
    print(f"  epsilon data:          {args.epsilon_data}")
    print(f"  num steps:             {args.num_steps}")
    print(f"  seed:                  {args.seed}")
    print(f"  num collocation:       {args.num_collocation}")
    print(f"  deriv zero epsilon:    {args.deriv_zero_epsilon}")
    print(f"  diag collocation:      {args.num_diagnostic_collocation}")
    print(f"  comp grad every:       {args.component_grad_every}")
    print()

    write_coverage_report(
        base_output_dir=base_output_dir,
        starts=starts,
        window_size_records=args.window_size_records,
    )

    for i, window_start in enumerate(starts, start=1):
        center = window_center_index(window_start, args.window_size_records)
        run_name = run_name_for_window(window_start, args.window_size_records)
        output_dir = base_output_dir / run_name

        print()
        print("=" * 80)
        print(f"Window {i}/{len(starts)}")
        print(f"  run name:      {run_name}")
        print(f"  start index:   {window_start}")
        print(f"  center index:  {center}")
        print(f"  output dir:    {output_dir}")

        model_final = output_dir / "model_final.pt"

        if args.skip_existing and model_final.exists():
            print("  skipping existing completed run")

            summary = read_final_summary(output_dir)

            row = {
                "run_name": run_name,
                "output_dir": str(output_dir),
                "window_start_index": window_start,
                "window_center_index": center,
                "window_size_records": args.window_size_records,
                "stride_records": args.stride_records,
                "target_xy_ratio": args.target_xy_ratio,
                "target_t_ratio": args.target_t_ratio,
                "epsilon_data": args.epsilon_data,
                "num_steps": args.num_steps,
                "seed": args.seed,
                "num_collocation": args.num_collocation,
                "deriv_zero_epsilon": args.deriv_zero_epsilon,
                "num_diagnostic_collocation": args.num_diagnostic_collocation,
                "component_grad_every": args.component_grad_every,
                "runtime_sec": "",
                **summary,
            }

            append_or_update_summary(summary_path, row)
            continue

        cmd = build_command(
            args=args,
            window_start=window_start,
            output_dir=output_dir,
        )

        print()
        print("Command:")
        print(" ".join(cmd))
        print()

        if args.dry_run:
            continue

        t0 = time.time()

        result = subprocess.run(cmd)

        runtime_sec = time.time() - t0

        if result.returncode != 0:
            print(f"Run failed with return code {result.returncode}")
            raise SystemExit(result.returncode)

        summary = read_final_summary(output_dir)

        row = {
            "run_name": run_name,
            "output_dir": str(output_dir),
            "window_start_index": window_start,
            "window_center_index": center,
            "window_size_records": args.window_size_records,
            "stride_records": args.stride_records,
            "target_xy_ratio": args.target_xy_ratio,
            "target_t_ratio": args.target_t_ratio,
            "epsilon_data": args.epsilon_data,
            "num_steps": args.num_steps,
            "seed": args.seed,
            "num_collocation": args.num_collocation,
            "deriv_zero_epsilon": args.deriv_zero_epsilon,
            "num_diagnostic_collocation": args.num_diagnostic_collocation,
            "component_grad_every": args.component_grad_every,
            "runtime_sec": runtime_sec,
            **summary,
        }

        append_or_update_summary(summary_path, row)

        print()
        print("Finished window:")
        print(f"  runtime_sec: {runtime_sec:.1f}")
        print(f"  status:      {summary['status']}")
        print(f"  final data:  {summary.get('final_data_loss', '')}")
        print(f"  final xyRef: {summary.get('final_xy_over_data_reference', '')}")
        print(f"  final tRef:  {summary.get('final_t_over_data_reference', '')}")

    print()
    print("DONE")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()