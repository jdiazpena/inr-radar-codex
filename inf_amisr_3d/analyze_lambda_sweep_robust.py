# -*- coding: utf-8 -*-
"""
analyze_lambda_sweep_robust.py

Robust analysis of lambda sweeps for windowed 3D INR training.

This script does NOT use final-step ratios.

Instead, for each run it reads the run's history.csv and computes
late-training statistics after the regularization ramp:

    data_med = median(data_loss over late training)
    xy_med   = median(curv_xy_weighted over late training)
    t_med    = median(curv_t_weighted over late training)

Then:

    xy_over_data  = xy_med / data_med
    t_over_data   = t_med  / data_med
    reg_over_data = (xy_med + t_med) / data_med

This is meant to answer:

    Are the curvature losses acting as soft background priors,
    while data_loss remains the main term?

Run from inside inf_amisr_3d:

    python3 analyze_lambda_sweep_robust.py \
      --summary_csv outputs/lambda_sweep_window100_size11_focused/sweep_summary.csv \
      --output_dir outputs/lambda_sweep_window100_size11_focused/robust_analysis

Optional:

    python3 analyze_lambda_sweep_robust.py \
      --summary_csv outputs/lambda_sweep_window100_size11_focused/sweep_summary.csv \
      --output_dir outputs/lambda_sweep_window100_size11_focused/robust_analysis \
      --late_frac 0.2 \
      --reg_ramp_frac 0.2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# Helpers
# ============================================================

def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def sci_label(x: float) -> str:
    """
    Compact scientific label for plot axes.
    """

    x = float(x)

    if x == 0.0:
        return "0"

    s = f"{x:.1e}"
    s = s.replace(".0e", "e")
    s = s.replace("e-0", "e-")
    s = s.replace("e+0", "e")
    s = s.replace("e+", "e")

    return s


def resolve_history_path(row: pd.Series, summary_csv: Path) -> Path | None:
    """
    Find history.csv for one sweep row.

    Priority:
        1. row["output_dir"]/history.csv
        2. summary_csv.parent / run_name / history.csv
    """

    if "output_dir" in row and isinstance(row["output_dir"], str):
        p = Path(row["output_dir"]) / "history.csv"
        if p.exists():
            return p

    if "run_name" in row and isinstance(row["run_name"], str):
        p = summary_csv.parent / row["run_name"] / "history.csv"
        if p.exists():
            return p

    return None


def require_columns(df: pd.DataFrame, cols: list[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]

    if missing:
        raise KeyError(f"{where} is missing required columns: {missing}")


def choose_late_window(
    hist: pd.DataFrame,
    num_steps: int | None,
    reg_ramp_frac: float,
    late_frac: float,
) -> pd.DataFrame:
    """
    Select logged history rows after ramp and inside the late-training region.

    late_start_step = max(ramp_steps, late_frac start)
    """

    require_columns(hist, ["step"], "history.csv")

    hist = hist.copy()
    hist["step"] = pd.to_numeric(hist["step"], errors="coerce")
    hist = hist.dropna(subset=["step"]).copy()

    if len(hist) == 0:
        return hist

    if num_steps is None:
        max_step = int(hist["step"].max())
    else:
        max_step = int(num_steps)

    ramp_steps = int(reg_ramp_frac * max_step)
    late_start = int((1.0 - late_frac) * max_step)

    start_step = max(ramp_steps, late_start)

    late = hist[hist["step"] >= start_step].copy()

    # If too few logged rows, fall back to all after ramp.
    if len(late) < 3:
        late = hist[hist["step"] >= ramp_steps].copy()

    # If still empty, use all history.
    if len(late) == 0:
        late = hist.copy()

    return late


def compute_run_stats(
    row: pd.Series,
    summary_csv: Path,
    reg_ramp_frac: float,
    late_frac: float,
) -> dict:
    """
    Compute robust late-training statistics for one run.
    """

    out = row.to_dict()

    history_path = resolve_history_path(row, summary_csv)

    out["history_path"] = str(history_path) if history_path is not None else ""
    out["analysis_status"] = "ok"

    if history_path is None:
        out["analysis_status"] = "missing_history"
        return out

    hist = pd.read_csv(history_path)

    required = [
        "step",
        "total_loss",
        "data_loss",
        "curv_xy_weighted",
        "curv_t_weighted",
        "curv_xy_raw",
        "curv_t_raw",
        "rmse_log10",
    ]

    missing = [c for c in required if c not in hist.columns]
    if missing:
        out["analysis_status"] = f"missing_history_columns:{missing}"
        return out

    num_steps = None

    if "num_steps" in row:
        num_steps = int(safe_float(row["num_steps"], default=np.nan)) if np.isfinite(safe_float(row["num_steps"])) else None

    late = choose_late_window(
        hist=hist,
        num_steps=num_steps,
        reg_ramp_frac=reg_ramp_frac,
        late_frac=late_frac,
    )

    out["late_n_rows"] = int(len(late))
    out["late_step_min"] = int(late["step"].min())
    out["late_step_max"] = int(late["step"].max())

    # Medians: primary robust statistics.
    for col in [
        "total_loss",
        "data_loss",
        "curv_xy_weighted",
        "curv_t_weighted",
        "curv_xy_raw",
        "curv_t_raw",
        "rmse_log10",
    ]:
        out[f"{col}_late_med"] = float(late[col].median())
        out[f"{col}_late_mean"] = float(late[col].mean())
        out[f"{col}_late_p25"] = float(late[col].quantile(0.25))
        out[f"{col}_late_p75"] = float(late[col].quantile(0.75))
        out[f"{col}_late_min"] = float(late[col].min())
        out[f"{col}_late_max"] = float(late[col].max())

    data_med = out["data_loss_late_med"]
    xy_med = out["curv_xy_weighted_late_med"]
    t_med = out["curv_t_weighted_late_med"]
    total_med = out["total_loss_late_med"]

    if data_med > 0:
        out["xy_over_data_late_med"] = xy_med / data_med
        out["t_over_data_late_med"] = t_med / data_med
        out["reg_over_data_late_med"] = (xy_med + t_med) / data_med
    else:
        out["xy_over_data_late_med"] = np.nan
        out["t_over_data_late_med"] = np.nan
        out["reg_over_data_late_med"] = np.nan

    if total_med > 0:
        out["data_fraction_late_med"] = data_med / total_med
        out["xy_fraction_late_med"] = xy_med / total_med
        out["t_fraction_late_med"] = t_med / total_med
    else:
        out["data_fraction_late_med"] = np.nan
        out["xy_fraction_late_med"] = np.nan
        out["t_fraction_late_med"] = np.nan

    # Stability of data loss in late phase.
    # Large values mean final-step ratios are likely unreliable.
    data_p25 = out["data_loss_late_p25"]
    data_p75 = out["data_loss_late_p75"]

    if data_med > 0:
        out["data_iqr_over_med"] = (data_p75 - data_p25) / data_med
    else:
        out["data_iqr_over_med"] = np.nan

    return out


def classify_candidate(
    row: pd.Series,
    xy_min: float,
    xy_max: float,
    t_min: float,
    t_max: float,
    reg_min: float,
    reg_max: float,
) -> str:
    """
    Classify one row based on late-median loss balance.
    """

    xy = row["xy_over_data_late_med"]
    tt = row["t_over_data_late_med"]
    reg = row["reg_over_data_late_med"]

    if not np.isfinite(xy) or not np.isfinite(tt) or not np.isfinite(reg):
        return "bad_stats"

    if row["lambda_curv_xy"] <= 0 or row["lambda_curv_t"] <= 0:
        return "zero_lambda_control"

    if reg > reg_max:
        return "regularization_too_strong"

    if reg < reg_min:
        return "regularization_too_weak"

    if xy < xy_min:
        return "xy_too_weak"

    if xy > xy_max:
        return "xy_too_strong"

    if tt < t_min:
        return "time_too_weak"

    if tt > t_max:
        return "time_too_strong"

    return "balanced_candidate"


def balance_score(
    row: pd.Series,
    target_xy: float,
    target_t: float,
    target_reg: float,
) -> float:
    """
    Score closeness to target balance in log space.

    Smaller is better.
    """

    xy = row["xy_over_data_late_med"]
    tt = row["t_over_data_late_med"]
    reg = row["reg_over_data_late_med"]

    vals = np.array([xy, tt, reg], dtype=float)
    targets = np.array([target_xy, target_t, target_reg], dtype=float)

    if np.any(vals <= 0) or np.any(~np.isfinite(vals)):
        return np.inf

    return float(np.sqrt(np.mean((np.log10(vals) - np.log10(targets)) ** 2)))


def pivot_metric(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Pivot metric into lambda_xy x lambda_t table.

    Uses pivot_table so duplicate rows do not crash the analysis.
    """

    piv = df.pivot_table(
        index="lambda_curv_xy",
        columns="lambda_curv_t",
        values=value_col,
        aggfunc="median",
    )

    return piv.sort_index(axis=0).sort_index(axis=1)


def plot_heatmap(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    out_path: Path,
    log_values: bool = False,
    fmt: str = ".2g",
) -> None:
    piv = pivot_metric(df, value_col)

    values = piv.to_numpy(dtype=float)

    if log_values:
        plot_values = np.log10(values)
        color_label = f"log10({value_col})"
    else:
        plot_values = values
        color_label = value_col

    fig, ax = plt.subplots(figsize=(9, 6))

    im = ax.imshow(
        plot_values,
        origin="lower",
        aspect="auto",
    )

    fig.colorbar(im, ax=ax, label=color_label)

    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_yticks(np.arange(len(piv.index)))

    ax.set_xticklabels([sci_label(x) for x in piv.columns], rotation=45, ha="right")
    ax.set_yticklabels([sci_label(x) for x in piv.index])

    ax.set_xlabel("lambda_curv_t")
    ax.set_ylabel("lambda_curv_xy")
    ax.set_title(title)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]

            if not np.isfinite(val):
                text = "nan"
            elif log_values:
                text = f"{np.log10(val):.2f}"
            else:
                text = format(val, fmt)

            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_loss_composition(df: pd.DataFrame, out_path: Path) -> None:
    """
    Stacked bars using late-median loss terms normalized by data loss.
    """

    dfp = df.copy()
    dfp = dfp.sort_values("reg_over_data_late_med")

    labels = [
        f"xy={sci_label(x)}\nt={sci_label(t)}"
        for x, t in zip(dfp["lambda_curv_xy"], dfp["lambda_curv_t"])
    ]

    data = np.ones(len(dfp))
    xy = dfp["xy_over_data_late_med"].to_numpy(dtype=float)
    tt = dfp["t_over_data_late_med"].to_numpy(dtype=float)

    x = np.arange(len(dfp))

    fig, ax = plt.subplots(figsize=(max(12, 0.5 * len(dfp)), 6))

    ax.bar(x, data, label="data loss")
    ax.bar(x, xy, bottom=data, label="xy curvature")
    ax.bar(x, tt, bottom=data + xy, label="time curvature")

    ax.axhline(2.0, linestyle="--", linewidth=1.0, label="reg/data = 1 line")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right")
    ax.set_ylabel("loss contribution normalized by late-median data_loss")
    ax.set_title("Robust late-median loss composition")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_xy_t_scatter(df: pd.DataFrame, out_path: Path) -> None:
    """
    Scatter of xy/data vs t/data.
    """

    fig, ax = plt.subplots(figsize=(7, 6))

    sc = ax.scatter(
        df["xy_over_data_late_med"],
        df["t_over_data_late_med"],
        c=df["reg_over_data_late_med"],
        s=80,
    )

    fig.colorbar(sc, ax=ax, label="reg/data")

    ax.axvline(0.05, linestyle="--", linewidth=1)
    ax.axvline(0.50, linestyle="--", linewidth=1)
    ax.axhline(0.05, linestyle="--", linewidth=1)
    ax.axhline(0.50, linestyle="--", linewidth=1)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("xy curvature / data loss")
    ax.set_ylabel("time curvature / data loss")
    ax.set_title("Late-median balance: xy prior vs time prior")
    ax.grid(True, which="both", alpha=0.3)

    for _, r in df.iterrows():
        label = f"{sci_label(r['lambda_curv_xy'])},{sci_label(r['lambda_curv_t'])}"
        ax.annotate(label, (r["xy_over_data_late_med"], r["t_over_data_late_med"]), fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--summary_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--reg_ramp_frac", type=float, default=0.2)
    parser.add_argument("--late_frac", type=float, default=0.2)

    # Balance target range.
    parser.add_argument("--xy_min", type=float, default=0.05)
    parser.add_argument("--xy_max", type=float, default=0.50)
    parser.add_argument("--t_min", type=float, default=0.05)
    parser.add_argument("--t_max", type=float, default=0.50)
    parser.add_argument("--reg_min", type=float, default=0.10)
    parser.add_argument("--reg_max", type=float, default=0.80)

    # Target values for score, not hard thresholds.
    parser.add_argument("--target_xy", type=float, default=0.15)
    parser.add_argument("--target_t", type=float, default=0.15)
    parser.add_argument("--target_reg", type=float, default=0.30)

    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_sum = pd.read_csv(summary_csv)

    require_columns(
        df_sum,
        [
            "run_name",
            "output_dir",
            "lambda_curv_xy",
            "lambda_curv_t",
        ],
        "summary_csv",
    )

    # Remove exact duplicate run_names, keeping the last summary.
    before = len(df_sum)
    df_sum = df_sum.drop_duplicates(subset=["run_name"], keep="last").copy()
    after = len(df_sum)

    print(f"Loaded summary: {summary_csv}")
    print(f"Rows before duplicate cleanup: {before}")
    print(f"Rows after duplicate cleanup:  {after}")

    rows = []

    for _, row in df_sum.iterrows():
        rows.append(
            compute_run_stats(
                row=row,
                summary_csv=summary_csv,
                reg_ramp_frac=args.reg_ramp_frac,
                late_frac=args.late_frac,
            )
        )

    df = pd.DataFrame(rows)

    # Use numeric lambda columns.
    df["lambda_curv_xy"] = pd.to_numeric(df["lambda_curv_xy"], errors="coerce")
    df["lambda_curv_t"] = pd.to_numeric(df["lambda_curv_t"], errors="coerce")

    # Keep only successful rows with nonzero lambdas for candidate analysis.
    df_all = df.copy()

    df_valid = df[
        (df["analysis_status"] == "ok")
        & (df["lambda_curv_xy"] > 0)
        & (df["lambda_curv_t"] > 0)
        & np.isfinite(df["xy_over_data_late_med"])
        & np.isfinite(df["t_over_data_late_med"])
        & np.isfinite(df["reg_over_data_late_med"])
    ].copy()

    df_valid["balance_class"] = df_valid.apply(
        classify_candidate,
        axis=1,
        xy_min=args.xy_min,
        xy_max=args.xy_max,
        t_min=args.t_min,
        t_max=args.t_max,
        reg_min=args.reg_min,
        reg_max=args.reg_max,
    )

    df_valid["balance_score"] = df_valid.apply(
        balance_score,
        axis=1,
        target_xy=args.target_xy,
        target_t=args.target_t,
        target_reg=args.target_reg,
    )

    df_valid = df_valid.sort_values(
        ["balance_class", "balance_score", "reg_over_data_late_med"]
    ).reset_index(drop=True)

    candidates = df_valid[df_valid["balance_class"] == "balanced_candidate"].copy()
    candidates = candidates.sort_values("balance_score").reset_index(drop=True)

    # Save tables.
    all_path = output_dir / "robust_lambda_all_rows.csv"
    valid_path = output_dir / "robust_lambda_valid_nonzero.csv"
    cand_path = output_dir / "robust_lambda_candidates.csv"

    df_all.to_csv(all_path, index=False)
    df_valid.to_csv(valid_path, index=False)
    candidates.to_csv(cand_path, index=False)

    # Plots.
    if len(df_valid) > 0:
        plot_heatmap(
            df_valid,
            value_col="data_loss_late_med",
            title="Late-median data loss",
            out_path=output_dir / "heatmap_data_loss_late_med.png",
            log_values=True,
        )

        plot_heatmap(
            df_valid,
            value_col="curv_xy_weighted_late_med",
            title="Late-median weighted x-y curvature",
            out_path=output_dir / "heatmap_xy_weighted_late_med.png",
            log_values=True,
        )

        plot_heatmap(
            df_valid,
            value_col="curv_t_weighted_late_med",
            title="Late-median weighted time curvature",
            out_path=output_dir / "heatmap_t_weighted_late_med.png",
            log_values=True,
        )

        plot_heatmap(
            df_valid,
            value_col="xy_over_data_late_med",
            title="Late-median x-y curvature / data loss",
            out_path=output_dir / "heatmap_xy_over_data_late_med.png",
            log_values=False,
            fmt=".2f",
        )

        plot_heatmap(
            df_valid,
            value_col="t_over_data_late_med",
            title="Late-median time curvature / data loss",
            out_path=output_dir / "heatmap_t_over_data_late_med.png",
            log_values=False,
            fmt=".2f",
        )

        plot_heatmap(
            df_valid,
            value_col="reg_over_data_late_med",
            title="Late-median total regularization / data loss",
            out_path=output_dir / "heatmap_reg_over_data_late_med.png",
            log_values=False,
            fmt=".2f",
        )

        plot_heatmap(
            df_valid,
            value_col="rmse_log10_late_med",
            title="Late-median RMSE log10 sanity check",
            out_path=output_dir / "heatmap_rmse_log10_late_med.png",
            log_values=True,
        )

        plot_loss_composition(
            df_valid,
            out_path=output_dir / "loss_composition_late_median.png",
        )

        plot_xy_t_scatter(
            df_valid,
            out_path=output_dir / "scatter_xy_vs_t_late_median.png",
        )

    # Write text summary.
    summary_txt = output_dir / "robust_balance_summary.txt"

    with open(summary_txt, "w") as f:
        f.write("Robust lambda sweep analysis\n")
        f.write("============================\n\n")
        f.write(f"summary_csv: {summary_csv}\n")
        f.write(f"reg_ramp_frac: {args.reg_ramp_frac}\n")
        f.write(f"late_frac: {args.late_frac}\n\n")

        f.write("Target ranges:\n")
        f.write(f"  xy/data:  {args.xy_min} to {args.xy_max}\n")
        f.write(f"  t/data:   {args.t_min} to {args.t_max}\n")
        f.write(f"  reg/data: {args.reg_min} to {args.reg_max}\n\n")

        f.write(f"valid nonzero runs: {len(df_valid)}\n")
        f.write(f"balanced candidates: {len(candidates)}\n\n")

        if len(candidates) > 0:
            f.write("Balanced candidates, sorted by balance_score:\n\n")
            cols = [
                "run_name",
                "lambda_curv_xy",
                "lambda_curv_t",
                "xy_over_data_late_med",
                "t_over_data_late_med",
                "reg_over_data_late_med",
                "data_loss_late_med",
                "rmse_log10_late_med",
                "balance_score",
            ]

            f.write(candidates[cols].to_string(index=False))
            f.write("\n")
        else:
            f.write("No candidates found inside the requested target ranges.\n\n")
            f.write("Closest nonzero runs by balance_score:\n\n")

            cols = [
                "run_name",
                "lambda_curv_xy",
                "lambda_curv_t",
                "xy_over_data_late_med",
                "t_over_data_late_med",
                "reg_over_data_late_med",
                "data_loss_late_med",
                "rmse_log10_late_med",
                "balance_score",
                "balance_class",
            ]

            f.write(df_valid.sort_values("balance_score")[cols].head(10).to_string(index=False))
            f.write("\n")

    print()
    print("Saved:")
    print(f"  {all_path}")
    print(f"  {valid_path}")
    print(f"  {cand_path}")
    print(f"  {summary_txt}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()