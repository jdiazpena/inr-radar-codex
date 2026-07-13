# -*- coding: utf-8 -*-
"""
plot_lambda_balance.py

Analyze a lambda sweep for the 3D windowed AMISR INR.

Purpose:
    This script is NOT trying to find the lowest-RMSE model.
    It is meant to find lambda values that give a reasonable loss balance:

        data_loss should remain the main term
        spatial curvature should be active but secondary
        temporal curvature should be active but secondary

Main ratios:
    xy_over_data  = curv_xy_weighted / data_loss
    t_over_data   = curv_t_weighted  / data_loss
    reg_over_data = (curv_xy_weighted + curv_t_weighted) / data_loss

Default target balance:
    xy_over_data  between 0.05 and 0.30
    t_over_data   between 0.05 and 0.50
    reg_over_data between 0.10 and 0.80

Run from inside inf_amisr_3d:

    python3 plot_lambda_balance.py \
      --summary_csv outputs/lambda_sweep_window100_size11/sweep_summary.csv \
      --output_dir outputs/lambda_sweep_window100_size11/balance_plots

Optional:
    --include_zero
    --metric final
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sci_label(x: float) -> str:
    if x == 0:
        return "0"
    return f"{x:.0e}"


def ratio_safe(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return num / den


def get_metric_columns(metric: str) -> dict[str, str]:
    """
    Select which columns to analyze.

    For this use case, final is the safest default because the sweep_summary
    best_total values may include ramp-time checkpoints if they were generated
    before the after-ramp checkpoint logic was corrected.
    """

    if metric == "final":
        return {
            "data": "final_data_loss",
            "xy": "final_curv_xy_weighted",
            "t": "final_curv_t_weighted",
            "total": "final_total_loss",
            "rmse": "final_rmse_log10",
            "step": "final_step",
        }

    if metric == "best_total":
        return {
            "data": "best_total_data_loss",
            "xy": "best_total_curv_xy_weighted",
            "t": "best_total_curv_t_weighted",
            "total": "best_total_loss",
            "rmse": "best_total_rmse_log10",
            "step": "best_total_step",
        }

    raise ValueError(f"Unknown metric: {metric}")


def compute_balance_table(
    df: pd.DataFrame,
    metric: str,
    include_zero: bool,
    xy_min: float,
    xy_max: float,
    t_min: float,
    t_max: float,
    reg_min: float,
    reg_max: float,
) -> pd.DataFrame:
    cols = get_metric_columns(metric)

    required = [
        "lambda_curv_xy",
        "lambda_curv_t",
        cols["data"],
        cols["xy"],
        cols["t"],
        cols["total"],
        cols["rmse"],
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = df.copy()

    if "status" in out.columns:
        out = out[out["status"] == "ok"].copy()

    if not include_zero:
        out = out[
            (out["lambda_curv_xy"] > 0.0)
            & (out["lambda_curv_t"] > 0.0)
        ].copy()

    out["data_loss"] = out[cols["data"]]
    out["xy_weighted"] = out[cols["xy"]]
    out["t_weighted"] = out[cols["t"]]
    out["total_loss"] = out[cols["total"]]
    out["rmse_log10"] = out[cols["rmse"]]

    out["xy_over_data"] = ratio_safe(out["xy_weighted"], out["data_loss"])
    out["t_over_data"] = ratio_safe(out["t_weighted"], out["data_loss"])
    out["reg_over_data"] = ratio_safe(
        out["xy_weighted"] + out["t_weighted"],
        out["data_loss"],
    )
    out["data_fraction_of_total"] = ratio_safe(out["data_loss"], out["total_loss"])

    out["xy_in_range"] = out["xy_over_data"].between(xy_min, xy_max)
    out["t_in_range"] = out["t_over_data"].between(t_min, t_max)
    out["reg_in_range"] = out["reg_over_data"].between(reg_min, reg_max)
    out["balance_ok"] = (
        out["xy_in_range"]
        & out["t_in_range"]
        & out["reg_in_range"]
    )

    # Score is only for sorting balance candidates.
    # It penalizes distance from the center of the desired ratio ranges.
    xy_target = np.sqrt(xy_min * xy_max)
    t_target = np.sqrt(t_min * t_max)
    reg_target = np.sqrt(reg_min * reg_max)

    eps = 1e-30
    out["balance_score"] = (
        np.abs(np.log10((out["xy_over_data"] + eps) / xy_target))
        + np.abs(np.log10((out["t_over_data"] + eps) / t_target))
        + np.abs(np.log10((out["reg_over_data"] + eps) / reg_target))
    )

    keep_cols = [
        "run_name",
        "output_dir",
        "lambda_curv_xy",
        "lambda_curv_t",
        "data_loss",
        "xy_weighted",
        "t_weighted",
        "total_loss",
        "xy_over_data",
        "t_over_data",
        "reg_over_data",
        "data_fraction_of_total",
        "rmse_log10",
        "balance_ok",
        "balance_score",
    ]

    available = [c for c in keep_cols if c in out.columns]
    out = out[available].copy()

    out = out.sort_values(
        ["balance_ok", "balance_score", "reg_over_data"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    return out


def pivot_metric(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    piv = df.pivot(
        index="lambda_curv_xy",
        columns="lambda_curv_t",
        values=value_col,
    )
    piv = piv.sort_index(axis=0).sort_index(axis=1)
    return piv


def plot_heatmap(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    out_path: Path,
    fmt: str = ".2g",
) -> None:
    piv = pivot_metric(df, value_col)

    values = piv.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(values, aspect="auto")
    fig.colorbar(im, ax=ax, label=value_col)

    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_yticks(np.arange(len(piv.index)))

    ax.set_xticklabels([sci_label(v) for v in piv.columns], rotation=45, ha="right")
    ax.set_yticklabels([sci_label(v) for v in piv.index])

    ax.set_xlabel("lambda_curv_t")
    ax.set_ylabel("lambda_curv_xy")
    ax.set_title(title)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isfinite(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_balance_scatter(
    df: pd.DataFrame,
    out_path: Path,
    xy_min: float,
    xy_max: float,
    t_min: float,
    t_max: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    sc = ax.scatter(
        df["xy_over_data"],
        df["t_over_data"],
        s=80,
        c=df["reg_over_data"],
    )
    fig.colorbar(sc, ax=ax, label="reg_over_data")

    # Target rectangle.
    ax.plot(
        [xy_min, xy_max, xy_max, xy_min, xy_min],
        [t_min, t_min, t_max, t_max, t_min],
        linewidth=2,
    )

    for _, row in df.iterrows():
        label = f"xy={sci_label(row['lambda_curv_xy'])}\nt={sci_label(row['lambda_curv_t'])}"
        ax.annotate(label, (row["xy_over_data"], row["t_over_data"]), fontsize=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("xy curvature / data loss")
    ax.set_ylabel("time curvature / data loss")
    ax.set_title("Loss-balance map")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_loss_composition(df: pd.DataFrame, out_path: Path) -> None:
    """
    Plot normalized loss composition.

    Bars are normalized by data_loss, so data is always 1.
    This directly shows whether priors are background pressure or dominant.
    """

    plot_df = df.sort_values("reg_over_data").copy()

    labels = [
        f"xy={sci_label(x)}\nt={sci_label(t)}"
        for x, t in zip(plot_df["lambda_curv_xy"], plot_df["lambda_curv_t"])
    ]

    x = np.arange(len(plot_df))

    data = np.ones(len(plot_df))
    xy = plot_df["xy_over_data"].to_numpy(dtype=float)
    tt = plot_df["t_over_data"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(plot_df)), 6))

    ax.bar(x, data, label="data loss")
    ax.bar(x, xy, bottom=data, label="xy curvature")
    ax.bar(x, tt, bottom=data + xy, label="time curvature")

    ax.axhline(2.0, linestyle="--", linewidth=1)
    ax.set_ylabel("loss contribution normalized by data_loss")
    ax.set_title("Loss composition: data term should remain the main term")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_text_summary(
    table: pd.DataFrame,
    out_path: Path,
    xy_min: float,
    xy_max: float,
    t_min: float,
    t_max: float,
    reg_min: float,
    reg_max: float,
) -> None:
    ok = table[table["balance_ok"]].copy()

    lines = []
    lines.append("Lambda balance analysis")
    lines.append("")
    lines.append("Goal:")
    lines.append("  data_loss remains the main term")
    lines.append("  xy curvature is active but secondary")
    lines.append("  time curvature is active but secondary")
    lines.append("")
    lines.append("Target ranges:")
    lines.append(f"  xy_over_data:  {xy_min:g} to {xy_max:g}")
    lines.append(f"  t_over_data:   {t_min:g} to {t_max:g}")
    lines.append(f"  reg_over_data: {reg_min:g} to {reg_max:g}")
    lines.append("")
    lines.append(f"Rows analyzed: {len(table)}")
    lines.append(f"Rows inside target balance: {len(ok)}")
    lines.append("")

    if len(ok) > 0:
        lines.append("Best balance candidates:")
        for _, row in ok.head(10).iterrows():
            lines.append(
                "  "
                f"lambda_xy={row['lambda_curv_xy']:.3e}, "
                f"lambda_t={row['lambda_curv_t']:.3e}, "
                f"xy/data={row['xy_over_data']:.3g}, "
                f"t/data={row['t_over_data']:.3g}, "
                f"reg/data={row['reg_over_data']:.3g}, "
                f"rmse={row['rmse_log10']:.3g}"
            )
    else:
        lines.append("No rows satisfied the target balance.")
        lines.append("Closest rows by balance_score:")
        for _, row in table.head(10).iterrows():
            lines.append(
                "  "
                f"lambda_xy={row['lambda_curv_xy']:.3e}, "
                f"lambda_t={row['lambda_curv_t']:.3e}, "
                f"xy/data={row['xy_over_data']:.3g}, "
                f"t/data={row['t_over_data']:.3g}, "
                f"reg/data={row['reg_over_data']:.3g}, "
                f"rmse={row['rmse_log10']:.3g}, "
                f"score={row['balance_score']:.3g}"
            )

    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--summary_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--metric",
        type=str,
        default="final",
        choices=["final", "best_total"],
        help="Which stored summary values to analyze. Use final by default.",
    )

    parser.add_argument(
        "--include_zero",
        action="store_true",
        help="Include zero-lambda baseline/control rows.",
    )

    parser.add_argument("--xy_min", type=float, default=0.05)
    parser.add_argument("--xy_max", type=float, default=0.30)
    parser.add_argument("--t_min", type=float, default=0.05)
    parser.add_argument("--t_max", type=float, default=0.50)
    parser.add_argument("--reg_min", type=float, default=0.10)
    parser.add_argument("--reg_max", type=float, default=0.80)

    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)

    table = compute_balance_table(
        df=df,
        metric=args.metric,
        include_zero=args.include_zero,
        xy_min=args.xy_min,
        xy_max=args.xy_max,
        t_min=args.t_min,
        t_max=args.t_max,
        reg_min=args.reg_min,
        reg_max=args.reg_max,
    )

    balance_csv = output_dir / "lambda_balance_table.csv"
    table.to_csv(balance_csv, index=False)

    ok_csv = output_dir / "lambda_balance_candidates.csv"
    table[table["balance_ok"]].to_csv(ok_csv, index=False)

    plot_heatmap(
        table,
        "xy_over_data",
        "Spatial curvature / data loss",
        output_dir / "heatmap_xy_over_data.png",
        fmt=".2g",
    )

    plot_heatmap(
        table,
        "t_over_data",
        "Temporal curvature / data loss",
        output_dir / "heatmap_t_over_data.png",
        fmt=".2g",
    )

    plot_heatmap(
        table,
        "reg_over_data",
        "Total regularization / data loss",
        output_dir / "heatmap_reg_over_data.png",
        fmt=".2g",
    )

    plot_heatmap(
        table,
        "data_fraction_of_total",
        "Data fraction of total loss",
        output_dir / "heatmap_data_fraction_of_total.png",
        fmt=".2g",
    )

    plot_heatmap(
        table,
        "rmse_log10",
        "RMSE log10(Ne), for sanity only",
        output_dir / "heatmap_rmse_log10_sanity.png",
        fmt=".2g",
    )

    plot_balance_scatter(
        table,
        output_dir / "scatter_xy_vs_t_balance.png",
        xy_min=args.xy_min,
        xy_max=args.xy_max,
        t_min=args.t_min,
        t_max=args.t_max,
    )

    plot_loss_composition(
        table,
        output_dir / "loss_composition_normalized_by_data.png",
    )

    write_text_summary(
        table,
        output_dir / "balance_summary.txt",
        xy_min=args.xy_min,
        xy_max=args.xy_max,
        t_min=args.t_min,
        t_max=args.t_max,
        reg_min=args.reg_min,
        reg_max=args.reg_max,
    )

    print("DONE")
    print(f"Balance table:      {balance_csv}")
    print(f"Candidate table:    {ok_csv}")
    print(f"Text summary:       {output_dir / 'balance_summary.txt'}")
    print(f"Plots saved in:     {output_dir}")

    print()
    print("Top rows by balance score:")
    show_cols = [
        "lambda_curv_xy",
        "lambda_curv_t",
        "xy_over_data",
        "t_over_data",
        "reg_over_data",
        "rmse_log10",
        "balance_ok",
    ]
    print(table[show_cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
