# -*- coding: utf-8 -*-
"""
plot_lambda_groups.py

Grouped loss-composition plots for lambda sweep analysis.

This script uses robust late-median ratios, not final-step ratios.

It creates:

1. For each fixed spatial lambda:
       one bar plot showing all time lambdas in increasing order

2. For each fixed time lambda:
       one bar plot showing all spatial lambdas in increasing order

Each bar is normalized by data_loss:

    data = 1
    xy   = curv_xy_weighted_late_med / data_loss_late_med
    t    = curv_t_weighted_late_med  / data_loss_late_med

The dashed line at y=2 means:

    regularization = data

So below y=2:
    data is still the largest contribution

Run from inf_amisr_3d:

    python3 plot_lambda_groups.py \
      --robust_csv outputs/lambda_sweep_window100_size11_focused/robust_analysis/robust_lambda_valid_nonzero.csv \
      --output_dir outputs/lambda_sweep_window100_size11_focused/robust_analysis/grouped_lambda_plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def sci_label(x: float) -> str:
    x = float(x)

    if x == 0.0:
        return "0"

    s = f"{x:.1e}"
    s = s.replace(".0e", "e")
    s = s.replace("e-0", "e-")
    s = s.replace("e+0", "e")
    s = s.replace("e+", "e")

    return s


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def get_global_ylim(df: pd.DataFrame, margin: float = 1.10) -> float:
    total_height = (
        1.0
        + df["xy_over_data_late_med"].to_numpy(dtype=float)
        + df["t_over_data_late_med"].to_numpy(dtype=float)
    )

    ymax = float(np.nanmax(total_height))

    if not np.isfinite(ymax):
        return 2.0

    ymax = max(2.1, ymax * margin)

    return ymax


def plot_group(
    group_df: pd.DataFrame,
    varying_col: str,
    fixed_label: str,
    title: str,
    out_path: Path,
    ylim: float,
) -> None:
    group_df = group_df.sort_values(varying_col).copy()

    labels = [sci_label(x) for x in group_df[varying_col]]

    data = np.ones(len(group_df))
    xy = group_df["xy_over_data_late_med"].to_numpy(dtype=float)
    tt = group_df["t_over_data_late_med"].to_numpy(dtype=float)

    x = np.arange(len(group_df))

    fig_width = max(8, 0.7 * len(group_df))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))

    ax.bar(x, data, label="data loss")
    ax.bar(x, xy, bottom=data, label="x-y curvature")
    ax.bar(x, tt, bottom=data + xy, label="time curvature")

    ax.axhline(
        2.0,
        linestyle="--",
        linewidth=1.2,
        label="reg/data = 1 line",
    )

    ax.set_ylim(0.0, ylim)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_xlabel(varying_col)
    ax.set_ylabel("loss contribution normalized by late-median data_loss")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")

    # Put the actual ratios above bars.
    for i, (_, row) in enumerate(group_df.iterrows()):
        total = 1.0 + row["xy_over_data_late_med"] + row["t_over_data_late_med"]
        txt = (
            f"xy={row['xy_over_data_late_med']:.2g}\n"
            f"t={row['t_over_data_late_med']:.2g}"
        )
        ax.text(
            i,
            total + 0.03 * ylim,
            txt,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_all_fixed_xy(df: pd.DataFrame, output_dir: Path, ylim: float) -> None:
    out_dir = output_dir / "fixed_xy"
    out_dir.mkdir(parents=True, exist_ok=True)

    for lambda_xy in sorted(df["lambda_curv_xy"].unique()):
        group = df[df["lambda_curv_xy"] == lambda_xy].copy()

        title = (
            f"Loss composition for fixed spatial curvature lambda\n"
            f"lambda_curv_xy = {sci_label(lambda_xy)}"
        )

        out_path = out_dir / f"fixed_xy_{sci_label(lambda_xy).replace('-', 'm')}.png"

        plot_group(
            group_df=group,
            varying_col="lambda_curv_t",
            fixed_label=f"lambda_curv_xy={sci_label(lambda_xy)}",
            title=title,
            out_path=out_path,
            ylim=ylim,
        )


def plot_all_fixed_t(df: pd.DataFrame, output_dir: Path, ylim: float) -> None:
    out_dir = output_dir / "fixed_t"
    out_dir.mkdir(parents=True, exist_ok=True)

    for lambda_t in sorted(df["lambda_curv_t"].unique()):
        group = df[df["lambda_curv_t"] == lambda_t].copy()

        title = (
            f"Loss composition for fixed temporal curvature lambda\n"
            f"lambda_curv_t = {sci_label(lambda_t)}"
        )

        out_path = out_dir / f"fixed_t_{sci_label(lambda_t).replace('-', 'm')}.png"

        plot_group(
            group_df=group,
            varying_col="lambda_curv_xy",
            fixed_label=f"lambda_curv_t={sci_label(lambda_t)}",
            title=title,
            out_path=out_path,
            ylim=ylim,
        )


def plot_combined_panels_fixed_xy(df: pd.DataFrame, output_dir: Path, ylim: float) -> None:
    lambdas_xy = sorted(df["lambda_curv_xy"].unique())

    n = len(lambdas_xy)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(13, 4.2 * nrows),
        squeeze=False,
    )

    for ax in axes.ravel():
        ax.axis("off")

    for ax, lambda_xy in zip(axes.ravel(), lambdas_xy):
        ax.axis("on")

        group = df[df["lambda_curv_xy"] == lambda_xy].sort_values("lambda_curv_t")

        labels = [sci_label(x) for x in group["lambda_curv_t"]]
        x = np.arange(len(group))

        data = np.ones(len(group))
        xy = group["xy_over_data_late_med"].to_numpy(dtype=float)
        tt = group["t_over_data_late_med"].to_numpy(dtype=float)

        ax.bar(x, data, label="data loss")
        ax.bar(x, xy, bottom=data, label="x-y curvature")
        ax.bar(x, tt, bottom=data + xy, label="time curvature")

        ax.axhline(2.0, linestyle="--", linewidth=1.0)

        ax.set_ylim(0.0, ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(f"fixed xy={sci_label(lambda_xy)}")
        ax.set_xlabel("lambda_curv_t")
        ax.set_ylabel("normalized loss")
        ax.grid(True, axis="y", alpha=0.3)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)

    fig.suptitle(
        "Loss composition grouped by fixed spatial lambda",
        y=0.995,
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    path = output_dir / "combined_fixed_xy.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_combined_panels_fixed_t(df: pd.DataFrame, output_dir: Path, ylim: float) -> None:
    lambdas_t = sorted(df["lambda_curv_t"].unique())

    n = len(lambdas_t)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(13, 4.2 * nrows),
        squeeze=False,
    )

    for ax in axes.ravel():
        ax.axis("off")

    for ax, lambda_t in zip(axes.ravel(), lambdas_t):
        ax.axis("on")

        group = df[df["lambda_curv_t"] == lambda_t].sort_values("lambda_curv_xy")

        labels = [sci_label(x) for x in group["lambda_curv_xy"]]
        x = np.arange(len(group))

        data = np.ones(len(group))
        xy = group["xy_over_data_late_med"].to_numpy(dtype=float)
        tt = group["t_over_data_late_med"].to_numpy(dtype=float)

        ax.bar(x, data, label="data loss")
        ax.bar(x, xy, bottom=data, label="x-y curvature")
        ax.bar(x, tt, bottom=data + xy, label="time curvature")

        ax.axhline(2.0, linestyle="--", linewidth=1.0)

        ax.set_ylim(0.0, ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(f"fixed t={sci_label(lambda_t)}")
        ax.set_xlabel("lambda_curv_xy")
        ax.set_ylabel("normalized loss")
        ax.grid(True, axis="y", alpha=0.3)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)

    fig.suptitle(
        "Loss composition grouped by fixed temporal lambda",
        y=0.995,
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    path = output_dir / "combined_fixed_t.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--robust_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--ylim",
        type=float,
        default=None,
        help="Optional fixed y-limit for all plots. If omitted, computed globally.",
    )

    args = parser.parse_args()

    robust_csv = Path(args.robust_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(robust_csv)

    require_columns(
        df,
        [
            "lambda_curv_xy",
            "lambda_curv_t",
            "xy_over_data_late_med",
            "t_over_data_late_med",
            "reg_over_data_late_med",
        ],
    )

    df["lambda_curv_xy"] = pd.to_numeric(df["lambda_curv_xy"], errors="coerce")
    df["lambda_curv_t"] = pd.to_numeric(df["lambda_curv_t"], errors="coerce")

    df = df[
        (df["lambda_curv_xy"] > 0)
        & (df["lambda_curv_t"] > 0)
        & np.isfinite(df["xy_over_data_late_med"])
        & np.isfinite(df["t_over_data_late_med"])
    ].copy()

    df = df.drop_duplicates(
        subset=["lambda_curv_xy", "lambda_curv_t"],
        keep="last",
    )

    if len(df) == 0:
        raise RuntimeError("No valid nonzero lambda rows found.")

    if args.ylim is None:
        ylim = get_global_ylim(df)
    else:
        ylim = float(args.ylim)

    print("Grouped lambda plots")
    print(f"  robust_csv: {robust_csv}")
    print(f"  output_dir: {output_dir}")
    print(f"  rows:       {len(df)}")
    print(f"  y-limit:    {ylim}")

    plot_all_fixed_xy(df, output_dir, ylim)
    plot_all_fixed_t(df, output_dir, ylim)

    plot_combined_panels_fixed_xy(df, output_dir, ylim)
    plot_combined_panels_fixed_t(df, output_dir, ylim)

    print()
    print("Saved:")
    print(f"  {output_dir / 'combined_fixed_xy.png'}")
    print(f"  {output_dir / 'combined_fixed_t.png'}")
    print(f"  {output_dir / 'fixed_xy'}")
    print(f"  {output_dir / 'fixed_t'}")


if __name__ == "__main__":
    main()