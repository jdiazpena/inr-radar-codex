# -*- coding: utf-8 -*-
"""Compare synthetic regularization sweep runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = [
    ("data_only", Path("outputs/codex_sweep_high_amp_data_only_15000")),
    ("xy030_t030", Path("outputs/codex_sweep_high_amp_xy030_t030_15000")),
    ("xy070_t030", Path("outputs/codex_sweep_high_amp_xy070_t030_15000")),
    ("xy070_t070", Path("outputs/codex_sweep_high_amp_xy070_t070_15000")),
]
COMPARISON_DIR = Path("outputs/comparison")


def mean_by_run_region(regional: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "density_rmse",
        "density_mae",
        "density_bias",
        "density_p95_abs",
        "density_max_abs",
        "dx_gradient_rmse",
        "dy_gradient_rmse",
        "dt_gradient_rmse",
        "combined_gradient_magnitude_rmse",
        "gradient_p95_abs",
    ]
    cols = [c for c in numeric if c in regional.columns]
    return regional.groupby(["run", "region"], as_index=False)[cols].mean()


def plot_grouped(summary: pd.DataFrame, metric: str, ylabel: str, out_path: Path) -> None:
    regions = list(summary["region"].drop_duplicates())
    runs = [label for label, _ in RUNS]
    x = np.arange(len(regions))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, run in enumerate(runs):
        vals = []
        for region in regions:
            m = summary[(summary["run"] == run) & (summary["region"] == region)]
            vals.append(float(m[metric].iloc[0]) if len(m) else np.nan)
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=run)

    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_loss_histories(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, run_dir in RUNS:
        path = run_dir / "history.csv"
        if not path.exists():
            continue
        hist = pd.read_csv(path)
        ax.plot(hist["step"], hist["data_loss"], label=f"{label} data")
        if "total_loss" in hist.columns:
            ax.plot(hist["step"], hist["total_loss"], linestyle="--", alpha=0.7, label=f"{label} total")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_derivative_diagnostics(out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    cols = [("fxx_rms", "fxx RMS"), ("fxy_rms", "fxy RMS"), ("fyy_rms", "fyy RMS"), ("ftt_rms", "ftt RMS")]
    for ax, (col, title) in zip(axes.ravel(), cols):
        for label, run_dir in RUNS:
            path = run_dir / "history.csv"
            if not path.exists():
                continue
            hist = pd.read_csv(path)
            if col in hist.columns:
                y = pd.to_numeric(hist[col], errors="coerce")
                if y.notna().any():
                    ax.plot(hist["step"], y, label=label)
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("step")
    axes[-1, 1].set_xlabel("step")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def load_final_training_rows() -> pd.DataFrame:
    rows = []
    for label, run_dir in RUNS:
        hist_path = run_dir / "history.csv"
        if not hist_path.exists():
            continue
        hist = pd.read_csv(hist_path)
        last = hist.iloc[-1].to_dict()
        last["run"] = label
        last["run_dir"] = str(run_dir)
        rows.append(last)
    return pd.DataFrame(rows)


def write_findings(summary: pd.DataFrame, out_path: Path) -> None:
    full = summary[summary["region"] == "full_domain"].copy()
    best_density = full.sort_values("density_rmse").iloc[0] if len(full) else None
    best_gradient = full.sort_values("combined_gradient_magnitude_rmse").iloc[0] if "combined_gradient_magnitude_rmse" in full and len(full) else None

    lines = ["# Synthetic Regularization Sweep Findings", ""]
    if best_density is not None:
        lines.append(f"Best full-domain density RMSE: `{best_density['run']}` ({best_density['density_rmse']:.6e}).")
    if best_gradient is not None:
        lines.append(f"Best full-domain combined gradient RMSE: `{best_gradient['run']}` ({best_gradient['combined_gradient_magnitude_rmse']:.6e}).")
    if best_density is not None and best_gradient is not None:
        same = best_density["run"] == best_gradient["run"]
        lines.append(f"Best density and best gradient run are {'the same' if same else 'different'}.")
    lines.extend([
        "",
        "Use the generated figures and CSVs to answer whether data-only oscillates between observations, whether 0.30/0.30 reduces artifacts without smearing, whether 0.70/0.30 improves gradients, and whether 0.70/0.70 oversmooths or improves temporal consistency.",
        "",
    ])
    out_path.write_text("\n".join(lines))


def main() -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    regional_path = COMPARISON_DIR / "regional_error_summary.csv"
    if not regional_path.exists():
        raise FileNotFoundError(f"Run analyze_synthetic_sweep.py first; missing {regional_path}")

    regional = pd.read_csv(regional_path)
    summary = mean_by_run_region(regional)
    final_train = load_final_training_rows()

    summary.to_csv(COMPARISON_DIR / "regularization_sweep_summary.csv", index=False)
    final_train.to_csv(COMPARISON_DIR / "training_final_rows.csv", index=False)

    plot_grouped(summary, "density_rmse", "density RMSE [m^-3]", COMPARISON_DIR / "density_rmse_by_region.png")
    plot_grouped(summary, "density_p95_abs", "density p95 abs error [m^-3]", COMPARISON_DIR / "density_p95_by_region.png")
    if "combined_gradient_magnitude_rmse" in summary.columns:
        plot_grouped(summary, "combined_gradient_magnitude_rmse", "combined gradient RMSE", COMPARISON_DIR / "gradient_rmse_by_region.png")
    plot_loss_histories(COMPARISON_DIR / "loss_history_comparison.png")
    plot_derivative_diagnostics(COMPARISON_DIR / "derivative_diagnostic_comparison.png")
    write_findings(summary, COMPARISON_DIR / "regularization_sweep_findings.md")

    print(f"Saved comparison outputs in {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
