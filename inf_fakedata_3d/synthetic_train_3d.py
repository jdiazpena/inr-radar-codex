#-*- coding: utf-8 -*-
"""
synthetic_train_3d_window_reference_reg_diagnostic.py

Train a windowed 3D INR on synthetic plasma data.

3D here means:
    x, y, time

not:
    x, y, z

Current experiment:
    f_theta(x_norm, y_norm, t_norm) -> normalized log10(Ne)

Loss:
    total_loss =
        data_loss
        + lambda_curv_xy_eff * curv_xy_loss
        + lambda_curv_t_eff  * curv_t_loss

This version can use either fixed lambdas or reference-ratio lambdas.

Fixed-lambda mode:
    lambda_curv_xy_eff and lambda_curv_t_eff are set from command-line
    values, with the usual ramp.

Reference-ratio mode:
    lambdas are adjusted so the weighted curvature terms target a chosen
    fraction of a stable data reference:

        data_reference = max(data_loss_ema, epsilon_data)

        lambda_xy_target = target_xy_ratio * data_reference / curv_xy_raw_ema
        lambda_t_target  = target_t_ratio  * data_reference / curv_t_raw_ema

    The epsilon floor prevents the priors from disappearing when the measured
    radar points become easy to fit.

where:
    data_loss:
        MSE at measured radar points.

    curv_xy_loss:
        spatial curvature penalty:
            mean(fxx^2 + 2 fxy^2 + fyy^2)

    curv_t_loss:
        temporal curvature penalty:
            mean(ftt^2)

Important:
    temporal curvature is NOT temporal gradient.
    It allows linear time evolution but penalizes time wiggles.

This script trains ONE synthetic temporal window.
A later wrapper can loop over many windows.
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

from synthetic_dataset import SyntheticPlasmaTimeDataset
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


def ramp_weight(
    target_weight: float,
    step: int,
    num_steps: int,
    ramp_frac: float,
) -> float:
    if target_weight <= 0.0:
        return 0.0

    if ramp_frac <= 0.0:
        return float(target_weight)

    ramp_steps = max(1, int(ramp_frac * num_steps))
    factor = min(1.0, step / ramp_steps)

    return float(target_weight * factor)



def update_ema_scalar(
    old_value: float | None,
    new_value: float,
    beta: float,
) -> float:
    """
    Exponential moving average for scalar diagnostics.

    beta close to 1 gives a slow/stable average.
    """

    new_value = float(new_value)

    if old_value is None or not np.isfinite(old_value):
        return new_value

    return float(beta * old_value + (1.0 - beta) * new_value)


def clamp_float(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp a float to [min_value, max_value].
    """

    value = float(value)
    min_value = float(min_value)
    max_value = float(max_value)

    if max_value < min_value:
        raise ValueError("max_value must be >= min_value")

    return float(min(max(value, min_value), max_value))


def safe_ratio(numer: float, denom: float, eps: float = 1e-30) -> float:
    """
    Numerically safe scalar ratio.
    """

    numer = float(numer)
    denom = float(denom)

    if not np.isfinite(numer) or not np.isfinite(denom):
        return float("nan")

    if abs(denom) < eps:
        return float("nan")

    return float(numer / denom)


# ============================================================
# GRID / MASK HELPERS
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
    dataset: SyntheticPlasmaTimeDataset,
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
    Tn = 2.0 * (float(t_sec) - t_min) / (t_max - t_min) - 1.0

    coords_grid = np.column_stack(
        [
            Xn.ravel(),
            Yn.ravel(),
            np.full(X.size, Tn, dtype=np.float64),
        ]
    ).astype(np.float32)

    return coords_grid


def make_collocation_pool(
    dataset: SyntheticPlasmaTimeDataset,
    df: pd.DataFrame,
    grid_nx: int,
    grid_ny: int,
    padding_frac: float,
    nearest_radius_factor: float,
) -> tuple[torch.Tensor, float, float, int]:
    """
    Build x-y-t collocation points.

    For each existing time in the current window:
        make the same x-y grid
        mask it by nearest radar-point distance
        assign that time value

    These collocation points do not have data targets.
    They are only used for derivative losses.
    """

    measured_xy = (
        df[["x_km", "y_km"]]
        .drop_duplicates()
        .to_numpy(dtype=float)
    )

    X, Y = make_query_grid_from_points(
        x_km=df["x_km"].to_numpy(dtype=float),
        y_km=df["y_km"].to_numpy(dtype=float),
        nx=grid_nx,
        ny=grid_ny,
        padding_frac=padding_frac,
    )

    nearest_radius_km = estimate_nearest_radius(
        measured_xy,
        factor=nearest_radius_factor,
    )

    valid_mask, _ = nearest_distance_mask(
        X,
        Y,
        measured_xy,
        radius_km=nearest_radius_km,
    )

    xy_valid = np.column_stack([X.ravel(), Y.ravel()])[valid_mask.ravel()]

    unique_times = np.sort(df["t_sec"].unique())

    all_coords = []

    for t_sec in unique_times:
        Xv = xy_valid[:, 0]
        Yv = xy_valid[:, 1]

        coords_t = normalize_xy_t_grid_with_dataset(
            dataset=dataset,
            X=Xv.reshape(-1, 1),
            Y=Yv.reshape(-1, 1),
            t_sec=float(t_sec),
        )

        all_coords.append(coords_t)

    coords_col_np = np.concatenate(all_coords, axis=0).astype(np.float32)

    if coords_col_np.shape[0] == 0:
        raise RuntimeError("No valid collocation points were created.")

    coords_col = torch.from_numpy(coords_col_np)

    valid_fraction = float(valid_mask.mean())

    return coords_col, nearest_radius_km, valid_fraction, int(unique_times.size)


def sample_collocation_points(
    collocation_pool: torch.Tensor,
    num_collocation: int,
) -> torch.Tensor:
    n_total = collocation_pool.shape[0]

    if num_collocation <= 0 or num_collocation >= n_total:
        idx = torch.arange(n_total, device=collocation_pool.device)
    else:
        idx = torch.randperm(n_total, device=collocation_pool.device)[:num_collocation]

    return collocation_pool[idx]


# ============================================================
# DERIVATIVE LOSSES
# ============================================================

def curvature_losses_xy_t(
    model: torch.nn.Module,
    coords_col: torch.Tensor,
    use_xy: bool,
    use_t: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute spatial x-y curvature and temporal curvature.

    coords_col has columns:
        0: x_norm
        1: y_norm
        2: t_norm

    curv_xy:
        mean(fxx^2 + 2 fxy^2 + fyy^2)

    curv_t:
        mean(ftt^2)

    Derivatives are with respect to normalized coordinates.
    """

    coords_col = coords_col.detach().clone().requires_grad_(True)

    pred = model(coords_col)

    grad = torch.autograd.grad(
        outputs=pred,
        inputs=coords_col,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    zero = pred.new_tensor(0.0)

    curv_xy = zero
    curv_t = zero

    if use_xy:
        fx = grad[:, 0:1]
        fy = grad[:, 1:2]

        grad_fx = torch.autograd.grad(
            outputs=fx,
            inputs=coords_col,
            grad_outputs=torch.ones_like(fx),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_fy = torch.autograd.grad(
            outputs=fy,
            inputs=coords_col,
            grad_outputs=torch.ones_like(fy),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        fxx = grad_fx[:, 0:1]
        fxy = grad_fx[:, 1:2]
        fyy = grad_fy[:, 1:2]

        curv_xy = torch.mean(fxx ** 2 + 2.0 * fxy ** 2 + fyy ** 2)

    if use_t:
        ft = grad[:, 2:3]

        grad_ft = torch.autograd.grad(
            outputs=ft,
            inputs=coords_col,
            grad_outputs=torch.ones_like(ft),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        ftt = grad_ft[:, 2:3]

        curv_t = torch.mean(ftt ** 2)

    return curv_xy, curv_t




def tensor_near_zero_stats(
    tensor: torch.Tensor | None,
    prefix: str,
    eps: float,
) -> dict[str, float]:
    """
    Compute numerical diagnostics for one derivative tensor.

    The near-zero fraction uses |tensor| < eps.
    The exact-zero fraction uses tensor == 0 exactly and is mostly a bug detector.
    """

    keys = [
        f"{prefix}_rms",
        f"{prefix}_meanabs",
        f"{prefix}_maxabs",
        f"{prefix}_frac_near_zero",
        f"{prefix}_frac_exact_zero",
    ]

    if tensor is None:
        return {key: float("nan") for key in keys}

    x = tensor.detach()

    if x.numel() == 0:
        return {key: float("nan") for key in keys}

    abs_x = x.abs()

    return {
        f"{prefix}_rms": float(torch.sqrt(torch.mean(x ** 2)).item()),
        f"{prefix}_meanabs": float(torch.mean(abs_x).item()),
        f"{prefix}_maxabs": float(torch.max(abs_x).item()),
        f"{prefix}_frac_near_zero": float(torch.mean((abs_x < eps).float()).item()),
        f"{prefix}_frac_exact_zero": float(torch.mean((x == 0.0).float()).item()),
    }


def derivative_diagnostics_xy_t(
    model: torch.nn.Module,
    coords_col: torch.Tensor,
    use_xy: bool,
    use_t: bool,
    zero_eps: float,
) -> dict[str, float]:
    """
    Compute fixed-probe diagnostics for the second derivatives used by the loss.

    This is diagnostic only. It does not change training.

    coords_col has columns:
        0: x_norm
        1: y_norm
        2: t_norm

    Logged components:
        fxx, fxy, fyy, ftt

    Logged aggregate probe losses:
        diag_curv_xy_probe = mean(fxx^2 + 2 fxy^2 + fyy^2)
        diag_curv_t_probe  = mean(ftt^2)
    """

    coords_col = coords_col.detach().clone().requires_grad_(True)

    pred = model(coords_col)

    grad = torch.autograd.grad(
        outputs=pred,
        inputs=coords_col,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    fxx = None
    fxy = None
    fyy = None
    ftt = None

    diag_curv_xy_probe = float("nan")
    diag_curv_t_probe = float("nan")

    if use_xy:
        fx = grad[:, 0:1]
        fy = grad[:, 1:2]

        grad_fx = torch.autograd.grad(
            outputs=fx,
            inputs=coords_col,
            grad_outputs=torch.ones_like(fx),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_fy = torch.autograd.grad(
            outputs=fy,
            inputs=coords_col,
            grad_outputs=torch.ones_like(fy),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        fxx = grad_fx[:, 0:1]
        fxy = grad_fx[:, 1:2]
        fyy = grad_fy[:, 1:2]

        diag_curv_xy_probe = float(
            torch.mean(fxx ** 2 + 2.0 * fxy ** 2 + fyy ** 2).detach().item()
        )

    if use_t:
        ft = grad[:, 2:3]

        grad_ft = torch.autograd.grad(
            outputs=ft,
            inputs=coords_col,
            grad_outputs=torch.ones_like(ft),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        ftt = grad_ft[:, 2:3]

        diag_curv_t_probe = float(torch.mean(ftt ** 2).detach().item())

    out: dict[str, float] = {
        "diag_curv_xy_probe": diag_curv_xy_probe,
        "diag_curv_t_probe": diag_curv_t_probe,
    }

    out.update(tensor_near_zero_stats(fxx, "fxx", zero_eps))
    out.update(tensor_near_zero_stats(fxy, "fxy", zero_eps))
    out.update(tensor_near_zero_stats(fyy, "fyy", zero_eps))
    out.update(tensor_near_zero_stats(ftt, "ftt", zero_eps))

    return out


def parameter_grad_norm_from_loss(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    retain_graph: bool = True,
) -> float:
    """
    L2 norm of d(loss)/d(theta).

    This does not populate parameter .grad fields because it uses autograd.grad.
    """

    if not isinstance(loss, torch.Tensor):
        return float("nan")

    if not loss.requires_grad:
        return 0.0

    grads = torch.autograd.grad(
        outputs=loss,
        inputs=parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )

    total = 0.0

    for grad in grads:
        if grad is None:
            continue

        total += float(torch.sum(grad.detach() ** 2).item())

    return float(np.sqrt(total))


def parameter_grad_norm_from_existing_grads(
    parameters: list[torch.nn.Parameter],
) -> float:
    """
    L2 norm of the gradients currently stored in parameter .grad fields.
    """

    total = 0.0

    for param in parameters:
        if param.grad is None:
            continue

        total += float(torch.sum(param.grad.detach() ** 2).item())

    return float(np.sqrt(total))
@torch.no_grad()
def evaluate_model_on_coords(
    model: torch.nn.Module,
    coords_np: np.ndarray,
    dataset: SyntheticPlasmaTimeDataset,
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
# PLOTTING
# ============================================================

def select_plot_time_indices(
    df: pd.DataFrame,
    num_plot_times: int,
) -> list[int]:
    unique_times = np.sort(df["time_index"].unique())

    if num_plot_times <= 0:
        return []

    if num_plot_times >= unique_times.size:
        return [int(x) for x in unique_times]

    picks = np.linspace(0, unique_times.size - 1, num_plot_times)
    picks = np.round(picks).astype(int)

    return [int(unique_times[i]) for i in picks]


def plot_history(
    history_path: Path,
    out_dir: Path,
) -> None:
    hist = pd.read_csv(history_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(hist["step"], hist["total_loss"], marker="o", markersize=3, label="total")
    ax.plot(hist["step"], hist["data_loss"], marker="o", markersize=3, label="data")
    ax.plot(hist["step"], hist["curv_xy_weighted"], marker="o", markersize=3, label="xy curv weighted")
    ax.plot(hist["step"], hist["curv_t_weighted"], marker="o", markersize=3, label="t curv weighted")

    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Synthetic 3D window INR training loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = out_dir / "training_history.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved history plot: {path}")



def plot_diagnostics(
    history_path: Path,
    out_dir: Path,
) -> None:
    """
    Plot derivative and parameter-gradient diagnostics if present in history.csv.
    """

    hist = pd.read_csv(history_path)

    # ------------------------------------------------------------
    # Derivative RMS diagnostics
    # ------------------------------------------------------------
    deriv_cols = [
        ("fxx_rms", r"$f_{xx}$ RMS"),
        ("fxy_rms", r"$f_{xy}$ RMS"),
        ("fyy_rms", r"$f_{yy}$ RMS"),
        ("ftt_rms", r"$f_{tt}$ RMS"),
    ]

    available_deriv = [
        (col, label)
        for col, label in deriv_cols
        if col in hist.columns
    ]

    if available_deriv:
        fig, ax = plt.subplots(figsize=(8, 5))

        for col, label in available_deriv:
            ax.plot(hist["step"], hist[col], marker="o", markersize=3, label=label)

        ax.set_xlabel("step")
        ax.set_ylabel("derivative RMS")
        ax.set_title("Fixed-probe second-derivative diagnostics")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

        path = out_dir / "derivative_rms_diagnostics.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved derivative RMS diagnostics: {path}")

    # ------------------------------------------------------------
    # Near-zero fractions
    # ------------------------------------------------------------
    zero_cols = [
        ("fxx_frac_near_zero", r"$f_{xx}$ near-zero fraction"),
        ("fxy_frac_near_zero", r"$f_{xy}$ near-zero fraction"),
        ("fyy_frac_near_zero", r"$f_{yy}$ near-zero fraction"),
        ("ftt_frac_near_zero", r"$f_{tt}$ near-zero fraction"),
    ]

    available_zero = [
        (col, label)
        for col, label in zero_cols
        if col in hist.columns
    ]

    if available_zero:
        fig, ax = plt.subplots(figsize=(8, 5))

        for col, label in available_zero:
            ax.plot(hist["step"], hist[col], marker="o", markersize=3, label=label)

        ax.set_xlabel("step")
        ax.set_ylabel("fraction")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("Fraction of fixed-probe derivatives near zero")
        ax.grid(True, alpha=0.3)
        ax.legend()

        path = out_dir / "derivative_near_zero_fraction.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved derivative near-zero diagnostics: {path}")

    # ------------------------------------------------------------
    # Parameter-gradient norms
    # ------------------------------------------------------------
    grad_cols = [
        ("grad_norm_total", "total"),
        ("grad_norm_data", "data"),
        ("grad_norm_xy_weighted", "xy weighted"),
        ("grad_norm_t_weighted", "t weighted"),
    ]

    available_grad = [
        (col, label)
        for col, label in grad_cols
        if col in hist.columns
    ]

    if available_grad:
        fig, ax = plt.subplots(figsize=(8, 5))

        any_plotted = False

        for col, label in available_grad:
            y = pd.to_numeric(hist[col], errors="coerce")

            if y.notna().any():
                ax.plot(hist["step"], y, marker="o", markersize=3, label=label)
                any_plotted = True

        if any_plotted:
            ax.set_xlabel("step")
            ax.set_ylabel("parameter-gradient L2 norm")
            ax.set_title("Loss-component gradient norms")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend()

            path = out_dir / "gradient_norm_diagnostics.png"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            print(f"Saved gradient norm diagnostics: {path}")
        else:
            plt.close(fig)


def plot_xy_prediction_at_time(
    model: torch.nn.Module,
    dataset: SyntheticPlasmaTimeDataset,
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
    vmin: float,
    vmax: float,
) -> None:
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
        f"Synthetic 3D INR | time_index={time_index} | t={t_min:.1f} min"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    fig_path = out_dir / f"xy_time_index_{time_index:04d}.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved x-y time plot: {fig_path}")

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

    config = vars(args).copy()

    config_path = out_dir / "run_config.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    print(f"Using device: {device}")

    # ------------------------------------------------------------
    # 1. Load one time window
    # ------------------------------------------------------------
    dataset = SyntheticPlasmaTimeDataset(
        csv_path=args.synthetic_csv,
        target_col=args.target_col,
        window_start_index=args.window_start_index,
        window_size_records=args.window_size_records,
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
    # 2. Collocation pool for derivative losses
    # ------------------------------------------------------------
    if args.reference_loss_weights:
        use_xy_curv = args.target_xy_ratio > 0.0
        use_t_curv = args.target_t_ratio > 0.0
    else:
        use_xy_curv = args.lambda_curv_xy > 0.0
        use_t_curv = args.lambda_curv_t > 0.0

    diagnostic_use_xy = args.num_diagnostic_collocation != 0
    diagnostic_use_t = args.num_diagnostic_collocation != 0
    needs_collocation_pool = use_xy_curv or use_t_curv or diagnostic_use_xy or diagnostic_use_t

    if needs_collocation_pool:
        collocation_pool, collocation_radius_km, collocation_valid_fraction, collocation_n_times = make_collocation_pool(
            dataset=dataset,
            df=df,
            grid_nx=args.collocation_grid_nx,
            grid_ny=args.collocation_grid_ny,
            padding_frac=args.grid_padding_frac,
            nearest_radius_factor=args.nearest_radius_factor,
        )

        collocation_pool = collocation_pool.to(device)

        diagnostic_collocation_points = sample_collocation_points(
            collocation_pool=collocation_pool,
            num_collocation=args.num_diagnostic_collocation,
        ).detach()

        print()
        print("Collocation points:")
        print(f"  pool size:             {collocation_pool.shape[0]}")
        print(f"  sample per step:       {args.num_collocation}")
        print(f"  diagnostic probe size: {diagnostic_collocation_points.shape[0]}")
        print(f"  time records used:     {collocation_n_times}")
        print(f"  nearest radius [km]:   {collocation_radius_km:.3f}")
        print(f"  valid grid fraction:   {collocation_valid_fraction:.3f}")
        print(f"  curvature loss xy:     {use_xy_curv}")
        print(f"  curvature loss t:      {use_t_curv}")
        print(f"  diagnostic xy:         {diagnostic_use_xy}")
        print(f"  diagnostic t:          {diagnostic_use_t}")
        print(f"  lambda_curv_xy:        {args.lambda_curv_xy}")
        print(f"  lambda_curv_t:         {args.lambda_curv_t}")
        print(f"  reg_ramp_frac:         {args.reg_ramp_frac}")
        print(f"  reference mode:        {args.reference_loss_weights}")
        if args.reference_loss_weights:
            print(f"  target_xy_ratio:       {args.target_xy_ratio}")
            print(f"  target_t_ratio:        {args.target_t_ratio}")
            print(f"  epsilon_data:          {args.epsilon_data}")
            print(f"  loss_ema_beta:         {args.loss_ema_beta}")
            print(f"  lambda_smoothing:      {args.lambda_smoothing}")
            print(f"  lambda_update_every:   {args.lambda_update_every}")
            print(f"  lambda_warmup_steps:   {args.lambda_warmup_steps}")
            print(f"  freeze_after_step:     {args.freeze_lambdas_after_step}")
    else:
        collocation_pool = None
        diagnostic_collocation_points = None

    # ------------------------------------------------------------
    # 3. Build model
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
    # 4. Train
    # ------------------------------------------------------------
    history_fields = [
        "step",
        "total_loss",
        "data_loss",
        "curv_xy_raw",
        "curv_xy_weighted",
        "lambda_curv_xy_base",
        "lambda_curv_xy_eff",
        "lambda_curv_xy_target",
        "curv_t_raw",
        "curv_t_weighted",
        "lambda_curv_t_base",
        "lambda_curv_t_eff",
        "lambda_curv_t_target",
        "data_loss_ema",
        "data_reference",
        "curv_xy_raw_ema",
        "curv_t_raw_ema",
        "xy_over_data_inst",
        "t_over_data_inst",
        "xy_over_data_ref",
        "t_over_data_ref",
        "reference_loss_weights",
        "lambda_update_active",
        "lambda_frozen",
        "target_xy_ratio",
        "target_t_ratio",
        "epsilon_data",

        # Fixed-probe derivative diagnostics.
        "diag_curv_xy_probe",
        "diag_curv_t_probe",
        "fxx_rms",
        "fxx_meanabs",
        "fxx_maxabs",
        "fxx_frac_near_zero",
        "fxx_frac_exact_zero",
        "fxy_rms",
        "fxy_meanabs",
        "fxy_maxabs",
        "fxy_frac_near_zero",
        "fxy_frac_exact_zero",
        "fyy_rms",
        "fyy_meanabs",
        "fyy_maxabs",
        "fyy_frac_near_zero",
        "fyy_frac_exact_zero",
        "ftt_rms",
        "ftt_meanabs",
        "ftt_maxabs",
        "ftt_frac_near_zero",
        "ftt_frac_exact_zero",

        # Parameter-gradient diagnostics.
        "component_grad_computed",
        "grad_norm_total",
        "grad_norm_data",
        "grad_norm_xy_weighted",
        "grad_norm_t_weighted",

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

    derivative_diag_keys = [
        "diag_curv_xy_probe",
        "diag_curv_t_probe",
        "fxx_rms",
        "fxx_meanabs",
        "fxx_maxabs",
        "fxx_frac_near_zero",
        "fxx_frac_exact_zero",
        "fxy_rms",
        "fxy_meanabs",
        "fxy_maxabs",
        "fxy_frac_near_zero",
        "fxy_frac_exact_zero",
        "fyy_rms",
        "fyy_meanabs",
        "fyy_maxabs",
        "fyy_frac_near_zero",
        "fyy_frac_exact_zero",
        "ftt_rms",
        "ftt_meanabs",
        "ftt_maxabs",
        "ftt_frac_near_zero",
        "ftt_frac_exact_zero",
    ]

    grad_diag_keys = [
        "component_grad_computed",
        "grad_norm_total",
        "grad_norm_data",
        "grad_norm_xy_weighted",
        "grad_norm_t_weighted",
    ]

    latest_derivative_diag = {
        key: float("nan")
        for key in derivative_diag_keys
    }

    latest_grad_diag = {
        "component_grad_computed": False,
        "grad_norm_total": float("nan"),
        "grad_norm_data": float("nan"),
        "grad_norm_xy_weighted": float("nan"),
        "grad_norm_t_weighted": float("nan"),
    }
    # ------------------------------------------------------------
    # Best checkpoints after regularization ramp
    # ------------------------------------------------------------
    if args.reg_ramp_frac > 0.0:
        ramp_steps = max(1, int(args.reg_ramp_frac * args.num_steps))
    else:
        ramp_steps = 0

    best_total_after_ramp = float("inf")
    best_data_after_ramp = float("inf")

    best_total_step = None
    best_data_step = None

    best_total_path = out_dir / "model_best_total_after_ramp.pt"
    best_data_path = out_dir / "model_best_data_after_ramp.pt"

    # ------------------------------------------------------------
    # Reference-ratio lambda state
    # ------------------------------------------------------------
    data_loss_ema = None
    curv_xy_raw_ema = None
    curv_t_raw_ema = None

    lambda_curv_xy_base = float(args.lambda_curv_xy)
    lambda_curv_t_base = float(args.lambda_curv_t)

    lambda_curv_xy_target = float(args.lambda_curv_xy)
    lambda_curv_t_target = float(args.lambda_curv_t)

    lambda_frozen = False

    if args.reference_loss_weights:
        lambda_curv_xy_base = 0.0
        lambda_curv_t_base = 0.0
        lambda_curv_xy_target = 0.0
        lambda_curv_t_target = 0.0

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
        data_loss = F.mse_loss(pred, batch_values)

        if use_xy_curv or use_t_curv:
            coords_col = sample_collocation_points(
                collocation_pool=collocation_pool,
                num_collocation=args.num_collocation,
            )

            curv_xy_raw, curv_t_raw = curvature_losses_xy_t(
                model=model,
                coords_col=coords_col,
                use_xy=use_xy_curv,
                use_t=use_t_curv,
            )
        else:
            curv_xy_raw = data_loss.new_tensor(0.0)
            curv_t_raw = data_loss.new_tensor(0.0)

        # --------------------------------------------------------
        # Update EMA statistics used by reference-ratio weighting.
        # These are diagnostics in fixed-lambda mode and controls in
        # reference mode.
        # --------------------------------------------------------
        data_scalar_current = float(data_loss.detach().item())
        curv_xy_scalar_current = float(curv_xy_raw.detach().item())
        curv_t_scalar_current = float(curv_t_raw.detach().item())

        data_loss_ema = update_ema_scalar(
            old_value=data_loss_ema,
            new_value=data_scalar_current,
            beta=args.loss_ema_beta,
        )

        curv_xy_raw_ema = update_ema_scalar(
            old_value=curv_xy_raw_ema,
            new_value=curv_xy_scalar_current,
            beta=args.loss_ema_beta,
        )

        curv_t_raw_ema = update_ema_scalar(
            old_value=curv_t_raw_ema,
            new_value=curv_t_scalar_current,
            beta=args.loss_ema_beta,
        )

        data_reference = max(float(data_loss_ema), float(args.epsilon_data))

        # --------------------------------------------------------
        # Reference-ratio lambda update.
        #
        # The controller targets weighted curvature terms relative to
        # data_reference, not relative to the collapsing instantaneous
        # data loss.
        # --------------------------------------------------------
        lambda_update_active = False

        if args.reference_loss_weights:
            if use_xy_curv and curv_xy_raw_ema > args.curvature_ema_floor:
                lambda_curv_xy_target = (
                    args.target_xy_ratio * data_reference / curv_xy_raw_ema
                )
                lambda_curv_xy_target = clamp_float(
                    lambda_curv_xy_target,
                    args.lambda_curv_xy_min,
                    args.lambda_curv_xy_max,
                )
            else:
                lambda_curv_xy_target = 0.0

            if use_t_curv and curv_t_raw_ema > args.curvature_ema_floor:
                lambda_curv_t_target = (
                    args.target_t_ratio * data_reference / curv_t_raw_ema
                )
                lambda_curv_t_target = clamp_float(
                    lambda_curv_t_target,
                    args.lambda_curv_t_min,
                    args.lambda_curv_t_max,
                )
            else:
                lambda_curv_t_target = 0.0

            if args.freeze_lambdas_after_step > 0 and step >= args.freeze_lambdas_after_step:
                lambda_frozen = True

            can_update = (
                step > args.lambda_warmup_steps
                and not lambda_frozen
                and (step % args.lambda_update_every == 0 or step == 1)
            )

            if can_update:
                lambda_update_active = True

                s = float(args.lambda_smoothing)

                lambda_curv_xy_base = (
                    (1.0 - s) * lambda_curv_xy_base
                    + s * lambda_curv_xy_target
                )

                lambda_curv_t_base = (
                    (1.0 - s) * lambda_curv_t_base
                    + s * lambda_curv_t_target
                )
        else:
            lambda_curv_xy_base = float(args.lambda_curv_xy)
            lambda_curv_t_base = float(args.lambda_curv_t)
            lambda_curv_xy_target = float(args.lambda_curv_xy)
            lambda_curv_t_target = float(args.lambda_curv_t)

        lambda_curv_xy_eff = ramp_weight(
            target_weight=lambda_curv_xy_base,
            step=step,
            num_steps=args.num_steps,
            ramp_frac=args.reg_ramp_frac,
        )

        lambda_curv_t_eff = ramp_weight(
            target_weight=lambda_curv_t_base,
            step=step,
            num_steps=args.num_steps,
            ramp_frac=args.reg_ramp_frac,
        )

        curv_xy_weighted = lambda_curv_xy_eff * curv_xy_raw
        curv_t_weighted = lambda_curv_t_eff * curv_t_raw

        xy_weighted_scalar = float(curv_xy_weighted.detach().item())
        t_weighted_scalar = float(curv_t_weighted.detach().item())

        xy_over_data_inst = safe_ratio(xy_weighted_scalar, data_scalar_current)
        t_over_data_inst = safe_ratio(t_weighted_scalar, data_scalar_current)
        xy_over_data_ref = safe_ratio(xy_weighted_scalar, data_reference)
        t_over_data_ref = safe_ratio(t_weighted_scalar, data_reference)

        # total_loss = data_loss + curv_xy_weighted + curv_t_weighted

        # optimizer.zero_grad(set_to_none=True)
        # total_loss.backward()
        # optimizer.step()

        total_loss = data_loss + curv_xy_weighted + curv_t_weighted

        # --------------------------------------------------------
        # Save best checkpoints after the regularization ramp.
        #
        # This saves the model BEFORE the optimizer step, so the saved
        # weights correspond to the loss values used for the decision.
        # --------------------------------------------------------
        if step > ramp_steps:
            total_scalar = float(total_loss.detach().item())
            data_scalar = float(data_loss.detach().item())

            if total_scalar < best_total_after_ramp:
                best_total_after_ramp = total_scalar
                best_total_step = step

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "coord_scalers": dataset.coord_scalers,
                        "target_scaler": dataset.target_scaler,
                        "checkpoint_type": "best_total_after_ramp",
                        "step": step,
                        "losses": {
                            "total_loss": total_scalar,
                            "data_loss": data_scalar,
                            "curv_xy_raw": float(curv_xy_raw.detach().item()),
                            "curv_xy_weighted": float(curv_xy_weighted.detach().item()),
                            "lambda_curv_xy_eff": float(lambda_curv_xy_eff),
                            "curv_t_raw": float(curv_t_raw.detach().item()),
                            "curv_t_weighted": float(curv_t_weighted.detach().item()),
                            "lambda_curv_t_eff": float(lambda_curv_t_eff),
                            "lambda_curv_xy_base": float(lambda_curv_xy_base),
                            "lambda_curv_t_base": float(lambda_curv_t_base),
                            "lambda_curv_xy_target": float(lambda_curv_xy_target),
                            "lambda_curv_t_target": float(lambda_curv_t_target),
                            "data_loss_ema": float(data_loss_ema),
                            "data_reference": float(data_reference),
                            "curv_xy_raw_ema": float(curv_xy_raw_ema),
                            "curv_t_raw_ema": float(curv_t_raw_ema),
                            "xy_over_data_ref": float(xy_over_data_ref),
                            "t_over_data_ref": float(t_over_data_ref),
                        },
                    },
                    best_total_path,
                )

            if data_scalar < best_data_after_ramp:
                best_data_after_ramp = data_scalar
                best_data_step = step

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "coord_scalers": dataset.coord_scalers,
                        "target_scaler": dataset.target_scaler,
                        "checkpoint_type": "best_data_after_ramp",
                        "step": step,
                        "losses": {
                            "total_loss": total_scalar,
                            "data_loss": data_scalar,
                            "curv_xy_raw": float(curv_xy_raw.detach().item()),
                            "curv_xy_weighted": float(curv_xy_weighted.detach().item()),
                            "lambda_curv_xy_eff": float(lambda_curv_xy_eff),
                            "curv_t_raw": float(curv_t_raw.detach().item()),
                            "curv_t_weighted": float(curv_t_weighted.detach().item()),
                            "lambda_curv_t_eff": float(lambda_curv_t_eff),
                            "lambda_curv_xy_base": float(lambda_curv_xy_base),
                            "lambda_curv_t_base": float(lambda_curv_t_base),
                            "lambda_curv_xy_target": float(lambda_curv_xy_target),
                            "lambda_curv_t_target": float(lambda_curv_t_target),
                            "data_loss_ema": float(data_loss_ema),
                            "data_reference": float(data_reference),
                            "curv_xy_raw_ema": float(curv_xy_raw_ema),
                            "curv_t_raw_ema": float(curv_t_raw_ema),
                            "xy_over_data_ref": float(xy_over_data_ref),
                            "t_over_data_ref": float(t_over_data_ref),
                        },
                    },
                    best_data_path,
                )

        # --------------------------------------------------------
        # Optional parameter-gradient diagnostics.
        #
        # These answer whether each loss component is still pushing
        # the network weights. They are more expensive than the normal
        # training step, so they are computed only every
        # component_grad_every steps.
        # --------------------------------------------------------
        compute_component_grad_diag = (
            args.component_grad_every > 0
            and (
                step == 1
                or step % args.component_grad_every == 0
                or step == args.num_steps
            )
        )

        latest_grad_diag = {
            "component_grad_computed": bool(compute_component_grad_diag),
            "grad_norm_total": float("nan"),
            "grad_norm_data": float("nan"),
            "grad_norm_xy_weighted": float("nan"),
            "grad_norm_t_weighted": float("nan"),
        }

        trainable_params = [
            param
            for param in model.parameters()
            if param.requires_grad
        ]

        if compute_component_grad_diag:
            latest_grad_diag["grad_norm_data"] = parameter_grad_norm_from_loss(
                loss=data_loss,
                parameters=trainable_params,
                retain_graph=True,
            )

            latest_grad_diag["grad_norm_xy_weighted"] = parameter_grad_norm_from_loss(
                loss=curv_xy_weighted,
                parameters=trainable_params,
                retain_graph=True,
            )

            latest_grad_diag["grad_norm_t_weighted"] = parameter_grad_norm_from_loss(
                loss=curv_t_weighted,
                parameters=trainable_params,
                retain_graph=True,
            )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        if compute_component_grad_diag:
            latest_grad_diag["grad_norm_total"] = parameter_grad_norm_from_existing_grads(
                parameters=trainable_params,
            )

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

            if diagnostic_collocation_points is not None:
                latest_derivative_diag = derivative_diagnostics_xy_t(
                    model=model,
                    coords_col=diagnostic_collocation_points,
                    use_xy=diagnostic_use_xy,
                    use_t=diagnostic_use_t,
                    zero_eps=args.deriv_zero_epsilon,
                )
            else:
                latest_derivative_diag = {
                    key: float("nan")
                    for key in derivative_diag_keys
                }

            row = {
                "step": step,
                "total_loss": float(total_loss.item()),
                "data_loss": float(data_loss.item()),
                "curv_xy_raw": float(curv_xy_raw.item()),
                "curv_xy_weighted": float(curv_xy_weighted.item()),
                "lambda_curv_xy_base": float(lambda_curv_xy_base),
                "lambda_curv_xy_eff": float(lambda_curv_xy_eff),
                "lambda_curv_xy_target": float(lambda_curv_xy_target),
                "curv_t_raw": float(curv_t_raw.item()),
                "curv_t_weighted": float(curv_t_weighted.item()),
                "lambda_curv_t_base": float(lambda_curv_t_base),
                "lambda_curv_t_eff": float(lambda_curv_t_eff),
                "lambda_curv_t_target": float(lambda_curv_t_target),
                "data_loss_ema": float(data_loss_ema),
                "data_reference": float(data_reference),
                "curv_xy_raw_ema": float(curv_xy_raw_ema),
                "curv_t_raw_ema": float(curv_t_raw_ema),
                "xy_over_data_inst": float(xy_over_data_inst),
                "t_over_data_inst": float(t_over_data_inst),
                "xy_over_data_ref": float(xy_over_data_ref),
                "t_over_data_ref": float(t_over_data_ref),
                "reference_loss_weights": bool(args.reference_loss_weights),
                "lambda_update_active": bool(lambda_update_active),
                "lambda_frozen": bool(lambda_frozen),
                "target_xy_ratio": float(args.target_xy_ratio),
                "target_t_ratio": float(args.target_t_ratio),
                "epsilon_data": float(args.epsilon_data),

                "diag_curv_xy_probe": latest_derivative_diag["diag_curv_xy_probe"],
                "diag_curv_t_probe": latest_derivative_diag["diag_curv_t_probe"],
                "fxx_rms": latest_derivative_diag["fxx_rms"],
                "fxx_meanabs": latest_derivative_diag["fxx_meanabs"],
                "fxx_maxabs": latest_derivative_diag["fxx_maxabs"],
                "fxx_frac_near_zero": latest_derivative_diag["fxx_frac_near_zero"],
                "fxx_frac_exact_zero": latest_derivative_diag["fxx_frac_exact_zero"],
                "fxy_rms": latest_derivative_diag["fxy_rms"],
                "fxy_meanabs": latest_derivative_diag["fxy_meanabs"],
                "fxy_maxabs": latest_derivative_diag["fxy_maxabs"],
                "fxy_frac_near_zero": latest_derivative_diag["fxy_frac_near_zero"],
                "fxy_frac_exact_zero": latest_derivative_diag["fxy_frac_exact_zero"],
                "fyy_rms": latest_derivative_diag["fyy_rms"],
                "fyy_meanabs": latest_derivative_diag["fyy_meanabs"],
                "fyy_maxabs": latest_derivative_diag["fyy_maxabs"],
                "fyy_frac_near_zero": latest_derivative_diag["fyy_frac_near_zero"],
                "fyy_frac_exact_zero": latest_derivative_diag["fyy_frac_exact_zero"],
                "ftt_rms": latest_derivative_diag["ftt_rms"],
                "ftt_meanabs": latest_derivative_diag["ftt_meanabs"],
                "ftt_maxabs": latest_derivative_diag["ftt_maxabs"],
                "ftt_frac_near_zero": latest_derivative_diag["ftt_frac_near_zero"],
                "ftt_frac_exact_zero": latest_derivative_diag["ftt_frac_exact_zero"],

                "component_grad_computed": latest_grad_diag["component_grad_computed"],
                "grad_norm_total": latest_grad_diag["grad_norm_total"],
                "grad_norm_data": latest_grad_diag["grad_norm_data"],
                "grad_norm_xy_weighted": latest_grad_diag["grad_norm_xy_weighted"],
                "grad_norm_t_weighted": latest_grad_diag["grad_norm_t_weighted"],

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
                f"tot={total_loss.item():.2e} "
                f"data={data_loss.item():.2e} "
                f"ref={data_reference:.1e} "
                f"xyW={curv_xy_weighted.item():.2e} "
                f"tW={curv_t_weighted.item():.2e} "
                f"xyRef={xy_over_data_ref:.2f} "
                f"tRef={t_over_data_ref:.2f} "
                f"lxy={lambda_curv_xy_eff:.1e} "
                f"lt={lambda_curv_t_eff:.1e} "
                f"rmse={latest_metrics['rmse']:.2e}"
            )

    # ------------------------------------------------------------
    # 5. Save model and measured-point predictions
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

    if best_total_step is not None:
        print(
            f"Saved best-total checkpoint: {best_total_path} "
            f"(step {best_total_step}, total={best_total_after_ramp:.8e})"
        )

    if best_data_step is not None:
        print(
            f"Saved best-data checkpoint: {best_data_path} "
            f"(step {best_data_step}, data={best_data_after_ramp:.8e})"
        )

    print()
    print("Final measured-point metrics in log10(Ne):")
    for key, value in final_metrics.items():
        print(f"  {key:12s}: {value:.8e}")

    # ------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------
    if not args.no_plots:
        plot_history(history_path, out_dir)
        plot_diagnostics(history_path, out_dir)

        plot_time_indices = select_plot_time_indices(
            df=df,
            num_plot_times=args.num_plot_times,
        )

        vmin = float(df["log10_Ne"].min())
        vmax = float(df["log10_Ne"].max())

        print()
        print("Plot time indices:")
        print(plot_time_indices)
        print(f"Fixed color scale: vmin={vmin:.6f}, vmax={vmax:.6f}")

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
                vmin=vmin,
                vmax=vmax,
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
        "--synthetic_csv",
        type=str,
        default="outputs/synthetic_smoke_test/synthetic_observations.csv",
        help="Synthetic observations CSV generated by synthetic_plasma.py.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="log10_Ne",
        help="Target column in the synthetic CSV.",
    )

    # Window
    parser.add_argument("--window_start_index", type=int, default=0)
    parser.add_argument("--window_size_records", type=int, default=11)

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

    # Regularization: fixed-lambda fallback
    parser.add_argument("--lambda_curv_xy", type=float, default=0.0)
    parser.add_argument("--lambda_curv_t", type=float, default=0.0)

    # Regularization: reference-ratio lambda mode
    parser.add_argument(
        "--reference_loss_weights",
        action="store_true",
        help="Use target ratios and data_reference=max(data_loss_ema, epsilon_data) to set lambdas.",
    )
    parser.add_argument("--target_xy_ratio", type=float, default=0.30)
    parser.add_argument("--target_t_ratio", type=float, default=0.30)
    parser.add_argument(
        "--epsilon_data",
        type=float,
        default=1e-6,
        help="Minimum data reference used for lambda calibration.",
    )
    parser.add_argument(
        "--loss_ema_beta",
        type=float,
        default=0.99,
        help="EMA beta for data and raw curvature losses.",
    )
    parser.add_argument(
        "--curvature_ema_floor",
        type=float,
        default=1e-30,
        help="Floor for raw curvature EMA denominator.",
    )
    parser.add_argument(
        "--lambda_smoothing",
        type=float,
        default=0.05,
        help="Fraction of target lambda blended into base lambda at each update.",
    )
    parser.add_argument("--lambda_update_every", type=int, default=10)
    parser.add_argument(
        "--lambda_warmup_steps",
        type=int,
        default=500,
        help="Steps before lambdas are allowed to update in reference mode.",
    )
    parser.add_argument(
        "--freeze_lambdas_after_step",
        type=int,
        default=0,
        help="0 means never freeze. Otherwise freeze lambdas at/after this step.",
    )
    parser.add_argument("--lambda_curv_xy_min", type=float, default=0.0)
    parser.add_argument("--lambda_curv_xy_max", type=float, default=1e-6)
    parser.add_argument("--lambda_curv_t_min", type=float, default=0.0)
    parser.add_argument("--lambda_curv_t_max", type=float, default=1e-6)

    parser.add_argument(
        "--num_collocation",
        type=int,
        default=8192,
        help="0 means use all collocation points.",
    )
    parser.add_argument("--collocation_grid_nx", type=int, default=80)
    parser.add_argument("--collocation_grid_ny", type=int, default=80)
    parser.add_argument("--reg_ramp_frac", type=float, default=0.2)

    # Logging
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--summary_every", type=int, default=250)
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--resume_history", action="store_true")

    # Diagnostics
    parser.add_argument(
        "--deriv_zero_epsilon",
        type=float,
        default=1e-12,
        help="Threshold for derivative near-zero fractions.",
    )
    parser.add_argument(
        "--num_diagnostic_collocation",
        type=int,
        default=4096,
        help="Fixed collocation probe size for derivative diagnostics.",
    )
    parser.add_argument(
        "--component_grad_every",
        type=int,
        default=500,
        help="Compute expensive per-loss-component parameter gradient norms every N steps. 0 disables.",
    )

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
        default="outputs/synthetic_3d_window_reference_reg_diagnostic",
    )
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--save_grid_csv", action="store_true")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)