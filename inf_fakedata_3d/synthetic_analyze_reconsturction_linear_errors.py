# -*- coding: utf-8 -*-
"""
synthetic_analyze_reconstruction.py

Dense error analysis for ONE trained synthetic INR run.

This script is intentionally configured from the top of the file, not from
command-line arguments. The training scripts are command-line driven because
we will launch many runs from bash. This analysis script is usually run once
for one completed training folder, so the paths/settings are kept here.

What it does:
    1. Loads one trained model checkpoint.
    2. Reconstructs the analytical synthetic truth from synthetic_config.json.
    3. Evaluates truth and INR prediction on a dense x-y grid at selected times.
    4. Saves density error maps and CSV summaries.
    5. Optionally computes first-gradient errors using autograd.

Run from inside your synthetic project folder, for example:
    cd ~/postdoc/inr-radar/inf_fakedata_3d
    python3 synthetic_analyze_reconstruction.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models import MLPINR


# ============================================================
# USER SETTINGS: EDIT THESE FOR EACH ANALYSIS RUN
# ============================================================

# Folder produced by synthetic_train_3d_window_reference_reg_diagnostic.py
RUN_DIR = Path("outputs/synthetic_train_high_amp_win0_diag_1500")

# Synthetic case used to train that model.
SYNTHETIC_CSV = Path("outputs/synthetic_high_amp_left_right/synthetic_observations.csv")
SYNTHETIC_CONFIG = Path("outputs/synthetic_high_amp_left_right/synthetic_config.json")

# Checkpoint inside RUN_DIR. Usually model_final.pt is fine.
CHECKPOINT_NAME = "model_final.pt"

# Output folder inside RUN_DIR.
ANALYSIS_SUBDIR = "error_analysis"

# Dense evaluation grid. This is independent from the training collocation grid.
GRID_NX = 250
GRID_NY = 250

# Domain mode:
#   "full_domain"      -> use [-domain_size/2, +domain_size/2] from synthetic_config.json.
#                         This shows interpolation/extrapolation over the whole synthetic box.
#   "training_extent"  -> use the x/y min/max stored in the model coordinate scalers.
#                         This avoids evaluating outside the observed coordinate range.
DOMAIN_MODE = "full_domain"

# Time selection:
#   "first_middle_last" -> analyze first, middle, and last available time records.
#   "all"               -> analyze all available time records.
#   list[int]            -> explicit time_index list, e.g. [0, 10, 20, 30].
TIME_SELECTION: str | list[int] = "first_middle_last"

# Model evaluation chunk size. Lower if GPU memory is tight.
PREDICT_CHUNK_SIZE = 65536
GRADIENT_CHUNK_SIZE = 16384

# Compute model first derivatives and compare against analytical truth.
# This is more expensive than density-only error, but useful for INR validation.
COMPUTE_GRADIENT_ERRORS = True

# Save dense CSV files for each analyzed time.
SAVE_DENSE_CSV = True

# Plotting.
CMAP_FIELD = "plasma"
CMAP_ERROR = "RdBu_r"
DPI = 200

# Device. "auto" uses cuda if available.
DEVICE = "auto"


# ============================================================
# SYNTHETIC TRUTH EVALUATION
# ============================================================

@dataclass
class MovingGaussianPatch:
    name: str = "patch_1"
    amplitude_m3: float = 1.0e12
    sigma_x_km: float = 45.0
    sigma_y_km: float = 45.0
    x0_km: float = -180.0
    y0_km: float = 0.0
    vx_km_s: float = 360.0 / 3600.0
    vy_km_s: float = 0.0

    def center(self, t_sec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t_sec = np.asarray(t_sec, dtype=np.float64)
        xc = self.x0_km + self.vx_km_s * t_sec
        yc = self.y0_km + self.vy_km_s * t_sec
        return xc, yc

    def evaluate_delta_and_derivatives(
        self,
        x_km: np.ndarray,
        y_km: np.ndarray,
        t_sec: np.ndarray,
    ) -> dict[str, np.ndarray]:
        x = np.asarray(x_km, dtype=np.float64)
        y = np.asarray(y_km, dtype=np.float64)
        t = np.asarray(t_sec, dtype=np.float64)

        xc, yc = self.center(t)
        dx = x - xc
        dy = y - yc

        sx2 = self.sigma_x_km ** 2
        sy2 = self.sigma_y_km ** 2

        exponent = -0.5 * ((dx ** 2) / sx2 + (dy ** 2) / sy2)
        delta_ne = self.amplitude_m3 * np.exp(exponent)

        d_delta_dx = delta_ne * (-dx / sx2)
        d_delta_dy = delta_ne * (-dy / sy2)
        d_delta_dt = delta_ne * (
            dx * self.vx_km_s / sx2
            + dy * self.vy_km_s / sy2
        )

        return {
            "delta_ne": delta_ne,
            "d_delta_dx_km": d_delta_dx,
            "d_delta_dy_km": d_delta_dy,
            "d_delta_dt_sec": d_delta_dt,
            "center_x_km": xc,
            "center_y_km": yc,
        }


def evaluate_synthetic_plasma(
    x_km: np.ndarray,
    y_km: np.ndarray,
    t_sec: np.ndarray,
    background_ne_m3: float,
    patches: list[MovingGaussianPatch],
    min_ne_m3: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Evaluate total Ne, log10(Ne), and analytical first derivatives of log10(Ne).
    """

    x = np.asarray(x_km, dtype=np.float64)
    y = np.asarray(y_km, dtype=np.float64)
    t = np.asarray(t_sec, dtype=np.float64)

    out_shape = np.broadcast_shapes(x.shape, y.shape, t.shape)
    ne = np.full(out_shape, float(background_ne_m3), dtype=np.float64)
    d_ne_dx = np.zeros_like(ne)
    d_ne_dy = np.zeros_like(ne)
    d_ne_dt = np.zeros_like(ne)

    x_b = np.broadcast_to(x, out_shape)
    y_b = np.broadcast_to(y, out_shape)
    t_b = np.broadcast_to(t, out_shape)

    patch_centers: dict[str, np.ndarray] = {}

    for patch in patches:
        vals = patch.evaluate_delta_and_derivatives(x_b, y_b, t_b)
        ne += vals["delta_ne"]
        d_ne_dx += vals["d_delta_dx_km"]
        d_ne_dy += vals["d_delta_dy_km"]
        d_ne_dt += vals["d_delta_dt_sec"]
        patch_centers[f"{patch.name}_center_x_km"] = vals["center_x_km"]
        patch_centers[f"{patch.name}_center_y_km"] = vals["center_y_km"]

    ne = np.maximum(ne, float(min_ne_m3))
    log10_ne = np.log10(ne)

    ln10 = np.log(10.0)
    dlog_dx = d_ne_dx / (ne * ln10)
    dlog_dy = d_ne_dy / (ne * ln10)
    dlog_dt = d_ne_dt / (ne * ln10)

    out = {
        "Ne": ne,
        "log10_Ne": log10_ne,
        "true_dlog10Ne_dx_km": dlog_dx,
        "true_dlog10Ne_dy_km": dlog_dy,
        "true_dlog10Ne_dt_sec": dlog_dt,
    }
    out.update(patch_centers)
    return out


def evaluate_integration_averaged_plasma(
    x_km: np.ndarray,
    y_km: np.ndarray,
    t_sec: np.ndarray,
    integration_time_sec: float,
    integration_samples: int,
    background_ne_m3: float,
    patches: list[MovingGaussianPatch],
) -> dict[str, np.ndarray]:
    """Evaluate the linear-density average represented by an ISR exposure."""

    if integration_time_sec <= 0.0:
        return evaluate_synthetic_plasma(
            x_km=x_km,
            y_km=y_km,
            t_sec=t_sec,
            background_ne_m3=background_ne_m3,
            patches=patches,
        )

    integration_samples = int(integration_samples)
    if integration_samples < 2:
        raise ValueError("integration_samples must be >= 2 for an integrated truth")
    offsets = np.linspace(
        -0.5 * integration_time_sec,
        0.5 * integration_time_sec,
        integration_samples,
        dtype=np.float64,
    )
    vals = evaluate_synthetic_plasma(
        x_km=np.asarray(x_km, dtype=np.float64)[..., None],
        y_km=np.asarray(y_km, dtype=np.float64)[..., None],
        t_sec=np.asarray(t_sec, dtype=np.float64)[..., None] + offsets,
        background_ne_m3=background_ne_m3,
        patches=patches,
    )
    weights = np.ones(integration_samples, dtype=np.float64)
    weights[[0, -1]] = 0.5
    weights /= np.sum(weights)
    ne = np.sum(vals["Ne"] * weights, axis=-1)
    ln10 = np.log(10.0)
    d_ne_dx = np.sum(vals["true_dlog10Ne_dx_km"] * vals["Ne"] * ln10 * weights, axis=-1)
    d_ne_dy = np.sum(vals["true_dlog10Ne_dy_km"] * vals["Ne"] * ln10 * weights, axis=-1)
    d_ne_dt = np.sum(vals["true_dlog10Ne_dt_sec"] * vals["Ne"] * ln10 * weights, axis=-1)
    denom = np.maximum(ne * ln10, 1.0e-30)
    return {
        "Ne": ne,
        "log10_Ne": np.log10(ne),
        "true_dlog10Ne_dx_km": d_ne_dx / denom,
        "true_dlog10Ne_dy_km": d_ne_dy / denom,
        "true_dlog10Ne_dt_sec": d_ne_dt / denom,
    }


# ============================================================
# GENERAL HELPERS
# ============================================================

def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def torch_load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def choose_device() -> torch.device:
    if DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(DEVICE)


def build_model_from_checkpoint(checkpoint: dict, device: torch.device) -> MLPINR:
    cfg = checkpoint.get("config", {})

    coord_scalers = checkpoint.get("coord_scalers")
    target_scaler = checkpoint.get("target_scaler")

    if coord_scalers is None:
        raise KeyError("Checkpoint is missing coord_scalers.")
    if target_scaler is None:
        raise KeyError("Checkpoint is missing target_scaler.")

    in_features = len(coord_scalers)
    out_features = 1

    model = MLPINR(
        in_features=in_features,
        out_features=out_features,
        hidden_features=int(cfg.get("hidden_features", 256)),
        hidden_layers=int(cfg.get("hidden_layers", 3)),
        activation=str(cfg.get("activation", "sine")),
        first_omega_0=float(cfg.get("first_omega_0", 5.0)),
        hidden_omega_0=float(cfg.get("hidden_omega_0", 5.0)),
        outermost_linear=True,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def patches_from_config(config: dict) -> list[MovingGaussianPatch]:
    patches_cfg = config.get("patches")
    if not patches_cfg:
        raise KeyError("synthetic_config.json is missing the 'patches' list.")

    patches = []
    for p in patches_cfg:
        patches.append(
            MovingGaussianPatch(
                name=str(p.get("name", "patch")),
                amplitude_m3=float(p["amplitude_m3"]),
                sigma_x_km=float(p["sigma_x_km"]),
                sigma_y_km=float(p["sigma_y_km"]),
                x0_km=float(p["x0_km"]),
                y0_km=float(p["y0_km"]),
                vx_km_s=float(p["vx_km_s"]),
                vy_km_s=float(p["vy_km_s"]),
            )
        )
    return patches


def denormalize_target(pred_norm: np.ndarray, target_scaler: dict) -> np.ndarray:
    values_norm = np.asarray(pred_norm, dtype=np.float64)
    vmin = float(target_scaler["min"])
    vmax = float(target_scaler["max"])
    return 0.5 * (values_norm + 1.0) * (vmax - vmin) + vmin


def normalize_coords_from_scalers(
    x_km: np.ndarray,
    y_km: np.ndarray,
    t_sec: np.ndarray,
    coord_scalers: dict,
) -> np.ndarray:
    x_min = float(coord_scalers["x_km"]["min"])
    x_max = float(coord_scalers["x_km"]["max"])
    y_min = float(coord_scalers["y_km"]["min"])
    y_max = float(coord_scalers["y_km"]["max"])
    t_min = float(coord_scalers["t_sec"]["min"])
    t_max = float(coord_scalers["t_sec"]["max"])

    x_norm = 2.0 * (x_km - x_min) / (x_max - x_min) - 1.0
    y_norm = 2.0 * (y_km - y_min) / (y_max - y_min) - 1.0
    t_norm = 2.0 * (t_sec - t_min) / (t_max - t_min) - 1.0

    return np.column_stack([x_norm.ravel(), y_norm.ravel(), t_norm.ravel()]).astype(np.float32)


def predict_log10_on_coords(
    model: torch.nn.Module,
    coords_norm: np.ndarray,
    target_scaler: dict,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    outputs = []
    n = coords_norm.shape[0]

    with torch.no_grad():
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            coords = torch.from_numpy(coords_norm[start:end]).to(device)
            pred_norm = model(coords).detach().cpu().numpy()
            outputs.append(pred_norm)

    pred_norm_all = np.concatenate(outputs, axis=0)[:, 0]
    return denormalize_target(pred_norm_all, target_scaler=target_scaler)


def predict_log10_and_gradients_on_coords(
    model: torch.nn.Module,
    coords_norm: np.ndarray,
    coord_scalers: dict,
    target_scaler: dict,
    device: torch.device,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    """
    Evaluate model log10(Ne) and first derivatives with respect to physical units.

    The model outputs normalized log10(Ne) as a function of normalized coords.
    Convert derivatives using chain rule:

        d log10Ne / dx_km
        = target_scale * d pred_norm / d x_norm * d x_norm / dx_km
    """

    pred_list = []
    dx_list = []
    dy_list = []
    dt_list = []

    target_scale = 0.5 * (float(target_scaler["max"]) - float(target_scaler["min"]))

    dxnorm_dx = 2.0 / (float(coord_scalers["x_km"]["max"]) - float(coord_scalers["x_km"]["min"]))
    dynorm_dy = 2.0 / (float(coord_scalers["y_km"]["max"]) - float(coord_scalers["y_km"]["min"]))
    dtnorm_dt = 2.0 / (float(coord_scalers["t_sec"]["max"]) - float(coord_scalers["t_sec"]["min"]))

    n = coords_norm.shape[0]

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        coords = torch.from_numpy(coords_norm[start:end]).to(device)
        coords = coords.detach().clone().requires_grad_(True)

        pred_norm = model(coords)

        grad_norm = torch.autograd.grad(
            outputs=pred_norm,
            inputs=coords,
            grad_outputs=torch.ones_like(pred_norm),
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]

        pred_log10 = denormalize_target(
            pred_norm.detach().cpu().numpy()[:, 0],
            target_scaler=target_scaler,
        )

        pred_list.append(pred_log10)
        dx_list.append((target_scale * grad_norm[:, 0].detach().cpu().numpy() * dxnorm_dx).astype(np.float64))
        dy_list.append((target_scale * grad_norm[:, 1].detach().cpu().numpy() * dynorm_dy).astype(np.float64))
        dt_list.append((target_scale * grad_norm[:, 2].detach().cpu().numpy() * dtnorm_dt).astype(np.float64))

    return {
        "pred_log10_Ne": np.concatenate(pred_list, axis=0),
        "pred_dlog10Ne_dx_km": np.concatenate(dx_list, axis=0),
        "pred_dlog10Ne_dy_km": np.concatenate(dy_list, axis=0),
        "pred_dlog10Ne_dt_sec": np.concatenate(dt_list, axis=0),
    }


def compute_metrics(error: np.ndarray, prefix: str = "") -> dict[str, float]:
    error = np.asarray(error, dtype=np.float64)
    abs_error = np.abs(error)

    out = {
        f"{prefix}mse": float(np.mean(error ** 2)),
        f"{prefix}rmse": float(np.sqrt(np.mean(error ** 2))),
        f"{prefix}mae": float(np.mean(abs_error)),
        f"{prefix}bias": float(np.mean(error)),
        f"{prefix}max_abs": float(np.max(abs_error)),
        f"{prefix}p95_abs": float(np.quantile(abs_error, 0.95)),
        f"{prefix}p99_abs": float(np.quantile(abs_error, 0.99)),
    }
    return out


def select_time_indices(obs_df: pd.DataFrame) -> list[int]:
    available = [int(x) for x in np.sort(obs_df["time_index"].unique())]

    if isinstance(TIME_SELECTION, list):
        requested = [int(x) for x in TIME_SELECTION]
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"Requested time_index values are unavailable: {missing}")
        return requested

    if TIME_SELECTION == "all":
        return available

    if TIME_SELECTION == "first_middle_last":
        if len(available) <= 3:
            return available
        return [available[0], available[len(available) // 2], available[-1]]

    raise ValueError(f"Unsupported TIME_SELECTION: {TIME_SELECTION}")


def make_dense_grid(config: dict, coord_scalers: dict) -> tuple[np.ndarray, np.ndarray]:
    if DOMAIN_MODE == "full_domain":
        half = 0.5 * float(config["domain_size_km"])
        x = np.linspace(-half, half, int(GRID_NX), dtype=np.float64)
        y = np.linspace(-half, half, int(GRID_NY), dtype=np.float64)
    elif DOMAIN_MODE == "training_extent":
        x = np.linspace(
            float(coord_scalers["x_km"]["min"]),
            float(coord_scalers["x_km"]["max"]),
            int(GRID_NX),
            dtype=np.float64,
        )
        y = np.linspace(
            float(coord_scalers["y_km"]["min"]),
            float(coord_scalers["y_km"]["max"]),
            int(GRID_NY),
            dtype=np.float64,
        )
    else:
        raise ValueError(f"Unsupported DOMAIN_MODE: {DOMAIN_MODE}")

    X, Y = np.meshgrid(x, y)
    return X, Y


def plot_truth_pred_error(
    df_time: pd.DataFrame,
    obs_time: pd.DataFrame,
    time_index: int,
    out_path: Path,
) -> None:
    x_unique = np.sort(df_time["x_km"].unique())
    y_unique = np.sort(df_time["y_km"].unique())

    nx = len(x_unique)
    ny = len(y_unique)

    X = df_time["x_km"].to_numpy().reshape(ny, nx)
    Y = df_time["y_km"].to_numpy().reshape(ny, nx)
    truth = df_time["true_log10_Ne"].to_numpy().reshape(ny, nx)
    pred = df_time["pred_log10_Ne"].to_numpy().reshape(ny, nx)

    # Error map shown in the third panel:
    # compute the difference in physical density units first, then compress
    # the dynamic range with a signed log transform.  This avoids treating
    # pred_log10 - true_log10 as an additive density error.
    err = df_time["signed_log10_abs_error_Ne"].to_numpy().reshape(ny, nx)

    vmin = float(min(np.nanmin(truth), np.nanmin(pred)))
    vmax = float(max(np.nanmax(truth), np.nanmax(pred)))
    err_abs = float(np.nanmax(np.abs(err)))
    if err_abs <= 0 or not np.isfinite(err_abs):
        err_abs = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)

    im0 = axes[0].pcolormesh(X, Y, truth, shading="auto", cmap=CMAP_FIELD, vmin=vmin, vmax=vmax)
    axes[0].scatter(obs_time["x_km"], obs_time["y_km"], c="none", edgecolor="k", s=18, linewidth=0.4)
    axes[0].set_title("Truth log10(Ne)")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(X, Y, pred, shading="auto", cmap=CMAP_FIELD, vmin=vmin, vmax=vmax)
    axes[1].scatter(obs_time["x_km"], obs_time["y_km"], c="none", edgecolor="k", s=18, linewidth=0.4)
    axes[1].set_title("INR prediction log10(Ne)")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(X, Y, err, shading="auto", cmap=CMAP_ERROR, vmin=-err_abs, vmax=err_abs)
    axes[2].scatter(obs_time["x_km"], obs_time["y_km"], c="none", edgecolor="k", s=18, linewidth=0.4)
    axes[2].set_title("Signed log10(|pred_Ne - true_Ne| + 1)")
    fig.colorbar(im2, ax=axes[2])

    t_min = float(df_time["t_min"].median())
    for ax in axes:
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"Synthetic reconstruction | time_index={time_index} | t={t_min:.1f} min")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_gradient_error(
    df_time: pd.DataFrame,
    obs_time: pd.DataFrame,
    time_index: int,
    out_path: Path,
) -> None:
    x_unique = np.sort(df_time["x_km"].unique())
    y_unique = np.sort(df_time["y_km"].unique())
    nx = len(x_unique)
    ny = len(y_unique)

    X = df_time["x_km"].to_numpy().reshape(ny, nx)
    Y = df_time["y_km"].to_numpy().reshape(ny, nx)

    err_grad_mag = df_time["error_grad_xy_mag"].to_numpy().reshape(ny, nx)
    pred_grad_mag = df_time["pred_grad_xy_mag"].to_numpy().reshape(ny, nx)
    true_grad_mag = df_time["true_grad_xy_mag"].to_numpy().reshape(ny, nx)

    vmax_grad = float(max(np.nanmax(pred_grad_mag), np.nanmax(true_grad_mag)))
    vmax_err = float(np.nanmax(err_grad_mag))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)

    im0 = axes[0].pcolormesh(X, Y, true_grad_mag, shading="auto", cmap=CMAP_FIELD, vmin=0.0, vmax=vmax_grad)
    axes[0].scatter(obs_time["x_km"], obs_time["y_km"], c="none", edgecolor="k", s=18, linewidth=0.4)
    axes[0].set_title("Truth |grad_xy log10Ne|")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(X, Y, pred_grad_mag, shading="auto", cmap=CMAP_FIELD, vmin=0.0, vmax=vmax_grad)
    axes[1].scatter(obs_time["x_km"], obs_time["y_km"], c="none", edgecolor="k", s=18, linewidth=0.4)
    axes[1].set_title("INR |grad_xy log10Ne|")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(X, Y, err_grad_mag, shading="auto", cmap=CMAP_FIELD, vmin=0.0, vmax=vmax_err)
    axes[2].scatter(obs_time["x_km"], obs_time["y_km"], c="none", edgecolor="k", s=18, linewidth=0.4)
    axes[2].set_title("|pred grad_xy - truth grad_xy|")
    fig.colorbar(im2, ax=axes[2])

    t_min = float(df_time["t_min"].median())
    for ax in axes:
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"Synthetic gradient error | time_index={time_index} | t={t_min:.1f} min")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main() -> None:
    run_dir = Path(RUN_DIR)
    synthetic_csv = Path(SYNTHETIC_CSV)
    synthetic_config = Path(SYNTHETIC_CONFIG)
    checkpoint_path = run_dir / CHECKPOINT_NAME
    out_dir = run_dir / ANALYSIS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not synthetic_csv.exists():
        raise FileNotFoundError(f"Synthetic CSV not found: {synthetic_csv}")
    if not synthetic_config.exists():
        raise FileNotFoundError(f"Synthetic config not found: {synthetic_config}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = choose_device()
    print("Synthetic reconstruction analysis")
    print(f"  run_dir:          {run_dir}")
    print(f"  checkpoint:       {checkpoint_path}")
    print(f"  synthetic_csv:    {synthetic_csv}")
    print(f"  synthetic_config: {synthetic_config}")
    print(f"  output_dir:       {out_dir}")
    print(f"  device:           {device}")
    print(f"  grid:             {GRID_NX} x {GRID_NY}")
    print(f"  domain mode:      {DOMAIN_MODE}")
    print(f"  gradients:        {COMPUTE_GRADIENT_ERRORS}")
    print()

    obs_df = pd.read_csv(synthetic_csv)
    config = load_json(synthetic_config)
    patches = patches_from_config(config)

    checkpoint = torch_load_checkpoint(checkpoint_path, device=device)
    model = build_model_from_checkpoint(checkpoint, device=device)

    coord_scalers = checkpoint["coord_scalers"]
    target_scaler = checkpoint["target_scaler"]

    time_indices = select_time_indices(obs_df)
    print(f"Time indices selected: {time_indices}")

    X, Y = make_dense_grid(config=config, coord_scalers=coord_scalers)

    summary_rows = []

    for time_index in time_indices:
        obs_time = obs_df[obs_df["time_index"] == time_index].copy()
        if len(obs_time) == 0:
            raise ValueError(f"No observations for time_index={time_index}")

        t_sec = float(obs_time["t_sec"].median())
        t_min = t_sec / 60.0
        T = np.full_like(X, t_sec, dtype=np.float64)

        midpoint_truth = evaluate_synthetic_plasma(
            x_km=X,
            y_km=Y,
            t_sec=T,
            background_ne_m3=float(config["background_ne_m3"]),
            patches=patches,
        )
        integration_time_sec = float(config.get("integration_time_sec", 0.0))
        integration_samples = int(config.get("integration_samples", 1))
        truth = evaluate_integration_averaged_plasma(
            x_km=X,
            y_km=Y,
            t_sec=T,
            integration_time_sec=integration_time_sec,
            integration_samples=integration_samples,
            background_ne_m3=float(config["background_ne_m3"]),
            patches=patches,
        )

        coords_norm = normalize_coords_from_scalers(
            x_km=X.ravel(),
            y_km=Y.ravel(),
            t_sec=T.ravel(),
            coord_scalers=coord_scalers,
        )

        if COMPUTE_GRADIENT_ERRORS:
            pred = predict_log10_and_gradients_on_coords(
                model=model,
                coords_norm=coords_norm,
                coord_scalers=coord_scalers,
                target_scaler=target_scaler,
                device=device,
                chunk_size=GRADIENT_CHUNK_SIZE,
            )
            pred_log10 = pred["pred_log10_Ne"]
        else:
            pred_log10 = predict_log10_on_coords(
                model=model,
                coords_norm=coords_norm,
                target_scaler=target_scaler,
                device=device,
                chunk_size=PREDICT_CHUNK_SIZE,
            )
            pred = {"pred_log10_Ne": pred_log10}

        true_log10 = truth["log10_Ne"].ravel()

        # The network predicts log10(Ne), because that is the training target.
        # There are two different, useful errors:
        #
        #   1. error_log10_Ne = pred_log10 - true_log10
        #      This is a dex/log-ratio error: log10(pred_Ne / true_Ne).
        #      It is useful, but it is NOT an additive physical density error.
        #
        #   2. error_Ne = pred_Ne - true_Ne
        #      This is the physical electron-density error in m^-3.
        #      For plotting, signed_log10_abs_error_Ne compresses this physical
        #      error after the linear-space subtraction.
        true_Ne = truth["Ne"].ravel()
        pred_Ne = np.power(10.0, pred_log10)

        error_log10_Ne = pred_log10 - true_log10
        abs_error_log10_Ne = np.abs(error_log10_Ne)

        error_Ne = pred_Ne - true_Ne
        abs_error_Ne = np.abs(error_Ne)
        rel_error_Ne = error_Ne / np.maximum(true_Ne, 1.0e-30)
        abs_rel_error_Ne = np.abs(rel_error_Ne)

        # Log-compressed physical error for signed error maps.
        # The +1 avoids log10(0). The sign preserves over/under prediction.
        signed_log10_abs_error_Ne = np.sign(error_Ne) * np.log10(abs_error_Ne + 1.0)

        midpoint_true_Ne = midpoint_truth["Ne"].ravel()
        midpoint_true_log10 = midpoint_truth["log10_Ne"].ravel()
        midpoint_error_Ne = pred_Ne - midpoint_true_Ne
        midpoint_error_log10_Ne = pred_log10 - midpoint_true_log10

        dense_df = pd.DataFrame(
            {
                "time_index": np.full(X.size, int(time_index), dtype=int),
                "t_sec": np.full(X.size, t_sec, dtype=float),
                "t_min": np.full(X.size, t_min, dtype=float),
                "integration_time_sec": np.full(X.size, integration_time_sec, dtype=float),
                "x_km": X.ravel(),
                "y_km": Y.ravel(),
                "true_Ne": true_Ne,
                "pred_Ne": pred_Ne,
                "error_Ne": error_Ne,
                "abs_error_Ne": abs_error_Ne,
                "rel_error_Ne": rel_error_Ne,
                "abs_rel_error_Ne": abs_rel_error_Ne,
                "signed_log10_abs_error_Ne": signed_log10_abs_error_Ne,
                "true_log10_Ne": true_log10,
                "pred_log10_Ne": pred_log10,
                "error_log10_Ne": error_log10_Ne,
                "abs_error_log10_Ne": abs_error_log10_Ne,
                "midpoint_true_Ne": midpoint_true_Ne,
                "midpoint_true_log10_Ne": midpoint_true_log10,
                "midpoint_error_Ne": midpoint_error_Ne,
                "midpoint_abs_error_Ne": np.abs(midpoint_error_Ne),
                "midpoint_error_log10_Ne": midpoint_error_log10_Ne,
                "true_dlog10Ne_dx_km": truth["true_dlog10Ne_dx_km"].ravel(),
                "true_dlog10Ne_dy_km": truth["true_dlog10Ne_dy_km"].ravel(),
                "true_dlog10Ne_dt_sec": truth["true_dlog10Ne_dt_sec"].ravel(),
            }
        )

        row = {
            "time_index": int(time_index),
            "t_sec": t_sec,
            "t_min": t_min,
            "n_grid_points": int(X.size),
            "domain_mode": DOMAIN_MODE,
            "grid_nx": int(GRID_NX),
            "grid_ny": int(GRID_NY),
            "integration_time_sec": integration_time_sec,
        }
        # Keep the original log-space/dex metrics, but also add physical
        # linear-density metrics.
        row.update(compute_metrics(error_log10_Ne, prefix="log10_"))
        row.update(compute_metrics(error_Ne, prefix="Ne_"))
        row.update(compute_metrics(rel_error_Ne, prefix="rel_Ne_"))
        row.update(compute_metrics(signed_log10_abs_error_Ne, prefix="signed_log10_abs_Ne_"))
        row.update(compute_metrics(midpoint_error_Ne, prefix="midpoint_Ne_"))
        row.update(compute_metrics(midpoint_error_log10_Ne, prefix="midpoint_log10_"))

        if COMPUTE_GRADIENT_ERRORS:
            dense_df["pred_dlog10Ne_dx_km"] = pred["pred_dlog10Ne_dx_km"]
            dense_df["pred_dlog10Ne_dy_km"] = pred["pred_dlog10Ne_dy_km"]
            dense_df["pred_dlog10Ne_dt_sec"] = pred["pred_dlog10Ne_dt_sec"]

            dense_df["error_dlog10Ne_dx_km"] = dense_df["pred_dlog10Ne_dx_km"] - dense_df["true_dlog10Ne_dx_km"]
            dense_df["error_dlog10Ne_dy_km"] = dense_df["pred_dlog10Ne_dy_km"] - dense_df["true_dlog10Ne_dy_km"]
            dense_df["error_dlog10Ne_dt_sec"] = dense_df["pred_dlog10Ne_dt_sec"] - dense_df["true_dlog10Ne_dt_sec"]

            dense_df["true_grad_xy_mag"] = np.sqrt(
                dense_df["true_dlog10Ne_dx_km"] ** 2
                + dense_df["true_dlog10Ne_dy_km"] ** 2
            )
            dense_df["pred_grad_xy_mag"] = np.sqrt(
                dense_df["pred_dlog10Ne_dx_km"] ** 2
                + dense_df["pred_dlog10Ne_dy_km"] ** 2
            )
            dense_df["error_grad_xy_mag"] = np.sqrt(
                dense_df["error_dlog10Ne_dx_km"] ** 2
                + dense_df["error_dlog10Ne_dy_km"] ** 2
            )

            row.update(compute_metrics(dense_df["error_dlog10Ne_dx_km"].to_numpy(), prefix="grad_x_"))
            row.update(compute_metrics(dense_df["error_dlog10Ne_dy_km"].to_numpy(), prefix="grad_y_"))
            row.update(compute_metrics(dense_df["error_dlog10Ne_dt_sec"].to_numpy(), prefix="grad_t_"))
            row.update(compute_metrics(dense_df["error_grad_xy_mag"].to_numpy(), prefix="grad_xy_mag_"))

        summary_rows.append(row)

        plot_truth_pred_error(
            df_time=dense_df,
            obs_time=obs_time,
            time_index=int(time_index),
            out_path=out_dir / f"truth_pred_error_time_{int(time_index):04d}.png",
        )

        if COMPUTE_GRADIENT_ERRORS:
            plot_gradient_error(
                df_time=dense_df,
                obs_time=obs_time,
                time_index=int(time_index),
                out_path=out_dir / f"gradient_error_time_{int(time_index):04d}.png",
            )

        if SAVE_DENSE_CSV:
            dense_path = out_dir / f"dense_reconstruction_time_{int(time_index):04d}.csv"
            dense_df.to_csv(dense_path, index=False)

        print(
            f"time_index={time_index:04d} | "
            f"log10 RMSE={row['log10_rmse']:.4e} | "
            f"Ne RMSE={row['Ne_rmse']:.4e} m^-3 | "
            f"midpoint Ne RMSE={row['midpoint_Ne_rmse']:.4e} m^-3 | "
            f"rel RMSE={row['rel_Ne_rmse']:.4e} | "
            f"signed-log-phys p95={row['signed_log10_abs_Ne_p95_abs']:.4e}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "error_summary_by_time.csv"
    summary_df.to_csv(summary_path, index=False)

    # Also save an all-time average row for quick reference.
    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
    mean_row = summary_df[numeric_cols].mean().to_dict()
    mean_row["time_index"] = -1
    mean_row["t_sec"] = np.nan
    mean_row["t_min"] = np.nan
    mean_row["label"] = "mean_over_selected_times"
    mean_path = out_dir / "error_summary_mean.csv"
    pd.DataFrame([mean_row]).to_csv(mean_path, index=False)

    print()
    print("Saved:")
    print(f"  {summary_path}")
    print(f"  {mean_path}")
    print(f"  plots/CSVs in {out_dir}")
    print("DONE")


if __name__ == "__main__":
    main()
