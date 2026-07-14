# -*- coding: utf-8 -*-
"""
synthetic_plasma.py

Synthetic x-y-time plasma generator for INR tests.

Initial goal:
    Generate a known electron-density field Ne(x, y, t) with a moving
    Gaussian patch on top of a uniform background, then sample it at sparse
    fake-radar points.

Coordinates:
    x_km: horizontal coordinate [km]
    y_km: horizontal coordinate [km]
    t_sec: time [s]

Target used by the INR:
    log10_Ne = log10(Ne)

This file does not train the network. It only creates synthetic truth and
observation tables that can later be connected to the existing trainer.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Field components
# ============================================================

@dataclass
class MovingGaussianPatch:
    """
    Moving anisotropic Gaussian patch/depletion in Ne.

    Positive amplitude gives an enhancement.
    Negative amplitude gives a depletion. The final Ne is clipped by the
    generator to stay positive.
    """

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
        """
        Return this patch contribution and its physical-coordinate derivatives.

        Derivatives are for the patch contribution DeltaNe, not log10(Ne):
            d_delta_dx_km
            d_delta_dy_km
            d_delta_dt_sec
        """

        x = np.asarray(x_km, dtype=np.float64)
        y = np.asarray(y_km, dtype=np.float64)
        t = np.asarray(t_sec, dtype=np.float64)

        xc, yc = self.center(t)

        dx = x - xc
        dy = y - yc

        sx2 = self.sigma_x_km ** 2
        sy2 = self.sigma_y_km ** 2

        exponent = -0.5 * ((dx ** 2) / sx2 + (dy ** 2) / sy2)
        gaussian = np.exp(exponent)
        delta_ne = self.amplitude_m3 * gaussian

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


# ============================================================
# Synthetic field evaluation
# ============================================================

def evaluate_synthetic_plasma(
    x_km: np.ndarray,
    y_km: np.ndarray,
    t_sec: np.ndarray,
    background_ne_m3: float,
    patches: list[MovingGaussianPatch],
    min_ne_m3: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Evaluate total Ne, log10(Ne), and first derivatives of log10(Ne).

    The first derivatives are analytical and use physical units:
        dlog10Ne_dx_km
        dlog10Ne_dy_km
        dlog10Ne_dt_sec
    """

    x = np.asarray(x_km, dtype=np.float64)
    y = np.asarray(y_km, dtype=np.float64)
    t = np.asarray(t_sec, dtype=np.float64)

    ne = np.full(np.broadcast_shapes(x.shape, y.shape, t.shape), float(background_ne_m3), dtype=np.float64)
    d_ne_dx = np.zeros_like(ne)
    d_ne_dy = np.zeros_like(ne)
    d_ne_dt = np.zeros_like(ne)

    # Broadcast to common shape explicitly.
    x_b = np.broadcast_to(x, ne.shape)
    y_b = np.broadcast_to(y, ne.shape)
    t_b = np.broadcast_to(t, ne.shape)

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


# ============================================================
# Sampling geometry
# ============================================================

def make_time_array(duration_sec: float, n_times: int) -> np.ndarray:
    if n_times < 2:
        raise ValueError("n_times must be >= 2")

    return np.linspace(0.0, float(duration_sec), int(n_times), dtype=np.float64)


def make_sparse_grid_xy(
    domain_size_km: float,
    nx: int = 7,
    ny: int = 6,
    margin_frac: float = 0.10,
) -> pd.DataFrame:
    """
    Fixed sparse observation geometry over the 500 x 500 km box.

    Default nx=7, ny=6 gives 42 sample locations, similar to the number of
    beams in the real slice experiments but without pretending to be AMISR.
    """

    half = 0.5 * float(domain_size_km)
    margin = float(margin_frac) * float(domain_size_km)

    x = np.linspace(-half + margin, half - margin, int(nx))
    y = np.linspace(-half + margin, half - margin, int(ny))
    X, Y = np.meshgrid(x, y)

    return pd.DataFrame(
        {
            "beam_id": np.arange(X.size, dtype=int),
            "x_km": X.ravel(),
            "y_km": Y.ravel(),
            "sample_geometry": "sparse_grid",
        }
    )


def make_sparse_23_xy(
    domain_size_km: float,
    margin_frac: float = 0.10,
) -> pd.DataFrame:
    """
    Fixed 23-point observation geometry over the 500 x 500 km box.

    The layout is deterministic and approximately uniform: a 5 by 5 grid with
    the two far-corner points removed. This gives a sparser companion geometry
    to the existing 42-point sparse grid while preserving broad domain coverage.
    """

    half = 0.5 * float(domain_size_km)
    margin = float(margin_frac) * float(domain_size_km)

    x = np.linspace(-half + margin, half - margin, 5)
    y = np.linspace(-half + margin, half - margin, 5)
    X, Y = np.meshgrid(x, y)

    points = pd.DataFrame({"x_km": X.ravel(), "y_km": Y.ravel()})
    drop = (
        ((points["x_km"] == x[0]) & (points["y_km"] == y[0]))
        | ((points["x_km"] == x[-1]) & (points["y_km"] == y[-1]))
    )
    points = points.loc[~drop].reset_index(drop=True)

    return pd.DataFrame(
        {
            "beam_id": np.arange(len(points), dtype=int),
            "x_km": points["x_km"].to_numpy(dtype=float),
            "y_km": points["y_km"].to_numpy(dtype=float),
            "sample_geometry": "sparse_23",
        }
    )


def make_sparse_11_xy(
    domain_size_km: float,
    margin_frac: float = 0.10,
) -> pd.DataFrame:
    """Fixed 11-point stress-test geometry with broad domain coverage."""

    half = 0.5 * float(domain_size_km)
    margin = float(margin_frac) * float(domain_size_km)
    edge = half - margin

    base = [(x, y) for y in (-edge, 0.0, edge) for x in (-edge, 0.0, edge)]
    points = base + [(0.0, -0.5 * edge), (0.0, 0.5 * edge)]

    return pd.DataFrame(
        {
            "beam_id": np.arange(len(points), dtype=int),
            "x_km": [p[0] for p in points],
            "y_km": [p[1] for p in points],
            "sample_geometry": "sparse_11",
        }
    )


def make_fan_radar_xy(
    n_ranges: int = 6,
    n_angles: int = 7,
    range_min_km: float = 40.0,
    range_max_km: float = 230.0,
    angle_min_deg: float = -50.0,
    angle_max_deg: float = 50.0,
    y_offset_km: float = -70.0,
) -> pd.DataFrame:
    """
    Simple fan-like radar geometry. This is not a real AMISR layout, but it is
    more radar-like than a Cartesian grid.
    """

    ranges = np.linspace(float(range_min_km), float(range_max_km), int(n_ranges))
    angles = np.deg2rad(np.linspace(float(angle_min_deg), float(angle_max_deg), int(n_angles)))

    rows = []
    beam_id = 0
    for r in ranges:
        for a in angles:
            x = r * np.sin(a)
            y = y_offset_km + r * np.cos(a)
            rows.append(
                {
                    "beam_id": beam_id,
                    "x_km": x,
                    "y_km": y,
                    "sample_geometry": "fan",
                }
            )
            beam_id += 1

    return pd.DataFrame(rows)


def make_random_xy(
    domain_size_km: float,
    n_points: int = 42,
    margin_frac: float = 0.10,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    half = 0.5 * float(domain_size_km)
    margin = float(margin_frac) * float(domain_size_km)

    x = rng.uniform(-half + margin, half - margin, int(n_points))
    y = rng.uniform(-half + margin, half - margin, int(n_points))

    return pd.DataFrame(
        {
            "beam_id": np.arange(int(n_points), dtype=int),
            "x_km": x,
            "y_km": y,
            "sample_geometry": "random",
        }
    )


def make_observation_geometry(
    mode: str,
    domain_size_km: float,
    seed: int,
) -> pd.DataFrame:
    mode = mode.lower()

    if mode == "sparse_grid":
        return make_sparse_grid_xy(domain_size_km=domain_size_km)

    if mode == "sparse_23":
        return make_sparse_23_xy(domain_size_km=domain_size_km)

    if mode == "sparse_11":
        return make_sparse_11_xy(domain_size_km=domain_size_km)

    if mode == "fan":
        return make_fan_radar_xy()

    if mode == "random":
        return make_random_xy(domain_size_km=domain_size_km, seed=seed)

    raise ValueError(f"Unknown sample geometry: {mode}")


# ============================================================
# DataFrame builders
# ============================================================

def make_observation_dataframe(
    xy_df: pd.DataFrame,
    times_sec: np.ndarray,
    background_ne_m3: float,
    patches: list[MovingGaussianPatch],
    integration_time_sec: float = 0.0,
    integration_samples: int = 1,
    noise_std_log10: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Sample the field at fixed locations, optionally averaging Ne in linear space
    over a centered integration window before converting to log10(Ne).
    """

    rng = np.random.default_rng(seed)
    rows = []

    for time_index, t_sec in enumerate(times_sec):
        x = xy_df["x_km"].to_numpy(dtype=np.float64)
        y = xy_df["y_km"].to_numpy(dtype=np.float64)
        t = np.full_like(x, float(t_sec), dtype=np.float64)

        midpoint_vals = evaluate_synthetic_plasma(
            x_km=x,
            y_km=y,
            t_sec=t,
            background_ne_m3=background_ne_m3,
            patches=patches,
        )

        vals = evaluate_integration_averaged_plasma(
            x_km=x,
            y_km=y,
            t_sec=t,
            integration_time_sec=integration_time_sec,
            integration_samples=integration_samples,
            background_ne_m3=background_ne_m3,
            patches=patches,
        )

        log10_ne_obs = vals["log10_Ne"].copy()
        if noise_std_log10 > 0.0:
            log10_ne_obs += rng.normal(0.0, float(noise_std_log10), size=log10_ne_obs.shape)

        for i in range(len(xy_df)):
            rows.append(
                {
                    "time_index": int(time_index),
                    "t_sec": float(t_sec),
                    "t_min": float(t_sec) / 60.0,
                    "integration_time_sec": float(integration_time_sec),
                    "integration_start_sec": float(t_sec - 0.5 * integration_time_sec),
                    "integration_end_sec": float(t_sec + 0.5 * integration_time_sec),
                    "beam_id": int(xy_df.iloc[i]["beam_id"]),
                    "x_km": float(x[i]),
                    "y_km": float(y[i]),
                    "Ne": float(vals["Ne"][i]),
                    "log10_Ne_true": float(vals["log10_Ne"][i]),
                    "log10_Ne": float(log10_ne_obs[i]),
                    "Ne_midpoint": float(midpoint_vals["Ne"][i]),
                    "log10_Ne_midpoint": float(midpoint_vals["log10_Ne"][i]),
                    "true_dlog10Ne_dx_km": float(vals["true_dlog10Ne_dx_km"][i]),
                    "true_dlog10Ne_dy_km": float(vals["true_dlog10Ne_dy_km"][i]),
                    "true_dlog10Ne_dt_sec": float(vals["true_dlog10Ne_dt_sec"][i]),
                    "midpoint_dlog10Ne_dx_km": float(midpoint_vals["true_dlog10Ne_dx_km"][i]),
                    "midpoint_dlog10Ne_dy_km": float(midpoint_vals["true_dlog10Ne_dy_km"][i]),
                    "midpoint_dlog10Ne_dt_sec": float(midpoint_vals["true_dlog10Ne_dt_sec"][i]),
                    "sample_geometry": str(xy_df.iloc[i]["sample_geometry"]),
                    "is_observed": True,
                }
            )

    return pd.DataFrame(rows)


def make_truth_grid_dataframe(
    domain_size_km: float,
    grid_nx: int,
    grid_ny: int,
    times_sec: np.ndarray,
    background_ne_m3: float,
    patches: list[MovingGaussianPatch],
    integration_time_sec: float = 0.0,
    integration_samples: int = 1,
) -> pd.DataFrame:
    """
    Dense truth grid for selected times. Useful for validation and plotting.
    """

    half = 0.5 * float(domain_size_km)
    x = np.linspace(-half, half, int(grid_nx))
    y = np.linspace(-half, half, int(grid_ny))
    X, Y = np.meshgrid(x, y)

    rows = []

    for time_index, t_sec in enumerate(times_sec):
        T = np.full_like(X, float(t_sec), dtype=np.float64)

        midpoint_vals = evaluate_synthetic_plasma(
            x_km=X,
            y_km=Y,
            t_sec=T,
            background_ne_m3=background_ne_m3,
            patches=patches,
        )
        vals = evaluate_integration_averaged_plasma(
            x_km=X,
            y_km=Y,
            t_sec=T,
            integration_time_sec=integration_time_sec,
            integration_samples=integration_samples,
            background_ne_m3=background_ne_m3,
            patches=patches,
        )

        rows.append(
            pd.DataFrame(
                {
                    "time_index": np.full(X.size, int(time_index), dtype=int),
                    "t_sec": np.full(X.size, float(t_sec), dtype=float),
                    "t_min": np.full(X.size, float(t_sec) / 60.0, dtype=float),
                    "x_km": X.ravel(),
                    "y_km": Y.ravel(),
                    "Ne": vals["Ne"].ravel(),
                    "log10_Ne": vals["log10_Ne"].ravel(),
                    "Ne_midpoint": midpoint_vals["Ne"].ravel(),
                    "log10_Ne_midpoint": midpoint_vals["log10_Ne"].ravel(),
                    "true_dlog10Ne_dx_km": vals["true_dlog10Ne_dx_km"].ravel(),
                    "true_dlog10Ne_dy_km": vals["true_dlog10Ne_dy_km"].ravel(),
                    "true_dlog10Ne_dt_sec": vals["true_dlog10Ne_dt_sec"].ravel(),
                    "is_observed": np.full(X.size, False, dtype=bool),
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def evaluate_integration_averaged_plasma(
    x_km: np.ndarray,
    y_km: np.ndarray,
    t_sec: np.ndarray,
    integration_time_sec: float,
    integration_samples: int,
    background_ne_m3: float,
    patches: list[MovingGaussianPatch],
) -> dict[str, np.ndarray]:
    """Average Ne and its derivatives over a centered exposure window."""

    integration_time_sec = float(integration_time_sec)
    integration_samples = int(integration_samples)
    if integration_time_sec <= 0.0:
        return evaluate_synthetic_plasma(
            x_km=x_km,
            y_km=y_km,
            t_sec=t_sec,
            background_ne_m3=background_ne_m3,
            patches=patches,
        )
    if integration_samples < 2:
        raise ValueError("integration_samples must be >= 2 when integration_time_sec > 0")

    offsets = np.linspace(
        -0.5 * integration_time_sec,
        0.5 * integration_time_sec,
        integration_samples,
        dtype=np.float64,
    )
    x = np.asarray(x_km, dtype=np.float64)[..., None]
    y = np.asarray(y_km, dtype=np.float64)[..., None]
    t = np.asarray(t_sec, dtype=np.float64)[..., None] + offsets
    vals = evaluate_synthetic_plasma(
        x_km=x,
        y_km=y,
        t_sec=t,
        background_ne_m3=background_ne_m3,
        patches=patches,
    )

    weights = np.ones(integration_samples, dtype=np.float64)
    weights[[0, -1]] = 0.5
    weights /= np.sum(weights)
    ne = np.sum(vals["Ne"] * weights, axis=-1)
    d_ne_dx = np.sum(vals["true_dlog10Ne_dx_km"] * vals["Ne"] * np.log(10.0) * weights, axis=-1)
    d_ne_dy = np.sum(vals["true_dlog10Ne_dy_km"] * vals["Ne"] * np.log(10.0) * weights, axis=-1)
    d_ne_dt = np.sum(vals["true_dlog10Ne_dt_sec"] * vals["Ne"] * np.log(10.0) * weights, axis=-1)
    denom = np.maximum(ne * np.log(10.0), 1.0e-30)

    return {
        "Ne": ne,
        "log10_Ne": np.log10(ne),
        "true_dlog10Ne_dx_km": d_ne_dx / denom,
        "true_dlog10Ne_dy_km": d_ne_dy / denom,
        "true_dlog10Ne_dt_sec": d_ne_dt / denom,
    }


# ============================================================
# Plotting
# ============================================================

def plot_truth_slice(
    truth_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    time_index: int,
    out_path: Path,
) -> None:
    df_t = truth_df[truth_df["time_index"] == time_index]
    obs_t = obs_df[obs_df["time_index"] == time_index]

    if len(df_t) == 0:
        raise ValueError(f"No truth rows for time_index={time_index}")

    x_unique = np.sort(df_t["x_km"].unique())
    y_unique = np.sort(df_t["y_km"].unique())

    X = df_t["x_km"].to_numpy().reshape(len(y_unique), len(x_unique))
    Y = df_t["y_km"].to_numpy().reshape(len(y_unique), len(x_unique))
    Z = df_t["log10_Ne"].to_numpy().reshape(len(y_unique), len(x_unique))

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    im = ax.pcolormesh(X, Y, Z, shading="auto", cmap="plasma")
    fig.colorbar(im, ax=ax, label="true log10(Ne)")

    ax.scatter(
        obs_t["x_km"],
        obs_t["y_km"],
        c=obs_t["log10_Ne"],
        cmap="plasma",
        s=28,
        edgecolor="k",
        linewidth=0.35,
    )

    t_min = float(df_t["t_min"].median())
    ax.set_title(f"Synthetic truth and sparse samples | time_index={time_index} | t={t_min:.1f} min")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def choose_plot_time_indices(n_times: int) -> list[int]:
    if n_times <= 3:
        return list(range(n_times))
    return [0, n_times // 2, n_times - 1]


# ============================================================
# Default cases and CLI
# ============================================================

def make_default_patches(
    motion: str,
    duration_sec: float,
    patch_amplitude_m3: float,
    patch_sigma_x_km: float,
    patch_sigma_y_km: float,
    speed_km_s: float | None = None,
    direction_deg: float = 0.0,
) -> list[MovingGaussianPatch]:
    motion = motion.lower()

    if speed_km_s is None:
        vx = 360.0 / float(duration_sec)
        vy = 0.0
    else:
        theta = np.deg2rad(float(direction_deg))
        vx = float(speed_km_s) * float(np.cos(theta))
        vy = float(speed_km_s) * float(np.sin(theta))

    if motion == "left_right":
        return [
            MovingGaussianPatch(
                name="patch_1",
                amplitude_m3=float(patch_amplitude_m3),
                sigma_x_km=float(patch_sigma_x_km),
                sigma_y_km=float(patch_sigma_y_km),
                x0_km=-0.5 * float(duration_sec) * vx if speed_km_s is not None else -180.0,
                y0_km=-0.5 * float(duration_sec) * vy if speed_km_s is not None else 0.0,
                vx_km_s=vx,
                vy_km_s=vy,
            )
        ]

    if motion == "diagonal":
        if speed_km_s is None:
            vx = 360.0 / float(duration_sec)
            vy = 300.0 / float(duration_sec)
        return [
            MovingGaussianPatch(
                name="patch_1",
                amplitude_m3=float(patch_amplitude_m3),
                sigma_x_km=float(patch_sigma_x_km),
                sigma_y_km=float(patch_sigma_y_km),
                x0_km=-0.5 * float(duration_sec) * vx if speed_km_s is not None else -180.0,
                y0_km=-0.5 * float(duration_sec) * vy if speed_km_s is not None else -150.0,
                vx_km_s=vx,
                vy_km_s=vy,
            )
        ]

    if motion == "static":
        return [
            MovingGaussianPatch(
                name="patch_1",
                amplitude_m3=float(patch_amplitude_m3),
                sigma_x_km=float(patch_sigma_x_km),
                sigma_y_km=float(patch_sigma_y_km),
                x0_km=0.0,
                y0_km=0.0,
                vx_km_s=0.0,
                vy_km_s=0.0,
            )
        ]

    if motion == "flow_reversal":
        shear_speed = float(speed_km_s if speed_km_s is not None else 1.0)
        return [
            MovingGaussianPatch(
                name="north_arc",
                amplitude_m3=0.75 * float(patch_amplitude_m3),
                sigma_x_km=max(80.0, float(patch_sigma_x_km)),
                sigma_y_km=min(35.0, float(patch_sigma_y_km)),
                x0_km=-0.5 * float(duration_sec) * shear_speed,
                y0_km=70.0,
                vx_km_s=shear_speed,
                vy_km_s=0.0,
            ),
            MovingGaussianPatch(
                name="south_arc",
                amplitude_m3=0.75 * float(patch_amplitude_m3),
                sigma_x_km=max(80.0, float(patch_sigma_x_km)),
                sigma_y_km=min(35.0, float(patch_sigma_y_km)),
                x0_km=0.5 * float(duration_sec) * shear_speed,
                y0_km=-70.0,
                vx_km_s=-shear_speed,
                vy_km_s=0.0,
            ),
        ]

    raise ValueError(f"Unknown default motion case: {motion}")


def generate_synthetic_case(args: argparse.Namespace) -> dict[str, Path]:
    if args.integration_time_sec < 0.0:
        raise ValueError("integration_time_sec must be >= 0")
    if args.speed_km_s is not None and args.speed_km_s < 0.0:
        raise ValueError("speed_km_s is a magnitude and must be >= 0")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    times_sec = make_time_array(
        duration_sec=args.duration_sec,
        n_times=args.n_times,
    )

    patches = make_default_patches(
        motion=args.motion,
        duration_sec=args.duration_sec,
        patch_amplitude_m3=args.patch_amplitude_m3,
        patch_sigma_x_km=args.patch_sigma_x_km,
        patch_sigma_y_km=args.patch_sigma_y_km,
        speed_km_s=args.speed_km_s,
        direction_deg=args.direction_deg,
    )

    xy_df = make_observation_geometry(
        mode=args.sample_geometry,
        domain_size_km=args.domain_size_km,
        seed=args.seed,
    )

    obs_df = make_observation_dataframe(
        xy_df=xy_df,
        times_sec=times_sec,
        background_ne_m3=args.background_ne_m3,
        patches=patches,
        integration_time_sec=args.integration_time_sec,
        integration_samples=args.integration_samples,
        noise_std_log10=args.noise_std_log10,
        seed=args.seed,
    )

    plot_indices = choose_plot_time_indices(args.n_times)
    plot_times_sec = times_sec[plot_indices]

    truth_df = make_truth_grid_dataframe(
        domain_size_km=args.domain_size_km,
        grid_nx=args.truth_grid_nx,
        grid_ny=args.truth_grid_ny,
        times_sec=plot_times_sec,
        background_ne_m3=args.background_ne_m3,
        patches=patches,
        integration_time_sec=args.integration_time_sec,
        integration_samples=args.integration_samples,
    )

    # Keep original time_index numbers in selected truth grid.
    remap = {local_i: original_i for local_i, original_i in enumerate(plot_indices)}
    truth_df["time_index"] = truth_df["time_index"].map(remap).astype(int)

    obs_path = out_dir / "synthetic_observations.csv"
    truth_path = out_dir / "synthetic_truth_selected_times.csv"
    geom_path = out_dir / "synthetic_sample_geometry.csv"
    config_path = out_dir / "synthetic_config.json"

    obs_df.to_csv(obs_path, index=False)
    truth_df.to_csv(truth_path, index=False)
    xy_df.to_csv(geom_path, index=False)

    config = vars(args).copy()
    config["patches"] = [asdict(p) for p in patches]
    config["plot_time_indices"] = plot_indices
    config["integration_window_convention"] = "centered_on_t_sec"
    config["observed_target"] = "linear_Ne_trapezoidal_average_then_log10"
    config["instantaneous_reference"] = "field_at_integration_midpoint"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    for time_index in plot_indices:
        plot_truth_slice(
            truth_df=truth_df,
            obs_df=obs_df,
            time_index=time_index,
            out_path=out_dir / f"truth_samples_time_{time_index:04d}.png",
        )

    return {
        "output_dir": out_dir,
        "observations_csv": obs_path,
        "truth_selected_csv": truth_path,
        "geometry_csv": geom_path,
        "config_json": config_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="outputs/synthetic_left_right_test")
    parser.add_argument("--domain_size_km", type=float, default=500.0)
    parser.add_argument("--duration_sec", type=float, default=3600.0)
    parser.add_argument("--n_times", type=int, default=31)
    parser.add_argument("--background_ne_m3", type=float, default=1.0e9)
    parser.add_argument(
        "--patch_amplitude_m3",
        type=float,
        default=1.0e12,
        help="Peak Gaussian enhancement above background. 1e12 gives peak log10(Ne) near 12.",
    )
    parser.add_argument("--patch_sigma_x_km", type=float, default=45.0)
    parser.add_argument("--patch_sigma_y_km", type=float, default=45.0)
    parser.add_argument(
        "--motion",
        type=str,
        default="left_right",
        choices=["left_right", "diagonal", "static", "flow_reversal"],
    )
    parser.add_argument("--speed_km_s", type=float, default=None)
    parser.add_argument("--direction_deg", type=float, default=0.0)
    parser.add_argument("--integration_time_sec", type=float, default=0.0)
    parser.add_argument("--integration_samples", type=int, default=41)
    parser.add_argument(
        "--sample_geometry",
        type=str,
        default="sparse_grid",
        choices=["sparse_grid", "sparse_23", "sparse_11", "fan", "random"],
    )
    parser.add_argument("--truth_grid_nx", type=int, default=151)
    parser.add_argument("--truth_grid_ny", type=int, default=151)
    parser.add_argument("--noise_std_log10", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    paths = generate_synthetic_case(args)

    obs_df = pd.read_csv(paths["observations_csv"])

    print("Synthetic plasma case generated")
    print(f"  output_dir:       {paths['output_dir']}")
    print(f"  observations:     {paths['observations_csv']}")
    print(f"  truth selected:   {paths['truth_selected_csv']}")
    print(f"  geometry:         {paths['geometry_csv']}")
    print(f"  config:           {paths['config_json']}")
    print()
    print("Observation summary:")
    print(f"  rows:             {len(obs_df)}")
    print(f"  time records:     {obs_df['time_index'].nunique()}")
    print(f"  beams/locations:  {obs_df['beam_id'].nunique()}")
    print(f"  x range [km]:     {obs_df['x_km'].min():.3f} to {obs_df['x_km'].max():.3f}")
    print(f"  y range [km]:     {obs_df['y_km'].min():.3f} to {obs_df['y_km'].max():.3f}")
    print(f"  log10_Ne range:   {obs_df['log10_Ne'].min():.6f} to {obs_df['log10_Ne'].max():.6f}")
    print(f"  Ne range [m^-3]:   {obs_df['Ne'].min():.6e} to {obs_df['Ne'].max():.6e}")
    print()
    print("DONE")


if __name__ == "__main__":
    main()
