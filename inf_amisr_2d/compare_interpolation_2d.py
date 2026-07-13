# -*- coding: utf-8 -*-
"""
compare_interpolation_2d.py

Compare classical 2D interpolation methods on one AMISR selected slice.

Input:
    data/slice111748_h330_best_slice.csv

Methods:
    nearest
    linear
    cubic
    idw
    natural_neighbor, if MetPy is installed

Important:
    The grid is rectangular in memory, but we mask it using a Delaunay domain:
        valid grid point = inside triangulation of measured x/y points

    This is the interpolation-style meaning of:
        "inside the area where there is data"

    Outside the triangulation, values are set to NaN.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import Delaunay


# ============================================================
# Optional natural neighbor
# ============================================================

try:
    from metpy.interpolate import natural_neighbor_to_grid
    HAS_METPY = True
except Exception:
    natural_neighbor_to_grid = None
    HAS_METPY = False


# ============================================================
# Helpers
# ============================================================

def read_slice(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = ["x_km", "y_km", "log10_Ne"]

    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required).copy()

    if len(df) < 3:
        raise ValueError("Need at least 3 valid points for 2D interpolation.")

    return df


def make_grid(
    x: np.ndarray,
    y: np.ndarray,
    nx: int,
    ny: int,
    padding_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))

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


def delaunay_domain_mask(points_xy: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    True where grid points are inside the Delaunay triangulation.

    This is the same support idea used by scattered linear interpolation:
    inside the triangulated data domain = valid.
    outside = NaN.
    """

    tri = Delaunay(points_xy)

    query_xy = np.column_stack([X.ravel(), Y.ravel()])
    inside = tri.find_simplex(query_xy) >= 0

    return inside.reshape(X.shape)


def interpolate_idw(
    points_xy: np.ndarray,
    values: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    power: float = 2.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Inverse-distance weighting interpolation.

    This is not natural neighbor.
    It is just a useful smooth local baseline.
    """

    query_xy = np.column_stack([X.ravel(), Y.ravel()])

    diff = query_xy[:, None, :] - points_xy[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    # Exact match handling.
    exact = dist < eps

    weights = 1.0 / np.maximum(dist, eps) ** power
    pred = np.sum(weights * values[None, :], axis=1) / np.sum(weights, axis=1)

    if exact.any():
        exact_rows = np.where(exact.any(axis=1))[0]
        for row in exact_rows:
            col = np.where(exact[row])[0][0]
            pred[row] = values[col]

    return pred.reshape(X.shape)


def run_interpolations(
    points_xy: np.ndarray,
    values: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    domain_mask: np.ndarray,
    methods: list[str],
    idw_power: float,
) -> dict[str, np.ndarray]:
    out = {}

    for method in methods:
        method = method.lower()

        if method in ["nearest", "linear", "cubic"]:
            print(f"Interpolating: {method}")

            Z = griddata(
                points_xy,
                values,
                (X, Y),
                method=method,
            )

        elif method == "idw":
            print(f"Interpolating: idw, power={idw_power}")

            Z = interpolate_idw(
                points_xy=points_xy,
                values=values,
                X=X,
                Y=Y,
                power=idw_power,
            )

        elif method in ["natural", "natural_neighbor", "natural-neighbor"]:
            if not HAS_METPY:
                print("Skipping natural_neighbor: MetPy is not installed.")
                print("Install with: conda install -c conda-forge metpy")
                continue

            print("Interpolating: natural_neighbor using MetPy")

            Z = natural_neighbor_to_grid(
                points_xy[:, 0],
                points_xy[:, 1],
                values,
                X,
                Y,
            )

        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        Z = np.asarray(Z, dtype=float)

        # Make all methods use the same interpolation-style support.
        Z[~domain_mask] = np.nan

        out[method] = Z

    return out


def save_grid_csv(
    out_dir: Path,
    X: np.ndarray,
    Y: np.ndarray,
    domain_mask: np.ndarray,
    fields: dict[str, np.ndarray],
) -> None:
    data = {
        "x_km": X.ravel(),
        "y_km": Y.ravel(),
        "domain_mask": domain_mask.ravel(),
    }

    for name, Z in fields.items():
        data[f"{name}_log10_Ne"] = Z.ravel()

    df = pd.DataFrame(data)

    path = out_dir / "interpolation_grid_predictions.csv"
    df.to_csv(path, index=False)

    print(f"Saved: {path}")


def plot_single_method(
    name: str,
    Z: np.ndarray,
    df: pd.DataFrame,
    X: np.ndarray,
    Y: np.ndarray,
    out_dir: Path,
    vmin: float,
    vmax: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.pcolormesh(
        X,
        Y,
        Z,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    fig.colorbar(im, ax=ax, label="interpolated log10(Ne)")

    ax.scatter(
        df["x_km"],
        df["y_km"],
        c=df["log10_Ne"],
        s=35,
        edgecolor="k",
        linewidth=0.4,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel("x east [km]")
    ax.set_ylabel("y north [km]")
    ax.set_title(f"{name} interpolation")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    path = out_dir / f"interp_{name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"Saved: {path}")


def plot_panel(
    fields: dict[str, np.ndarray],
    df: pd.DataFrame,
    X: np.ndarray,
    Y: np.ndarray,
    out_dir: Path,
    vmin: float,
    vmax: float,
) -> None:
    names = list(fields.keys())

    if len(names) == 0:
        return

    n = len(names)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 5.5 * nrows),
        squeeze=False,
    )

    last_im = None

    for i, name in enumerate(names):
        r = i // ncols
        c = i % ncols

        ax = axes[r][c]
        Z = fields[name]

        im = ax.pcolormesh(
            X,
            Y,
            Z,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )

        last_im = im

        ax.scatter(
            df["x_km"],
            df["y_km"],
            c=df["log10_Ne"],
            s=25,
            edgecolor="k",
            linewidth=0.3,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(name)
        ax.set_xlabel("x east [km]")
        ax.set_ylabel("y north [km]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    for j in range(n, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), label="log10(Ne)")

    path = out_dir / "interpolation_comparison_panel.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"Saved: {path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/slice111748_h330_best_slice.csv",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/interpolation_compare_h330",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=["nearest", "linear", "cubic", "idw", "natural_neighbor"],
        help="Interpolation methods to run.",
    )

    parser.add_argument("--grid_nx", type=int, default=250)
    parser.add_argument("--grid_ny", type=int, default=250)
    parser.add_argument("--grid_padding_frac", type=float, default=0.05)

    parser.add_argument("--idw_power", type=float, default=2.0)

    parser.add_argument("--save_grid_csv", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_slice(args.csv_path)

    points_xy = df[["x_km", "y_km"]].to_numpy(dtype=float)
    values = df["log10_Ne"].to_numpy(dtype=float)

    print("Loaded slice:")
    print(f"  csv_path: {args.csv_path}")
    print(f"  points:   {len(df)}")
    print(f"  x range:  {points_xy[:, 0].min():.3f} to {points_xy[:, 0].max():.3f} km")
    print(f"  y range:  {points_xy[:, 1].min():.3f} to {points_xy[:, 1].max():.3f} km")
    print(f"  log10_Ne: {values.min():.6f} to {values.max():.6f}")

    X, Y = make_grid(
        x=points_xy[:, 0],
        y=points_xy[:, 1],
        nx=args.grid_nx,
        ny=args.grid_ny,
        padding_frac=args.grid_padding_frac,
    )

    domain_mask = delaunay_domain_mask(points_xy, X, Y)

    print()
    print("Grid:")
    print(f"  shape:               {X.shape}")
    print(f"  domain mask:          Delaunay triangulation support")
    print(f"  valid grid fraction:  {domain_mask.mean():.3f}")

    fields = run_interpolations(
        points_xy=points_xy,
        values=values,
        X=X,
        Y=Y,
        domain_mask=domain_mask,
        methods=args.methods,
        idw_power=args.idw_power,
    )

    if args.save_grid_csv:
        save_grid_csv(
            out_dir=out_dir,
            X=X,
            Y=Y,
            domain_mask=domain_mask,
            fields=fields,
        )

    vmin = float(np.min(values))
    vmax = float(np.max(values))

    for name, Z in fields.items():
        plot_single_method(
            name=name,
            Z=Z,
            df=df,
            X=X,
            Y=Y,
            out_dir=out_dir,
            vmin=vmin,
            vmax=vmax,
        )

    plot_panel(
        fields=fields,
        df=df,
        X=X,
        Y=Y,
        out_dir=out_dir,
        vmin=vmin,
        vmax=vmax,
    )

    print()
    print("DONE")


if __name__ == "__main__":
    main()