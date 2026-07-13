# -*- coding: utf-8 -*-
"""
datasets_synthetic.py

Dataset utilities for synthetic x-y-time plasma INR experiments.

This file is intentionally independent from the AMISR HDF5 reader.
It reads the CSV produced by synthetic_plasma.py / synthetic_plasma_v2.py
and exposes the same basic interface used by the INR trainers:

    coords = [x_norm, y_norm, t_norm]
    values = normalized log10_Ne

The raw dataframe keeps the analytical truth columns when present, so later
we can compute density and gradient errors against known ground truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ============================================================
# Normalization helpers
# ============================================================

def normalize_minus1_plus1(
    values: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    name: str = "values",
) -> tuple[np.ndarray, float, float]:
    """
    Normalize values to [-1, 1].

    Formula:
        y_norm = 2 * (y - ymin) / (ymax - ymin) - 1
    """

    values = np.asarray(values, dtype=np.float64)

    if vmin is None:
        vmin = float(np.nanmin(values))

    if vmax is None:
        vmax = float(np.nanmax(values))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError(f"{name}: non-finite normalization limits.")

    if vmax <= vmin:
        raise ValueError(
            f"{name}: cannot normalize because vmax <= vmin "
            f"({vmax} <= {vmin})."
        )

    values_norm = 2.0 * (values - vmin) / (vmax - vmin) - 1.0

    return values_norm, vmin, vmax


def denormalize_minus1_plus1(
    values_norm: np.ndarray,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """
    Convert normalized values from [-1, 1] back to the original scale.
    """

    values_norm = np.asarray(values_norm, dtype=np.float64)

    return 0.5 * (values_norm + 1.0) * (vmax - vmin) + vmin


# ============================================================
# Synthetic dataset
# ============================================================

class SyntheticPlasmaTimeDataset(Dataset):
    """
    Dataset for synthetic x-y-time plasma observations.

    Expected input CSV columns:
        x_km
        y_km
        t_sec
        time_index
        log10_Ne

    Optional columns are preserved, for example:
        Ne
        log10_Ne_true
        true_dlog10Ne_dx_km
        true_dlog10Ne_dy_km
        true_dlog10Ne_dt_sec
        beam_id
        sample_geometry
        is_observed

    dataset[0] returns the full coordinate-value set.
    """

    def __init__(
        self,
        csv_path: str | Path,
        coord_cols: Sequence[str] = ("x_km", "y_km", "t_sec"),
        target_col: str = "log10_Ne",
        normalize_coords: bool = True,
        normalize_values: bool = True,
        drop_bad_rows: bool = True,
        window_start_index: int | None = None,
        window_size_records: int | None = None,
        verbose: bool = True,
    ):
        super().__init__()

        self.csv_path = Path(csv_path)
        self.coord_cols = tuple(coord_cols)
        self.target_col = str(target_col)
        self.normalize_coords = bool(normalize_coords)
        self.normalize_values = bool(normalize_values)
        self.window_start_index = window_start_index
        self.window_size_records = window_size_records

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Synthetic CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        required_cols = list(self.coord_cols) + [self.target_col]

        if "time_index" not in df.columns:
            raise KeyError("Synthetic CSV must contain a time_index column.")

        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Required column missing from synthetic CSV: {col}")

        # --------------------------------------------------------
        # Optional temporal window selection by record index.
        # This mirrors the real AMISR dataset behavior.
        # --------------------------------------------------------
        unique_times = np.sort(df["time_index"].unique())

        if window_start_index is not None or window_size_records is not None:
            if window_start_index is None:
                window_start_index = 0
            if window_size_records is None:
                raise ValueError(
                    "window_size_records must be provided when using window_start_index."
                )

            start = int(window_start_index)
            size = int(window_size_records)

            if start < 0:
                raise ValueError("window_start_index must be >= 0")
            if size <= 0:
                raise ValueError("window_size_records must be > 0")
            if start >= unique_times.size:
                raise ValueError(
                    f"window_start_index={start} is outside available time records "
                    f"0..{unique_times.size - 1}."
                )

            stop = min(start + size, unique_times.size)
            selected_times = unique_times[start:stop]
            df = df[df["time_index"].isin(selected_times)].copy()

            if verbose:
                print(
                    "SyntheticPlasmaTimeDataset window: "
                    f"time-record positions {start}:{stop} "
                    f"({len(selected_times)} records)"
                )

        # --------------------------------------------------------
        # Clean rows
        # --------------------------------------------------------
        df = df.replace([np.inf, -np.inf], np.nan)

        if drop_bad_rows:
            before = len(df)
            df = df.dropna(subset=required_cols).copy()
            after = len(df)

            if verbose and after < before:
                print(f"SyntheticPlasmaTimeDataset: dropped {before - after} bad rows.")

        if len(df) == 0:
            raise ValueError("No valid synthetic rows left after cleaning.")

        self.df = df.reset_index(drop=True)

        # --------------------------------------------------------
        # Extract raw arrays
        # --------------------------------------------------------
        coords_raw = self.df.loc[:, self.coord_cols].to_numpy(dtype=np.float64)
        values_raw = self.df.loc[:, [self.target_col]].to_numpy(dtype=np.float64)

        self.coords_raw = coords_raw.astype(np.float64)
        self.values_raw = values_raw.astype(np.float64)

        # --------------------------------------------------------
        # Normalize coordinates
        # --------------------------------------------------------
        self.coord_scalers: dict[str, dict[str, float]] = {}
        coords_out = np.zeros_like(coords_raw, dtype=np.float64)

        for i, col in enumerate(self.coord_cols):
            if self.normalize_coords:
                coords_out[:, i], vmin, vmax = normalize_minus1_plus1(
                    coords_raw[:, i],
                    name=col,
                )
            else:
                coords_out[:, i] = coords_raw[:, i]
                vmin = float(np.nanmin(coords_raw[:, i]))
                vmax = float(np.nanmax(coords_raw[:, i]))

            self.coord_scalers[col] = {
                "min": vmin,
                "max": vmax,
            }

        # --------------------------------------------------------
        # Normalize target
        # --------------------------------------------------------
        if self.normalize_values:
            values_out, target_min, target_max = normalize_minus1_plus1(
                values_raw,
                name=self.target_col,
            )
        else:
            values_out = values_raw
            target_min = float(np.nanmin(values_raw))
            target_max = float(np.nanmax(values_raw))

        self.target_scaler = {
            "column": self.target_col,
            "min": target_min,
            "max": target_max,
        }

        self.coords = torch.from_numpy(coords_out.astype(np.float32))
        self.values = torch.from_numpy(values_out.astype(np.float32))

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx != 0:
            raise IndexError("SyntheticPlasmaTimeDataset contains only one item: index 0.")

        return {
            "coords": self.coords,
            "values": self.values,
        }

    @property
    def n_samples(self) -> int:
        return self.coords.shape[0]

    @property
    def in_features(self) -> int:
        return self.coords.shape[1]

    @property
    def out_features(self) -> int:
        return self.values.shape[1]

    def denormalize_target(self, values_norm: np.ndarray) -> np.ndarray:
        if not self.normalize_values:
            return np.asarray(values_norm, dtype=np.float64)

        return denormalize_minus1_plus1(
            values_norm,
            vmin=self.target_scaler["min"],
            vmax=self.target_scaler["max"],
        )

    def denormalize_coords(self, coords_norm: np.ndarray) -> np.ndarray:
        coords_norm = np.asarray(coords_norm, dtype=np.float64)

        if coords_norm.ndim != 2:
            raise ValueError(f"coords_norm must be 2D, got shape {coords_norm.shape}")

        if coords_norm.shape[1] != len(self.coord_cols):
            raise ValueError(
                f"coords_norm has {coords_norm.shape[1]} columns, "
                f"expected {len(self.coord_cols)}."
            )

        if not self.normalize_coords:
            return coords_norm

        coords_raw = np.zeros_like(coords_norm, dtype=np.float64)

        for i, col in enumerate(self.coord_cols):
            vmin = self.coord_scalers[col]["min"]
            vmax = self.coord_scalers[col]["max"]

            coords_raw[:, i] = denormalize_minus1_plus1(
                coords_norm[:, i],
                vmin=vmin,
                vmax=vmax,
            )

        return coords_raw

    def make_prediction_dataframe(
        self,
        pred_values_norm: np.ndarray,
        pred_col: str = "pred_log10_Ne",
        resid_col: str = "resid_log10_Ne",
    ) -> pd.DataFrame:
        pred_values_norm = np.asarray(pred_values_norm, dtype=np.float64)

        if pred_values_norm.ndim == 2:
            if pred_values_norm.shape[1] != 1:
                raise ValueError(
                    f"pred_values_norm must have shape [N, 1], got {pred_values_norm.shape}"
                )
            pred_values_norm = pred_values_norm[:, 0]

        if pred_values_norm.shape[0] != self.n_samples:
            raise ValueError(
                f"Prediction length {pred_values_norm.shape[0]} does not match "
                f"dataset length {self.n_samples}."
            )

        pred_raw = self.denormalize_target(pred_values_norm)
        true_raw = self.df[self.target_col].to_numpy(dtype=np.float64)
        residual = pred_raw - true_raw

        out = self.df.copy()
        out[pred_col] = pred_raw
        out[resid_col] = residual

        if "log10_Ne_true" in out.columns:
            out["resid_log10_Ne_true"] = pred_raw - out["log10_Ne_true"].to_numpy(dtype=np.float64)

        return out

    def get_time_indices(self) -> np.ndarray:
        return np.sort(self.df["time_index"].unique())

    def dataframe_for_time_index(self, time_index: int) -> pd.DataFrame:
        return self.df[self.df["time_index"] == time_index].copy()

    def summary(self) -> None:
        print("SyntheticPlasmaTimeDataset")
        print(f"  csv_path:     {self.csv_path}")
        print(f"  n_samples:    {self.n_samples}")
        print(f"  coords shape: {tuple(self.coords.shape)}")
        print(f"  values shape: {tuple(self.values.shape)}")
        print(f"  coord cols:   {self.coord_cols}")
        print(f"  target col:   {self.target_col}")
        print()

        print("Coordinate scalers:")
        for col, scaler in self.coord_scalers.items():
            print(f"  {col}: min={scaler['min']:.6f}, max={scaler['max']:.6f}")

        print()
        print("Target scaler:")
        print(
            f"  {self.target_col}: "
            f"min={self.target_scaler['min']:.6f}, "
            f"max={self.target_scaler['max']:.6f}"
        )

        n_times = self.df["time_index"].nunique()
        print()
        print(f"Time records: {n_times}")
        print(
            f"t_sec range: "
            f"{self.df['t_sec'].min():.3f} to "
            f"{self.df['t_sec'].max():.3f}"
        )

        if "beam_id" in self.df.columns:
            print(f"Unique sample locations / beam_id: {self.df['beam_id'].nunique()}")

        if "log10_Ne_true" in self.df.columns:
            err = self.df[self.target_col].to_numpy(dtype=float) - self.df["log10_Ne_true"].to_numpy(dtype=float)
            print()
            print("Observation noise relative to truth:")
            print(f"  mean: {np.mean(err):.6e}")
            print(f"  std:  {np.std(err):.6e}")
