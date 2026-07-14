# -*- coding: utf-8 -*-
"""
datasets.py

Dataset utilities for 3D AMISR INR experiments.

3D here means:
    x, y, time

not:
    x, y, z

This dataset reads the real AMISR HDF5 through amisr_h5_reader_3d.py,
then converts the extracted dataframe into normalized PyTorch tensors.

Current model input:
    coords = [x_norm, y_norm, t_norm]

Current target:
    values = normalized log10(Ne)

This file does NOT:
    - train the model
    - plot results
    - save derived cache files
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from amisr_h5_reader_3d import read_amisr_h5_3d_altitude_band


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
# Dataset
# ============================================================

class RadarTimeH5Dataset(Dataset):
    """
    Dataset for a 3D AMISR time experiment.

    The raw source is the AMISR HDF5 file.

    The reader extracts one selected altitude-band sample per beam per time.
    The dataset then normalizes:

        [x_km, y_km, t_sec] -> [x_norm, y_norm, t_norm]
        log10_Ne            -> normalized log10_Ne

    dataset[0] returns the full coordinate-value set.
    """

    def __init__(
        self,
        h5_path: str | Path,
        h0_km: float,
        half_width_km: float,
        time_start_utc: str | float | int | None = None,
        time_end_utc: str | float | int | None = None,
        record_stride: int = 1,
        max_records: int | None = None,
        coord_cols: Sequence[str] = ("x_km", "y_km", "t_sec"),
        target_col: str = "log10_Ne",
        normalize_coords: bool = True,
        normalize_values: bool = True,
        drop_bad_rows: bool = True,
        verbose: bool = True,
        window_start_index: int | None = None,
        window_size_records: int | None = None,
    ):
        super().__init__()

        self.h5_path = Path(h5_path)
        self.h0_km = float(h0_km)
        self.half_width_km = float(half_width_km)
        self.time_start_utc = time_start_utc
        self.time_end_utc = time_end_utc
        self.record_stride = int(record_stride)
        self.max_records = max_records

        self.coord_cols = tuple(coord_cols)
        self.target_col = target_col
        self.normalize_coords = normalize_coords
        self.normalize_values = normalize_values

        self.window_start_index = window_start_index
        self.window_size_records = window_size_records

        # --------------------------------------------------------
        # Read HDF5 through the dedicated reader
        # --------------------------------------------------------
        df = read_amisr_h5_3d_altitude_band(
            h5_path=self.h5_path,
            h0_km=self.h0_km,
            half_width_km=self.half_width_km,
            time_start_utc=self.time_start_utc,
            time_end_utc=self.time_end_utc,
            record_stride=self.record_stride,
            max_records=self.max_records,
            verbose=verbose,
            window_start_index=self.window_start_index,
            window_size_records=self.window_size_records,
        )

        # --------------------------------------------------------
        # Validate target
        # --------------------------------------------------------
        if self.target_col not in df.columns:
            if self.target_col == "log10_Ne" and "Ne" in df.columns:
                if (df["Ne"] <= 0).any():
                    raise ValueError(
                        "Cannot create log10_Ne because some Ne values are <= 0."
                    )

                df["log10_Ne"] = np.log10(df["Ne"].astype(float))

            else:
                raise KeyError(f"Target column not found: {self.target_col}")

        required_cols = list(self.coord_cols) + [self.target_col]

        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Required column missing from dataframe: {col}")

        # --------------------------------------------------------
        # Clean rows
        # --------------------------------------------------------
        df = df.replace([np.inf, -np.inf], np.nan)

        if drop_bad_rows:
            before = len(df)
            df = df.dropna(subset=required_cols).copy()
            after = len(df)

            if after < before:
                print(f"RadarTimeH5Dataset: dropped {before - after} bad rows.")

        if len(df) == 0:
            raise ValueError("No valid rows left after cleaning the dataframe.")

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

        # --------------------------------------------------------
        # Torch tensors
        # --------------------------------------------------------
        self.coords = torch.from_numpy(coords_out.astype(np.float32))
        self.values = torch.from_numpy(values_out.astype(np.float32))

    def __len__(self) -> int:
        """
        Return 1 because this dataset represents one full coordinate-value field.
        """

        return 1

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Return the full coordinate-value set.
        """

        if idx != 0:
            raise IndexError("RadarTimeH5Dataset contains only one item: index 0.")

        return {
            "coords": self.coords,   # [N, 3]
            "values": self.values,   # [N, 1]
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
        """
        Convert predicted normalized values back to log10(Ne).
        """

        if not self.normalize_values:
            return np.asarray(values_norm, dtype=np.float64)

        return denormalize_minus1_plus1(
            values_norm,
            vmin=self.target_scaler["min"],
            vmax=self.target_scaler["max"],
        )

    def denormalize_coords(self, coords_norm: np.ndarray) -> np.ndarray:
        """
        Convert normalized coordinates back to physical coordinates.

        For this dataset:
            [x_norm, y_norm, t_norm] -> [x_km, y_km, t_sec]
        """

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
        """
        Create a dataframe containing the original HDF5-derived data plus predictions.
        """

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

        return out

    def get_time_indices(self) -> np.ndarray:
        """
        Return sorted unique AMISR time indices in the dataset.
        """

        if "time_index" not in self.df.columns:
            raise KeyError("Dataset dataframe has no time_index column.")

        return np.sort(self.df["time_index"].unique())

    def dataframe_for_time_index(self, time_index: int) -> pd.DataFrame:
        """
        Return dataframe rows for one AMISR time index.
        Useful for plotting x-y maps at a selected time.
        """

        if "time_index" not in self.df.columns:
            raise KeyError("Dataset dataframe has no time_index column.")

        return self.df[self.df["time_index"] == time_index].copy()

    def summary(self) -> None:
        """
        Print a short dataset summary.
        """

        print("RadarTimeH5Dataset")
        print(f"  h5_path:      {self.h5_path}")
        print(f"  h0_km:        {self.h0_km}")
        print(f"  half_width:   {self.half_width_km}")
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

        if "time_index" in self.df.columns:
            n_times = self.df["time_index"].nunique()
            print()
            print(f"Time records: {n_times}")

        if "unix_mid" in self.df.columns:
            print(
                f"Unix midpoint range: "
                f"{self.df['unix_mid'].min():.3f} to "
                f"{self.df['unix_mid'].max():.3f}"
            )

        if "t_sec" in self.df.columns:
            print(
                f"t_sec range: "
                f"{self.df['t_sec'].min():.3f} to "
                f"{self.df['t_sec'].max():.3f}"
            )