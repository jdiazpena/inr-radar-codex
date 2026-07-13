# -*- coding: utf-8 -*-
"""
datasets.py

Dataset utilities for radar INR experiments.

Current purpose:
    Load one prepared AMISR 2D slice CSV and convert it into the same
    coordinate-value structure used by the image INR experiments.

Input CSV:
    This file is produced by the AMISR slice preparation step.

    Expected columns:
        x_km
        y_km
        log10_Ne

    Useful optional columns:
        beamcode
        beam_index
        range_index
        altitude_km
        z_km
        Ne
        dNe

Output used by train_radar_2d.py:
    sample = dataset[0]

    sample["coords"] -> torch.Tensor [N, 2]
        normalized x/y coordinates in [-1, 1]

    sample["values"] -> torch.Tensor [N, 1]
        normalized log10(Ne) values in [-1, 1]

Important:
    This file does NOT read raw AMISR HDF5.
    This file does NOT choose the altitude band.
    This file does NOT train the model.

    Raw HDF5 -> selected CSV happens before this.
    selected CSV -> coords/values happens here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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

    Returns:
        normalized values
        vmin
        vmax

    The min/max are returned so train.py can later convert predictions
    back to physical units.
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

    Formula:
        y = 0.5 * (y_norm + 1) * (ymax - ymin) + ymin
    """
    values_norm = np.asarray(values_norm, dtype=np.float64)
    return 0.5 * (values_norm + 1.0) * (vmax - vmin) + vmin


class RadarSliceDataset(Dataset):
    """
    Dataset for one prepared 2D AMISR slice.

    This follows the same idea as ImageINRDataset:
        one dataset item contains the full coordinate-value set.

    For the first radar experiment:
        coords = [x_norm, y_norm]
        values = normalized log10(Ne)

    The CSV path is passed by train_radar_2d.py.
    This keeps the dataset reusable for different slices.
    """

    def __init__(
        self,
        csv_path: str | Path,
        coord_cols: Sequence[str] = ("x_km", "y_km"),
        target_col: str = "log10_Ne",
        normalize_coords: bool = True,
        normalize_values: bool = True,
        drop_bad_rows: bool = True,
    ):
        super().__init__()

        self.csv_path = Path(csv_path)
        self.coord_cols = tuple(coord_cols)
        self.target_col = target_col
        self.normalize_coords = normalize_coords
        self.normalize_values = normalize_values

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        # Load the prepared slice CSV.
        # This CSV should already contain one selected altitude band
        # and one selected time.
        df = pd.read_csv(self.csv_path)

        # If log10_Ne is missing but Ne exists, create log10_Ne here.
        # This is only a convenience. The preparation step should normally
        # already create log10_Ne.
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
                raise KeyError(f"Required column missing from CSV: {col}")

        # Clean obvious invalid rows.
        # The preparation code should already do most of this, but keeping this
        # here prevents silent NaN propagation into PyTorch.
        df = df.replace([np.inf, -np.inf], np.nan)

        if drop_bad_rows:
            before = len(df)
            df = df.dropna(subset=required_cols).copy()
            after = len(df)

            if after < before:
                print(f"RadarSliceDataset: dropped {before - after} bad rows.")

        if len(df) == 0:
            raise ValueError("No valid rows left after cleaning the CSV.")

        # Store cleaned dataframe.
        # train.py can use this later to save predictions with beamcode,
        # altitude, etc.
        self.df = df.reset_index(drop=True)

        # Raw physical coordinates and target.
        coords_raw = self.df.loc[:, self.coord_cols].to_numpy(dtype=np.float64)
        values_raw = self.df.loc[:, [self.target_col]].to_numpy(dtype=np.float64)

        # Normalize coordinates dimension by dimension.
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

        # Normalize target values.
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

        # Torch tensors used by train.py.
        self.coords = torch.from_numpy(coords_out.astype(np.float32))
        self.values = torch.from_numpy(values_out.astype(np.float32))

        # Keep raw numpy arrays for metrics and debugging.
        self.coords_raw = coords_raw.astype(np.float64)
        self.values_raw = values_raw.astype(np.float64)

    def __len__(self) -> int:
        """
        Return 1 because this dataset represents one full coordinate-value field.

        This matches the old ImageINRDataset pattern:
            dataset[0] returns all points.
        """
        return 1

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Return the full coordinate-value set.

        train.py will do:
            sample = dataset[0]
            coords = sample["coords"]
            values = sample["values"]
        """
        if idx != 0:
            raise IndexError("RadarSliceDataset contains only one item: index 0.")

        return {
            "coords": self.coords,   # [N, 2]
            "values": self.values,   # [N, 1]
        }

    @property
    def n_samples(self) -> int:
        """Number of coordinate-value samples."""
        return self.coords.shape[0]

    @property
    def in_features(self) -> int:
        """Coordinate dimension. For the first 2D slice this is 2."""
        return self.coords.shape[1]

    @property
    def out_features(self) -> int:
        """Target dimension. For log10(Ne) this is 1."""
        return self.values.shape[1]

    def denormalize_target(self, values_norm: np.ndarray) -> np.ndarray:
        """
        Convert predicted normalized target values back to physical target scale.

        For the current dataset:
            normalized prediction -> log10(Ne)
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

        For the current dataset:
            [x_norm, y_norm] -> [x_km, y_km]
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
        Create a dataframe containing the original slice plus model predictions.

        Input:
            pred_values_norm:
                model output on normalized scale, shape [N] or [N, 1]

        Output:
            dataframe with:
                original columns
                pred_log10_Ne
                resid_log10_Ne

        This will be useful in train_radar_2d.py after evaluating the model.
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

    def summary(self) -> None:
        """
        Print a short dataset summary.

        Useful before writing train_radar_2d.py.
        """
        print("RadarSliceDataset")
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