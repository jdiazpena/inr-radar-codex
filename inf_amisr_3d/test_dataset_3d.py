# -*- coding: utf-8 -*-
"""
test_dataset_3d.py

Small test for the 3D AMISR HDF5 dataset.

Run from inside inf_amisr_3d:

    python3 test_dataset_3d.py

This does not train anything.
It only checks that the HDF5 reader + dataset produce:

    coords: [N, 3] = x_norm, y_norm, t_norm
    values: [N, 1] = normalized log10(Ne)
"""

import numpy as np
import torch

from datasets import RadarTimeH5Dataset


# ============================================================
# USER SETTINGS
# ============================================================

H5_PATH = "../data/20120122.001_lp_5min.h5"

H0_KM = 330.0
HALF_WIDTH_KM = 15.0

# Keep this small for the first test.
# Once it works, set MAX_RECORDS = None to use all selected times.
MAX_RECORDS = None

RECORD_STRIDE = 1

TIME_START_UTC = None
TIME_END_UTC = None

WINDOW_START_INDEX = 100
WINDOW_SIZE_RECORDS = 11


# ============================================================
# TEST
# ============================================================

def main():
    dataset = RadarTimeH5Dataset(
        h5_path=H5_PATH,
        h0_km=H0_KM,
        half_width_km=HALF_WIDTH_KM,
        time_start_utc=TIME_START_UTC,
        time_end_utc=TIME_END_UTC,
        record_stride=RECORD_STRIDE,
        max_records=MAX_RECORDS,
        verbose=True,
        window_start_index=WINDOW_START_INDEX,
        window_size_records=WINDOW_SIZE_RECORDS,
    )

    print()
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    dataset.summary()

    sample = dataset[0]

    coords = sample["coords"]
    values = sample["values"]

    print()
    print("=" * 60)
    print("TENSOR CHECKS")
    print("=" * 60)

    print("coords type:", type(coords))
    print("values type:", type(values))
    print("coords shape:", coords.shape)
    print("values shape:", values.shape)
    print("coords dtype:", coords.dtype)
    print("values dtype:", values.dtype)

    assert isinstance(coords, torch.Tensor)
    assert isinstance(values, torch.Tensor)

    assert coords.ndim == 2
    assert values.ndim == 2

    assert coords.shape[1] == 3
    assert values.shape[1] == 1
    assert coords.shape[0] == values.shape[0]

    assert torch.isfinite(coords).all()
    assert torch.isfinite(values).all()

    print("shape checks passed")
    print("finite checks passed")

    print()
    print("=" * 60)
    print("NORMALIZED RANGE CHECKS")
    print("=" * 60)

    coord_names = dataset.coord_cols

    for i, name in enumerate(coord_names):
        cmin = coords[:, i].min().item()
        cmax = coords[:, i].max().item()
        print(f"{name}_norm min/max: {cmin:.6f}, {cmax:.6f}")

        assert cmin >= -1.0001
        assert cmax <= 1.0001

    vmin = values.min().item()
    vmax = values.max().item()

    print(f"value_norm min/max: {vmin:.6f}, {vmax:.6f}")

    assert vmin >= -1.0001
    assert vmax <= 1.0001

    print("normalization checks passed")

    print()
    print("=" * 60)
    print("DATAFRAME CHECKS")
    print("=" * 60)

    df = dataset.df

    print("dataframe rows:", len(df))
    print("columns:")
    print(list(df.columns))

    print()
    print("first rows:")
    print(df.head())

    print()
    print("rows per time_index:")
    print(df.groupby("time_index").size())

    print()
    print("unique time records:", df["time_index"].nunique())
    print("unique beams:", df["beam_index"].nunique())

    print()
    print("x_km range:", df["x_km"].min(), df["x_km"].max())
    print("y_km range:", df["y_km"].min(), df["y_km"].max())
    print("t_sec range:", df["t_sec"].min(), df["t_sec"].max())
    print("altitude_km range:", df["altitude_km"].min(), df["altitude_km"].max())
    print("log10_Ne range:", df["log10_Ne"].min(), df["log10_Ne"].max())

    print()
    print("=" * 60)
    print("DENORMALIZATION CHECK")
    print("=" * 60)

    coords_raw_recovered = dataset.denormalize_coords(coords.numpy())
    values_raw_recovered = dataset.denormalize_target(values.numpy())

    coords_raw_original = dataset.coords_raw
    values_raw_original = dataset.values_raw

    for i, name in enumerate(dataset.coord_cols):
        diff = np.nanmax(
            np.abs(coords_raw_recovered[:, i] - coords_raw_original[:, i])
        )
        print(f"max {name} denorm difference:", diff)

        if name in ["x_km", "y_km"]:
            assert diff < 1e-4      # 0.1 meter
        elif name == "t_sec":
            assert diff < 1e-2      # 0.01 seconds
        else:
            assert diff < 1e-4

    value_diff = np.nanmax(np.abs(values_raw_recovered - values_raw_original))

    print("max value denorm difference:", value_diff)

    assert value_diff < 1e-6

    print("denormalization checks passed")


if __name__ == "__main__":
    main()