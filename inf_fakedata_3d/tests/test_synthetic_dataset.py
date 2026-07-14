# -*- coding: utf-8 -*-
"""
test_synthetic_dataset.py

Smoke test for SyntheticPlasmaTimeDataset.

Run from the synthetic project folder after generating data:

    python3 test_synthetic_dataset.py \
        --csv outputs/synthetic_smoke_test/synthetic_observations.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthetic_dataset import SyntheticPlasmaTimeDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="outputs/synthetic_smoke_test/synthetic_observations.csv",
    )
    parser.add_argument("--window_start_index", type=int, default=0)
    parser.add_argument("--window_size_records", type=int, default=31)
    args = parser.parse_args()

    dataset = SyntheticPlasmaTimeDataset(
        csv_path=args.csv,
        window_start_index=args.window_start_index,
        window_size_records=args.window_size_records,
        verbose=True,
    )

    dataset.summary()

    sample = dataset[0]
    coords = sample["coords"]
    values = sample["values"]

    assert coords.ndim == 2
    assert coords.shape[1] == 3
    assert values.ndim == 2
    assert values.shape[1] == 1
    assert coords.shape[0] == values.shape[0]
    assert np.isfinite(coords.numpy()).all()
    assert np.isfinite(values.numpy()).all()
    assert coords.min() >= -1.00001
    assert coords.max() <= 1.00001
    assert values.min() >= -1.00001
    assert values.max() <= 1.00001

    # Check denormalization shape and range.
    values_raw = dataset.denormalize_target(values.numpy())
    assert values_raw.shape == values.numpy().shape
    assert np.isfinite(values_raw).all()

    pred_df = dataset.make_prediction_dataframe(values.numpy())
    assert "pred_log10_Ne" in pred_df.columns
    assert "resid_log10_Ne" in pred_df.columns
    assert np.nanmax(np.abs(pred_df["resid_log10_Ne"].to_numpy())) < 1e-5

    print()
    print("Synthetic dataset smoke test passed")
    print(f"  csv:       {Path(args.csv)}")
    print(f"  samples:   {dataset.n_samples}")
    print(f"  n_times:   {dataset.df['time_index'].nunique()}")


if __name__ == "__main__":
    main()
