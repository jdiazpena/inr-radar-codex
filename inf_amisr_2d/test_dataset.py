# -*- coding: utf-8 -*-

from pathlib import Path

# ============================================================
# USER SETTINGS
# ============================================================

CSV_PATH = Path(__file__).resolve().parent / "data" / "slice111748_h330_best_slice.csv"

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import torch

from datasets import RadarSliceDataset


# ============================================================
# LOAD DATASET
# ============================================================

dataset = RadarSliceDataset(csv_path=CSV_PATH)

dataset.summary()

sample = dataset[0]

coords = sample["coords"]
values = sample["values"]


# ============================================================
# BASIC SHAPE TESTS
# ============================================================

print()
print("Basic tensor checks:")
print("  coords type:", type(coords))
print("  values type:", type(values))
print("  coords shape:", coords.shape)
print("  values shape:", values.shape)
print("  coords dtype:", coords.dtype)
print("  values dtype:", values.dtype)

assert isinstance(coords, torch.Tensor)
assert isinstance(values, torch.Tensor)

assert coords.ndim == 2
assert values.ndim == 2

assert coords.shape[1] == 2
assert values.shape[1] == 1

assert coords.shape[0] == values.shape[0]

print("  shape checks passed")


# ============================================================
# FINITE VALUE TESTS
# ============================================================

assert torch.isfinite(coords).all()
assert torch.isfinite(values).all()

print("  finite checks passed")


# ============================================================
# NORMALIZATION TESTS
# ============================================================

coords_np = coords.numpy()
values_np = values.numpy()

print()
print("Normalized ranges:")
print("  x_norm min/max:", coords_np[:, 0].min(), coords_np[:, 0].max())
print("  y_norm min/max:", coords_np[:, 1].min(), coords_np[:, 1].max())
print("  value_norm min/max:", values_np[:, 0].min(), values_np[:, 0].max())

tol = 1e-5

assert coords_np.min() >= -1.0 - tol
assert coords_np.max() <=  1.0 + tol
assert values_np.min() >= -1.0 - tol
assert values_np.max() <=  1.0 + tol

print("  normalization checks passed")


# ============================================================
# DENORMALIZATION TESTS
# ============================================================

# Check that normalized target can be converted back to log10_Ne.
recovered_log10_ne = dataset.denormalize_target(values_np)

true_log10_ne = dataset.df["log10_Ne"].to_numpy()[:, None]

max_diff = np.max(np.abs(recovered_log10_ne - true_log10_ne))

print()
print("Denormalization check:")
print("  max abs difference:", max_diff)

assert max_diff < 1e-6

print("  target denormalization passed")


# ============================================================
# DATAFRAME OUTPUT TEST
# ============================================================

# Pretend the model predicted exactly the true normalized values.
pred_df = dataset.make_prediction_dataframe(values_np)

print()
print("Prediction dataframe test:")
print(pred_df.head())

assert "pred_log10_Ne" in pred_df.columns
assert "resid_log10_Ne" in pred_df.columns

max_resid = np.max(np.abs(pred_df["resid_log10_Ne"].to_numpy()))

print("  max fake residual:", max_resid)

assert max_resid < 1e-6

print()
print("ALL DATASET TESTS PASSED")