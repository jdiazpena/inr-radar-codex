# -*- coding: utf-8 -*-

"""
test_model.py

Small test for:
    datasets.py -> RadarSliceDataset
    models.py   -> MLPINR

This does NOT train the model.
It only checks that the model accepts coords [N, 2]
and returns predictions [N, 1].
"""

import torch
import torch.nn.functional as F

from datasets import RadarSliceDataset
from models import MLPINR


# ============================================================
# USER SETTINGS
# ============================================================

CSV_PATH = "data/slice111748_h330_best_slice.csv"

HIDDEN_FEATURES = 128
HIDDEN_LAYERS = 3
FIRST_OMEGA_0 = 30.0
HIDDEN_OMEGA_0 = 30.0


# ============================================================
# LOAD DATASET
# ============================================================

dataset = RadarSliceDataset(csv_path=CSV_PATH)
dataset.summary()

sample = dataset[0]

coords = sample["coords"]
values = sample["values"]

print()
print("Input tensors:")
print("  coords:", coords.shape, coords.dtype)
print("  values:", values.shape, values.dtype)


# ============================================================
# BUILD MODEL
# ============================================================

model = MLPINR(
    in_features=dataset.in_features,
    out_features=dataset.out_features,
    hidden_features=HIDDEN_FEATURES,
    hidden_layers=HIDDEN_LAYERS,
    activation="sine",
    first_omega_0=FIRST_OMEGA_0,
    hidden_omega_0=HIDDEN_OMEGA_0,
    outermost_linear=True,
)

print()
print("Model:")
print(model)


# ============================================================
# FORWARD PASS TEST
# ============================================================

pred = model(coords)

print()
print("Forward pass:")
print("  pred shape:", pred.shape)
print("  expected:  ", values.shape)

assert pred.shape == values.shape
assert torch.isfinite(pred).all()

print("  forward pass OK")


# ============================================================
# LOSS TEST
# ============================================================

loss = F.mse_loss(pred, values)

print()
print("Loss:")
print("  mse:", loss.item())

assert torch.isfinite(loss)

print("  loss OK")


# ============================================================
# BACKWARD PASS TEST
# ============================================================

loss.backward()

bad_grads = []

for name, param in model.named_parameters():
    if param.grad is None:
        bad_grads.append(name)
    elif not torch.isfinite(param.grad).all():
        bad_grads.append(name)

if bad_grads:
    raise RuntimeError(f"Bad gradients in: {bad_grads}")

print()
print("Backward pass:")
print("  gradients OK")

print()
print("ALL MODEL TESTS PASSED")