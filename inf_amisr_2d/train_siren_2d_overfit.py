# -*- coding: utf-8 -*-

# ============================================================
# USER SETTINGS
# ============================================================

CSV_FILE = r"D:\JMDP\GDrive\PostDoc UAI\Python\slice_111748_h330_best_slice.csv"

SAVE_FIGURES = True
SHOW_FIGURES = True

NUM_STEPS = 5000
LEARNING_RATE = 1e-4

HIDDEN_FEATURES = 256
HIDDEN_LAYERS = 3

FIRST_OMEGA_0 = 30.0
HIDDEN_OMEGA_0 = 30.0

PRINT_EVERY = 250

# Use a real external plot window.
# If TkAgg fails, try "QtAgg".
MATPLOTLIB_BACKEND = "TkAgg"


# ============================================================
# IMPORTS
# ============================================================

import math
from pathlib import Path

import matplotlib
matplotlib.use(MATPLOTLIB_BACKEND)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# SIREN MODEL
# ============================================================

class Sine(nn.Module):
    def __init__(self, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.omega_0 * x)


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega_0=30.0,
    ):
        super().__init__()

        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0

            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class Siren(nn.Module):
    def __init__(
        self,
        in_features=2,
        out_features=1,
        hidden_features=128,
        hidden_layers=3,
        first_omega_0=30.0,
        hidden_omega_0=30.0,
    ):
        super().__init__()

        layers = []

        layers.append(
            SineLayer(
                in_features,
                hidden_features,
                is_first=True,
                omega_0=first_omega_0,
            )
        )

        for _ in range(hidden_layers):
            layers.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        final_linear = nn.Linear(hidden_features, out_features)

        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_features) / hidden_omega_0
            final_linear.weight.uniform_(-bound, bound)

        layers.append(final_linear)

        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        return self.net(coords)


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize_minus1_plus1(values):
    values = np.asarray(values, dtype=np.float64)

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Non-finite min/max during normalization.")

    if vmax <= vmin:
        raise ValueError(f"Cannot normalize because vmax <= vmin: {vmax} <= {vmin}")

    norm = 2.0 * (values - vmin) / (vmax - vmin) - 1.0

    return norm, vmin, vmax


def denormalize_minus1_plus1(values_norm, vmin, vmax):
    return 0.5 * (values_norm + 1.0) * (vmax - vmin) + vmin


# ============================================================
# LOAD DATA
# ============================================================

csv_path = Path(CSV_FILE)

if not csv_path.exists():
    raise FileNotFoundError(f"Could not find CSV file: {csv_path}")

df = pd.read_csv(csv_path)

required = ["x_km", "y_km", "log10_Ne", "beamcode"]

for col in required:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["x_km", "y_km", "log10_Ne"]).copy()

print("Loaded slice:")
print(f"  file: {csv_path}")
print(f"  rows: {len(df)}")
print(f"  unique beams: {df['beamcode'].nunique()}")
print()
print(df[["x_km", "y_km", "log10_Ne"]].describe())


# ============================================================
# BUILD INR ARRAYS
# ============================================================

x_norm, x_min, x_max = normalize_minus1_plus1(df["x_km"].to_numpy())
y_norm, y_min, y_max = normalize_minus1_plus1(df["y_km"].to_numpy())
target_norm, target_min, target_max = normalize_minus1_plus1(df["log10_Ne"].to_numpy())

coords_np = np.stack([x_norm, y_norm], axis=1).astype(np.float32)
values_np = target_norm[:, None].astype(np.float32)

print()
print("INR arrays:")
print("  coords:", coords_np.shape)
print("  values:", values_np.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("  device:", device)

coords = torch.from_numpy(coords_np).to(device)
values = torch.from_numpy(values_np).to(device)


# ============================================================
# TRAIN
# ============================================================

model = Siren(
    in_features=2,
    out_features=1,
    hidden_features=HIDDEN_FEATURES,
    hidden_layers=HIDDEN_LAYERS,
    first_omega_0=FIRST_OMEGA_0,
    hidden_omega_0=HIDDEN_OMEGA_0,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

history = []

for step in range(1, NUM_STEPS + 1):
    pred = model(coords)
    loss = loss_fn(pred, values)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step == 1 or step % PRINT_EVERY == 0 or step == NUM_STEPS:
        loss_value = float(loss.item())
        history.append({"step": step, "mse_norm": loss_value})
        print(f"step {step:6d} | mse_norm {loss_value:.8e}")


# ============================================================
# EVALUATE AT MEASURED POINTS
# ============================================================

model.eval()

with torch.no_grad():
    pred_norm = model(coords).detach().cpu().numpy().[:, 0]