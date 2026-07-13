# models.py

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn


ActivationName = Literal["relu", "tanh", "softplus", "sine"]


class Sine(nn.Module):
    """
    Sine activation used in SIREN-style networks.

    The factor w0 controls the frequency scale:
        y = sin(w0 * x)

    The original SIREN implementation uses w0 = 30 for all sine layers.
    """
    def __init__(self, w0: float = 30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


def get_activation(name: ActivationName, w0: float = 1.0) -> nn.Module:
    """
    Return an activation module by name.
    """
    name = name.lower()

    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "tanh":
        return nn.Tanh()
    if name == "softplus":
        return nn.Softplus()
    if name == "sine":
        return Sine(w0=w0)

    raise ValueError(f"Unsupported activation: {name}")


def init_linear(
    layer: nn.Linear,
    activation: ActivationName,
    is_first: bool = False,
    w0: float = 1.0,
) -> None:
    """
    Initialize one Linear layer.

    For sine, use SIREN-style initialization.
    For the others, use common standard initializations.
    """
    if not isinstance(layer, nn.Linear):
        raise TypeError("init_linear expects an nn.Linear layer.")

    with torch.no_grad():
        in_features = layer.in_features

        # if activation == "sine":
        #     if is_first:
        #         # First SIREN layer
        #         bound = 1.0 / in_features
        #     else:
        #         # Hidden SIREN layers
        #         bound = math.sqrt(6.0 / in_features) / w0

        #     layer.weight.uniform_(-bound, bound)
        #     if layer.bias is not None:
        #         layer.bias.uniform_(-bound, bound)

        if activation == "sine":
            if is_first:
                # First SIREN layer
                bound = 1.0 / in_features
            else:
                # Hidden/output SIREN layers
                bound = math.sqrt(6.0 / in_features) / w0

            layer.weight.uniform_(-bound, bound)

            # Match the official SIREN repo more closely:
            # initialize only weights here and leave nn.Linear biases at their default.

        elif activation == "relu":
            nn.init.kaiming_uniform_(layer.weight, a=0.0, nonlinearity="relu")
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1.0 / math.sqrt(fan_in)
                layer.bias.uniform_(-bound, bound)

        elif activation == "tanh":
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        elif activation == "softplus":
            # Softplus is smooth and often behaves reasonably with Xavier init
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        else:
            raise ValueError(f"Unsupported activation for initialization: {activation}")


class MLPINR(nn.Module):
    """
    Minimal implicit neural representation MLP.

    Maps coordinates -> signal value

    Example for grayscale image:
        input:  [N, 2]  -> (y, x)
        output: [N, 1]  -> intensity
    """
    def __init__(
        self,
        in_features: int = 2,
        out_features: int = 1,
        hidden_features: int = 256,
        hidden_layers: int = 3,
        activation: ActivationName = "relu",
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        outermost_linear: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0
        self.outermost_linear = outermost_linear

        layers = []

        # ---- First layer ----
        first_linear = nn.Linear(in_features, hidden_features)
        init_linear(
            first_linear,
            activation=activation,
            is_first=True,
            w0=first_omega_0,
        )
        layers.append(first_linear)

        if activation == "sine":
            layers.append(get_activation("sine", w0=first_omega_0))
        else:
            layers.append(get_activation(activation))

        # ---- Hidden layers ----
        for _ in range(hidden_layers):
            hidden_linear = nn.Linear(hidden_features, hidden_features)
            init_linear(
                hidden_linear,
                activation=activation,
                is_first=False,
                w0=hidden_omega_0,
            )
            layers.append(hidden_linear)

            if activation == "sine":
                layers.append(get_activation("sine", w0=hidden_omega_0))
            else:
                layers.append(get_activation(activation))

        # ---- Output layer ----
        final_linear = nn.Linear(hidden_features, out_features)

        # Final layer initialization:
        # keep it simple and controlled
        if activation == "sine":
            init_linear(
                final_linear,
                activation="sine",
                is_first=False,
                w0=hidden_omega_0,
            )
        else:
            init_linear(
                final_linear,
                activation=activation,
                is_first=False,
                w0=1.0,
            )

        layers.append(final_linear)

        # Optional output activation could go here later, but not for now.
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        coords can be:
            [N, in_features]
        or
            [B, N, in_features]

        nn.Linear works on the last dimension, so both are fine.
        """
        return self.net(coords)