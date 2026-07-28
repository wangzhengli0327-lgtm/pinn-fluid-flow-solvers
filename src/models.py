from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np
import torch
from torch import nn


def set_seed(seed: int = 2024) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    """Fully-connected network for PINNs.

    Uses SiLU activation, 6 hidden layers x 80 neurons, Xavier uniform init.
    Inputs are normalized from [lb, ub] to [-1, 1].
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_layers: int = 6,
        hidden_dim: int = 80,
        activation: str = "silu",
        lb: Sequence[float] | None = None,
        ub: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")

        act = self._activation(activation)
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), act]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), self._activation(activation)]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

        if lb is None:
            lb = [-1.0] * in_dim
        if ub is None:
            ub = [1.0] * in_dim
        self.register_buffer("lb", torch.tensor(lb, dtype=torch.float32).view(1, -1))
        self.register_buffer("ub", torch.tensor(ub, dtype=torch.float32).view(1, -1))

        self.apply(self._init_weights)

    @staticmethod
    def _activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "tanh":
            return nn.Tanh()
        if name == "silu":
            return nn.SiLU()
        if name == "gelu":
            return nn.GELU()
        raise ValueError(f"Unsupported activation: {name}")

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalize(x))
