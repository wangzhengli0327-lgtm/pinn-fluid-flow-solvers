from __future__ import annotations

import numpy as np
import torch


def relative_l2(pred: np.ndarray | torch.Tensor, ref: np.ndarray | torch.Tensor, eps: float = 1e-12) -> float:
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(ref, torch.Tensor):
        ref = ref.detach().cpu().numpy()
    return float(np.linalg.norm(pred.reshape(-1) - ref.reshape(-1)) / (np.linalg.norm(ref.reshape(-1)) + eps))


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)
