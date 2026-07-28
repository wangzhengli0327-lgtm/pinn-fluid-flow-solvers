from __future__ import annotations

import torch


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """dy/dx for scalar or vector y wrt x.

    Returns a tensor with the same shape as x. y must have shape (N, 1) or (N,).
    """
    if y.ndim == 1:
        y = y.view(-1, 1)
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


def column_grad(y: torch.Tensor, x: torch.Tensor, col: int) -> torch.Tensor:
    return grad(y, x)[:, col : col + 1]
