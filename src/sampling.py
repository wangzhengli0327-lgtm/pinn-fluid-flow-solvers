from __future__ import annotations

import torch


def sample_uniform(lower: list[float] | tuple[float, ...], upper: list[float] | tuple[float, ...], n: int, device: torch.device) -> torch.Tensor:
    lb = torch.tensor(lower, dtype=torch.float32, device=device).view(1, -1)
    ub = torch.tensor(upper, dtype=torch.float32, device=device).view(1, -1)
    return lb + (ub - lb) * torch.rand((n, len(lower)), dtype=torch.float32, device=device)


def sample_boundary_1d(x_value: float, t_lower: float, t_upper: float, n: int, device: torch.device) -> torch.Tensor:
    t = t_lower + (t_upper - t_lower) * torch.rand((n, 1), dtype=torch.float32, device=device)
    x = torch.full_like(t, float(x_value))
    return torch.cat([x, t], dim=1)


def sample_initial_1d(x_lower: float, x_upper: float, n: int, device: torch.device) -> torch.Tensor:
    x = x_lower + (x_upper - x_lower) * torch.rand((n, 1), dtype=torch.float32, device=device)
    t = torch.zeros_like(x)
    return torch.cat([x, t], dim=1)


def sample_cavity_boundaries(n_each: int, device: torch.device, eps: float = 1e-4) -> dict[str, torch.Tensor]:
    x_top = eps + (1.0 - 2.0 * eps) * torch.rand((n_each, 1), dtype=torch.float32, device=device)
    y_top = torch.ones_like(x_top)
    top = torch.cat([x_top, y_top], dim=1)

    x_bottom = torch.rand((n_each, 1), dtype=torch.float32, device=device)
    y_bottom = torch.zeros_like(x_bottom)
    bottom = torch.cat([x_bottom, y_bottom], dim=1)

    y_left = torch.rand((n_each, 1), dtype=torch.float32, device=device)
    x_left = torch.zeros_like(y_left)
    left = torch.cat([x_left, y_left], dim=1)

    y_right = torch.rand((n_each, 1), dtype=torch.float32, device=device)
    x_right = torch.ones_like(y_right)
    right = torch.cat([x_right, y_right], dim=1)

    return {"top": top, "bottom": bottom, "left": left, "right": right}
