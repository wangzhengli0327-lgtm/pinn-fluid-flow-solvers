from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_txt_three_columns(path: str | Path, skip_header: int = 1) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.loadtxt(path, skiprows=skip_header, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Expected at least three columns in {path}, got shape {data.shape}")
    return data[:, :3]


def load_burgers(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = load_txt_three_columns(path)
    return data[:, 0:1], data[:, 1:2], data[:, 2:3]


def load_cavity_speed(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = load_txt_three_columns(path)
    return data[:, 0:1], data[:, 1:2], data[:, 2:3]


def to_tensor(array: np.ndarray, device: torch.device, requires_grad: bool = False) -> torch.Tensor:
    t = torch.tensor(array, dtype=torch.float32, device=device)
    t.requires_grad_(requires_grad)
    return t


def random_choice(n_total: int, n_sample: int, seed: int = 2024) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_sample = min(n_sample, n_total)
    return rng.choice(n_total, size=n_sample, replace=False)


def batch_predict(model: torch.nn.Module, xy: torch.Tensor, batch_size: int = 65536) -> torch.Tensor:
    outputs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, xy.shape[0], batch_size):
            outputs.append(model(xy[i : i + batch_size]).detach().cpu())
    return torch.cat(outputs, dim=0)
