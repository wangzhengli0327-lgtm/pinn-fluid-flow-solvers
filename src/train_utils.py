from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class History:
    def __init__(self) -> None:
        self.data: dict[str, list[float]] = {}

    def add(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            self.data.setdefault(k, []).append(float(v))

    def latest(self) -> str:
        return ", ".join(f"{k}={v[-1]:.3e}" for k, v in self.data.items() if v)

    def save(self, path: str | Path) -> None:
        save_json(self.data, path)
