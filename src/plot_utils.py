from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_loss_curve(history: dict[str, list[float]], path: str | Path, title: str = "Training loss") -> None:
    path = _ensure_parent(path)
    plt.figure(figsize=(7, 5))
    for k, v in history.items():
        if len(v) > 0:
            plt.semilogy(v, label=k)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _grid_from_scatter(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)
    ux = np.unique(np.round(x, decimals=10))
    uy = np.unique(np.round(y, decimals=10))
    if ux.size * uy.size == x.size:
        order = np.lexsort((x, y))
        zz = z[order].reshape(uy.size, ux.size)
        return ux, uy, zz
    return None


def save_field(x: np.ndarray, y: np.ndarray, z: np.ndarray, path: str | Path, title: str, xlabel: str = "x", ylabel: str = "t") -> None:
    path = _ensure_parent(path)
    grid = _grid_from_scatter(x, y, z)
    plt.figure(figsize=(7, 5))
    if grid is not None:
        ux, uy, zz = grid
        plt.imshow(zz, extent=[ux.min(), ux.max(), uy.min(), uy.max()], origin="lower", aspect="auto")
    else:
        plt.tricontourf(x.reshape(-1), y.reshape(-1), z.reshape(-1), levels=80)
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_line_profiles(x: np.ndarray, t: np.ndarray, pred: np.ndarray, ref: np.ndarray, times: list[float], path: str | Path) -> None:
    path = _ensure_parent(path)
    x = x.reshape(-1)
    t = t.reshape(-1)
    pred = pred.reshape(-1)
    ref = ref.reshape(-1)
    plt.figure(figsize=(8, 5))
    for tt in times:
        unique_t = np.unique(t)
        nearest = unique_t[np.argmin(np.abs(unique_t - tt))]
        mask = np.isclose(t, nearest)
        idx = np.argsort(x[mask])
        plt.plot(x[mask][idx], ref[mask][idx], linestyle="--", label=f"ref t={nearest:.2f}")
        plt.plot(x[mask][idx], pred[mask][idx], label=f"PINN t={nearest:.2f}")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("Line profiles")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_vector_field(x: np.ndarray, y: np.ndarray, u: np.ndarray, v: np.ndarray, path: str | Path, title: str = "Velocity field", step: int = 20) -> None:
    path = _ensure_parent(path)
    x = x.reshape(-1)
    y = y.reshape(-1)
    u = u.reshape(-1)
    v = v.reshape(-1)
    grid_u = _grid_from_scatter(x, y, u)
    grid_v = _grid_from_scatter(x, y, v)
    plt.figure(figsize=(6, 6))
    if grid_u is not None and grid_v is not None:
        ux, uy, U = grid_u
        _, _, V = grid_v
        X, Y = np.meshgrid(ux, uy)
        plt.quiver(X[::step, ::step], Y[::step, ::step], U[::step, ::step], V[::step, ::step])
    else:
        idx = np.arange(0, x.size, max(1, x.size // 2000))
        plt.quiver(x[idx], y[idx], u[idx], v[idx])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_streamlines(x: np.ndarray, y: np.ndarray, u: np.ndarray, v: np.ndarray, path: str | Path, title: str = "Streamlines") -> None:
    path = _ensure_parent(path)
    x = x.reshape(-1)
    y = y.reshape(-1)
    u = u.reshape(-1)
    v = v.reshape(-1)
    grid_u = _grid_from_scatter(x, y, u)
    grid_v = _grid_from_scatter(x, y, v)
    plt.figure(figsize=(6, 6))
    if grid_u is not None and grid_v is not None:
        ux, uy, U = grid_u
        _, _, V = grid_v
        X, Y = np.meshgrid(ux, uy)
        speed = np.sqrt(U ** 2 + V ** 2)
        plt.contourf(X, Y, speed, levels=50)
        plt.colorbar()
        plt.streamplot(ux, uy, U, V, density=1.5)
    else:
        idx = np.arange(0, x.size, max(1, x.size // 3000))
        speed = np.sqrt(u[idx] ** 2 + v[idx] ** 2)
        plt.tricontourf(x[idx], y[idx], speed, levels=50)
        plt.colorbar()
        plt.quiver(x[idx], y[idx], u[idx], v[idx])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
