from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_utils import batch_predict, to_tensor
from src.gradients import grad
from src.metrics import mse, relative_l2
from src.models import MLP, set_seed
from src.plot_utils import save_field, save_line_profiles, save_loss_curve
from src.sampling import sample_boundary_1d, sample_initial_1d, sample_uniform
from src.train_utils import History, default_device, ensure_dir, save_json


def config(mode: str) -> dict:
    if mode == "debug":
        return dict(n_r=512, n_0=64, n_b=64, epochs=50, print_every=10, n_test_x=101, n_test_t=101, lbfgs_steps=5)
    if mode == "light":
        return dict(n_r=6000, n_0=200, n_b=200, epochs=10000, print_every=200, n_test_x=128, n_test_t=101, lbfgs_steps=500)
    return dict(n_r=12000, n_0=300, n_b=300, epochs=20000, print_every=500, n_test_x=256, n_test_t=201, lbfgs_steps=2000)


def stable_exp(s: torch.Tensor, max_value: float = 60.0) -> torch.Tensor:
    return torch.exp(torch.clamp(s, max=max_value))


def initial_condition(x: torch.Tensor, re: float) -> torch.Tensor:
    exponent = re * x ** 2 / 4.0 - re / 16.0
    return x / (1.0 + stable_exp(exponent))


def exact_solution_torch(xt: torch.Tensor, re: float) -> torch.Tensor:
    x = xt[:, 0:1]
    t = xt[:, 1:2]
    exponent = 0.5 * torch.log(t + 1.0) - re / 16.0 + re * x ** 2 / (4.0 * t + 4.0)
    numerator = x / (t + 1.0)
    return numerator / (1.0 + stable_exp(exponent))


def burgers_residual(model: MLP, xt: torch.Tensor, nu: float) -> torch.Tensor:
    xt = xt.clone().detach().requires_grad_(True)
    u = model(xt)
    gu = grad(u, xt)
    u_x = gu[:, 0:1]
    u_t = gu[:, 1:2]
    u_xx = grad(u_x, xt)[:, 0:1]
    return u_t + u * u_x - nu * u_xx


def compute_loss(model, xt_r, xt_0, xt_l, xt_rbc, nu, args):
    """Compute total loss for a given batch of collocation/BC/IC points."""
    f = burgers_residual(model, xt_r, nu)
    loss_pde = torch.mean(f ** 2)
    loss_ic = mse(model(xt_0), initial_condition(xt_0[:, 0:1], args.re))
    loss_bc = torch.mean(model(xt_l) ** 2) + torch.mean(model(xt_rbc) ** 2)
    return loss_pde + args.w_ic * loss_ic + args.w_bc * loss_bc


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1: forward PINN for Burgers equation with exact solution.")
    parser.add_argument("--mode", choices=["debug", "light", "full"], default="debug")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--re", type=float, default=500.0)
    parser.add_argument("--w-ic", type=float, default=5.0)
    parser.add_argument("--w-bc", type=float, default=5.0)
    parser.add_argument("--lr-step", type=int, default=5000)
    parser.add_argument("--lr-gamma", type=float, default=0.9)
    parser.add_argument("--lbfgs-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = config(args.mode)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.lbfgs_steps is not None:
        cfg["lbfgs_steps"] = args.lbfgs_steps

    set_seed(args.seed)
    device = default_device()
    out_root = ROOT
    fig_dir = ensure_dir(out_root / "figures" / "task1")
    ckpt_dir = ensure_dir(out_root / "checkpoints")
    log_dir = ensure_dir(out_root / "logs")

    nu = 1.0 / args.re
    model = MLP(2, 1, hidden_layers=6, hidden_dim=80, activation="silu", lb=[0, 0], ub=[1, 1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(opt, step_size=args.lr_step, gamma=args.lr_gamma)
    hist = History()

    for ep in range(1, cfg["epochs"] + 1):
        xt_r = sample_uniform([0.0, 0.0], [1.0, 1.0], cfg["n_r"], device)
        xt_0 = sample_initial_1d(0.0, 1.0, cfg["n_0"], device)
        xt_l = sample_boundary_1d(0.0, 0.0, 1.0, cfg["n_b"], device)
        xt_rbc = sample_boundary_1d(1.0, 0.0, 1.0, cfg["n_b"], device)

        opt.zero_grad(set_to_none=True)
        loss = compute_loss(model, xt_r, xt_0, xt_l, xt_rbc, nu, args)
        loss.backward()
        opt.step()
        scheduler.step()

        hist.add(total=loss.item())
        if ep % cfg["print_every"] == 0 or ep == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"[Task1][{ep:06d}/{cfg['epochs']}] lr={lr_now:.2e} loss={loss.item():.6e}")

    # --- L-BFGS fine-tuning ---
    if cfg["lbfgs_steps"] > 0:
        print(f"[Task1] Adam done. Starting L-BFGS fine-tuning ({cfg['lbfgs_steps']} steps)...")
        opt_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            max_iter=cfg["lbfgs_steps"],
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
        )
        def closure():
            opt_lbfgs.zero_grad()
            xt_r = sample_uniform([0.0, 0.0], [1.0, 1.0], cfg["n_r"], device)
            xt_0 = sample_initial_1d(0.0, 1.0, cfg["n_0"], device)
            xt_l = sample_boundary_1d(0.0, 0.0, 1.0, cfg["n_b"], device)
            xt_rbc = sample_boundary_1d(1.0, 0.0, 1.0, cfg["n_b"], device)
            loss_val = compute_loss(model, xt_r, xt_0, xt_l, xt_rbc, nu, args)
            loss_val.backward()
            return loss_val
        opt_lbfgs.step(closure)
        print("[Task1] L-BFGS done.")

    x = np.linspace(0.0, 1.0, cfg["n_test_x"])
    t = np.linspace(0.0, 1.0, cfg["n_test_t"])
    X, T = np.meshgrid(x, t)
    xt_np = np.hstack([X.reshape(-1, 1), T.reshape(-1, 1)])
    xt = to_tensor(xt_np, device)
    pred = batch_predict(model, xt).numpy()
    with torch.no_grad():
        exact = exact_solution_torch(xt, args.re).detach().cpu().numpy()
    rel = relative_l2(pred, exact)
    print(f"[Task1] Relative L2 error = {rel:.6e}")

    save_field(xt_np[:, 0:1], xt_np[:, 1:2], exact, fig_dir / "exact.png", "Exact solution", ylabel="t")
    save_field(xt_np[:, 0:1], xt_np[:, 1:2], pred, fig_dir / "prediction.png", "PINN prediction", ylabel="t")
    save_field(xt_np[:, 0:1], xt_np[:, 1:2], np.abs(pred - exact), fig_dir / "abs_error.png", "Absolute error", ylabel="t")
    save_line_profiles(xt_np[:, 0:1], xt_np[:, 1:2], pred, exact, [0.25, 0.50, 0.75, 1.00], fig_dir / "profiles.png")
    save_loss_curve(hist.data, fig_dir / "loss.png", "Task 1 loss")

    torch.save(model.state_dict(), ckpt_dir / "task1_burgers_exact.pt")
    hist.save(log_dir / "task1_history.json")
    save_json({"relative_l2": rel, "Re": args.re, "nu": nu, "config": cfg, "mode": args.mode}, log_dir / "task1_metrics.json")


if __name__ == "__main__":
    main()
