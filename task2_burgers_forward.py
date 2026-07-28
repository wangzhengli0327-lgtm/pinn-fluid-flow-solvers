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

from src.data_utils import batch_predict, load_burgers, to_tensor
from src.gradients import grad
from src.metrics import mse, relative_l2
from src.models import MLP, set_seed
from src.plot_utils import save_field, save_line_profiles, save_loss_curve
from src.sampling import sample_boundary_1d, sample_initial_1d, sample_uniform
from src.train_utils import History, default_device, ensure_dir, save_json


def config(mode: str) -> dict:
    if mode == "debug":
        return dict(n_r=512, n_0=64, n_b=64, epochs=50, print_every=10, lbfgs_steps=5)
    if mode == "light":
        return dict(n_r=6000, n_0=200, n_b=200, epochs=10000, print_every=200, lbfgs_steps=500)
    return dict(n_r=12000, n_0=300, n_b=300, epochs=20000, print_every=500, lbfgs_steps=2000)


def burgers_residual(model: MLP, xt: torch.Tensor, nu: float) -> torch.Tensor:
    xt = xt.clone().detach().requires_grad_(True)
    u = model(xt)
    gu = grad(u, xt)
    u_x = gu[:, 0:1]
    u_t = gu[:, 1:2]
    u_xx = grad(u_x, xt)[:, 0:1]
    return u_t + u * u_x - nu * u_xx


def periodic_bc_loss(model: MLP, t_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_l = torch.full_like(t_b, -1.0)
    x_r = torch.full_like(t_b, 1.0)
    xt_l = torch.cat([x_l, t_b], dim=1).requires_grad_(True)
    xt_r = torch.cat([x_r, t_b], dim=1).requires_grad_(True)
    u_l = model(xt_l)
    u_r = model(xt_r)
    ux_l = grad(u_l, xt_l)[:, 0:1]
    ux_r = grad(u_r, xt_r)[:, 0:1]
    return mse(u_l, u_r), mse(ux_l, ux_r)


def compute_loss(model, xt_r, xt_0, t_b, nu, args, xt_data=None, u_data=None):
    f = burgers_residual(model, xt_r, nu)
    loss_pde = torch.mean(f ** 2)
    u0_pred = model(xt_0)
    u0_true = -torch.sin(math.pi * xt_0[:, 0:1])
    loss_ic = mse(u0_pred, u0_true)
    loss_bc, loss_bc_grad = periodic_bc_loss(model, t_b)
    loss = loss_pde + args.w_ic * loss_ic + args.w_bc * loss_bc + args.w_grad_bc * loss_bc_grad
    if args.use_data and xt_data is not None and u_data is not None:
        loss = loss + args.w_data * mse(model(xt_data), u_data)
    return loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2: forward PINN for 1D Burgers equation.")
    parser.add_argument("--mode", choices=["debug", "light", "full"], default="debug")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data", type=str, default=str(ROOT / "data" / "Burgers_star_data.txt"))
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--use-data", action="store_true")
    parser.add_argument("--n-data", type=int, default=2000)
    parser.add_argument("--w-ic", type=float, default=5.0)
    parser.add_argument("--w-bc", type=float, default=5.0)
    parser.add_argument("--w-grad-bc", type=float, default=1.0)
    parser.add_argument("--w-data", type=float, default=1.0)
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
    fig_dir = ensure_dir(out_root / "figures" / "task2")
    ckpt_dir = ensure_dir(out_root / "checkpoints")
    log_dir = ensure_dir(out_root / "logs")

    nu = 0.01 / math.pi
    model = MLP(2, 1, hidden_layers=6, hidden_dim=80, activation="silu", lb=[-1, 0], ub=[1, 1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(opt, step_size=args.lr_step, gamma=args.lr_gamma)

    x_ref, t_ref, u_ref = load_burgers(args.data)
    xt_ref = to_tensor(np.hstack([x_ref, t_ref]), device)

    if args.use_data:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(x_ref.shape[0], min(args.n_data, x_ref.shape[0]), replace=False)
        xt_data = to_tensor(np.hstack([x_ref[idx], t_ref[idx]]), device)
        u_data = to_tensor(u_ref[idx], device)
    else:
        xt_data = None
        u_data = None

    hist = History()
    for ep in range(1, cfg["epochs"] + 1):
        xt_r = sample_uniform([-1.0, 0.0], [1.0, 1.0], cfg["n_r"], device)
        xt_0 = sample_initial_1d(-1.0, 1.0, cfg["n_0"], device)
        t_b = torch.rand((cfg["n_b"], 1), dtype=torch.float32, device=device)

        opt.zero_grad(set_to_none=True)
        loss = compute_loss(model, xt_r, xt_0, t_b, nu, args, xt_data, u_data)
        loss.backward()
        opt.step()
        scheduler.step()

        hist.add(total=loss.item())
        if ep % cfg["print_every"] == 0 or ep == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"[Task2][{ep:06d}/{cfg['epochs']}] lr={lr_now:.2e} loss={loss.item():.6e}")

    if cfg["lbfgs_steps"] > 0:
        print(f"[Task2] Adam done. Starting L-BFGS fine-tuning ({cfg['lbfgs_steps']} steps)...")
        opt_lbfgs = torch.optim.LBFGS(
            model.parameters(), max_iter=cfg["lbfgs_steps"],
            line_search_fn="strong_wolfe", tolerance_grad=1e-7,
        )
        def closure2():
            opt_lbfgs.zero_grad()
            xt_r2 = sample_uniform([-1.0, 0.0], [1.0, 1.0], cfg["n_r"], device)
            xt_02 = sample_initial_1d(-1.0, 1.0, cfg["n_0"], device)
            t_b2 = torch.rand((cfg["n_b"], 1), dtype=torch.float32, device=device)
            loss_val = compute_loss(model, xt_r2, xt_02, t_b2, nu, args, xt_data, u_data)
            loss_val.backward()
            return loss_val
        opt_lbfgs.step(closure2)
        print("[Task2] L-BFGS done.")

    pred = batch_predict(model, xt_ref).numpy()
    rel = relative_l2(pred, u_ref)
    print(f"[Task2] Relative L2 error = {rel:.6e}")

    save_field(x_ref, t_ref, u_ref, fig_dir / "reference.png", "Reference numerical solution", ylabel="t")
    save_field(x_ref, t_ref, pred, fig_dir / "prediction.png", "PINN prediction", ylabel="t")
    save_field(x_ref, t_ref, np.abs(pred - u_ref), fig_dir / "abs_error.png", "Absolute error", ylabel="t")
    save_line_profiles(x_ref, t_ref, pred, u_ref, [0.25, 0.50, 0.75, 1.00], fig_dir / "profiles.png")
    save_loss_curve(hist.data, fig_dir / "loss.png", "Task 2 loss")

    torch.save(model.state_dict(), ckpt_dir / "task2_burgers_forward.pt")
    hist.save(log_dir / "task2_history.json")
    save_json({"relative_l2": rel, "nu": nu, "config": cfg, "mode": args.mode, "use_data": args.use_data}, log_dir / "task2_metrics.json")


if __name__ == "__main__":
    main()
