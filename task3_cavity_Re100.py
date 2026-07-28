from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_utils import batch_predict, load_cavity_speed, to_tensor
from src.gradients import grad
from src.metrics import mse, relative_l2
from src.models import MLP, set_seed
from src.plot_utils import save_field, save_loss_curve, save_streamlines, save_vector_field
from src.sampling import sample_cavity_boundaries, sample_uniform
from src.train_utils import History, default_device, ensure_dir, save_json


def config(mode: str) -> dict:
    if mode == "debug":
        return dict(n_r=16, n_b_each=16, epochs=20, print_every=5, eval_grid=31, n_speed_data=128, lbfgs_steps=5)
    if mode == "light":
        return dict(n_r=1500, n_b_each=150, epochs=10000, print_every=200, eval_grid=71, n_speed_data=2000, lbfgs_steps=500)
    return dict(n_r=3000, n_b_each=300, epochs=20000, print_every=500, eval_grid=101, n_speed_data=5000, lbfgs_steps=2000)


def cavity_residual(model: MLP, xy: torch.Tensor, re: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xy = xy.clone().detach().requires_grad_(True)
    out = model(xy)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    gu, gv, gp = grad(u, xy), grad(v, xy), grad(p, xy)
    u_x, u_y = gu[:, 0:1], gu[:, 1:2]
    v_x, v_y = gv[:, 0:1], gv[:, 1:2]
    u_xx, u_yy = grad(u_x, xy)[:, 0:1], grad(u_y, xy)[:, 1:2]
    v_xx, v_yy = grad(v_x, xy)[:, 0:1], grad(v_y, xy)[:, 1:2]
    return (
        u_x + v_y,
        u * u_x + v * u_y + gp[:, 0:1] - (u_xx + u_yy) / re,
        u * v_x + v * v_y + gp[:, 1:2] - (v_xx + v_yy) / re,
    )


def boundary_loss_fn(model: MLP, b: dict[str, torch.Tensor]) -> torch.Tensor:
    top = model(b["top"])
    bottom, left, right = model(b["bottom"]), model(b["left"]), model(b["right"])
    loss_top = torch.mean((top[:, 0:1] - 1.0) ** 2) + torch.mean(top[:, 1:2] ** 2)
    loss_walls = torch.mean(bottom[:, 0:2] ** 2) + torch.mean(left[:, 0:2] ** 2) + torch.mean(right[:, 0:2] ** 2)
    return loss_top + loss_walls


def compute_loss(model, xy_r, b, re, args, xy_speed=None, vmag_speed=None, w_speed=1.0):
    f_c, f_u, f_v = cavity_residual(model, xy_r, re)
    loss_pde = torch.mean(f_c ** 2) + torch.mean(f_u ** 2) + torch.mean(f_v ** 2)
    loss_bc = boundary_loss_fn(model, b)
    p0 = torch.zeros((1, 2), dtype=torch.float32, device=xy_r.device)
    loss_p = torch.mean(model(p0)[:, 2:3] ** 2)
    loss = loss_pde + args.w_bc * loss_bc + args.w_p * loss_p
    if args.use_speed_data and xy_speed is not None and vmag_speed is not None:
        out_speed = model(xy_speed)
        speed_pred = torch.sqrt(out_speed[:, 0:1] ** 2 + out_speed[:, 1:2] ** 2 + 1e-12)
        loss = loss + w_speed * mse(speed_pred, vmag_speed)
    return loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3: PINN for steady lid-driven cavity flow.")
    parser.add_argument("--mode", choices=["debug", "light", "full"], default="debug")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data", type=str, default=str(ROOT / "data" / "Cavity_star_data_Re100.txt"))
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--re", type=float, default=100.0)
    parser.add_argument("--w-bc", type=float, default=5.0)
    parser.add_argument("--w-p", type=float, default=1.0)
    parser.add_argument("--use-speed-data", action="store_true")
    parser.add_argument("--w-speed", type=float, default=1.0)
    parser.add_argument("--max-eval", type=int, default=200000)
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
    fig_dir = ensure_dir(out_root / "figures" / "task3")
    ckpt_dir = ensure_dir(out_root / "checkpoints")
    log_dir = ensure_dir(out_root / "logs")

    model = MLP(2, 3, hidden_layers=6, hidden_dim=80, activation="silu", lb=[0, 0], ub=[1, 1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(opt, step_size=args.lr_step, gamma=args.lr_gamma)

    x_ref, y_ref, vmag_ref = load_cavity_speed(args.data)
    rng = np.random.default_rng(args.seed)
    eval_idx = rng.choice(x_ref.shape[0], min(args.max_eval, x_ref.shape[0]), replace=False)
    xy_ref_eval = to_tensor(np.hstack([x_ref[eval_idx], y_ref[eval_idx]]), device)
    vmag_ref_eval = vmag_ref[eval_idx]

    if args.use_speed_data:
        idx = rng.choice(x_ref.shape[0], min(cfg["n_speed_data"], x_ref.shape[0]), replace=False)
        xy_speed = to_tensor(np.hstack([x_ref[idx], y_ref[idx]]), device)
        vmag_speed = to_tensor(vmag_ref[idx], device)
    else:
        xy_speed, vmag_speed = None, None

    hist = History()
    for ep in range(1, cfg["epochs"] + 1):
        xy_r = sample_uniform([0.0, 0.0], [1.0, 1.0], cfg["n_r"], device)
        b = sample_cavity_boundaries(cfg["n_b_each"], device)
        opt.zero_grad(set_to_none=True)
        loss = compute_loss(model, xy_r, b, args.re, args, xy_speed, vmag_speed, args.w_speed)
        loss.backward()
        opt.step()
        scheduler.step()
        hist.add(total=loss.item())
        if ep % cfg["print_every"] == 0 or ep == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"[Task3][{ep:06d}/{cfg['epochs']}] lr={lr_now:.2e} loss={loss.item():.6e}")

    if cfg["lbfgs_steps"] > 0:
        print(f"[Task3] Adam done. Starting L-BFGS fine-tuning ({cfg['lbfgs_steps']} steps)...")
        opt_lbfgs = torch.optim.LBFGS(
            model.parameters(), max_iter=cfg["lbfgs_steps"],
            line_search_fn="strong_wolfe", tolerance_grad=1e-7,
        )
        def closure3():
            opt_lbfgs.zero_grad()
            xyr = sample_uniform([0.0, 0.0], [1.0, 1.0], cfg["n_r"], device)
            b3 = sample_cavity_boundaries(cfg["n_b_each"], device)
            loss_val = compute_loss(model, xyr, b3, args.re, args, xy_speed, vmag_speed, args.w_speed)
            loss_val.backward()
            return loss_val
        opt_lbfgs.step(closure3)
        print("[Task3] L-BFGS done.")

    pred_eval = batch_predict(model, xy_ref_eval).numpy()
    u_eval, v_eval, p_eval = pred_eval[:, 0:1], pred_eval[:, 1:2], pred_eval[:, 2:3]
    vmag_pred_eval = np.sqrt(u_eval ** 2 + v_eval ** 2)
    rel_v = relative_l2(vmag_pred_eval, vmag_ref_eval)
    print(f"[Task3] Speed magnitude relative L2 error = {rel_v:.6e}")

    grid_n = cfg["eval_grid"]
    gx = np.linspace(0.0, 1.0, grid_n)
    gy = np.linspace(0.0, 1.0, grid_n)
    X, Y = np.meshgrid(gx, gy)
    xy_grid_np = np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])
    xy_grid = to_tensor(xy_grid_np, device)
    pred_grid = batch_predict(model, xy_grid).numpy()
    u, v, p = pred_grid[:, 0:1], pred_grid[:, 1:2], pred_grid[:, 2:3]
    speed = np.sqrt(u ** 2 + v ** 2)

    save_field(xy_grid_np[:, 0:1], xy_grid_np[:, 1:2], u, fig_dir / "u.png", "Predicted u", ylabel="y")
    save_field(xy_grid_np[:, 0:1], xy_grid_np[:, 1:2], v, fig_dir / "v.png", "Predicted v", ylabel="y")
    save_field(xy_grid_np[:, 0:1], xy_grid_np[:, 1:2], p, fig_dir / "p.png", "Predicted p", ylabel="y")
    save_field(xy_grid_np[:, 0:1], xy_grid_np[:, 1:2], speed, fig_dir / "speed_prediction_grid.png", "Predicted speed magnitude", ylabel="y")
    save_field(x_ref[eval_idx], y_ref[eval_idx], vmag_ref_eval, fig_dir / "speed_reference_sample.png", "Reference speed magnitude sample", ylabel="y")
    save_field(x_ref[eval_idx], y_ref[eval_idx], np.abs(vmag_pred_eval - vmag_ref_eval), fig_dir / "speed_abs_error_sample.png", "Speed magnitude absolute error sample", ylabel="y")
    save_vector_field(xy_grid_np[:, 0:1], xy_grid_np[:, 1:2], u, v, fig_dir / "velocity_vector.png")
    save_streamlines(xy_grid_np[:, 0:1], xy_grid_np[:, 1:2], u, v, fig_dir / "streamlines.png")
    save_loss_curve(hist.data, fig_dir / "loss.png", "Task 3 loss")

    torch.save(model.state_dict(), ckpt_dir / "task3_cavity_Re100.pt")
    hist.save(log_dir / "task3_history.json")
    save_json({"relative_l2_speed": rel_v, "Re": args.re, "config": cfg, "mode": args.mode, "use_speed_data": args.use_speed_data}, log_dir / "task3_metrics.json")


if __name__ == "__main__":
    main()
