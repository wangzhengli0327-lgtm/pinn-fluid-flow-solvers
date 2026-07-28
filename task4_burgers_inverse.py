from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_utils import batch_predict, load_burgers, to_tensor
from src.gradients import grad
from src.metrics import mse, relative_l2
from src.models import MLP, set_seed
from src.plot_utils import save_field, save_line_profiles, save_loss_curve
from src.sampling import sample_uniform
from src.train_utils import History, default_device, ensure_dir, save_json


def config(mode: str) -> dict:
    if mode == "debug":
        return dict(n_f=512, n_data=200, epochs=50, print_every=10, lbfgs_steps=5)
    if mode == "light":
        return dict(n_f=6000, n_data=1000, epochs=10000, print_every=200, lbfgs_steps=500)
    return dict(n_f=12000, n_data=2000, epochs=20000, print_every=500, lbfgs_steps=2000)


def inverse_softplus(x: float) -> float:
    return float(math.log(math.exp(x) - 1.0))


def burgers_inverse_residual(model: MLP, xt: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    xt = xt.clone().detach().requires_grad_(True)
    u = model(xt)
    gu = grad(u, xt)
    return gu[:, 1:2] + u * gu[:, 0:1] - lam * grad(gu[:, 0:1], xt)[:, 0:1]


def compute_loss(model, xt_f, xt_data, u_data, alpha, args):
    lam = F.softplus(alpha)
    loss_data = mse(model(xt_data), u_data)
    f = burgers_inverse_residual(model, xt_f, lam)
    loss_pde = torch.mean(f ** 2)
    return loss_data + args.w_pde * loss_pde


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 4: inverse PINN for identifying lambda in Burgers equation.")
    parser.add_argument("--mode", choices=["debug", "light", "full"], default="debug")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data", type=str, default=str(ROOT / "data" / "Burgers_star_data.txt"))
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--init-lambda", type=float, default=0.01)
    parser.add_argument("--w-pde", type=float, default=1.0)
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
    fig_dir = ensure_dir(out_root / "figures" / "task4")
    ckpt_dir = ensure_dir(out_root / "checkpoints")
    log_dir = ensure_dir(out_root / "logs")

    x_ref, t_ref, u_ref = load_burgers(args.data)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(x_ref.shape[0], min(cfg["n_data"], x_ref.shape[0]), replace=False)
    xt_data = to_tensor(np.hstack([x_ref[idx], t_ref[idx]]), device)
    u_data = to_tensor(u_ref[idx], device)
    xt_ref = to_tensor(np.hstack([x_ref, t_ref]), device)

    model = MLP(2, 1, hidden_layers=6, hidden_dim=80, activation="silu", lb=[-1, 0], ub=[1, 1]).to(device)
    alpha = torch.nn.Parameter(torch.tensor(inverse_softplus(args.init_lambda), dtype=torch.float32, device=device))
    opt = torch.optim.Adam(list(model.parameters()) + [alpha], lr=args.lr)
    scheduler = StepLR(opt, step_size=args.lr_step, gamma=args.lr_gamma)

    lam_true = 0.01 / math.pi
    hist = History()
    for ep in range(1, cfg["epochs"] + 1):
        xt_f = sample_uniform([-1.0, 0.0], [1.0, 1.0], cfg["n_f"], device)
        opt.zero_grad(set_to_none=True)
        loss = compute_loss(model, xt_f, xt_data, u_data, alpha, args)
        loss.backward()
        opt.step()
        scheduler.step()
        lam_value = float(F.softplus(alpha).detach().cpu())
        hist.add(total=loss.item(), lambda_pred=lam_value)
        if ep % cfg["print_every"] == 0 or ep == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"[Task4][{ep:06d}/{cfg['epochs']}] lr={lr_now:.2e} loss={loss.item():.6e} lam={lam_value:.8e}")

    if cfg["lbfgs_steps"] > 0:
        print(f"[Task4] Adam done. Starting L-BFGS fine-tuning ({cfg['lbfgs_steps']} steps)...")
        opt_lbfgs = torch.optim.LBFGS(
            list(model.parameters()) + [alpha], max_iter=cfg["lbfgs_steps"],
            line_search_fn="strong_wolfe", tolerance_grad=1e-7,
        )
        def closure4():
            opt_lbfgs.zero_grad()
            xt_f2 = sample_uniform([-1.0, 0.0], [1.0, 1.0], cfg["n_f"], device)
            loss_val = compute_loss(model, xt_f2, xt_data, u_data, alpha, args)
            loss_val.backward()
            return loss_val
        opt_lbfgs.step(closure4)
        print("[Task4] L-BFGS done.")

    pred = batch_predict(model, xt_ref).numpy()
    rel_u = relative_l2(pred, u_ref)
    lam_pred = float(F.softplus(alpha).detach().cpu())
    rel_lam = abs(lam_pred - lam_true) / abs(lam_true)
    print(f"[Task4] lambda_pred={lam_pred:.8e}, lambda_true={lam_true:.8e}, relative_error={rel_lam:.6e}")
    print(f"[Task4] u relative L2 error = {rel_u:.6e}")

    save_field(x_ref, t_ref, u_ref, fig_dir / "reference.png", "Reference numerical solution", ylabel="t")
    save_field(x_ref, t_ref, pred, fig_dir / "prediction.png", "PINN prediction", ylabel="t")
    save_field(x_ref, t_ref, np.abs(pred - u_ref), fig_dir / "abs_error.png", "Absolute error", ylabel="t")
    save_line_profiles(x_ref, t_ref, pred, u_ref, [0.25, 0.50, 0.75, 1.00], fig_dir / "profiles.png")
    save_loss_curve(hist.data, fig_dir / "loss_and_lambda.png", "Task 4 loss and lambda")

    torch.save({"model": model.state_dict(), "alpha": alpha.detach().cpu()}, ckpt_dir / "task4_burgers_inverse.pt")
    hist.save(log_dir / "task4_history.json")
    save_json({"lambda_pred": lam_pred, "lambda_true": lam_true, "relative_lambda_error": rel_lam, "relative_l2_u": rel_u, "config": cfg, "mode": args.mode}, log_dir / "task4_metrics.json")


if __name__ == "__main__":
    main()
