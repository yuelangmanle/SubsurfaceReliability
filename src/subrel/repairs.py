"""Post-hoc conservation fine-tuning and residual-adaptive collocation methods.

Implements two lightweight boundary interventions:
  - Grid surrogates (U-Net, FNO): soft TPFA volume-conservation fine-tuning
    with loss = MSE(h) + beta * mean(r_TPFA^2) computed via the differentiable
    discrete finite-volume divergence operator.
  - Coordinate PINNs: residual-adaptive resampling (RAR), redistributing the
    lowest-residual collocation points toward high-residual regions before the
    final L-BFGS optimization stage.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from .data import spectrum_derivative_nodes, spectrum_to_nodes
from .metrics import flux_error, head_error
from .models import PINN, count_params, pinn_residual
from .pinn_solve import FieldSampler
from .train_grid import DEV, make_inputs, train_grid_model

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# Repair A: conservation fine-tuning for U-Net / FNO
# --------------------------------------------------------------------------
def tpfa_residual(h: torch.Tensor, K: torch.Tensor, d: float) -> torch.Tensor:
    """Differentiable discrete -div(K grad h) at interior cells.

    h: (B, 1, n, n) head prediction; K: (B, 1, n, n) conductivity; d = 1/n.
    Returns (B, 1, n-2, n-2): the TPFA divergence imbalance, which is exactly
    zero for the FV reference solution (source-free interior).
    """
    h = h[:, 0]
    Kc = K[:, 0]
    khx = 2.0 * Kc[:, :, :-1] * Kc[:, :, 1:] / (Kc[:, :, :-1] + Kc[:, :, 1:])
    khy = 2.0 * Kc[:, :-1, :] * Kc[:, 1:, :] / (Kc[:, :-1, :] + Kc[:, 1:, :])
    fx = khx * (h[:, :, 1:] - h[:, :, :-1])          # (B, n, n-1) vertical-face fluxes
    fy = khy * (h[:, 1:, :] - h[:, :-1, :])          # (B, n-1, n) horizontal-face fluxes
    # divergence imbalance at interior cells (i, j in 1..n-2): (B, n-2, n-2)
    xdiff = fx[:, 1:-1, 1:] - fx[:, 1:-1, :-1]
    ydiff = fy[:, 1:, 1:-1] - fy[:, :-1, 1:-1]
    div = xdiff + ydiff
    return (div / (d * d))[:, None, :, :]


def conservation_finetune(cell_path: Path, base_run: Path, cfg: dict | None = None,
                          run_dir: Path | None = None):
    """Load a trained grid model, fine-tune with the physics penalty, re-eval."""
    import torch.nn.functional as Fn

    from .models import FNO2d, UNet

    cfg = {**dict(beta=1.0, epochs=30, lr=1e-4, batch=32, seed=0, ft_n=128),
           **(cfg or {})}
    base = json.loads((base_run / "metrics.json").read_text())
    mcfg = base["cfg"]
    torch.manual_seed(cfg["seed"])

    with h5py.File(cell_path, "r") as f:
        attrs = dict(f.attrs)
        Y = f["Y_cell"][:]
        H = f["H"][:]
    n = int(attrs["n"])
    n_tr, n_te = int(attrs["n_train"]), int(attrs["n_test"])
    n_tot = Y.shape[0]           # test instances are the LAST n_te rows
    ymu, ystd = float(Y[:n_tr].mean()), float(Y[:n_tr].std())
    Kall = torch.from_numpy(np.exp(Y.astype(np.float32)))[:, None]
    X = torch.from_numpy(make_inputs(Y, ymu, ystd))
    T = torch.from_numpy(((H - 0.5) / 0.5).astype(np.float32))[:, None]

    model = (UNet(3, mcfg["width"], mcfg["depth"]) if mcfg["model"] == "unet"
             else FNO2d(3, mcfg["width"], mcfg["modes"]))
    model.load_state_dict(torch.load(base_run / "model.pt", map_location="cpu"))
    model.to(DEV)

    # Scale-invariant physics weight: the raw TPFA residual carries a 1/d^2
    # factor (~1.6e4 at n=128), so an absolute beta would be meaningless.
    # Normalize by the initial mean squared residual so beta is the
    # dimensionless physics/data trade-off weight.
    d = 1.0 / n
    with torch.no_grad():
        acc = []
        for i in range(0, cfg["ft_n"], 256):
            hp = model(X[i:i + 256]) * 0.5 + 0.5
            acc.append(tpfa_residual(hp, Kall[i:i + 256].to(DEV), d).pow(2).mean())
        r0 = torch.stack(acc).mean().clamp_min(1e-12)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        # fine-tune on the SAME instances the base model trained on (ft_n),
        # so the only changed variable is the loss, not the data budget
        perm = torch.randperm(cfg["ft_n"])
        for i in range(0, cfg["ft_n"], cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            xb, tb = X[idx].to(DEV), T[idx].to(DEV)
            kb = Kall[idx].to(DEV)
            hp = model(xb)
            loss = Fn.mse_loss(hp, tb)
            if cfg["beta"] > 0:
                res = tpfa_residual(hp * 0.5 + 0.5, kb, d)
                loss = loss + cfg["beta"] * res.pow(2).mean() / r0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    ft_s = time.time() - t0

    model.eval()
    heads = []
    with torch.no_grad():
        for i in range(n_tot - n_te, n_tot, cfg["batch"]):
            out = model(X[i:i + cfg["batch"]]).numpy()
            heads.append(out[:, 0] * 0.5 + 0.5)
    h_pred = np.concatenate(heads)

    per_inst = []
    for i in range(n_te):
        j = n_tot - n_te + i
        he = head_error(h_pred[i], H[j])
        fe = flux_error(np.exp(Y[j]), h_pred[i], H[j])
        per_inst.append(dict(inst=j, head_err=he, flux_err=fe))
    result = dict(model=mcfg["model"] + ("+consFT" if cfg["beta"] > 0 else "+mseFT"),
                  sigma2=float(attrs["sigma2"]),
                  lam=float(attrs["lam"]), n=n, params=count_params(model),
                  ft_seconds=round(ft_s, 1),
                  median_head_err=float(np.median([p["head_err"] for p in per_inst])),
                  median_flux_err=float(np.median([p["flux_err"] for p in per_inst])),
                  mean_head_err=float(np.mean([p["head_err"] for p in per_inst])),
                  mean_flux_err=float(np.mean([p["flux_err"] for p in per_inst])),
                  per_inst=per_inst, ft_n=cfg["ft_n"], eval_slice="last_n_test",
                  cfg=cfg)
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), run_dir / "model.pt")
        _atomic_json(run_dir / "metrics.json", result)
    return result


def _atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Repair B: RAR for PINN
# --------------------------------------------------------------------------
def solve_instance_pinn_rar(cell_path: Path, inst: int, cfg: dict | None = None,
                            run_dir: Path | None = None):
    from scipy.stats import qmc

    cfg = {**dict(width=64, depth=4, m_colloc=16384, m_bc=1024, adam_iters=10000,
                  lr=1e-3, lbfgs_iters=500, rar_every=500, rar_frac=0.2,
                  seed=0), **(cfg or {})}
    torch.manual_seed(cfg["seed"] + inst)
    with h5py.File(cell_path, "r") as f:
        attrs = dict(f.attrs)
        yhat = f["Yhat_re"][inst] + 1j * f["Yhat_im"][inst]
        H = f["H"][inst]
        Y_cell = f["Y_cell"][inst]
    n = int(attrs["n"])

    sampler = FieldSampler(yhat)
    m = cfg["m_colloc"]
    xy_c = torch.from_numpy(
        qmc.LatinHypercube(d=2, seed=cfg["seed"] + inst).random(m).astype(np.float32))
    nb = cfg["m_bc"] // 2
    xb = np.random.default_rng(cfg["seed"] + inst).random((nb, 1))
    xy_b = torch.from_numpy(np.concatenate(
        [np.concatenate([xb, np.zeros((nb, 1))], 1),
         np.concatenate([xb, np.ones((nb, 1))], 1)], 0).astype(np.float32))

    model = PINN(cfg["width"], cfg["depth"])
    Yc, Yxc, Yyc = sampler.sample(xy_c)
    Yb, Ybx, Yby = None, None, None  # boundary penalty only needs dh/dy

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["adam_iters"])
    rng = np.random.default_rng(cfg["seed"] + inst + 1)
    t0 = time.time()
    for it in range(cfg["adam_iters"]):
        if it > 0 and it % cfg["rar_every"] == 0:
            with torch.no_grad():
                res, _, _ = pinn_residual(model, xy_c, Yc, Yxc, Yyc)
                bad = torch.argsort(res.squeeze().abs(), descending=True)
                k = int(cfg["rar_frac"] * m)
                fresh = torch.from_numpy(rng.random((k, 2)).astype(np.float32))
                xy_c = torch.cat([xy_c[bad[k:]], fresh], dim=0)
                Yc, Yxc, Yyc = sampler.sample(xy_c)
            # pinn_residual's in-place requires_grad_ is inert under no_grad;
            # re-arm the leaf flag before the training pass
            xy_c = xy_c.clone().requires_grad_(True)
        opt.zero_grad(set_to_none=True)
        res, _, _ = pinn_residual(model, xy_c, Yc, Yxc, Yyc)
        xyb = xy_b.requires_grad_(True)
        hb = model(xyb)
        hy_b = torch.autograd.grad(hb.sum(), xyb, create_graph=True)[0][:, 1]
        (torch.mean(res**2) + torch.mean(hy_b**2)).backward()
        opt.step()
        sched.step()
    adam_s = time.time() - t0

    x = (np.arange(n) + 0.5) / n
    Xc, Yc2 = np.meshgrid(x, x, indexing="xy")
    pts = torch.from_numpy(np.stack([Xc.reshape(-1), Yc2.reshape(-1)], 1).astype(np.float32))
    with torch.no_grad():
        h_pred = model(pts).reshape(n, n).numpy()
    result = dict(model="pinn+RAR", inst=inst, sigma2=float(attrs["sigma2"]),
                  lam=float(attrs["lam"]), n=n, head_err=head_error(h_pred, H),
                  flux_err=flux_error(np.exp(Y_cell), h_pred, H),
                  adam_seconds=round(adam_s, 1),
                  cfg={k: v for k, v in cfg.items()})
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / f"pinn_rar_inst{inst}.json").write_text(json.dumps(result))
    return result


# --------------------------------------------------------------------------
# Standard residual-adaptive resampling (RAR) for PINN
# --------------------------------------------------------------------------
def solve_instance_pinn_rarstd(cell_path: Path, inst: int,
                               cfg: dict | None = None,
                               run_dir: Path | None = None):
    """Standard RAR: move the easiest 20% of collocation points toward the
    hardest regions every `rar_every` Adam iterations; Adam then L-BFGS with
    the same protocol as the vanilla per-instance solver."""
    from scipy.stats import qmc

    cfg = {**dict(width=64, depth=4, m_colloc=16384, m_bc=1024, adam_iters=10000,
                  lr=1e-3, lbfgs_iters=500, rar_every=500, rar_frac=0.2,
                  jitter=0.02, seed=0), **(cfg or {})}
    torch.manual_seed(cfg["seed"] + inst)
    with h5py.File(cell_path, "r") as f:
        attrs = dict(f.attrs)
        yhat = f["Yhat_re"][inst] + 1j * f["Yhat_im"][inst]
        H = f["H"][inst]
        Y_cell = f["Y_cell"][inst]
    n = int(attrs["n"])

    sampler = FieldSampler(yhat)
    m = cfg["m_colloc"]
    xy_c = torch.from_numpy(
        qmc.LatinHypercube(d=2, seed=cfg["seed"] + inst).random(m).astype(np.float32))
    nb = cfg["m_bc"] // 2
    xb = np.random.default_rng(cfg["seed"] + inst).random((nb, 1))
    xy_b = torch.from_numpy(np.concatenate(
        [np.concatenate([xb, np.zeros((nb, 1))], 1),
         np.concatenate([xb, np.ones((nb, 1))], 1)], 0).astype(np.float32))

    model = PINN(cfg["width"], cfg["depth"])
    Yc, Yxc, Yyc = sampler.sample(xy_c)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["adam_iters"])
    rng = np.random.default_rng(cfg["seed"] + inst + 1)
    k = int(cfg["rar_frac"] * m)
    t0 = time.time()
    for it in range(cfg["adam_iters"]):
        if it > 0 and it % cfg["rar_every"] == 0:
            with torch.no_grad():
                res, _, _ = pinn_residual(model, xy_c, Yc, Yxc, Yyc)
                easy = torch.argsort(res.squeeze().abs())[:k]
                pool = torch.from_numpy(
                    rng.random((2 * m, 2)).astype(np.float32))
                Yp, Ypx, Ypy = sampler.sample(pool)
                resp, _, _ = pinn_residual(model, pool, Yp, Ypx, Ypy)
                hard = pool[torch.argsort(resp.squeeze().abs(),
                                          descending=True)[:k]]
                fresh = (hard + cfg["jitter"] * torch.randn(k, 2)).clamp_(0.0, 1.0)
                keep = torch.ones(m, dtype=torch.bool)
                keep[easy] = False
                xy_c = torch.cat([xy_c[keep], fresh], dim=0)
                Yc, Yxc, Yyc = sampler.sample(xy_c)
            xy_c = xy_c.clone().requires_grad_(True)
        opt.zero_grad(set_to_none=True)
        res, _, _ = pinn_residual(model, xy_c, Yc, Yxc, Yyc)
        xyb = xy_b.requires_grad_(True)
        hb = model(xyb)
        hy_b = torch.autograd.grad(hb.sum(), xyb, create_graph=True)[0][:, 1]
        (torch.mean(res**2) + torch.mean(hy_b**2)).backward()
        opt.step()
        sched.step()
    adam_s = time.time() - t0

    t1 = time.time()
    opt2 = torch.optim.LBFGS(model.parameters(), max_iter=cfg["lbfgs_iters"],
                             history_size=50, line_search_fn="strong_wolfe",
                             tolerance_grad=1e-9, tolerance_change=1e-12)
    xy_cc, Ycc, Yxc2, Yyc2 = xy_c, Yc, Yxc, Yyc
    xy_bc = xy_b.detach()

    def closure():
        opt2.zero_grad()
        res, _, _ = pinn_residual(model, xy_cc, Ycc, Yxc2, Yyc2)
        loss_pde = torch.mean(res**2)
        xyb2 = xy_bc.requires_grad_(True)
        hb = model(xyb2)
        hy_b = torch.autograd.grad(hb.sum(), xyb2, create_graph=True)[0][:, 1]
        loss = loss_pde + torch.mean(hy_b**2)
        loss.backward()
        return loss

    opt2.step(closure)
    lbfgs_s = time.time() - t1

    x = (np.arange(n) + 0.5) / n
    Xc, Yc2 = np.meshgrid(x, x, indexing="xy")
    pts = torch.from_numpy(
        np.stack([Xc.reshape(-1), Yc2.reshape(-1)], 1).astype(np.float32))
    with torch.no_grad():
        h_pred = model(pts).reshape(n, n).numpy()
    result = dict(model="pinn+RARstd", inst=inst, sigma2=float(attrs["sigma2"]),
                  lam=float(attrs["lam"]), n=n, head_err=head_error(h_pred, H),
                  flux_err=flux_error(np.exp(Y_cell), h_pred, H),
                  adam_seconds=round(adam_s, 1), lbfgs_seconds=round(lbfgs_s, 1),
                  cfg={kk: v for kk, v in cfg.items()})
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(run_dir / f"pinn_rarstd_inst{inst}.json", result)
    return result
