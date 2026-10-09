"""Per-instance PINN solver (vanilla recipe: Adam then L-BFGS).

The PINN minimizes the physics residual of -div(K grad h)=0 with K=exp(Y)
the *same continuous field* the FV truth used (band-limited spectral
interpolant, sampled bilinearly from a 4n fine grid -- exact up to
interpolation of a smooth field). Dirichlet left/right is hard-constrained;
no-flow top/bottom is a boundary penalty. No solution data whatsoever.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.stats import qmc

from .data import spectrum_derivative_nodes, spectrum_to_nodes
from .metrics import flux_error, head_error, spectral_slope
from .models import PINN, count_params, pinn_residual

ROOT = Path(__file__).resolve().parents[2]
# CPU throughout: the MPS backend deadlocks on second-order autograd
# (create_graph=True through autograd.grad) used by the PINN residual.
DEV = torch.device("cpu")
torch.set_num_threads(int(__import__("os").environ.get("SUBREL_THREADS", "4")))

DEFAULTS = dict(width=64, depth=4, m_colloc=16384, m_bc=1024,
                adam_iters=10000, lr=1e-3, lbfgs_iters=500, seed=0,
                unnormalized_resid=False)


class FieldSampler:
    """Bilinear sampler of (Y, Yx, Yy) from the band-limited fine grid."""

    def __init__(self, yhat: np.ndarray, fine_factor: int = 4):
        n = yhat.shape[0]
        self.nf = fine_factor * n
        self.grids = [
            torch.from_numpy(g.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEV)
            for g in (spectrum_to_nodes(yhat, self.nf),
                      spectrum_derivative_nodes(yhat, self.nf, "x"),
                      spectrum_derivative_nodes(yhat, self.nf, "y"))
        ]

    def sample(self, xy: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Fine-grid node j sits at x=j/nf; with align_corners=True, u=-1 maps
        # to pixel 0 (x=0), so the exact affine map is u = 2 x nf/(nf-1) - 1.
        # Input height axis = array axis 0 = y, width = axis 1 = x.
        a = 2.0 * self.nf / (self.nf - 1.0)
        grid = torch.stack([a * xy[:, 0] - 1.0, a * xy[:, 1] - 1.0], dim=-1)
        grid = grid.reshape(1, xy.shape[0], 1, 2)   # (B, H_out, W_out, 2)
        out = []
        for img in self.grids:
            v = torch.nn.functional.grid_sample(
                img, grid, mode="bilinear", padding_mode="border", align_corners=True)
            out.append(v.reshape(-1))
        return tuple(out)


def solve_instance_pinn(cell_path: Path, inst: int, cfg: dict | None = None,
                        run_dir: Path | None = None):
    cfg = {**DEFAULTS, **(cfg or {})}
    torch.manual_seed(cfg["seed"] + inst)
    with h5py.File(cell_path, "r") as f:
        attrs = dict(f.attrs)
        yhat = f["Yhat_re"][inst] + 1j * f["Yhat_im"][inst]
        H = f["H"][inst]
        Y_cell = f["Y_cell"][inst]
    n = int(attrs["n"])

    sampler = FieldSampler(yhat)
    # fixed collocation set: Latin hypercube (deterministic given seed)
    lhs = qmc.LatinHypercube(d=2, seed=cfg["seed"] + inst).random(cfg["m_colloc"])
    xy_c = torch.from_numpy(lhs.astype(np.float32)).to(DEV)
    # boundary points on top/bottom faces for the no-flow penalty
    nb = cfg["m_bc"] // 2
    xb = np.random.default_rng(cfg["seed"] + inst).random((nb, 1))
    xy_b = torch.from_numpy(np.concatenate([
        np.concatenate([xb, np.zeros((nb, 1))], axis=1),
        np.concatenate([xb, np.ones((nb, 1))], axis=1)], axis=0).astype(np.float32)).to(DEV)

    model = PINN(width=cfg["width"], depth=cfg["depth"]).to(DEV)
    Yc, Yxc, Yyc = sampler.sample(xy_c)

    t0 = time.time()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["adam_iters"])
    for it in range(cfg["adam_iters"]):
        opt.zero_grad(set_to_none=True)
        res, _, _ = pinn_residual(model, xy_c, Yc, Yxc, Yyc,
                                  unnormalized=cfg["unnormalized_resid"])
        loss_pde = torch.mean(res**2)
        # no-flow: dh/dy = 0 at y=0,1
        xy_b = xy_b.requires_grad_(True)
        hb = model(xy_b)
        hy_b = torch.autograd.grad(hb.sum(), xy_b, create_graph=True)[0][:, 1]
        loss = loss_pde + torch.mean(hy_b**2)
        loss.backward()
        opt.step()
        sched.step()
    adam_s = time.time() - t0

    # L-BFGS refinement (same device)
    t1 = time.time()
    model_cpu = model
    xy_cc, Ycc, Yxc2, Yyc2 = xy_c.cpu(), Yc.cpu(), Yxc.cpu(), Yyc.cpu()
    xy_bc = xy_b.detach().cpu()
    opt2 = torch.optim.LBFGS(model_cpu.parameters(), max_iter=cfg["lbfgs_iters"],
                             history_size=50, line_search_fn="strong_wolfe",
                             tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt2.zero_grad()
        res, _, _ = pinn_residual(model_cpu, xy_cc, Ycc, Yxc2, Yyc2,
                                      unnormalized=cfg["unnormalized_resid"])
        loss_pde = torch.mean(res**2)
        xy_b2 = xy_bc.requires_grad_(True)
        hb = model_cpu(xy_b2)
        hy_b = torch.autograd.grad(hb.sum(), xy_b2, create_graph=True)[0][:, 1]
        loss = loss_pde + torch.mean(hy_b**2)
        loss.backward()
        return loss

    opt2.step(closure)
    lbfgs_s = time.time() - t1

    # evaluate at cell centers
    x = (np.arange(n) + 0.5) / n
    Xc, Yc2 = np.meshgrid(x, x, indexing="xy")   # Xc varies along axis 1
    pts = torch.from_numpy(np.stack([Xc.reshape(-1), Yc2.reshape(-1)], axis=1)
                           .astype(np.float32))
    with torch.no_grad():
        h_pred = model_cpu(pts).reshape(n, n).numpy()

    he = head_error(h_pred, H)
    fe = flux_error(np.exp(Y_cell), h_pred, H)
    result = dict(model="pinn", inst=inst, sigma2=float(attrs["sigma2"]),
                  lam=float(attrs["lam"]), n=n, params=count_params(model),
                  head_err=he, flux_err=fe, spec_slope_h=spectral_slope(H),
                  adam_seconds=round(adam_s, 1), lbfgs_seconds=round(lbfgs_s, 1),
                  cfg={k: v for k, v in cfg.items()})
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model_cpu.state_dict(), run_dir / f"pinn_inst{inst}.pt")
        tmp = run_dir / f"pinn_inst{inst}.json.tmp"
        tmp.write_text(json.dumps(result))
        os.replace(tmp, run_dir / f"pinn_inst{inst}.json")
    return result


if __name__ == "__main__":
    import sys
    cell = Path(sys.argv[1])
    inst = int(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "runs" / "pilot"
    cfg = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    r = solve_instance_pinn(cell, inst, cfg, out)
    print(json.dumps(r, indent=2))
