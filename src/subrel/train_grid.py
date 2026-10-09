"""Trainer for the amortized grid-based families (U-Net, FNO).

One model per atlas cell, trained on (Y_cell -> H) pairs with per-cell input
standardization. Test metrics use the shared metrics module. Nothing here is
tuned per cell: identical hyperparameters everywhere -- the atlas measures
default configurations, not per-cell tuning.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from .metrics import flux_error, head_error, spectral_slope
from .models import FNO2d, UNet, count_params

ROOT = Path(__file__).resolve().parents[2]
# Execute training deterministically on CPU to ensure reproducible floating-point reduction order across platforms.
DEV = torch.device("cpu")
torch.set_num_threads(int(__import__("os").environ.get("SUBREL_THREADS", "4")))

DEFAULTS = dict(model="unet", width=24, depth=3, modes=12, batch=32,
                epochs=80, lr=1e-3, patience=15, seed=0, alpha_flux=0.0)


def train_cap(n_train: int) -> int:
    """Effective training-set size (SUBREL_TRAIN_CAP, default 128): uniform
    across the atlas; a 256-instance sensitivity check is run separately."""
    import os
    return min(n_train, int(os.environ.get("SUBREL_TRAIN_CAP", "128")))


def make_inputs(Y: np.ndarray, ymu: float, ystd: float) -> np.ndarray:
    """[standardized logK, x, y] channels; axis convention matches solver
    (axis1 = x). Y may be (n,n) or (N,n,n)."""
    n = Y.shape[-1]
    x = (np.arange(n) + 0.5) / n
    Xc, Yc = np.meshgrid(x, x, indexing="xy")  # Xc varies along axis 1
    z = (Y - ymu) / ystd
    return np.stack([z, np.broadcast_to(Xc, z.shape), np.broadcast_to(Yc, z.shape)],
                    axis=-3).astype(np.float32)


def tpfa_faces(h: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable conservative face fluxes of h under K.

    h, K: (B, 1, n, n). Harmonic face conductivity, matching the reference
    solver exactly; the 1/d factor is omitted on both sides of the loss ratio.
    """
    h = h[:, 0]
    Kc = K[:, 0]
    khx = 2.0 * Kc[:, :, :-1] * Kc[:, :, 1:] / (Kc[:, :, :-1] + Kc[:, :, 1:])
    khy = 2.0 * Kc[:, :-1, :] * Kc[:, 1:, :] / (Kc[:, :-1, :] + Kc[:, 1:, :])
    return khx * (h[:, :, 1:] - h[:, :, :-1]), khy * (h[:, 1:, :] - h[:, :-1, :])


def flux_loss(hp: torch.Tensor, ht: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Relative L2 mismatch of conservative face fluxes (interior faces)."""
    qxp, qyp = tpfa_faces(hp, K)
    qxt, qyt = tpfa_faces(ht, K)
    s = (slice(None), slice(2, -2), slice(2, -2))
    num = ((qxp[s] - qxt[s]) ** 2).sum() + ((qyp[s] - qyt[s]) ** 2).sum()
    den = (qxt[s] ** 2).sum() + (qyt[s] ** 2).sum()
    return num / den.clamp_min(1e-12)


def train_grid_model(cell_path: Path, cfg: dict | None = None, run_dir: Path | None = None):
    cfg = {**DEFAULTS, **(cfg or {})}
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    with h5py.File(cell_path, "r") as f:
        attrs = dict(f.attrs)
        Y = f["Y_cell"][:]
        H = f["H"][:]
    n = int(attrs["n"])
    n_tr, n_va, n_te = int(attrs["n_train"]), int(attrs["n_val"]), int(attrs["n_test"])
    ymu, ystd = float(Y[:n_tr].mean()), float(Y[:n_tr].std())  # stats from full train pool

    n_eff = train_cap(n_tr)
    Xtr = torch.from_numpy(make_inputs(Y[:n_eff], ymu, ystd)).to(DEV)
    # (N, 1, n, n) to match model output -- a bare (N, n, n) target would
    # silently broadcast against (N, 1, n, n) into (N, N, n, n)
    Ttr = torch.from_numpy(((H[:n_eff] - 0.5) / 0.5).astype(np.float32))[:, None].to(DEV)
    Kall = torch.from_numpy(np.exp(Y[:n_eff].astype(np.float32)))[:, None].to(DEV)
    Xva = torch.from_numpy(make_inputs(Y[n_tr:n_tr + n_va], ymu, ystd)).to(DEV)
    Tva = torch.from_numpy(((H[n_tr:n_tr + n_va] - 0.5) / 0.5).astype(np.float32))[:, None].to(DEV)
    Xte = torch.from_numpy(make_inputs(Y[-n_te:], ymu, ystd)).to(DEV)
    Hte = H[-n_te:]

    if cfg["model"] == "unet":
        model = UNet(cin=3, width=cfg["width"], depth=cfg["depth"]).to(DEV)
    elif cfg["model"] == "fno":
        model = FNO2d(cin=3, width=cfg["width"], modes=cfg["modes"]).to(DEV)
    else:
        raise ValueError(cfg["model"])

    # The raw flux mismatch is O(1e5-1e6) at initialisation (an untrained
    # network's face-flux differences are amplified by K), so an absolute
    # weight would be meaningless. Normalise by the initial value: alpha is
    # then the dimensionless flux/head trade-off, as in the conservation
    # fine-tuning experiment.
    f0 = None
    if cfg["alpha_flux"] > 0:
        with torch.no_grad():
            acc = []
            for i in range(0, min(n_eff, 256), 64):
                pred0 = model(Xtr[i:i + 64])
                acc.append(flux_loss(pred0 * 0.5 + 0.5,
                                     Ttr[i:i + 64] * 0.5 + 0.5,
                                     Kall[i:i + 64]))
            f0 = torch.stack(acc).mean().clamp_min(1e-12)
        print(f"    flux_loss at init: {float(f0):.4g} (normaliser)", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    n_tr_samples = n_eff
    best_val, best_state, bad = np.inf, None, 0
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n_tr_samples)
        for i in range(0, n_tr_samples, cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            xb, tb = Xtr[idx], Ttr[idx]
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, tb)
            if cfg["alpha_flux"] > 0:
                kb = Kall[idx].to(DEV)
                hp_phys = pred * 0.5 + 0.5
                ht_phys = tb * 0.5 + 0.5
                loss = loss + cfg["alpha_flux"] * flux_loss(hp_phys, ht_phys, kb) / f0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xva)
            vloss = float(torch.nn.functional.mse_loss(vp, Tva))
        if vloss < best_val - 1e-6:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    train_s = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    # test evaluation
    model.eval()
    heads = []
    with torch.no_grad():
        for i in range(0, n_te, cfg["batch"]):
            out = model(Xte[i:i + cfg["batch"]]).numpy()
            heads.append(out[:, 0] * 0.5 + 0.5)
    h_pred = np.concatenate(heads)

    per_inst = []
    for i in range(n_te):
        K = np.exp(Y[-n_te + i])
        he = head_error(h_pred[i], Hte[i])
        fe = flux_error(K, h_pred[i], Hte[i])
        per_inst.append(dict(inst=n_tr + n_va + i, head_err=he, flux_err=fe,
                             spec_slope_h=spectral_slope(Hte[i])))
    med = lambda k: float(np.median([p[k] for p in per_inst]))

    result = dict(model=cfg["model"], sigma2=float(attrs["sigma2"]), lam=float(attrs["lam"]),
                  n=n, params=count_params(model), epochs_run=ep + 1,
                  best_val=float(best_val), train_seconds=round(train_s, 1),
                  median_head_err=med("head_err"), median_flux_err=med("flux_err"),
                  mean_head_err=float(np.mean([p["head_err"] for p in per_inst])),
                  mean_flux_err=float(np.mean([p["flux_err"] for p in per_inst])),
                  per_inst=per_inst, cfg={k: v for k, v in cfg.items()})

    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), run_dir / "model.pt")
        tmp = run_dir / "metrics.json.tmp"
        tmp.write_text(json.dumps(result))
        os.replace(tmp, run_dir / "metrics.json")
    return result


if __name__ == "__main__":
    import sys
    cell = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "runs" / "pilot"
    cfg = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    r = train_grid_model(cell, cfg, out)
    print(json.dumps({k: v for k, v in r.items() if k != "per_inst"}, indent=2))
