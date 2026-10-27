"""Flux Reliability Certificate (FRC).

The exact error-propagation relation is

    delta q = B_K delta h,      B_K = -M_K G

with G the cell-to-face gradient and M_K the face transmissibility built from
the conductivity. For a solver family m, the *shape* of its head error is a
property of the architecture and the training protocol; the *magnitude* is a
property of the regime. The certificate separates the two: it carries a
calibration set of unit-norm head-error fields from regimes already run, and
for a new medium K evaluates

    A_m(K) = sqrt( mean_c || B_K e_c ||^2 ) / || B_K h_0 ||

i.e. how strongly this medium amplifies the family's typical error shape,
normalized by the flux the no-information profile h_0 = 1-x produces. A_m is
computable before the target regime is trained: it needs only K, the
calibration error fields, and h_0.

The predicted flux/head error ratio is compared against the measured one,
with calibration held out by regime (leave-one-regime-out).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from .metrics import tpfa_face_fluxes
from .models import FNO2d, PINN, UNet
from .train_grid import make_inputs

ROOT = Path(__file__).resolve().parents[2]
CELLS = [(s2, lam) for s2 in (0.5, 1, 2, 4)
         for lam in (0.02, 0.03, 0.05, 0.08, 0.12, 0.2)]
N_PINN = 4


def cid_of(s2, lam):
    return f"s{s2:g}_l{lam:g}_n128"


def _flux_energy(K: np.ndarray, field: np.ndarray) -> float:
    """||B_K field||_2 over interior faces (B_K = TPFA face flux operator)."""
    qx, qy = tpfa_face_fluxes(K, field)
    s = (slice(2, -2), slice(2, -2))
    return float(np.linalg.norm(qx[s]) ** 2 + np.linalg.norm(qy[s]) ** 2)


def _predictions(cid: str, fam: str):
    """Head predictions for the test instances of one cell and family."""
    with h5py.File(ROOT / "data" / f"cell_{cid}.h5", "r") as f:
        Yall, Hall = f["Y_cell"][:], f["H"][:]
    n = Yall.shape[1]
    ymu, ystd = float(Yall[:256].mean()), float(Yall[:256].std())
    out = {}
    if fam in ("unet", "fno"):
        d = ROOT / "runs/atlas" / cid / fam
        meta = json.loads((d / "metrics.json").read_text())
        m = (UNet(3, meta["cfg"]["width"], meta["cfg"]["depth"]) if fam == "unet"
             else FNO2d(3, meta["cfg"]["width"], meta["cfg"]["modes"]))
        m.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
        m.eval()
        with torch.no_grad():
            hp = m(torch.from_numpy(make_inputs(Yall[-64:], ymu, ystd))).numpy()[:, 0] * 0.5 + 0.5
        for i in range(64):
            out[288 + i] = (Yall[-64 + i], Hall[-64 + i], hp[i])
    else:
        x = (np.arange(n) + 0.5) / n
        Xc, Yc = np.meshgrid(x, x, indexing="xy")
        pts = torch.from_numpy(np.stack([Xc.reshape(-1), Yc.reshape(-1)], 1).astype(np.float32))
        for inst in range(288, 288 + N_PINN):
            p = ROOT / "runs/atlas" / cid / "pinn" / f"pinn_inst{inst}.pt"
            if not p.exists():
                continue
            pm = PINN(64, 4)
            pm.load_state_dict(torch.load(p, map_location="cpu"))
            pm.eval()
            with torch.no_grad():
                h = pm(pts).reshape(n, n).numpy()
            out[inst] = (Yall[inst], Hall[inst], h)
    return out


def build(out_name: str = "frc.json") -> dict:
    h0_cache = {}
    rec = {}
    for s2, lam in CELLS:
        cid = cid_of(s2, lam)
        for fam in ("unet", "fno", "pinn"):
            preds = _predictions(cid, fam)
            if not preds:
                continue
            errs, heads, fluxes = [], [], []
            for inst, (Y, H, hp) in preds.items():
                K = np.exp(Y)
                e = hp - H
                errs.append(e)
                heads.append(np.linalg.norm(e[2:-2, 2:-2]) / np.linalg.norm(H[2:-2, 2:-2]))
                fluxes.append(np.sqrt(_flux_energy(K, e) / max(_flux_energy(K, H), 1e-30)))
                h0_cache[cid] = (K, 1.0 - ((np.arange(H.shape[0]) + 0.5) / H.shape[0])[None, :] * np.ones_like(H))
            K, h0 = h0_cache[cid]
            rec[f"{cid}|{fam}"] = dict(
                head_err=float(np.median(heads)),
                flux_err=float(np.median(fluxes)),
                b_h0=_flux_energy(K, h0),
                _shapes=[(e / max(np.linalg.norm(e), 1e-30)).astype(np.float32)
                         for e in errs[:4]],
            )
            print(f"{cid} {fam}: head {np.median(heads):.3f} flux {np.median(fluxes):.3f}", flush=True)
    return rec


def certify(rec: dict) -> dict:
    """Leave-one-regime-out: predict each cell's flux/head amplification ratio
    from the error shapes of the other 23 regimes and the target medium."""
    from scipy.stats import spearmanr

    out = {}
    for fam in ("unet", "fno", "pinn"):
        keys = [k for k in rec if k.endswith("|" + fam)]
        if not keys:
            continue
        cells = [k.split("|")[0] for k in keys]
        pred, actual = [], []
        for t in cells:
            with h5py.File(ROOT / "data" / f"cell_{t}.h5", "r") as f:
                Kt = np.exp(f["Y_cell"][288])
            n = Kt.shape[0]
            h0 = 1.0 - ((np.arange(n) + 0.5) / n)[None, :] * np.ones_like(Kt)
            num, cnt = 0.0, 0
            for c in cells:
                if c == t:
                    continue
                for e in rec[f"{c}|{fam}"]["_shapes"]:
                    num += _flux_energy(Kt, e.astype(np.float64))
                    cnt += 1
            A = float(np.sqrt(num / cnt / rec[f"{t}|{fam}"]["b_h0"]))
            pred.append(A)
            actual.append(rec[f"{t}|{fam}"]["flux_err"] / max(rec[f"{t}|{fam}"]["head_err"], 1e-12))
        pred, actual = np.array(pred), np.array(actual)
        rho = spearmanr(pred, actual).statistic
        lg = np.polyfit(np.log(pred), np.log(actual), 1)
        resid = np.log(actual) - np.polyval(lg, np.log(pred))
        r2 = 1 - resid.var() / np.log(actual).var()
        out[fam] = dict(spearman=float(rho), r2_loglog=float(r2),
                        slope=float(lg[0]), intercept=float(lg[1]),
                        pred=pred.tolist(), actual=actual.tolist(), cells=cells)
        print(f"{fam}: Spearman {rho:.3f} | log-log R2 {r2:.3f} | slope {lg[0]:.2f}",
              flush=True)
    return out


if __name__ == "__main__":
    rec = build()
    cert = certify(rec)
    slim = {k: {kk: vv for kk, vv in v.items() if kk != "_shapes"}
            for k, v in rec.items()}
    (ROOT / "runs" / "frc.json").write_text(json.dumps(dict(records=slim, certificate=cert)))
    print("wrote runs/frc.json")
