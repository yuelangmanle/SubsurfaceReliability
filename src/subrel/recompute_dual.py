"""Deterministic regeneration of held-out test realizations and dual-norm metrics.

Regenerates the held-out test conductivity fields and finite-volume reference
solutions from the deterministic regime seeds, then evaluates:
  - Component-sum relative Darcy-flux L2 error (primary metric)
  - Joint Euclidean vector-L2 relative Darcy-flux error
  - Per-component error and reference norms, head error, and spectral Phi

Per-instance scalars are written to runs/dual_norm/<cell>.json and cross-checked
against runs/atlas/<cell>/<model>/metrics.json.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from subrel.data import cell_seed, field_at_cell_centers  # noqa: E402
from subrel.field import gaussian_field  # noqa: E402
from subrel.metrics import head_error, interior  # noqa: E402
from subrel.models import FNO2d, UNet  # noqa: E402
from subrel.solver import cell_centered_velocity, solve_darcy  # noqa: E402

OUT_DIR = ROOT / "runs" / "dual_norm"
N = 128
N_TRAIN, N_VAL, N_TEST = 256, 32, 64
SIGMAS = (0.5, 1.0, 2.0, 4.0)
LAMS = (0.02, 0.03, 0.05, 0.08, 0.12, 0.2)


def atomic_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def generate_test_slice(sigma2: float, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regenerate the cell's 352 fields; return (Y, H) float32 arrays of the
    64 test instances plus the train-pool (Y, mu, std) used at eval time."""
    rng = np.random.default_rng(cell_seed(sigma2, lam, N))
    total = N_TRAIN + N_VAL + N_TEST
    ys, hs = [], []
    for i in range(total):
        y = gaussian_field(N, sigma2, lam, rng, cov="gauss")
        yhat = np.fft.fft2(y)
        y_cc = field_at_cell_centers(yhat)
        if i >= total - N_TEST:
            h, _ = solve_darcy(np.exp(y_cc))
            hs.append(h.astype(np.float32))
        ys.append(y_cc.astype(np.float32))
    Y = np.stack(ys)
    H = np.stack(hs)
    Ytr = Y[:N_TRAIN]
    ymu, ystd = float(Ytr.mean()), float(Ytr.std())  # stats from full train pool
    return Y[-N_TEST:], H, np.array([ymu, ystd], dtype=np.float64)


def make_inputs(Y: np.ndarray, ymu: float, ystd: float) -> np.ndarray:
    n = Y.shape[-1]
    x = (np.arange(n) + 0.5) / n
    Xc, Yc = np.meshgrid(x, x, indexing="xy")
    z = (Y - ymu) / ystd
    return np.stack([z, np.broadcast_to(Xc, z.shape), np.broadcast_to(Yc, z.shape)],
                    axis=-3).astype(np.float32)


def dual_metrics(K: np.ndarray, h_pred: np.ndarray, h_true: np.ndarray) -> dict:
    n = h_true.shape[0]
    s = interior(n)
    d = 1.0 / n
    vx_t, vy_t = cell_centered_velocity(K, h_true, d)
    vx_p, vy_p = cell_centered_velocity(K, h_pred, d)
    ex = float(np.linalg.norm((vx_p - vx_t)[s]))
    ey = float(np.linalg.norm((vy_p - vy_t)[s]))
    fx = float(np.linalg.norm(vx_t[s]))
    fy = float(np.linalg.norm(vy_t[s]))
    primary = (ex + ey) / (fx + fy)
    joint = float(np.hypot(ex, ey) / np.hypot(fx, fy))
    return {"flux_err": primary, "flux_err_joint": joint,
            "err_vx": ex, "err_vy": ey, "ref_vx": fx, "ref_vy": fy}


def phi_of_y(y_cc: np.ndarray) -> float:
    yhat = np.fft.fft2(y_cc)
    n = yhat.shape[0]
    k = np.fft.fftfreq(n, d=1.0 / n)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    P = np.abs(yhat) ** 2
    kk = np.sqrt(KX ** 2 + KY ** 2)
    tot = P[kk > 0].sum()
    return float(P[kk >= 8.0].sum() / tot) if tot > 0 else float("nan")


def run_cell(sigma2: float, lam: float) -> dict:
    cell = f"s{sigma2:g}_l{lam:g}_n128"
    out_path = OUT_DIR / f"{cell}.json"
    if out_path.exists():
        print(f"[skip] {cell} (done)")
        return json.loads(out_path.read_text())
    t0 = time.time()
    Y, H, (ymu, ystd) = generate_test_slice(sigma2, lam)
    K_all = np.exp(Y.astype(np.float64))

    # archived per-instance results for the cross-check
    archived = {}
    for model in ("unet", "fno"):
        mp = ROOT / "runs" / "atlas" / cell / model / "metrics.json"
        if mp.exists():
            archived[model] = {r["inst"]: r for r in json.loads(mp.read_text())["per_inst"]}

    per: dict = {}
    for model in ("unet", "fno"):
        ckpt = ROOT / "runs" / "atlas" / cell / model / "model.pt"
        if not ckpt.exists():
            print(f"[warn] {cell}/{model}: no checkpoint, skipping")
            continue
        if model == "unet":
            net = UNet(cin=3, width=24, depth=3)
        else:
            net = FNO2d(cin=3, width=24, modes=12)
        net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        net.eval()
        X = torch.from_numpy(make_inputs(Y, ymu, ystd))
        with torch.no_grad():
            pred = net(X).squeeze(1).numpy()
        h_pred = pred.astype(np.float64) * 0.5 + 0.5
        recs = []
        for i in range(len(Y)):
            inst = N_TRAIN + N_VAL + i
            m = dual_metrics(K_all[i], h_pred[i], H[i].astype(np.float64))
            m.update(inst=inst, head_err=head_error(h_pred[i], H[i].astype(np.float64)),
                     phi=phi_of_y(Y[i].astype(np.float64)),
                     sigma2=sigma2, lam=lam)
            if model in archived and inst in archived[model]:
                a = archived[model][inst]
                m["check_head"] = round(abs(m["head_err"] - a["head_err"]), 8)
                m["check_flux"] = round(abs(m["flux_err"] - a["flux_err"]), 8)
            recs.append(m)
        per[model] = recs
        print(f"  {cell}/{model}: {len(recs)} instances "
              f"(median flux {np.median([r['flux_err'] for r in recs]):.3f}, "
              f"joint {np.median([r['flux_err_joint'] for r in recs]):.3f})", flush=True)

    result = {"cell": cell, "sigma2": sigma2, "lam": lam, "n_test": len(Y),
              "ymu": float(ymu), "ystd": float(ystd),
              "seconds": round(time.time() - t0, 1), "per": per}
    atomic_json(result, out_path)
    print(f"[done] {cell} in {result['seconds']}s", flush=True)
    return result


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for s2 in SIGMAS:
        for lam in LAMS:
            if only and only not in f"s{s2:g}_l{lam:g}":
                continue
            run_cell(s2, lam)
    # summary pass
    cells = sorted(OUT_DIR.glob("s*_l*_n128.json"))
    if len(cells) < 24:
        print(f"summary deferred: {len(cells)}/24 cells done")
        return
    summary = {}
    for model in ("unet", "fno"):
        rows = [r for c in cells for r in json.loads(c.read_text())["per"].get(model, [])]
        prim = np.array([r["flux_err"] for r in rows])
        joint = np.array([r["flux_err_joint"] for r in rows])
        checks = [r[k] for r in rows for k in ("check_head", "check_flux") if k in r]
        summary[model] = {
            "n": len(rows),
            "fail_rate_primary": round(float((prim > 1).mean()), 6),
            "fail_rate_joint": round(float((joint > 1).mean()), 6),
            "median_primary": round(float(np.median(prim)), 4),
            "median_joint": round(float(np.median(joint)), 4),
            "label_disagreement": int(((prim > 1) != (joint > 1)).sum()),
            "max_reproduction_deviation": max(checks) if checks else None,
            "n_checked": len(checks),
        }
    atomic_json({"models": summary}, OUT_DIR / "summary.json")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
