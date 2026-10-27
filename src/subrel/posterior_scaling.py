"""Computational scaling and accuracy of the AMG a posteriori flux indicator.

Measures wall-clock runtime (TPFA matrix assembly, Ruge-Stuben AMG setup, and
a single V-cycle vs direct sparse factorization) and classification accuracy
across 128x128, 256x256, and 512x512 grids. Output is written to
runs/posterior_scaling.json.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from subrel.data import cell_seed, field_at_cell_centers  # noqa: E402
from subrel.field import gaussian_field  # noqa: E402
from subrel.posterior import estimate  # noqa: E402
from subrel.solver import solve_darcy  # noqa: E402

OUT = ROOT / "runs" / "posterior_scaling.json"
GRIDS = (128, 256, 512)
CELLS = ((1.0, 0.05), (2.0, 0.03), (0.5, 0.08))
INSTANCES = (0, 1)  # two media per cell
AMPS = (0.005, 0.01, 0.02, 0.05)  # head-relative perturbation amplitude
PERT_LAM = 0.1  # correlation length of the perturbation


def atomic_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def perturbation(n: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth unit-variance perturbation shape on the n-grid."""
    y = gaussian_field(n, 1.0, PERT_LAM, rng, cov="gauss")
    return y / y.std()


def run_point(n: int, sigma2: float, lam: float, inst: int, amp: float) -> dict:
    rng = np.random.default_rng(cell_seed(sigma2, lam, n) + 7919 * inst)
    y = gaussian_field(n, sigma2, lam, rng, cov="gauss")
    y_cc = field_at_cell_centers(np.fft.fft2(y))
    K = np.exp(y_cc)
    t0 = time.time()
    h_true, _ = solve_darcy(K)
    t_solve = time.time() - t0

    phi = perturbation(n, rng)
    head_scale = float(np.linalg.norm(h_true[2:-2, 2:-2]))
    # rescale so the head-relative amplitude is exactly `amp`
    phi *= amp * head_scale / float(np.linalg.norm(phi[2:-2, 2:-2]))
    h_hat = h_true + phi

    rec = {"n": n, "sigma2": sigma2, "lam": lam, "inst": inst, "amp": amp,
           "t_solver_s": round(t_solve, 3)}
    t0 = time.time()
    res = estimate(K, h_hat, h_true, cycles=1, method="amg")
    rec["t_amg_total_s"] = round(time.time() - t0, 3)
    t0 = time.time()
    estimate(K, h_hat, h_true, cycles=1, method="exact")
    rec["t_exact_s"] = round(time.time() - t0, 3)
    rec.update({k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in res.items()})
    return rec


def main() -> None:
    if OUT.exists():
        print(f"[skip] {OUT} exists; delete to rerun")
        return
    out: dict = {"grids": GRIDS, "amps": AMPS, "points": []}
    atomic_json(out, OUT)
    for n in GRIDS:
        for sigma2, lam in CELLS:
            for inst in INSTANCES:
                for amp in AMPS:
                    t0 = time.time()
                    rec = run_point(n, sigma2, lam, inst, amp)
                    out["points"].append(rec)
                    atomic_json(out, OUT)
                    print(f"[n={n}] s{sigma2:g} l{lam:g} i{inst} amp={amp:g} "
                          f"eta={rec['eta_flux']:.3f} true={rec['true_flux_rel']:.3f} "
                          f"({time.time() - t0:.1f}s)", flush=True)
    print(f"wrote {OUT} ({len(out['points'])} points)")


if __name__ == "__main__":
    main()
