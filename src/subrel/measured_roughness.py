"""Empirical log-conductivity gradient energy across all benchmark realizations.

Computes the dimensionless empirical roughness statistic
R_gradY = L^2 <|grad ln K|^2> for all 24 x 352 benchmark realizations using
deterministic seed regeneration and writes the per-instance scalars to
runs/measured_roughness.json.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from subrel.data import cell_seed, field_at_cell_centers  # noqa: E402
from subrel.field import gaussian_field  # noqa: E402

OUT = ROOT / "runs" / "measured_roughness.json"
N = 128
SIGMAS = (0.5, 1.0, 2.0, 4.0)
LAMS = (0.02, 0.03, 0.05, 0.08, 0.12, 0.2)
N_TOTAL = 352


def atomic_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def grad_energy(y: np.ndarray) -> float:
    """L^2 <|grad Y|^2> with second-order central differences (interior)."""
    gy, gx = np.gradient(y, 1.0 / y.shape[0])
    s = slice(2, -2)
    return float(np.mean(gx[s, s] ** 2 + gy[s, s] ** 2))


def main() -> None:
    if OUT.exists():
        print(f"[skip] {OUT} exists; delete to rerun")
        return
    out: dict = {"definition": "R = L^2 <|grad logK|^2>, central differences, "
                               "2-cell interior", "cells": {}}
    atomic_json(out, OUT)
    for s2 in SIGMAS:
        for lam in LAMS:
            cell = f"s{s2:g}_l{lam:g}_n128"
            rng = np.random.default_rng(cell_seed(s2, lam, N))
            recs = []
            for _ in range(N_TOTAL):
                y = gaussian_field(N, s2, lam, rng, cov="gauss")
                y_cc = field_at_cell_centers(np.fft.fft2(y)).astype(np.float64)
                recs.append(grad_energy(y_cc))
            # analytic E|grad Y|^2 = sigma^2/lambda^2 for the Gaussian
            # covariance convention used in the paper's Section 4.1; the
            # measured central-difference energy lands at ~2x that (the
            # paper's "theoretical factor of two")
            out["cells"][cell] = {
                "sigma2": s2, "lam": lam,
                "measured_mean": round(float(np.mean(recs)), 6),
                "measured_median": round(float(np.median(recs)), 6),
                "analytic_s2_over_lam2": round(s2 / lam ** 2, 6),
                "ratio_mean": round(float(np.mean(recs)) / (s2 / lam ** 2), 4),
                "per_instance": [round(r, 6) for r in recs]}
            atomic_json(out, OUT)
            print(f"[done] {cell}: mean {out['cells'][cell]['measured_mean']:.3f} "
                  f"ratio {out['cells'][cell]['ratio_mean']}", flush=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
