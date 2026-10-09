"""Finite-volume grid-convergence evaluation between 128x128 and 256x256 grids.

Solves the same band-limited log-conductivity realizations at 128x128 and
256x256 resolution and compares the resulting conservative TPFA face fluxes.
Output is saved to runs/grid_check.json.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from subrel.data import cell_seed, field_at_cell_centers  # noqa: E402
from subrel.field import gaussian_field  # noqa: E402
from subrel.metrics import interior  # noqa: E402
from subrel.solver import _fluxes, solve_darcy  # noqa: E402

OUT = ROOT / "runs" / "grid_check.json"


def face_rel_err(K, h1, h2):
    n = h1.shape[0]
    d = 1.0 / n
    f1 = _fluxes(K, h1, d)
    f2 = _fluxes(K, h2, d)
    s = (slice(2, -2), slice(2, -2))
    num = (float(np.linalg.norm((f1["qx"] - f2["qx"])[s] )) * 0
           + float(np.linalg.norm((f1["qx"][s] - f2["qx"][s])))
           + float(np.linalg.norm((f1["qy"][s] - f2["qy"][s]))))
    den = (float(np.linalg.norm(f2["qx"][s])) + float(np.linalg.norm(f2["qy"][s])))
    return num / den


def main() -> None:
    if OUT.exists():
        print(f"[skip] {OUT} exists")
        return
    out = {"note": "same realization re-solved at 256^2 (log K interpolated); "
                   "relative L2 change of conservative interior face fluxes"}
    cases = {"gauss": (4.0, 0.02), "gauss": (1.0, 0.03), "gauss": (0.5, 0.2)}
    rows = []
    for cov, (s2, lam) in [("gauss", (4.0, 0.02)), ("gauss", (1.0, 0.03)),
                           ("gauss", (0.5, 0.2)), ("m05", (2.0, 0.03))]:
        rng = np.random.default_rng(cell_seed(s2, lam, 128))
        y = gaussian_field(128, s2, lam, rng, cov=cov)
        y128 = field_at_cell_centers(np.fft.fft2(y)).astype(np.float64)
        K128 = np.exp(y128)
        h128, _ = solve_darcy(K128)
        n2 = 256
        xs = (np.arange(128) + 0.5) / 128
        interp = RegularGridInterpolator((xs, xs), np.log(K128), method="linear",
                                         bounds_error=False, fill_value=None)
        xf = (np.arange(n2) + 0.5) / n2
        Xc, Yc = np.meshgrid(xf, xf, indexing="xy")
        K256 = np.exp(interp(np.stack([Yc.ravel(), Xc.ravel()], axis=-1)).reshape(n2, n2))
        h256, _ = solve_darcy(K256)
        # sample the fine solution at the coarse cell centres (bilinear) and
        # compare the conservative face fluxes on the COARSE grid: the
        # difference isolates the discretization error of the 128^2 solve
        from scipy.interpolate import RegularGridInterpolator as _R
        xf2 = (np.arange(n2) + 0.5) / n2
        xc128 = (np.arange(128) + 0.5) / 128
        X128, Y128 = np.meshgrid(xc128, xc128, indexing="xy")
        h256_at_128 = _R((xf2, xf2), h256, method="linear",
                         bounds_error=False, fill_value=None)(
              np.stack([Y128.ravel(), X128.ravel()], axis=-1)).reshape(128, 128)
        # K must also be the interpolated medium on the coarse grid
        K128_from_fine = np.exp(interp(np.stack(
            [Yc.ravel(), Xc.ravel()], axis=-1)).reshape(n2, n2)).reshape(
            128, 2, 128, 2).mean(axis=(1, 3))
        f_coarse = _fluxes(K128_from_fine, h256_at_128, 1.0 / 128)
        f_ref = _fluxes(K128_from_fine, h128, 1.0 / 128)
        sl = (slice(2, -2), slice(2, -2))
        num = (float(np.linalg.norm(f_coarse["qx"][sl] - f_ref["qx"][sl]))
               + float(np.linalg.norm(f_coarse["qy"][sl] - f_ref["qy"][sl])))
        den = (float(np.linalg.norm(f_ref["qx"][sl]))
               + float(np.linalg.norm(f_ref["qy"][sl])))
        rel = num / den
        rows.append({"cov": cov, "sigma2": s2, "lam": lam,
                     "face_flux_rel_change_128_to_256": round(rel, 4)})
        print(rows[-1])
    out["cases"] = rows
    fd, tmp = tempfile.mkstemp(dir=str(OUT.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
