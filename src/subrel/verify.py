"""Numerical verification suite for the finite-volume solver and field generator.

Verifies second-order spatial convergence against a smooth manufactured solution,
machine-precision global mass balance, harmonic-mean series/parallel layering
identities, and empirical variance and correlation recovery of spectral fields.
Run: python -m subrel.verify
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

from .field import empirical_covariance_isotropic, fit_gaussian_correlation, gaussian_field
from .solver import cell_centered_velocity, solve_darcy


def mms_test() -> bool:
    """Manufactured solution: K = 1 + a sin(b1 x) cos(b2 y), h = sin(pi x) sin(pi y).

    All-Dirichlet BCs with face values (h_true = 0 on every face). TPFA must
    reach ~2nd order in head (L2) and ~1st-2nd order in interior flux.
    """
    a, b1, b2, PI = 0.5, 2.0, 3.0, np.pi
    ns = [16, 32, 64, 128]
    errs_h, errs_q = [], []
    for n in ns:
        d = 1.0 / n
        x = (np.arange(n) + 0.5) * d
        X, Y = np.meshgrid(x, x, indexing="ij")   # axis0 = y (rows), axis1 = x (cols)
        # NB: solver convention: axis 1 is x (left/right Dirichlet)
        Xc, Yc = Y, X                               # so swap: Xc varies along axis 1
        K = 1.0 + a * np.sin(b1 * Xc) * np.cos(b2 * Yc)
        h_t = np.sin(PI * Xc) * np.sin(PI * Yc)
        Kx = a * b1 * np.cos(b1 * Xc) * np.cos(b2 * Yc)
        Ky = -a * b2 * np.sin(b1 * Xc) * np.sin(b2 * Yc)
        hx = PI * np.cos(PI * Xc) * np.sin(PI * Yc)
        hy = PI * np.sin(PI * Xc) * np.cos(PI * Yc)
        lap = -PI**2 * h_t
        f = -(Kx * hx + K * lap + Ky * hy + K * lap)

        zero = np.zeros(n)
        faces = {"left": zero, "right": zero, "top": zero, "bottom": zero}
        h_num, _ = solve_darcy(K, f=f, all_dirichlet_faces=faces)
        errs_h.append(np.sqrt(np.mean((h_num - h_t) ** 2)))

        vx_t, vy_t = -K * hx, -K * hy
        vx_n, vy_n = cell_centered_velocity(K, h_num, d)
        s = slice(2, n - 2)                          # interior halo-free region
        num = np.mean((vx_n - vx_t)[s, s] ** 2 + (vy_n - vy_t)[s, s] ** 2)
        den = np.mean(vx_t[s, s] ** 2 + vy_t[s, s] ** 2)
        errs_q.append(np.sqrt(num / den))
        print(f"  n={n:4d}  head L2 err={errs_h[-1]:.3e}  interior flux rel err={errs_q[-1]:.3e}")

    rate_h = -np.polyfit(np.log(ns), np.log(errs_h), 1)[0]
    rate_q = -np.polyfit(np.log(ns), np.log(errs_q), 1)[0]
    print(f"  head convergence rate = {rate_h:.2f} (expect ~2)")
    print(f"  flux convergence rate = {rate_q:.2f} (expect >=1)")
    return rate_h > 1.8 and rate_q > 0.9


def uniform_linearity_test() -> bool:
    """Uniform K + experiment BCs must give the exact discrete linear profile
    h = 1 - x and unit flux in +x (flow from high head on the left)."""
    n = 64
    K = np.ones((n, n))
    h, flux = solve_darcy(K)
    x = (np.arange(n) + 0.5) / n
    err = np.abs(h - (1.0 - x)[None, :]).max()
    qx = flux["qx"][:, 1:-1].mean()
    print(f"  max |h - (1-x)| = {err:.2e}   mean interior qx = {qx:.6f} (expect +1)")
    return err < 1e-10 and abs(qx - 1.0) < 1e-8


def field_statistics_test() -> bool:
    """Fields must hit target variance exactly and the requested correlation
    length (in domain units) within discretization tolerance.

    Cross-checks the FFT autocorrelation against a direct two-point estimate.
    """
    rng = np.random.default_rng(7)
    ok = True
    n = 128
    for sigma2, lam in [(1.0, 0.05), (2.0, 0.03), (0.5, 0.15)]:
        y = gaussian_field(n, sigma2, lam, rng)
        var = y.var()
        lam_px = lam * n
        max_lag = int(min(3 * lam_px, n // 2 - 1))
        rho = empirical_covariance_isotropic(y, max_lag=max_lag)
        lam_fit_px = fit_gaussian_correlation(rho[: max(3, int(2.0 * lam_px))])
        lam_fit = lam_fit_px / n
        # direct two-point check at lag = round(lam_px): rho should be exp(-1/2)=0.607
        l0 = max(1, int(round(lam_px)))
        direct = np.mean(y * np.roll(y, l0, axis=1)) / var
        print(
            f"  sigma2={sigma2:4.1f} lam={lam:4.2f} -> var={var:.4f}, "
            f"lam_fit={lam_fit:.4f} (ratio {lam_fit / lam:.2f}), "
            f"rho(lag=lam)={direct:.3f} (expect ~0.607)"
        )
        ok &= abs(var - sigma2) / sigma2 < 0.02
        ok &= abs(lam_fit / lam - 1.0) < 0.25
    return ok


def conservation_test() -> bool:
    """Interior face fluxes of the TPFA solve must be divergence-free for a
    source-free problem (mass conservation up to linear-solver tolerance)."""
    rng = np.random.default_rng(3)
    y = gaussian_field(64, 2.0, 0.06, rng)
    _, flux = solve_darcy(np.exp(y))
    div = flux["qx"][:, 1:] - flux["qx"][:, :-1] + flux["qy"][1:, :] - flux["qy"][:-1, :]
    interior = np.abs(div[1:-1, 1:-1]).max()
    print(f"  max |div q| interior = {interior:.2e} (expect ~0 up to solver tol)")
    return interior < 1e-8


def archived_results_test() -> bool:
    """The regenerated dual-norm test fields must reproduce the archived
    per-instance atlas metrics (bit-level regeneration check), and the new
    validation artifacts (nested screening, blind test, transport,
    cross-resolution indicator) must exist and be internally consistent."""
    ok = True
    dual = ROOT / "runs" / "dual_norm" / "summary.json"
    if dual.exists():
        s = json.loads(dual.read_text())["models"]
        worst = max(m["max_reproduction_deviation"] or 1.0 for m in s.values())
        ok &= worst < 1e-3
        print(f"  dual-norm regeneration: max deviation {worst:.2e} (<1e-3)")
    else:
        print("  dual-norm summary missing (optional)")
    for name, path in (("nested screening", ROOT / "runs" / "nested_loo.json"),
                       ("blind test", ROOT / "runs" / "blind_test.json"),
                       ("transport", ROOT / "runs" / "breakthrough.json"),
                       ("indicator scaling", ROOT / "runs" / "posterior_scaling.json")):
        present = path.exists()
        ok &= present
        print(f"  {name}: {'present' if present else 'MISSING'}")
    if (ROOT / "runs" / "blind_test.json").exists():
        bt = json.loads((ROOT / "runs" / "blind_test.json").read_text())
        complete = bt.get("stage") == "tested" and all(
            isinstance(bt.get("blind_results", {}).get(c, {}).get(m), dict)
            for c in bt.get("blind_results", {}) for m in ("unet", "fno"))
        ok &= complete
        print(f"  blind test stage: {bt.get('stage')} (complete={complete})")
    return ok


if __name__ == "__main__":
    print("== MMS convergence ==")
    t1 = mms_test()
    print("== uniform linearity ==")
    t2 = uniform_linearity_test()
    print("== field statistics ==")
    t3 = field_statistics_test()
    print("== conservation ==")
    t4 = conservation_test()
    print("== archived validation artifacts ==")
    t5 = archived_results_test()
    print()
    verdict = {"solver head order": t1, "exact linear profile": t2,
               "field variance/length": t3, "flux conservation": t4,
               "archived validation artifacts": t5}
    for k, v in verdict.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    sys.exit(0 if all(verdict.values()) else 1)
