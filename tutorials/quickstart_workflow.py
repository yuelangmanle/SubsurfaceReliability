"""Quickstart tutorial demonstrating the two-stage SubsurfaceReliability workflow.

Walks through a complete, self-contained example on a heterogeneousporous medium:
  1. Generate a rough lognormal or channelized conductivity field K(x) = exp(Y(x)).
  2. Stage 1 (Pre-Training Roughness Screening): Compute parameter-free discrete
     gradient energy R_{grad Y} and roughness invariants to determine whether
     head-only training will suffer from flux blindness.
  3. Solve the reference TPFA finite-volume system and compare head error (E_h),
     TPFA face-flux error (E_{q,face}), local divergence imbalance (I_div), and
     Pollock streamline breakthrough before and after Stage 2 (1-cycle AMG
     residual screening eta_F and one-step defect correction).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyamg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subrel.breakthrough import summarize, travel_times  # noqa: E402
from subrel.data import cell_seed, field_at_cell_centers  # noqa: E402
from subrel.field import gaussian_field  # noqa: E402
from subrel.metrics import (  # noqa: E402
    face_flux_error, flux_error, head_error, mass_balance_residual
)
from subrel.posterior import _flux_norm  # noqa: E402
from subrel.solver import assemble, solve_darcy  # noqa: E402


def main() -> None:
    n = 128
    d = 1.0 / n
    sigma2, lam = 2.0, 0.03
    print(f"=== SubsurfaceReliability v1.0 Quickstart Tutorial (N={n}x{n}, sigma^2={sigma2}, lambda={lam}) ===")

    # 1. Generate heterogeneous conductivity realization
    rng = np.random.default_rng(cell_seed(sigma2, lam, n))
    y_spec = gaussian_field(n, sigma2, lam, rng, cov="gauss")
    Y = field_at_cell_centers(np.fft.fft2(y_spec)).astype(np.float64)
    K = np.exp(Y)

    # 2. Stage 1: Compute pre-training roughness invariants directly from Y(x)
    gx = (Y[:, 1:] - Y[:, :-1]) / d
    gy = (Y[1:, :] - Y[:-1, :]) / d
    r_grad_y = float(np.mean(gx[:-1, :] ** 2 + gy[:, :-1] ** 2))
    s2_over_lam = sigma2 / lam
    print(f"[Stage 1 Screen] sigma^2/lambda = {s2_over_lam:.2f} (U-Net cutoff 3.33), "
          f"sigma^2 = {sigma2:.2f} (FNO cutoff 0.75), R_{{grad Y}} = {r_grad_y:.1f}")
    print(" -> Result: FLAGGED as rough regime. Head-only L2 training will exhibit flux blindness; "
          "use TPFA face-flux supervision (alpha in [0.01, 0.10]) + Stage 2 AMG defect correction.")

    # 3. Reference TPFA solution and synthetic high-wavenumber head perturbation (~5% head error)
    H_ref, _ = solve_darcy(K)
    H_ref = H_ref.astype(np.float64)
    rng_noise = np.random.default_rng(7)
    high_k_noise = gaussian_field(n, 1.0, 0.025, rng_noise, cov="gauss")
    h_surrogate = np.clip(H_ref + 0.032 * high_k_noise / np.std(high_k_noise), 0.0, 1.0)

    eh_pre = head_error(h_surrogate, H_ref)
    eq_pre = flux_error(K, h_surrogate, H_ref)
    eqf_pre = face_flux_error(K, h_surrogate, H_ref)
    idiv_pre = mass_balance_residual(K, h_surrogate, H_ref)
    ts_ref = travel_times(K, H_ref)
    tr_pre = summarize(ts_ref, travel_times(K, h_surrogate))
    print(f"[Before Stage 2] Head error E_h = {eh_pre:.2%}, Cell flux E_q = {eq_pre:.2f}, "
          f"TPFA face flux E_{{q,face}} = {eqf_pre:.2f}, I_div = {idiv_pre:.2f}, "
          f"Pollock arrival = {tr_pre['arrived']}/41 ({100*tr_pre['arrived']/41:.1f}%)")

    # 4. Stage 2: Reference-free 1-cycle AMG dual-norm indicator eta_F and 1-step defect correction
    A, rhs, _, _ = assemble(K)
    ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=500)
    r = rhs - A @ h_surrogate.ravel()
    z1 = ml.solve(r.astype(np.float64), x0=np.zeros_like(r), tol=0.0, maxiter=1, accel=None, cycle="V")
    eta_f = _flux_norm(K, z1.reshape(n, n)) / max(_flux_norm(K, h_surrogate), 1e-12)
    h_corrected = h_surrogate + z1.reshape(n, n)

    eh_post = head_error(h_corrected, H_ref)
    eqf_post = face_flux_error(K, h_corrected, H_ref)
    idiv_post = mass_balance_residual(K, h_corrected, H_ref)
    tr_post = summarize(ts_ref, travel_times(K, h_corrected))
    print(f"[Stage 2 Screen] Reference-free 1-cycle AMG indicator eta_F = {eta_f:.3f} (flags E_{{q,face}} > 1.0)")
    print(f"[After 1 V-cycle] Head error E_h = {eh_post:.2%}, TPFA face flux E_{{q,face}} = {eqf_post:.3f} "
          f"({100*(1 - eqf_post/eqf_pre):.1f}% reduction), I_div = {idiv_post:.3f}, "
          f"Pollock arrival = {tr_post['arrived']}/41 ({100*tr_post['arrived']/41:.1f}%), "
          f"median travel-time ratio = {tr_post['median_paired_ratio']:.3f}")


if __name__ == "__main__":
    main()
