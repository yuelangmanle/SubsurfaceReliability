"""A posteriori flux-error estimation and correction from the TPFA residual.

The discrete system is A_K h = b with A_K = G^T T G. For a neural prediction
h_hat the residual r = b - A_K h_hat satisfies, exactly,

    A_K e = r,      e = h - h_hat,
    ||delta F||^2_{T^{-1}} = e^T A_K e = r^T A_K^{-1} r,

so the conservative flux error in the transmissibility-weighted energy norm
is the A^{-1}-dual norm of the residual. That is a discrete identity, not an
empirical heuristic: it needs no reference solution, only K, h_hat and the
boundary conditions.

A^{-1} is approximated with a few preconditioned iterations, z_m ~ A_K^{-1} r.
The same z_m yields (i) a posterior error indicator, (ii) a corrected head
h_hat + z_m, and (iii) an estimated flux correction -T G z_m. No trainable
parameters are added: the network is untouched.
"""
from __future__ import annotations

import numpy as np
import pyamg
import scipy.sparse.linalg as spla

from .metrics import tpfa_face_fluxes
from .solver import assemble


def _flux_norm(K: np.ndarray, field: np.ndarray) -> float:
    qx, qy = tpfa_face_fluxes(K, field)
    s = (slice(2, -2), slice(2, -2))
    return float(np.sqrt(np.linalg.norm(qx[s]) ** 2 + np.linalg.norm(qy[s]) ** 2))


def estimate(K: np.ndarray, h_hat: np.ndarray, h_true: np.ndarray,
             cycles: int = 1, method: str = "amg") -> dict:
    """Posterior flux-risk indicators plus the corrected prediction."""
    A, rhs, n, d = assemble(K)
    r = rhs - A @ h_hat.ravel()

    if method == "amg":
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=500)
        z = np.zeros_like(r, dtype=np.float64)
        for _ in range(cycles):
            z = ml.solve(r.astype(np.float64), x0=z, tol=0.0, maxiter=1,
                         accel=None, cycle="V")
    elif method == "jacobi":
        D = A.diagonal()
        z = np.zeros_like(r)
        for _ in range(cycles):
            z = z + (r - A @ z) / D
    elif method == "exact":
        z = spla.spsolve(A, r)
    else:
        raise ValueError(method)

    eta_energy = float(np.sqrt(max(r @ z, 0.0)))
    dF_hat = _flux_norm(K, z.reshape(n, n))
    F_hat = _flux_norm(K, h_hat)
    eta_flux = dF_hat / (F_hat + 1e-30)

    h_corr = h_hat + z.reshape(n, n)
    F_true = _flux_norm(K, h_true)
    return dict(
        eta_energy=eta_energy,
        eta_flux=eta_flux,
        r_l2=float(np.linalg.norm(r)),
        r_rel=float(np.linalg.norm(r) / (np.linalg.norm(rhs) + 1e-30)),
        true_flux_rel=_flux_norm(K, h_hat - h_true) / (F_true + 1e-30),
        corr_flux_rel=_flux_norm(K, h_corr - h_true) / (F_true + 1e-30),
        base_flux_rel=_flux_norm(K, h_hat - h_true) / (F_true + 1e-30),
        corr_head_rel=float(np.linalg.norm((h_corr - h_true)[2:-2, 2:-2])
                            / np.linalg.norm(h_true[2:-2, 2:-2])),
    )
