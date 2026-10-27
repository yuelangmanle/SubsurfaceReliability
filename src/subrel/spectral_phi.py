"""Empirical high-wavenumber spectral energy fraction of log-conductivity fields.

Computes Phi(k_c), the fraction of log-conductivity variance carried by radial
wavenumbers |k| >= k_c (cycles per domain), together with the 1D radial
integral scale of the empirical correlation function on the 128x128 grid.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def phi_from_yhat(yhat: np.ndarray, k_cut: float = 8.0) -> float:
    n = yhat.shape[0]
    k = np.fft.fftfreq(n, d=1.0 / n)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    P = np.abs(yhat) ** 2
    kk = np.sqrt(KX ** 2 + KY ** 2)
    tot = P[kk > 0].sum()
    return float(P[kk >= k_cut].sum() / tot) if tot > 0 else float("nan")


def integral_scale(y: np.ndarray) -> float:
    """1D radial integral scale of the correlation function, in domain units."""
    from .field import empirical_covariance_isotropic

    rho = empirical_covariance_isotropic(y)
    n = y.shape[0]
    return float(np.trapezoid(rho[:60], dx=1.0 / n))


def cell_phi(h5_path: Path, k_cut: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-realization (Phi, integral scale) for every stored field."""
    with h5py.File(h5_path, "r") as f:
        re, im = f["Yhat_re"][:], f["Yhat_im"][:]
        Y = f["Y_cell"][:]
    phis = np.array([phi_from_yhat(re[i] + 1j * im[i], k_cut) for i in range(len(re))])
    Ls = np.array([integral_scale(Y[i]) for i in range(0, len(Y), max(1, len(Y) // 32))])
    return phis, Ls
