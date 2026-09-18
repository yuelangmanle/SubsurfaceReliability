"""Shared evaluation metrics (applied identically to every solver family).

All errors are computed against the FV ground truth on the base grid,
over the interior halo-free region (2-cell margin), so boundary-condition
treatment differences do not contaminate the comparison.
"""
from __future__ import annotations

import numpy as np

from .solver import cell_centered_velocity


def interior(n: int, halo: int = 2) -> tuple[slice, slice]:
    s = slice(halo, n - halo)
    return s, s


def rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


def head_error(h_pred: np.ndarray, h_true: np.ndarray) -> float:
    n = h_true.shape[0]
    s = interior(n)
    return rel_l2(h_pred[s], h_true[s])


def flux_error(K: np.ndarray, h_pred: np.ndarray, h_true: np.ndarray) -> float:
    n = h_true.shape[0]
    s = interior(n)
    d = 1.0 / n
    vx_t, vy_t = cell_centered_velocity(K, h_true, d)
    vx_p, vy_p = cell_centered_velocity(K, h_pred, d)
    num = np.linalg.norm(vx_p[s] - vx_t[s]) + np.linalg.norm(vy_p[s] - vy_t[s])
    den = np.linalg.norm(vx_t[s]) + np.linalg.norm(vy_t[s])
    return float(num / den)


def radial_spectrum(field: np.ndarray, n_bins: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthally-averaged power spectrum E(|k|) and the |k| bin centers."""
    n = field.shape[0]
    f = field - field.mean()
    spec = np.abs(np.fft.fft2(f)) ** 2 / (n * n)
    kx = np.fft.fftfreq(n, d=1.0 / n)
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    kr = np.sqrt(KX**2 + KY**2)
    kmax = kr.max()
    edges = np.linspace(0, kmax, n_bins + 1)
    centers, energy = [], []
    for i in range(n_bins):
        m = (kr >= edges[i]) & (kr < edges[i + 1])
        if m.sum() == 0:
            continue
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        energy.append(spec[m].mean())
    return np.array(centers), np.array(energy)


def spectral_slope(field: np.ndarray) -> float:
    """Power-law exponent of the radial spectrum E(k) ~ k^-alpha (fit on
    log-log over the decade above the correlation scale)."""
    k, e = radial_spectrum(field, n_bins=48)
    ok = e > 0
    k, e = k[ok], e[ok]
    if len(k) < 8:
        return np.nan
    lo = np.percentile(k, 10)
    sel = (k >= lo) & (e > e.max() * 1e-8)
    if sel.sum() < 5:
        return np.nan
    return float(-np.polyfit(np.log(k[sel]), np.log(e[sel]), 1)[0])


def tpfa_face_fluxes(K: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Conservative TPFA face fluxes of the field h under conductivity K.

    Returns (qx, qy) with qx[i, j] on the face between cells (i,j) and
    (i,j+1) (shape n x n-1) and qy[i, j] between (i,j) and (i+1,j)
    (shape n-1 x n). Face conductivity is the harmonic mean, matching the
    reference solver exactly.

    "Conservative" refers to the shared-face construction (harmonic face
    transmissibility, matching the reference solver): adjacent cells use
    the SAME face flux.  For a head field that satisfies the discrete
    balance this also makes the per-cell fluxes divergence-free; for a
    surrogate prediction it does NOT -- local balance is then not
    guaranteed, which is exactly what the mass-balance diagnostic
    measures."""
    khx = 2.0 * K[:, :-1] * K[:, 1:] / (K[:, :-1] + K[:, 1:])
    khy = 2.0 * K[:-1, :] * K[1:, :] / (K[:-1, :] + K[1:, :])
    qx = khx * (h[:, 1:] - h[:, :-1])
    qy = khy * (h[1:, :] - h[:-1, :])
    return qx, qy


def face_flux_error(K: np.ndarray, h_pred: np.ndarray, h_true: np.ndarray) -> float:
    """Relative L2 error of the CONSERVATIVE face fluxes (the quantity the
    FV reference itself conserves), interior faces only."""
    n = h_true.shape[0]
    s = interior(n)
    qxt, qyt = tpfa_face_fluxes(K, h_true)
    qxp, qyp = tpfa_face_fluxes(K, h_pred)
    nx, ny = qxt.shape
    m = slice(2, nx - 2), slice(2, ny - 2)
    num = np.linalg.norm(qxp[m] - qxt[m]) + np.linalg.norm(qyp[m] - qyt[m])
    den = np.linalg.norm(qxt[m]) + np.linalg.norm(qyt[m])
    return float(num / den)


def mass_balance_residual(K: np.ndarray, h_pred: np.ndarray,
                          h_true: np.ndarray) -> float:
    """Local mass-balance imbalance of the predicted field, normalised by the
    reference face-flux magnitude. Zero for the FV solution."""
    n = h_true.shape[0]
    qx, qy = tpfa_face_fluxes(K, h_pred)
    div = np.zeros((n, n))
    div[:, 1:-1] += qx[:, 1:] - qx[:, :-1]
    div[1:-1, :] += qy[1:, :] - qy[:-1, :]
    s = interior(n)
    qxt, qyt = tpfa_face_fluxes(K, h_true)
    scale = np.linalg.norm(qxt) + np.linalg.norm(qyt)
    return float(np.linalg.norm(div[s]) / scale)
