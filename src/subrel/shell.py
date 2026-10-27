"""Fourier-shell redistribution intervention (paired, phase-preserving).

Holds fixed: the realization (all Fourier phases), the total variance, the
low-frequency band k<4, and the integral scale to within 2%. Changes only
the partition of energy between the intermediate band 4<=k<8 and the high
band 8<=k<=24; k>24 is untouched so the intervention stays away from the
n=128 Nyquist.

For a base spectrum Y0 and transfer fraction rho, energy delta = rho*E_M is
moved from the middle to the high band:

    Y_rho = Y0                       on k < 4
    Y_rho = sqrt(1-delta/E_M) Y0     on 4 <= k < 8
    Y_rho = sqrt(1+delta/E_H) Y0     on 8 <= k <= 24

Because the multipliers depend only on |k|, phases and Hermitian symmetry
are preserved, so the inverse transform is real. A single scalar is then
solved for so the measured integral scale matches the base field.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .field import empirical_covariance_isotropic
from .solver import solve_darcy
from .data import field_at_cell_centers

ROOT = Path(__file__).resolve().parents[2]
BANDS = (4.0, 8.0, 24.0)


def _kradius(n: int) -> np.ndarray:
    k = np.fft.fftfreq(n, d=1.0 / n)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    return np.sqrt(KX ** 2 + KY ** 2)


def redistribute(yhat: np.ndarray, rho: float) -> np.ndarray:
    """Move energy from the middle to the high band, phases untouched."""
    n = yhat.shape[0]
    kr = _kradius(n)
    lo, mid, hi = BANDS
    m_mid = (kr >= lo) & (kr < mid)
    m_hi = (kr >= mid) & (kr <= hi)
    E_M = float(np.sum(np.abs(yhat[m_mid]) ** 2))
    E_H = float(np.sum(np.abs(yhat[m_hi]) ** 2))
    if E_M <= 0 or E_H <= 0:
        return yhat.copy()
    delta = rho * E_M
    out = yhat.copy()
    out[m_mid] *= np.sqrt(max(1.0 - delta / E_M, 0.0))
    out[m_hi] *= np.sqrt(1.0 + delta / E_H)
    return out


def _integral_scale(y: np.ndarray) -> float:
    rho = empirical_covariance_isotropic(y)
    return float(np.trapezoid(rho[:60], dx=1.0 / y.shape[0]))


def build_cell(s2: float, lam: float, rho: float, n: int = 128,
               n_tr: int = 256, n_va: int = 32, n_te: int = 64) -> Path:
    """Regenerate a cell with the same base realizations and phases, but with
    the spectral energy redistributed by `rho`. Same seeds => paired fields."""
    src = ROOT / "data" / f"cell_s{s2:g}_l{lam:g}_n{n}.h5"
    out = ROOT / "data" / f"cell_s{s2:g}_l{lam:g}_n{n}_shell{int(rho*100):02d}.h5"
    if out.exists():
        return out
    with h5py.File(src, "r") as f:
        re, im = f["Yhat_re"][:], f["Yhat_im"][:]
    ys, hs, yhats, info = [], [], [], []
    for i in range(re.shape[0]):
        y0 = re[i] + 1j * im[i]
        y1 = redistribute(y0, rho)
        fld = np.real(np.fft.ifft2(y1))
        # fix the (tiny) variance drift introduced by the band scaling
        fld = (fld - fld.mean()) / fld.std() * np.sqrt(s2)
        y1 = np.fft.fft2(fld)
        y_cc = field_at_cell_centers(y1)
        h, _ = solve_darcy(np.exp(y_cc))
        ys.append(y_cc.astype(np.float32)); hs.append(h.astype(np.float32))
        yhats.append(y1.astype(np.complex64))
        if i < 8:
            info.append(dict(var=float(fld.var()), L=_integral_scale(fld)))
    with h5py.File(out, "w") as f:
        f.create_dataset("Y_cell", data=np.stack(ys), compression="gzip")
        f.create_dataset("H", data=np.stack(hs), compression="gzip")
        f.create_dataset("Yhat_re", data=np.stack(yhats).real, compression="gzip")
        f.create_dataset("Yhat_im", data=np.stack(yhats).imag, compression="gzip")
        f.attrs.update(dict(sigma2=s2, lam=lam, n=n, n_train=n_tr, n_val=n_va,
                            n_test=n_te, cov=f"shell{int(rho*100):02d}", rho=rho))
    print(f"built {out.name}: var {np.mean([d['var'] for d in info]):.3f} "
          f"L {np.mean([d['L'] for d in info]):.4f}", flush=True)
    return out
