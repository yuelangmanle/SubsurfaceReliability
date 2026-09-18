"""Spectral-method generator of lognormal hydraulic conductivity fields.

Y ~ N(0, sigma^2) stationary Gaussian field with Gaussian covariance
C(r) = sigma^2 exp(-r^2 / (2 lam^2)) on the periodic unit square, sampled on
an N x N cell-centered grid. K = exp(Y).

Fields are normalized to the exact target variance (periodic spectral sampling
deviates slightly); the effective correlation length is measured empirically by
verify_field.py rather than trusted.

This is standard practice for controlled numerical experiments; every consumer
of these fields must go through the statistics reported in the data manifest.
"""
from __future__ import annotations

import numpy as np


def gaussian_field(
    n: int,
    sigma2: float,
    lam: float,
    rng: np.random.Generator,
    aniso: float = 1.0,
    theta: float = 0.0,
    cov: str = "gauss",
) -> np.ndarray:
    """Draw one log-conductivity field Y.

    Parameters
    ----------
    n : grid size (n x n cells on [0,1]^2)
    sigma2 : target variance of Y
    lam : correlation length of the Gaussian covariance (domain units)
    rng : numpy Generator
    aniso : aspect ratio lambda_x / lambda_y (1.0 = isotropic)
    theta : rotation of the anisotropy axes (radians)
    """
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    # Matern families use range parameter ell = lam*sqrt(pi/2) so that all
    # families share the integral scale at a given lam (theta_G = lam*sqrt(pi/2),
    # theta_M = ell); only the spectral shape -- the high-k tail -- differs.
    # Match the 1D radial integral scale L = int_0^inf rho(r) dr across
    # families: L_gauss = lam*sqrt(pi/2), L_m05 = ell, L_m15 = 2*ell.
    # Analytical scale parameters ell_3/2 = I_Y / 2 and ell_1/2 = I_Y ensure identical 1D radial integral scale.
    if cov == "gauss":
        lam_eff = lam
    elif cov == "m05":
        lam_eff = lam * np.sqrt(np.pi / 2.0)
    elif cov == "m15":
        lam_eff = lam * np.sqrt(np.pi / 2.0) / 2.0
    else:
        raise ValueError(cov)
    if aniso != 1.0 or theta != 0.0:
        c, s = np.cos(theta), np.sin(theta)
        k1 = c * KX + s * KY          # along long axis
        k2 = -s * KX + c * KY
        lam1, lam2 = lam_eff * aniso, lam_eff
        k2_eff = k1**2 * lam1**2 + k2**2 * lam2**2
    else:
        k2_eff = (KX**2 + KY**2) * lam_eff**2

    if cov == "gauss":
        psd = np.exp(-k2_eff / 2.0)   # Gaussian covariance spectral density
    elif cov == "m05":                # Matern nu=0.5: S(k) ~ (1+k^2 l^2)^-(nu+1)
        psd = (1.0 + k2_eff) ** -1.5
    elif cov == "m15":                # Matern nu=1.5
        psd = (1.0 + k2_eff) ** -2.5
    else:
        raise ValueError(cov)
    psd[0, 0] = 0.0                   # zero-mean field

    noise = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    y_hat = noise * np.sqrt(psd)
    y = np.real(np.fft.ifft2(y_hat))
    y = (y - y.mean()) / y.std()      # exact unit variance
    return y.astype(np.float64) * np.sqrt(sigma2)


def conductivity(y: np.ndarray) -> np.ndarray:
    """K from log-conductivity Y."""
    return np.exp(y)


def empirical_covariance_isotropic(field: np.ndarray, max_lag: int = 16) -> np.ndarray:
    """Empirical azimuthally-averaged covariance C(r) of a periodic field."""
    n = field.shape[0]
    f = field - field.mean()
    spec = np.abs(np.fft.fft2(f)) ** 2
    # ifft2 of the power spectrum is the circular SUM of f(x) f(x+s); divide by
    # n^2 for the mean and by the variance for the correlation coefficient
    rho = np.real(np.fft.ifft2(spec)) / (n * n * f.var())
    # azimuthal average of rho over |shift| bins
    ks = np.fft.fftfreq(n, d=1.0 / n).astype(int)
    KX, KY = np.meshgrid(ks, ks, indexing="ij")
    r = np.sqrt(KX**2 + KY**2)
    out = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            out[0] = rho[0, 0]
            continue
        shell = (r >= lag - 0.5) & (r < lag + 0.5)
        out[lag] = rho[shell].mean()
    return out


def fit_gaussian_correlation(rho: np.ndarray) -> float:
    """Least-squares fit of the lag where Gaussian covariance crosses exp(-1/2).

    rho[l] ~ exp(-l^2 / (2 lam^2)); regress log(rho) on -l^2/2.
    """
    lags = np.arange(len(rho), dtype=float)
    keep = rho > 0.05
    if keep.sum() < 3:
        return np.nan
    t = -0.5 * lags[keep] ** 2
    # log rho = slope * (-l^2/2) with slope = 1/lam^2  =>  lam = 1/sqrt(slope)
    slope = np.sum(t * np.log(rho[keep])) / np.sum(t * t)
    return 1.0 / np.sqrt(abs(slope)) if slope > 0 else np.nan
