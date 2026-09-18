"""Per-cell dataset generation for the reliability atlas.

Design decision (rigor): the conductivity field is DEFINED by its Fourier
spectrum. The FV solver uses K at cell centers (exact spectral evaluation),
and the PINN optimizes the same continuous interpolant at arbitrary
collocation points -- all families therefore solve the identical problem.

Cell = (sigma2, lam) of the atlas grid. Storage: one HDF5 per cell with
Yhat (complex spectrum, float32 pairs), Y_cell, H_truth, plus metadata.
Seeds are derived from the cell parameters => fully reproducible.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np

from .field import gaussian_field
from .solver import solve_darcy

ROOT = Path(__file__).resolve().parents[2]


def cell_seed(sigma2: float, lam: float, n: int) -> int:
    """Deterministic seed from cell parameters."""
    return int(1e7 * round(sigma2, 3) + 1e5 * round(lam, 4) + n) % (2**31 - 1)


def spectrum_to_nodes(yhat: np.ndarray, nf: int) -> np.ndarray:
    """Band-limited values of the field on an nf x nf node grid (nf >= n)."""
    n = yhat.shape[0]
    big = np.zeros((nf, nf), dtype=complex)
    big[: n // 2, : n // 2] = yhat[: n // 2, : n // 2]
    big[: n // 2, -(n // 2):] = yhat[: n // 2, n // 2:]
    big[-(n // 2):, : n // 2] = yhat[n // 2:, : n // 2]
    big[-(n // 2):, -(n // 2):] = yhat[n // 2:, n // 2:]
    return np.real(np.fft.ifft2(big)) * (nf * nf) / (n * n)


def spectrum_derivative_nodes(yhat: np.ndarray, nf: int, which: str) -> np.ndarray:
    """Band-limited derivative on the nf grid; which='x' differentiates along
    array axis 1 (x), 'y' along axis 0. Angular frequencies 2 pi k."""
    n = yhat.shape[0]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    ky, kx = np.meshgrid(k, k, indexing="ij")  # kx varies along axis 1
    dyhat = 1j * (kx if which == "x" else ky) * yhat
    return spectrum_to_nodes(dyhat, nf)


def field_at_cell_centers(yhat: np.ndarray) -> np.ndarray:
    """Exact spectral values at cell centers x=(j+0.5)/n via the 2n grid."""
    n = yhat.shape[0]
    nodes2n = spectrum_to_nodes(yhat, 2 * n)
    return nodes2n[1::2, 1::2]  # odd nodes = cell centers of the n-grid


def generate_cell(sigma2: float, lam: float, n: int, n_train: int, n_val: int,
                  n_test: int, out_dir: Path | None = None,
                  cov: str = "gauss") -> Path:
    out_dir = out_dir or (ROOT / "data")
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = "" if cov == "gauss" else f"_{cov}"
    path = out_dir / f"cell_s{sigma2:g}_l{lam:g}_n{n}{sfx}.h5"
    # same seed for every covariance family: identical white noise, different
    # spectral shaping -> paired realizations across families
    rng = np.random.default_rng(cell_seed(sigma2, lam, n))
    total = n_train + n_val + n_test

    ys, hs, yhats = [], [], []
    t0 = time.time()
    for _ in range(total):
        y = gaussian_field(n, sigma2, lam, rng, cov=cov)
        yhat = np.fft.fft2(y)
        y_cc = field_at_cell_centers(yhat)
        h, _ = solve_darcy(np.exp(y_cc))
        ys.append(y_cc.astype(np.float32))
        hs.append(h.astype(np.float32))
        yhats.append(yhat.astype(np.complex64))
    gen_s = time.time() - t0

    with h5py.File(path, "w") as f:
        f.create_dataset("Y_cell", data=np.stack(ys), compression="gzip")
        f.create_dataset("H", data=np.stack(hs), compression="gzip")
        f.create_dataset("Yhat_re", data=np.stack(yhats).real, compression="gzip")
        f.create_dataset("Yhat_im", data=np.stack(yhats).imag, compression="gzip")
        f.attrs.update(dict(sigma2=sigma2, lam=lam, n=n, n_train=n_train,
                            n_val=n_val, n_test=n_test, cov=cov,
                            seed=cell_seed(sigma2, lam, n),
                            gen_seconds=round(gen_s, 2)))
    meta = dict(sigma2=sigma2, lam=lam, n=n, cov=cov, total=total,
                gen_seconds=round(gen_s, 2), path=str(path))
    (out_dir / f"cell_s{sigma2:g}_l{lam:g}_n{n}{sfx}.json").write_text(json.dumps(meta))
    return path


if __name__ == "__main__":
    import sys
    s2, lam, ntr, nv, nte, n = (float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]),
                                int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
    p = generate_cell(s2, lam, n, ntr, nv, nte)
    print("wrote", p)
