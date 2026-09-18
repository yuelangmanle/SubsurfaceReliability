"""Cell-centered finite-volume TPFA solver for 2D steady Darcy flow.

    -div( K grad h ) = f   on [0,1]^2

Default experiment BCs: h=1 left, h=0 right, no-flow top/bottom (the classic
mean-flow-through-a-random-medium setup of the PINN/neural-solver literature).
An all-Dirichlet mode exists for manufactured-solution verification.

Returns cell-centered heads (conservative by construction) and face fluxes.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _harmonic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 2.0 * a * b / (a + b)


def assemble(
    K: np.ndarray,
    h_left: float = 1.0,
    h_right: float = 0.0,
    f: np.ndarray | None = None,
    all_dirichlet_faces: dict | None = None,
):
    """Assemble the TPFA system A h = b (same discretization as solve_darcy).

    Exposed so that a posteriori estimators can form the residual b - A h_hat
    and apply approximate inverses without re-deriving the stencil.

    Parameters
    ----------
    K : (n, n) cell-centered conductivity
    f : (n, n) cell-centered source (default zero)
    all_dirichlet_faces : if given, dict with keys 'left','right','top','bottom',
        each an (n,) array of h values AT THE FACES (manufactured-solution
        mode, all four sides Dirichlet); otherwise experiment BCs (Dirichlet
        left/right with values h_left/h_right, no-flow top/bottom) are used.
    """
    n = K.shape[0]
    d = 1.0 / n
    idx = np.arange(n * n).reshape(n, n)

    # boundary face values (imposed at the faces)
    if all_dirichlet_faces is not None:
        hL = np.asarray(all_dirichlet_faces["left"], float)
        hR = np.asarray(all_dirichlet_faces["right"], float)
        hT = np.asarray(all_dirichlet_faces["top"], float)
        hB = np.asarray(all_dirichlet_faces["bottom"], float)
    else:
        hL = np.full(n, float(h_left))
        hR = np.full(n, float(h_right))
        hT = hB = None

    # --- vectorized assembly of the TPFA stencil -------------------------
    # interior x-faces between (i,j)-(i,j+1); transmissibility = harmonic mean
    # (face area d / center distance d)
    Tx = _harmonic(K[:, :-1], K[:, 1:])
    ax = idx[:, :-1].reshape(-1)
    bx = idx[:, 1:].reshape(-1)
    tx = Tx.reshape(-1)
    Ty = _harmonic(K[:-1, :], K[1:, :])
    ay = idx[:-1, :].reshape(-1)
    by = idx[1:, :].reshape(-1)
    ty = Ty.reshape(-1)

    # per interior face (a,b,T) the four stencil entries, laid out explicitly:
    #   (a,a,+T) (b,b,+T) (a,b,-T) (b,a,-T)
    rows_i = np.concatenate([ax, bx, ax, bx, ay, by, ay, by])
    cols_i = np.concatenate([ax, bx, bx, ax, ay, by, by, ay])
    vals_i = np.concatenate([tx, tx, -tx, -tx, ty, ty, -ty, -ty])

    # boundary faces: half-distance d/2 -> transmissibility 2 K
    Tl = 2.0 * K[:, 0]
    Tr = 2.0 * K[:, -1]
    a = np.concatenate([rows_i, idx[:, 0], idx[:, -1]])
    bcol = np.concatenate([cols_i, idx[:, 0], idx[:, -1]])
    tv = np.concatenate([vals_i, Tl, Tr])
    rhs = np.zeros(n * n)
    np.add.at(rhs, idx[:, 0], Tl * hL)
    np.add.at(rhs, idx[:, -1], Tr * hR)
    if hT is not None:
        Tt, Tbb = 2.0 * K[0, :], 2.0 * K[-1, :]
        a = np.concatenate([a, idx[0, :], idx[-1, :]])
        bcol = np.concatenate([bcol, idx[0, :], idx[-1, :]])
        tv = np.concatenate([tv, Tt, Tbb])
        np.add.at(rhs, idx[0, :], Tt * hT)
        np.add.at(rhs, idx[-1, :], Tbb * hB)

    if f is not None:
        rhs += f.reshape(-1) * d * d  # cell-integrated source

    A = sp.csr_matrix((tv, (a, bcol)), shape=(n * n, n * n))
    return A, rhs, n, d


def solve_darcy(
    K: np.ndarray,
    h_left: float = 1.0,
    h_right: float = 0.0,
    f: np.ndarray | None = None,
    all_dirichlet_faces: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Solve -div(K grad h) = f with TPFA on an n x n cell grid.

    Parameters
    ----------
    K : (n, n) cell-centered conductivity
    f : (n, n) cell-centered source (default zero)
    all_dirichlet_faces : if given, dict with keys 'left','right','top','bottom',
        each an (n,) array of h values AT THE FACES (manufactured-solution
        mode, all four sides Dirichlet); otherwise experiment BCs (Dirichlet
        left/right with values h_left/h_right, no-flow top/bottom) are used.
    """
    n = K.shape[0]
    d = 1.0 / n
    idx = np.arange(n * n).reshape(n, n)

    # boundary face values (imposed at the faces)
    if all_dirichlet_faces is not None:
        hL = np.asarray(all_dirichlet_faces["left"], float)
        hR = np.asarray(all_dirichlet_faces["right"], float)
        hT = np.asarray(all_dirichlet_faces["top"], float)
        hB = np.asarray(all_dirichlet_faces["bottom"], float)
    else:
        hL = np.full(n, float(h_left))
        hR = np.full(n, float(h_right))
        hT = hB = None

    # --- vectorized assembly of the TPFA stencil -------------------------
    # interior x-faces between (i,j)-(i,j+1); transmissibility = harmonic mean
    # (face area d / center distance d)
    Tx = _harmonic(K[:, :-1], K[:, 1:])
    ax = idx[:, :-1].reshape(-1)
    bx = idx[:, 1:].reshape(-1)
    tx = Tx.reshape(-1)
    Ty = _harmonic(K[:-1, :], K[1:, :])
    ay = idx[:-1, :].reshape(-1)
    by = idx[1:, :].reshape(-1)
    ty = Ty.reshape(-1)

    # per interior face (a,b,T) the four stencil entries, laid out explicitly:
    #   (a,a,+T) (b,b,+T) (a,b,-T) (b,a,-T)
    rows_i = np.concatenate([ax, bx, ax, bx, ay, by, ay, by])
    cols_i = np.concatenate([ax, bx, bx, ax, ay, by, by, ay])
    vals_i = np.concatenate([tx, tx, -tx, -tx, ty, ty, -ty, -ty])

    # boundary faces: half-distance d/2 -> transmissibility 2 K
    Tl = 2.0 * K[:, 0]
    Tr = 2.0 * K[:, -1]
    a = np.concatenate([rows_i, idx[:, 0], idx[:, -1]])
    bcol = np.concatenate([cols_i, idx[:, 0], idx[:, -1]])
    tv = np.concatenate([vals_i, Tl, Tr])
    rhs = np.zeros(n * n)
    np.add.at(rhs, idx[:, 0], Tl * hL)
    np.add.at(rhs, idx[:, -1], Tr * hR)
    if hT is not None:
        Tt, Tbb = 2.0 * K[0, :], 2.0 * K[-1, :]
        a = np.concatenate([a, idx[0, :], idx[-1, :]])
        bcol = np.concatenate([bcol, idx[0, :], idx[-1, :]])
        tv = np.concatenate([tv, Tt, Tbb])
        np.add.at(rhs, idx[0, :], Tt * hT)
        np.add.at(rhs, idx[-1, :], Tbb * hB)

    if f is not None:
        rhs += f.reshape(-1) * d * d  # cell-integrated source

    A, rhs, n, d = assemble(K, h_left, h_right, f, all_dirichlet_faces)
    h = spla.spsolve(A, rhs)
    return h.reshape(n, n).astype(np.float64), _fluxes(K, h.reshape(n, n), d)


def _fluxes(K: np.ndarray, h: np.ndarray, d: float) -> dict:
    """Conservative face fluxes (positive in +x / +y).

    Returns dict with 'qx' (n, n+1) fluxes on vertical faces and
    'qy' (n+1, n) fluxes on horizontal faces, Darcy velocity units.
    """
    n = K.shape[0]
    # interior x-face between (i,j),(i,j+1): q = -K_harm (h1-h0)/d
    qx = np.zeros((n, n + 1))
    kh = _harmonic(K[:, :-1], K[:, 1:])
    qx[:, 1:-1] = -kh * (h[:, 1:] - h[:, :-1]) / d
    # boundary faces stay zero: flux metrics use interior faces only
    qy = np.zeros((n + 1, n))
    kv = _harmonic(K[:-1, :], K[1:, :])
    qy[1:-1, :] = -kv * (h[1:, :] - h[:-1, :]) / d
    return {"qx": qx, "qy": qy}


def cell_centered_velocity(K: np.ndarray, h: np.ndarray, d: float) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centered Darcy velocity via central differences (for error metrics).

    Applied identically to reference and neural solutions, so flux errors
    measure gradient quality, not discretization.
    """
    n = K.shape[0]
    dh_dx = np.zeros_like(h)
    dh_dy = np.zeros_like(h)
    dh_dx[:, 1:-1] = (h[:, 2:] - h[:, :-2]) / (2 * d)
    dh_dy[1:-1, :] = (h[2:, :] - h[:-2, :]) / (2 * d)
    dh_dx[:, 0] = (h[:, 1] - h[:, 0]) / d          # one-sided at Dirichlet edges
    dh_dx[:, -1] = (h[:, -1] - h[:, -2]) / d
    dh_dy[0, :] = (h[1, :] - h[0, :]) / d
    dh_dy[-1, :] = (h[-1, :] - h[-2, :]) / d
    return -K * dh_dx, -K * dh_dy
