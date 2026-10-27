"""Hydrological consequence of flux error: advective travel times.

Head-accurate flux-blind surrogates can carry large flux errors; this script
turns that abstract error into the quantity a water-resources user consumes:
particle arrival times at the outflow boundary.

For each selected (cell, instance) the medium and the reference head are
regenerated deterministically (same code path as recompute_dual), and the
head fields are compared through the SAME transport operator:

- solver reference head,
- the atlas U-Net prediction (head-trained, head-accurate),
- the atlas FNO prediction (head-trained -- the within-architecture
  control that separates "some FNO prediction" from "the flux objective"),
- the flux-weighted FNO prediction (runs/fluxloss, alpha=0.01).

Particles are injected uniformly on the inflow edge and advected with the
semi-analytic Pollock scheme through the TPFA face fluxes of each head
field.  Output: runs/breakthrough.json.  Idempotent per tracker version.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from subrel.data import cell_seed, field_at_cell_centers  # noqa: E402
from subrel.field import gaussian_field  # noqa: E402
from subrel.metrics import head_error, interior  # noqa: E402
from subrel.models import FNO2d, UNet  # noqa: E402
from subrel.solver import cell_centered_velocity, solve_darcy  # noqa: E402

OUT = ROOT / "runs" / "breakthrough.json"
TRACKER_VERSION = ("pollock-2d-v2: two-coordinate analytic advance, "
                   "relative flatness, full-y domain")
N = 128
N_PARTICLES = 41
# fixed a priori selection rule: every 8th test instance (indices 0, 8,
# 16, ... of the 64-instance test slice), set before any transport run;
# four were computed in the first pass, four extend the sample
INSTANCES = (288, 296, 304, 312, 320, 328, 336, 344)
CELLS = ((1.0, 0.05), (2.0, 0.03), (0.5, 0.08))  # mid / rough / boundary
ALPHA = "0.01"  # flux-weighted FNO weight


def atomic_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def generate_one(sigma2: float, lam: float, inst: int):
    """Regenerate one medium and its reference solution deterministically."""
    rng = np.random.default_rng(cell_seed(sigma2, lam, N))
    total_here = inst + 1  # advance the rng through instances 0..inst
    for i in range(total_here):
        y = gaussian_field(N, sigma2, lam, rng, cov="gauss")
        yhat = np.fft.fft2(y)
        y_cc = field_at_cell_centers(yhat)
        if i == inst:
            h, _ = solve_darcy(np.exp(y_cc))
            return y_cc.astype(np.float32), h.astype(np.float32)
    raise RuntimeError("unreachable")


def face_fluxes(K: np.ndarray, h: np.ndarray):
    """Conservative TPFA face fluxes (Darcy velocity units, +x/+y positive)."""
    from subrel.solver import _fluxes
    return _fluxes(K, h, 1.0 / K.shape[0])


def _advance(s: float, v0: float, v1: float, d: float, dt: float) -> float:
    """Exact within-cell displacement over a time dt.

    The velocity varies linearly from v0 (at s=0) to v1 (at s=d):
    ds/dt = v0 + (v1-v0) s / d.  Both coordinates of a particle must be
    advanced with this SAME dt -- advancing only the crossed coordinate
    was the defect that made diagonal uniform flow come out row-locked.
    """
    k = v1 - v0
    if abs(k) <= 1e-10 * max(abs(v0), abs(v1), 1e-30):
        return s + v0 * dt
    return (s + d * v0 / k) * float(np.exp(k * dt / d)) - d * v0 / k


def _t_to_face(s: float, v0: float, v1: float, d: float,
               to_right: bool) -> float:
    """Pollock crossing time to a cell face.

    A particle sits at offset ``s`` from the LEFT face of a cell of length
    ``d`` in which the velocity varies linearly from ``v0`` (left face) to
    ``v1`` (right face).  Returns the exact time to reach the right face
    (``to_right=True``) or the left face; ``inf`` when the velocity field
    never carries the particle there (it can only approach an interior
    stagnation point).
    """
    k = v1 - v0
    vs = v0 + k * s / d  # velocity at the particle position
    # relative flatness: face fluxes reconstructed from float64 heads carry
    # ~1e-15 noise, so an absolute threshold misroutes uniform flow into
    # the ill-conditioned log branch (d/k ~ 1e12)
    # 1e-10 relative: far above float64 reconstruction noise (~1e-15) yet
    # far below any physically meaningful velocity gradient; raising it to
    # ~1e-8 would only matter for conditioning in (1e-10, 1e-8), where the
    # log branch loses ~7e-7 relative accuracy -- physically negligible.
    flat = abs(k) <= 1e-10 * max(abs(v0), abs(v1), 1e-30)
    if flat:
        if to_right:
            return (d - s) / v0 if v0 > 0 else float("inf")
        return s / (-v0) if v0 < 0 else float("inf")
    if to_right:
        if vs <= 0 or v1 <= 0:
            return float("inf")
        return d / k * float(np.log(v1 / vs))
    if vs >= 0 or v0 >= 0:
        return float("inf")
    return d / k * float(np.log(v0 / vs))


def travel_times(K: np.ndarray, h: np.ndarray) -> dict:
    """Two-dimensional Pollock particle tracking through TPFA face fluxes.

    Particles are injected uniformly on the inflow edge of the interior
    region (the same 2-cell-halo interior every flux metric uses) and
    advected with the semi-analytic cell-to-cell scheme: inside each cell
    the velocity varies linearly between the exact face fluxes, so crossing
    times are mesh-exact and no time step is introduced.  Outcomes:
    ``arrived`` (reached the interior outflow boundary), ``backwashed``
    (left through the interior inflow boundary), ``stalled`` (the cell's
    face fluxes offer no reachable exit -- possible for a predicted field,
    whose face fluxes are not divergence-free), or ``censored`` (maximum
    physical time).
    """
    n = K.shape[0]
    d = 1.0 / n
    F = face_fluxes(K, h)
    fx, fy = F["qx"], F["qy"]  # fx (n,n+1) x-faces; fy (n+1,n) y-faces
    lo, hi = 2.0 * d, 1.0 - 2.0 * d
    TMAX = 50.0  # physical censoring time (pore-volume units, phi = 1)
    tat = np.full(N_PARTICLES, np.nan)
    y_exit = np.full(N_PARTICLES, np.nan)
    fate = ["active"] * N_PARTICLES
    for p in range(N_PARTICLES):
        # y spans the FULL physical domain: the no-flow edges y=0,1 are
        # enforced by qy[0]=qy[n]=0, and the interior-band y-boundaries are
        # not physical -- reference streamlines legitimately cross them on
        # their way to the outflow.  Only the x-direction is confined to
        # the evaluation band [lo, hi] (arrival / backwash / x-stall).
        x = lo + 1e-12
        y = (p + 0.5) / N_PARTICLES
        t = 0.0
        for _ in range(2000):
            iy = min(max(int(y * n), 0), n - 1)
            jx = min(max(int(x * n), 2), n - 3)
            sx, sy = x - jx * d, y - iy * d
            vx0, vx1 = fx[iy, jx], fx[iy, jx + 1]
            vy0, vy1 = fy[iy, jx], fy[iy + 1, jx]
            cand = (
                _t_to_face(sx, fx[iy, jx], fx[iy, jx + 1], d, True),    # right
                _t_to_face(sx, fx[iy, jx], fx[iy, jx + 1], d, False),   # left
                _t_to_face(sy, fy[iy, jx], fy[iy + 1, jx], d, True),    # top
                _t_to_face(sy, fy[iy, jx], fy[iy + 1, jx], d, False),   # bottom
            )
            tmin = min(cand)
            if not np.isfinite(tmin):
                fate[p] = "stalled"
                break
            if t + tmin > TMAX:
                fate[p] = "censored"
                break
            t += tmin
            w = cand.index(tmin)
            # advance BOTH coordinates over the same dt (Pollock), then
            # snap the crossed coordinate exactly onto its face
            sx_new = _advance(sx, vx0, vx1, d, tmin)
            sy_new = _advance(sy, vy0, vy1, d, tmin)
            if w in (0, 1):
                sx_new = d if w == 0 else 0.0
            else:
                sy_new = d if w == 2 else 0.0
            x = jx * d + min(max(sx_new, 0.0), d)
            y = iy * d + min(max(sy_new, 0.0), d)
            y = min(max(y, 0.0), 1.0 - 1e-12)  # physical no-flow edges
            # nudge off the shared face so the next lookup enters the
            # neighbouring cell cleanly
            if w == 0:
                x += 1e-12
            elif w == 1:
                x -= 1e-12
            elif w == 2:
                y += 1e-12
            else:
                y -= 1e-12
            if w == 0 and (jx + 1) * d >= hi - 1e-12:
                fate[p] = "arrived"
                tat[p] = t
                y_exit[p] = y
                break
            if w == 1 and jx * d <= lo + 1e-12:
                fate[p] = "backwashed"
                break
        else:
            fate[p] = "censored"
        if fate[p] == "active":
            fate[p] = "censored"
    return {"t": tat, "fate": fate, "y_exit": y_exit, "tmax": TMAX}


def load_pred(model_name: str, sigma2: float, lam: float, Y: np.ndarray,
              ymu: float, ystd: float) -> np.ndarray:
    """fno_head: the ATLAS head-trained FNO -- same architecture, data,
    budget and seed as the flux-weighted model, differing only in the
    objective.  This is the within-architecture control that separates
    'some FNO prediction' from 'the flux objective' in transport."""
    if model_name == "unet":
        ckpt = ROOT / "runs" / "atlas" / f"s{sigma2:g}_l{lam:g}_n128" / "unet" / "model.pt"
        net = UNet(cin=3, width=24, depth=3)
    elif model_name == "fno_head":
        ckpt = ROOT / "runs" / "atlas" / f"s{sigma2:g}_l{lam:g}_n128" / "fno" / "model.pt"
        net = FNO2d(cin=3, width=24, modes=12)
    else:  # flux-weighted FNO
        ckpt = ROOT / "runs" / "fluxloss" / f"fno_a{ALPHA}_s{sigma2:g}_l{lam:g}_n128" / "model.pt"
        net = FNO2d(cin=3, width=24, modes=12)
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()
    n = Y.shape[-1]
    x = (np.arange(n) + 0.5) / n
    Xc, Yc = np.meshgrid(x, x, indexing="xy")
    z = (Y[None] - ymu) / ystd
    X = np.stack([z[0], Xc, Yc], axis=0).astype(np.float32)[None]
    with torch.no_grad():
        pred = net(torch.from_numpy(X)).squeeze().numpy()
    return pred.astype(np.float64) * 0.5 + 0.5


def summarize(ref: dict, ts: dict) -> dict:
    """Arrival statistics against the reference, over ALL particles.

    The arrived-fraction and the fate counts are first-class outputs: with
    censored/stalled/backwashed particles present, an arrived-only median
    ratio would silently condition on survival, so the raw fates are
    reported alongside and the time ratios are explicitly labelled as
    arrived-particle statistics.
    """
    rt, tt = ref["t"], ts["t"]
    fin = np.isfinite(tt)
    q = lambda a, p: float(np.nanquantile(a, p))
    med = float(np.nanmedian(tt)) if fin.any() else float("nan")
    med_r = float(np.nanmedian(rt))
    # paired per-particle ratio over COMMONLY ARRIVED particles; the
    # arrived-only median ratio below is survivor-biased and kept
    # separately labelled
    both = fin & np.isfinite(rt)
    paired = float(np.median(tt[both] / rt[both])) if both.any() else None
    fates = {k: int(ts["fate"].count(k)) for k in sorted(set(ts["fate"]))}
    ye = ts.get("y_exit")
    return {"arrived": int(fin.sum()),
            "fates": fates,
            "n_paired_arrived": int(both.sum()),
            "median_paired_ratio": round(paired, 4) if paired is not None else None,
            "t_median_arrived": round(med, 4) if fin.any() else None,
            "t_q05_arrived": round(q(tt, 0.05), 4) if fin.any() else None,
            "t_q95_arrived": round(q(tt, 0.95), 4) if fin.any() else None,
            "median_ratio_arrived_vs_ref": round(med / med_r, 4) if fin.any() else None,
            "y_exit_q05": round(float(np.nanquantile(ye, 0.05)), 4) if ye is not None and np.isfinite(ye).any() else None,
            "y_exit_q95": round(float(np.nanquantile(ye, 0.95)), 4) if ye is not None and np.isfinite(ye).any() else None,
            "tmax": ts["tmax"]}


def selftest() -> bool:
    """Two analytic gates the tracker must pass exactly.

    1. Uniform K, h = 1-x: unit uniform +x velocity; every particle
       arrives with the exact time hi - lo.
    2. Uniform K, h = 1 - x - a*y: uniform diagonal velocity (1, a) with
       |a| small enough that no streamline reaches a physical edge.  Every
       particle must arrive with the exact time hi-lo and the exact exit
       elevation y0 + a*(hi-lo).  The exit coordinate is the decisive
       check: the defective tracker that advanced only the crossed
       coordinate passed all arrival counts here while carrying every
       particle along a horizontal line (y_exit = y0 for all p).
    """
    n = 64
    d = 1.0 / n
    lo, hi = 2.0 * d, 1.0 - 2.0 * d
    K = np.ones((n, n))
    x = (np.arange(n) + 0.5) / n

    ok = True
    h1 = np.broadcast_to(1.0 - x[None, :], (n, n)).copy()
    r1 = travel_times(K, h1)
    err1 = float(np.nanmax(np.abs(r1["t"] - (hi - lo))))
    gate1 = err1 < 1e-9 and all(f == "arrived" for f in r1["fate"])
    print(f"  horizontal selftest: max |t-(hi-lo)| = {err1:.2e} "
          f"-> {'PASS' if gate1 else 'FAIL'}")
    ok &= gate1

    for a in (-0.05, 0.02, 0.05):
        h2 = (1.0 - x[None, :]) - a * x[:, None]
        r2 = travel_times(K, np.ascontiguousarray(h2))
        y0s = np.array([(p + 0.5) / N_PARTICLES for p in range(N_PARTICLES)])
        got_arr = np.array([f == "arrived" for f in r2["fate"]])
        t_err = float(np.nanmax(np.abs(r2["t"][got_arr] - (hi - lo)))) if got_arr.any() else np.inf
        y_ana = y0s + a * (hi - lo)
        # streamlines whose whole path stays clear of the boundary cells:
        # there the linear reconstruction is exact.  Inside the boundary
        # cells the reconstructed velocity ramps to zero at the no-flow
        # edge (standard Pollock/MODPATH behaviour; this synthetic field
        # violates the no-flow condition, so the ramp is a reconstruction
        # artifact), and for those particles we only require a legal exit.
        inside = (np.minimum(y0s, y_ana) > d) & (np.maximum(y0s, y_ana) < 1.0 - d)
        inside_idx = got_arr & inside
        ye_err = float(np.nanmax(np.abs(
            r2["y_exit"][inside_idx] - y_ana[inside_idx]))) if inside_idx.any() else np.inf
        edge_ok = all(0.0 <= r2["y_exit"][p] <= 1.0
                      for p in range(N_PARTICLES) if got_arr[p] and not inside[p])
        gate = (got_arr.all() and t_err < 1e-9 and ye_err < 1e-9 and edge_ok)
        print(f"  diagonal a={a:g}: arrived {int(got_arr.sum())}/{N_PARTICLES}, "
              f"max |t-(hi-lo)| = {t_err:.2e}, interior max |y_exit-y_ana| = {ye_err:.2e}, "
              f"edge exits legal = {edge_ok} -> {'PASS' if gate else 'FAIL'}")
        ok &= gate
    return ok


def main() -> None:
    out = {"n_particles": N_PARTICLES, "instances": INSTANCES,
           "tracker": TRACKER_VERSION,
           "cells": {}, "note": "arrival times in pore-volume units (phi=1); "
           "TPFA face fluxes reconstructed from each head field -- local "
           "balance is not guaranteed for surrogate predictions"}
    if OUT.exists():
        out = json.loads(OUT.read_text())
        if out.get("tracker") == TRACKER_VERSION:
            print(f"[skip] results already present for {TRACKER_VERSION}")
            return
        if "tracker" in out:
            print(f"[regen] archived tracker {out.get('tracker')!r} != "
                  f"{TRACKER_VERSION!r}; regenerating")
        out = {"n_particles": N_PARTICLES, "instances": INSTANCES,
               "tracker": TRACKER_VERSION, "cells": {}, "note": out.get("note")}
    out["cells"] = {}
    ok = selftest()
    if not ok:
        raise SystemExit("selftest failed; refusing to produce results")
    for sigma2, lam in CELLS:
        cell = f"s{sigma2:g}_l{lam:g}_n128"
        out["cells"].setdefault(cell, {})
        if all(k in out["cells"][cell] for k in map(str, INSTANCES)):
            print(f"[skip] {cell}")
            continue
        rng = np.random.default_rng(cell_seed(sigma2, lam, N))
        ys = []
        for _ in range(256):
            y = gaussian_field(N, sigma2, lam, rng, cov="gauss")
            ys.append(field_at_cell_centers(np.fft.fft2(y)).astype(np.float32))
        Ytr = np.stack(ys)
        ymu, ystd = float(Ytr.mean()), float(Ytr.std())
        del Ytr, ys

        for inst in INSTANCES:
            t0 = time.time()
            Y, H = generate_one(sigma2, lam, inst)
            K = np.exp(Y.astype(np.float64))
            rec = {"flux_err": {}, "head_err": {}, "travel": {}}
            fields = {"solver": H.astype(np.float64)}
            fields["unet"] = load_pred("unet", sigma2, lam, Y, ymu, ystd)
            fields["fno_head"] = load_pred("fno_head", sigma2, lam, Y, ymu, ystd)
            fields["fno_flux"] = load_pred("fno", sigma2, lam, Y, ymu, ystd)
            vx_t, vy_t = cell_centered_velocity(K, H.astype(np.float64), 1.0 / N)
            s = interior(N)
            for name, hp in fields.items():
                rec["head_err"][name] = round(head_error(hp, H.astype(np.float64)), 4)
                if name != "solver":
                    vx_p, vy_p = cell_centered_velocity(K, hp, 1.0 / N)
                    rec["flux_err"][name] = round(
                        (float(np.linalg.norm((vx_p - vx_t)[s]))
                         + float(np.linalg.norm((vy_p - vy_t)[s])))
                        / (float(np.linalg.norm(vx_t[s])) + float(np.linalg.norm(vy_t[s]))), 4)
            ts_ref = travel_times(K, H.astype(np.float64))
            rec["travel"]["solver"] = summarize(ts_ref, ts_ref)
            for name in ("unet", "fno_head", "fno_flux"):
                rec["travel"][name] = summarize(ts_ref, travel_times(K, fields[name]))
            out["cells"][cell][str(inst)] = rec
            atomic_json(out, OUT)
            u = rec["travel"]["unet"]
            fh = rec["travel"]["fno_head"]
            f = rec["travel"]["fno_flux"]
            print(f"[done] {cell}/{inst} in {time.time() - t0:.1f}s "
                  f"unet arr={u['arrived']}/41 fno_head arr={fh['arrived']}/41 "
                  f"fno_flux arr={f['arrived']}/41", flush=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
