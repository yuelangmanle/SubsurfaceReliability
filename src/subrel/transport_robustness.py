"""Grid-refinement and particle-count sensitivity checks for advective transport.

Evaluates the stability of Pollock streamline particle tracking under:
  1. Spatial grid refinement: tracking the same conductivity realization at
     128x128 and 256x256 resolution.
  2. Inflow injection density: tracking 41, 81, and 161 uniformly spaced
     particles along the inflow boundary.

Output is written to runs/transport_robustness.json.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from subrel.breakthrough import TRACKER_VERSION, travel_times  # noqa: E402
from subrel.data import cell_seed, field_at_cell_centers  # noqa: E402
from subrel.field import gaussian_field  # noqa: E402
from subrel.solver import solve_darcy  # noqa: E402

OUT = ROOT / "runs" / "transport_robustness.json"


def atomic_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def reference_field(n: int, sigma2: float, lam: float, inst: int):
    rng = np.random.default_rng(cell_seed(sigma2, lam, n))
    for i in range(inst + 1):
        y = gaussian_field(n, sigma2, lam, rng, cov="gauss")
        y_cc = field_at_cell_centers(np.fft.fft2(y))
    h, _ = solve_darcy(np.exp(y_cc))
    return np.exp(y_cc.astype(np.float64)), h.astype(np.float64)


def track(K: np.ndarray, h: np.ndarray, n_particles: int) -> dict:
    global N_PARTICLES
    import subrel.breakthrough as bt
    old = bt.N_PARTICLES
    bt.N_PARTICLES = n_particles
    try:
        res = bt.travel_times(K, h)
    finally:
        bt.N_PARTICLES = old
    t = res["t"]
    fin = np.isfinite(t)
    return {"arrived": int(fin.sum()),
            "n": int(len(t)),
            "fates": {k: res["fate"].count(k) for k in sorted(set(res["fate"]))},
            "t_median_arrived": round(float(np.median(t[fin])), 4) if fin.any() else None,
            "t_q05_arrived": round(float(np.quantile(t[fin], 0.05)), 4) if fin.any() else None,
            "t_q95_arrived": round(float(np.quantile(t[fin], 0.95)), 4) if fin.any() else None}


def main() -> None:
    # guard: never wipe an existing archive on a casual re-run; the file is
    # only rebuilt when its tracker version differs (delete to force)
    if OUT.exists():
        try:
            prev = json.loads(OUT.read_text())
            if prev.get("tracker") == TRACKER_VERSION and \
                    prev.get("linear_baseline_summary", {}).get("arrived"):
                print(f"[skip] {OUT} complete for {TRACKER_VERSION}; "
                      "delete to rerun")
                return
            print(f"[regen] archived tracker/version differs; regenerating")
            # regenerating the prediction-density stage requires surrogate
            # checkpoints; without them a partial rerun would DESTROY the
            # archived evidence this file carries.  Refuse unless present.
            need = ROOT / "runs" / "atlas" / "s1_l0.05_n128" / "unet" / "model.pt"
            if not need.exists():
                raise SystemExit(
                    "[refuse] surrogate checkpoints missing (archival zip); "
                    "delete runs/transport_robustness.json explicitly if you "
                    "really want to rebuild it without them")
        except Exception:
            print("[regen] archive unreadable; regenerating")
    out: dict = {"tracker": TRACKER_VERSION}
    atomic_json(out, OUT)

    # --- grid convergence on ONE physical realization (1.0, 0.05, inst 288)
    # NOTE: cell_seed depends on n, so generating at n=256 directly yields a
    # different medium.  The physical refinement interpolates the same 128^2
    # log-conductivity field onto the 256^2 cell centres, then re-solves and
    # re-tracks.
    from scipy.interpolate import RegularGridInterpolator
    K128, h128 = reference_field(128, 1.0, 0.05, 288)
    grid = {}
    grid["n128"] = track(K128, h128, 41)
    atomic_json(out, OUT)
    print(f"grid n=128: {grid['n128']['arrived']}/{grid['n128']['n']} "
          f"t_med={grid['n128']['t_median_arrived']}")
    n = 256
    xs = (np.arange(128) + 0.5) / 128
    xf = (np.arange(n) + 0.5) / n
    Xc, Yc = np.meshgrid(xf, xf, indexing="xy")  # Xc varies along axis 1 = x
    # arrays are indexed [y, x]: the interpolant grid and the sample
    # coordinates must both be ordered (y, x).  The physical refinement path
    # interpolates log-conductivity.
    yxs = (np.arange(128) + 0.5) / 128
    interp_logk = RegularGridInterpolator(
        (yxs, yxs), np.log(K128), method="linear",
        bounds_error=False, fill_value=None)
    pts = np.stack([Yc.ravel(), Xc.ravel()], axis=-1)  # (y, x) order
    logK256 = interp_logk(pts).reshape(n, n)
    K256 = np.exp(logK256)
    h256, _ = solve_darcy(K256)
    grid["n256"] = track(K256, h256, 41)
    # alignment diagnostics on the SAME quantity (log K)
    logK128_up = interp_logk(pts).reshape(n, n)
    logK128_down = logK128_up.reshape(128, 2, 128, 2).mean(axis=(1, 3))
    corr = float(np.corrcoef(logK128_down.ravel(), np.log(K128).ravel())[0, 1])
    grid["refinement_note"] = ("n256 is the SAME 128^2 realization: log K "
                               "linearly interpolated to 256^2 cell centres "
                               "and re-solved; (y,x) coordinate order")
    grid["logK_downsample_corr_vs_128"] = round(corr, 6)
    grid["interp_max_abs_logK_err"] = round(
        float(np.abs(logK256 - logK128_up).max()), 12)
    # effective two-way check: the refined field must also survive the
    # round trip back to 128^2 with its statistics intact
    grid["refined_vs_128_rel_rmse_logK"] = round(float(
        np.sqrt(np.mean((logK128_down - np.log(K128).astype(float)) ** 2))
        / np.std(np.log(K128))), 6)
    # also track the K-interpolated variant for interpolation-path sensitivity
    interp_k = RegularGridInterpolator(
        (yxs, yxs), K128, method="linear", bounds_error=False, fill_value=None)
    K256_kpath = interp_k(pts).reshape(n, n)
    h256k, _ = solve_darcy(K256_kpath)
    grid["n256_Kinterp"] = track(K256_kpath, h256k, 41)
    atomic_json(out, OUT)
    print(f"grid n=256 (refined same field): {grid['n256']['arrived']}/{grid['n256']['n']} "
          f"t_med={grid['n256']['t_median_arrived']}")
    if grid["n128"]["t_median_arrived"] and grid["n256"]["t_median_arrived"]:
        shift = abs(grid["n256"]["t_median_arrived"] / grid["n128"]["t_median_arrived"] - 1)
        grid["median_shift_fraction"] = round(shift, 4)
        print(f"grid median shift: {shift:.3f}")
    out["grid_convergence"] = grid

    # --- injection density on three archived instances (reference fields)
    dens = {}
    for sigma2, lam, inst in ((1.0, 0.05, 288), (2.0, 0.03, 288), (0.5, 0.08, 288)):
        K, h = reference_field(128, sigma2, lam, inst)
        key = f"s{sigma2:g}_l{lam:g}_i{inst}"
        dens[key] = {}
        for np_ in (41, 81, 161):
            dens[key][f"p{np_}"] = track(K, h, np_)
            atomic_json(out, OUT)
        print(f"density {key}: " + " ".join(
            f"p{k}={v['arrived']}/{v['n']}" for k, v in dens[key].items()))
    out["injection_density"] = dens

    # --- density sensitivity on PREDICTED fields (the fate shares, not the
    # reference, are where particle injection density is most sensitive)
    from subrel.breakthrough import load_pred, generate_one  # noqa: F811
    pred_dens = {}
    sigma2, lam, inst = 1.0, 0.05, 288
    ymu = ystd = 0.0
    rng = np.random.default_rng(cell_seed(sigma2, lam, 128))
    ys = []
    for _ in range(256):
        y = gaussian_field(128, sigma2, lam, rng, cov="gauss")
        ys.append(field_at_cell_centers(np.fft.fft2(y)).astype(np.float32))
    Ytr = np.stack(ys)
    ymu, ystd = float(Ytr.mean()), float(Ytr.std())
    del Ytr, ys
    Y, H = generate_one(sigma2, lam, inst)
    K = np.exp(Y.astype(np.float64))
    for name in ("unet", "fno_head", "fno_flux"):
        hp = load_pred(name, sigma2, lam, Y, ymu, ystd)
        key = f"s{sigma2:g}_l{lam:g}_i{inst}_{name}"
        pred_dens[key] = {}
        for np_ in (41, 81, 161):
            pred_dens[key][f"p{np_}"] = track(K, hp, np_)
            atomic_json(out, OUT)
        print(f"pred-density {key}: " + " ".join(
            f"p{k}={v['arrived']}/{v['n']}" for k, v in pred_dens[key].items()))
    out["injection_density_predictions"] = pred_dens

    # --- linear-head control h0 = 1-x: a zero-training baseline that
    # achieves full passage by construction -- arrival fraction alone
    # cannot establish transport fidelity.  Same 24 realizations, same
    # tracker, same injection rule.
    from subrel.breakthrough import load_pred  # noqa: F811
    lin = {}
    n128 = 128
    for sigma2, lam in ((1.0, 0.05), (2.0, 0.03), (0.5, 0.08)):
        cell = f"s{sigma2:g}_l{lam:g}_n128"
        for inst in (288, 296, 304, 312, 320, 328, 336, 344):
            K, H = reference_field(n128, sigma2, lam, inst)
            h0 = np.broadcast_to(1.0 - (np.arange(n128) + 0.5) / n128,
                                 (n128, n128)).copy()
            import subrel.breakthrough as _bt
            res = _bt.travel_times(K, h0)
            # paired ratio vs the true reference solution
            ts_ref = _bt.travel_times(K, H)
            both = np.isfinite(res["t"]) & np.isfinite(ts_ref["t"])
            pr = float(np.median(res["t"][both] / ts_ref["t"][both])) if both.any() else None
            # its own flux/head errors against the true solution
            from subrel.metrics import head_error, interior
            from subrel.solver import cell_centered_velocity
            vx_t, vy_t = cell_centered_velocity(K, H, 1.0 / n128)
            vx_p, vy_p = cell_centered_velocity(K, h0, 1.0 / n128)
            sl = interior(n128)
            fe = ((float(np.linalg.norm((vx_p - vx_t)[sl]))
                   + float(np.linalg.norm((vy_p - vy_t)[sl])))
                  / (float(np.linalg.norm(vx_t[sl])) + float(np.linalg.norm(vy_t[sl]))))
            he = head_error(h0, H)
            fin = np.isfinite(res["t"])
            lin.setdefault(cell, {})[str(inst)] = {
                "arrived": int(fin.sum()), "n": int(len(res["t"])),
                "paired_ratio": round(pr, 4) if pr else None,
                "flux_err": round(fe, 4), "head_err": round(he, 4)}
            atomic_json(out, OUT)
    arrs = [v["arrived"] for c in lin.values() for v in c.values()]
    prs = [v["paired_ratio"] for c in lin.values() for v in c.values() if v["paired_ratio"]]
    out["linear_baseline_h0"] = lin
    out["linear_baseline_summary"] = {
        "arrived": sum(arrs), "n_particles": sum(v["n"] for c in lin.values() for v in c.values()),
        "paired_ratio_median": round(float(np.median(prs)), 4) if prs else None,
        "paired_ratio_min": round(float(min(prs)), 4) if prs else None,
        "paired_ratio_max": round(float(max(prs)), 4) if prs else None,
        "flux_err_median": round(float(np.median([v["flux_err"] for c in lin.values() for v in c.values()])), 4),
        "head_err_median": round(float(np.median([v["head_err"] for c in lin.values() for v in c.values()])), 4)}
    print("linear h0 baseline:", out["linear_baseline_summary"])

    atomic_json(out, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
