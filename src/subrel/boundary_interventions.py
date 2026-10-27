"""Boundary repair experiments, grid-refinement checks, and covariance tests.

Executes sequentially and writes each output file atomically so interrupted
runs can resume without repeating completed evaluations:
  - Standard residual-adaptive resampling (RAR) for PINN instances
  - Soft TPFA volume-conservation fine-tuning and matched-budget MSE controls
  - Finite-volume grid-resolution comparison (128x128 vs 256x256)
  - Covariance-family comparisons (Matern nu=1/2 and nu=3/2)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import h5py
import numpy as np

from .data import generate_cell, spectrum_to_nodes
from .metrics import flux_error, head_error
from .pinn_solve import solve_instance_pinn
from .repairs import conservation_finetune, solve_instance_pinn_rarstd
from .solver import solve_darcy
from .train_grid import train_grid_model

ROOT = Path(__file__).resolve().parents[2]
DRY = os.environ.get("SUBREL_DRY", "0") == "1"
PINN_CFG = dict(adam_iters=10000)   # match the atlas protocol exactly

RAR_CELLS = [(1.0, 0.03), (4.0, 0.08)]
RAR_INSTS = [288, 289, 290, 291]
CONSFT = [("unet", 0.5, 0.08), ("unet", 0.5, 0.12), ("unet", 1.0, 0.12),
          ("fno", 1.0, 0.05), ("fno", 1.0, 0.08), ("fno", 1.0, 0.12)]
GRID_CELLS = [(4.0, 0.02), (1.0, 0.03), (0.5, 0.2)]
COVS = [("m05", 0.5), ("m15", 1.5)]
COV_CELLS = [(1.0, 0.05), (2.0, 0.03)]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cid_of(s2: float, lam: float) -> str:
    return f"s{s2:g}_l{lam:g}_n128"


def cell_h5(s2: float, lam: float, cov: str = "gauss") -> Path:
    sfx = "" if cov == "gauss" else f"_{cov}"
    return ROOT / "data" / f"cell_s{s2:g}_l{lam:g}_n128{sfx}.h5"


def unit(path: Path, name: str) -> bool:
    """True if this unit should run (result file missing)."""
    if path.exists():
        if DRY:
            log(f"  skip  {name}")
        return False
    if DRY:
        log(f"  PLAN  {name}")
    return True


def stage_b1() -> None:
    log("B1: standard RAR")
    for s2, lam in RAR_CELLS:
        cid = cid_of(s2, lam)
        rdir = ROOT / "runs" / "repairs" / f"rarstd_{cid}"
        for inst in RAR_INSTS:
            out = rdir / f"pinn_rarstd_inst{inst}.json"
            if not unit(out, f"rarstd {cid} #{inst - 288}"):
                continue
            if DRY:
                continue
            log(f"rarstd {cid} inst {inst}")
            solve_instance_pinn_rarstd(cell_h5(s2, lam), inst, PINN_CFG, rdir)


def stage_b2() -> None:
    log("Stage 2: conservation fine-tuning and matched-budget MSE continuation control")
    for fam, s2, lam in CONSFT:
        cid = cid_of(s2, lam)
        base = ROOT / "runs" / "atlas" / cid / fam
        for tag, beta in [("consft2", 0.3), ("mse2", 0.0)]:
            rdir = ROOT / "runs" / "repairs" / f"{tag}_{fam}_{cid}"
            if not unit(rdir / "metrics.json", f"{tag} {fam} {cid}"):
                continue
            if DRY:
                continue
            log(f"{tag} {fam} {cid}")
            conservation_finetune(cell_h5(s2, lam), base,
                                  cfg=dict(beta=beta, epochs=30, lr=1e-4,
                                           ft_n=128),
                                  run_dir=rdir)


def stage_b3() -> None:
    log("B3: grid-resolution check")
    out = ROOT / "runs" / "repairs" / "grid_check.json"
    if not unit(out, "grid check"):
        return
    if DRY:
        return
    rows = []
    for s2, lam in GRID_CELLS:
        cid = cid_of(s2, lam)
        with h5py.File(cell_h5(s2, lam), "r") as f:
            yhats = f["Yhat_re"][-16:] + 1j * f["Yhat_im"][-16:]
            H128 = f["H"][-16:]
            Y128 = f["Y_cell"][-16:]
        he, fe = [], []
        for yhat, h128, y128 in zip(yhats, H128, Y128):
            # the SAME band-limited realization evaluated on a 2x finer grid:
            # odd nodes of a 512-grid are the cell centers of a 256-grid
            nodes = spectrum_to_nodes(yhat, 512)[1::2, 1::2]
            h256, _ = solve_darcy(np.exp(nodes))
            # bilinear cell-average of the fine solution onto the 128 grid
            h_coarse = h256.reshape(128, 2, 128, 2).mean(axis=(1, 3))
            he.append(head_error(h128, h_coarse))
            fe.append(flux_error(np.exp(y128), h128, h_coarse))
        row = dict(sigma2=s2, lam=lam,
                   median_head_128vs256=float(np.median(he)),
                   median_flux_128vs256=float(np.median(fe)))
        rows.append(row)
        log(f"grid {cid}: head {row['median_head_128vs256']:.4f} "
            f"flux {row['median_flux_128vs256']:.4f}")
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows, indent=1))
    os.replace(tmp, out)


def stage_c() -> None:
    log("C: covariance intervention (spectral shape at fixed sigma2, integral scale)")
    for cov, _nu in COVS:
        for s2, lam in COV_CELLS:
            cid = cid_of(s2, lam)
            h5 = cell_h5(s2, lam, cov)
            if not h5.exists():
                if DRY:
                    log(f"  PLAN  datagen {cov} {cid}")
                else:
                    log(f"generate {cov} {cid}")
                    generate_cell(s2, lam, 128, 256, 32, 64, ROOT / "data", cov=cov)
            cdir = ROOT / "runs" / "cov" / f"{cov}_{cid}"
            for fam in ("unet", "fno"):
                if unit(cdir / fam / "metrics.json", f"{fam} {cov} {cid}"):
                    if not DRY:
                        log(f"train {fam} {cov} {cid}")
                        train_grid_model(h5, dict(model=fam), cdir / fam)
            for i in range(4):
                inst = 288 + i
                if unit(cdir / "pinn" / f"pinn_inst{inst}.json",
                        f"pinn {cov} {cid} #{i}"):
                    if not DRY:
                        log(f"pinn {cov} {cid} inst {inst}")
                        solve_instance_pinn(h5, inst, PINN_CFG, cdir / "pinn")


def main() -> None:
    log(f"boundary_interventions start (dry={DRY})")
    stage_b1()
    stage_b2()
    stage_b3()
    stage_c()
    log("BOUNDARY INTERVENTIONS COMPLETE")


if __name__ == "__main__":
    main()
