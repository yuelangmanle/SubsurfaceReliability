"""Boundary repair evaluations, training-set size sensitivity, and baseline errors.

Evaluates:
  - Soft TPFA volume-conservation fine-tuning on six regime-boundary cells
  - Residual-adaptive resampling (RAR) for PINN instances near the transition
  - Training-set size sensitivity (256 vs 128 training realizations)
  - Analytical linear head profile h_0(x) = 1 - x reference errors across cells
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .metrics import flux_error, head_error
from .repairs import conservation_finetune, solve_instance_pinn_rar
from .train_grid import train_grid_model

ROOT = Path(__file__).resolve().parents[2]

CONSFT_CELLS = [  # (sigma2, lam, model): base median flux err
    (0.5, 0.12, "unet", 1.002), (0.5, 0.08, "unet", 1.069), (1.0, 0.12, "unet", 1.781),
    (1.0, 0.08, "fno", 1.154), (1.0, 0.12, "fno", 1.089), (1.0, 0.05, "fno", 1.438),
]
RAR_CELLS = [(1.0, 0.03), (4.0, 0.08)]   # PINN boundary: base ~1.13, ~1.29
BUDGET_CELL = (0.5, 0.02)                 # hard cell, atlas used 128 train

LOG = ROOT / "runs" / "final_experiments.log"


def say(msg):
    line = f"[{__import__('time').strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def main():
    out = ROOT / "runs" / "repairs"
    out.mkdir(parents=True, exist_ok=True)

    # ---- trivial baseline for every cell (the ruler) --------------------
    say("computing trivial linear-profile baseline for all cells")
    rows = []
    for lam in [0.02, 0.03, 0.05, 0.08, 0.12, 0.2]:
        for s2 in [0.5, 1.0, 2.0, 4.0]:
            h5 = ROOT / "data" / f"cell_s{s2:g}_l{lam:g}_n128.h5"
            if not h5.exists():
                continue
            with h5py.File(h5) as f:
                n = int(f.attrs["n"])
                H, Y = f["H"][288:], f["Y_cell"][288:]
            x = (np.arange(n) + 0.5) / n
            h_lin = np.tile(1 - x, (n, 1))
            te = [head_error(h_lin, H[i]) for i in range(len(H))]
            tf = [flux_error(np.exp(Y[i]), h_lin, H[i]) for i in range(len(H))]
            rows.append(dict(sigma2=s2, lam=lam,
                             head_err=float(np.median(te)),
                             flux_err=float(np.median(tf))))
    (out / "trivial_baseline.json").write_text(json.dumps(rows, indent=1))
    say(f"trivial baseline done ({len(rows)} cells); "
        f"median flux = {np.median([r['flux_err'] for r in rows]):.2f}")

    # ---- Repair A: conservation fine-tuning on boundary cells -----------
    for s2, lam, model, base in CONSFT_CELLS:
        cid = f"s{s2:g}_l{lam:g}_n128"
        cell_h5 = ROOT / "data" / f"cell_{cid}.h5"
        base_run = ROOT / "runs" / "atlas" / cid / model
        rdir = out / f"consft_{model}_{cid}"
        if (rdir / "metrics.json").exists():
            say(f"skip consFT {model} {cid} (done)")
            continue
        say(f"consFT {model} {cid} (base flux {base})")
        r = conservation_finetune(cell_h5, base_run,
                                  cfg=dict(beta=0.3, epochs=30, lr=1e-4),
                                  run_dir=rdir)
        say(f"consFT {model} {cid}: head {r['median_head_err']:.3f} "
            f"flux {r['median_flux_err']:.3f} (base {base})")

    # ---- Repair B: RAR on PINN boundary cells ----------------------------
    for s2, lam in RAR_CELLS:
        cid = f"s{s2:g}_l{lam:g}_n128"
        cell_h5 = ROOT / "data" / f"cell_{cid}.h5"
        rdir = out / f"rar_{cid}"
        for k in range(4):
            if (rdir / f"pinn_rar_inst{288+k}.json").exists():
                continue
            say(f"RAR {cid} inst {k}")
            r = solve_instance_pinn_rar(cell_h5, 288 + k, {}, rdir)
            say(f"RAR {cid}#{k}: head {r['head_err']:.3f} flux {r['flux_err']:.3f}")
        # base comparison values
        base_dir = ROOT / "runs" / "atlas" / cid / "pinn"
        for k in range(4):
            bf = base_dir / f"pinn_inst{288+k}.json"
            if bf.exists():
                b = json.loads(bf.read_text())
                say(f"     base {cid}#{k}: head {b['head_err']:.3f} flux {b['flux_err']:.3f}")

    # ---- budget sensitivity: 256-instance training on the hard cell ------
    import os
    os.environ["SUBREL_TRAIN_CAP"] = "256"
    cid = f"s{BUDGET_CELL[0]:g}_l{BUDGET_CELL[1]:g}_n128"
    for model in ("unet", "fno"):
        rdir = out / f"budget256_{model}_{cid}"
        if (rdir / "metrics.json").exists():
            continue
        say(f"budget-256 {model} {cid}")
        r = train_grid_model(ROOT / "data" / f"cell_{cid}.h5",
                             dict(model=model, epochs=80, patience=15), rdir)
        say(f"budget-256 {model} {cid}: head {r['median_head_err']:.3f} "
            f"flux {r['median_flux_err']:.3f} (atlas-128 base: see results)")
    os.environ["SUBREL_TRAIN_CAP"] = "128"

    say("ALL FINAL EXPERIMENTS DONE")


if __name__ == "__main__":
    main()
