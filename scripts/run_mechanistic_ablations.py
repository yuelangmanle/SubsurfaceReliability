"""Extended validation suite: multi-seed stability and off-grid blind testing.

Executes three validation stages:
  - Stage S (multi-seed stability): 3 representative cells x 3 model configs x 3 seeds
  - Stage B (blind batch 2): 4 off-grid parameter pairs sampling the low-variance transition region
  - Stage P (PINN blind test): PINN evaluation on extreme off-grid parameter pairs
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subrel.data import ensure_cell  # noqa: E402
from subrel.pinn_solve import solve_instance_pinn  # noqa: E402
from subrel.train_grid import DEFAULTS, train_grid_model  # noqa: E402

SEED_CELLS = [(0.5, 0.2), (1.0, 0.05), (2.0, 0.03)]
SEED_CFGS = [
    ("unet", {"model": "unet", "flux_weight": 0.0}),
    ("fno", {"model": "fno", "flux_weight": 0.0}),
    ("fno_flux001", {"model": "fno", "flux_weight": 0.01}),
]
EXTRA_SEEDS = [1, 2, 3]

BLIND_V2_CELLS = [
    (0.5, 0.10),
    (0.75, 0.12),
    (0.75, 0.20),
    (1.0, 0.16),
]

PINN_BLIND_CELLS = [(0.75, 0.26), (6.0, 0.065)]
PINN_BLIND_N_INST = 4


def cell_id(s2: float, lam: float, n: int = 128) -> str:
    return f"s{s2:g}_l{lam:g}_n{n}"


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def run_seed_stability() -> None:
    out_root = ROOT / "runs" / "seeds"
    for s2, lam in SEED_CELLS:
        cid = cell_id(s2, lam)
        h5 = ensure_cell(s2, lam, 128)
        for tag, base_cfg in SEED_CFGS:
            for seed in EXTRA_SEEDS:
                run_dir = out_root / f"{cid}_{tag}_s{seed}"
                if (run_dir / "metrics.json").exists():
                    continue
                cfg = {**DEFAULTS, **base_cfg, "train_seed": seed}
                print(f"[seeds] {cid} {tag} seed={seed}")
                train_grid_model(h5, cfg, run_dir)


def run_blind_batch2() -> None:
    out_root = ROOT / "runs" / "blind_v2"
    manifest = out_root / "manifest.json"
    if not manifest.exists():
        _atomic_write_json(manifest, {
            "cells": [{"sigma2": s, "lam": l, "cell": cell_id(s, l)}
                      for s, l in BLIND_V2_CELLS],
            "description": "Prospective off-grid blind validation Batch 2 targeting the low-variance transition region",
        })
    for s2, lam in BLIND_V2_CELLS:
        cid = cell_id(s2, lam)
        h5 = ensure_cell(s2, lam, 128)
        for model in ("unet", "fno"):
            run_dir = out_root / cid / model
            if (run_dir / "metrics.json").exists():
                continue
            print(f"[blind_v2] {cid}/{model}")
            train_grid_model(h5, {**DEFAULTS, "model": model}, run_dir)


def run_pinn_blind() -> None:
    out_root = ROOT / "runs" / "blind"
    for s2, lam in PINN_BLIND_CELLS:
        cid = cell_id(s2, lam)
        h5 = ensure_cell(s2, lam, 128)
        pinn_dir = out_root / cid / "pinn"
        pinn_dir.mkdir(parents=True, exist_ok=True)
        for k in range(PINN_BLIND_N_INST):
            inst = 288 + k
            out_json = pinn_dir / f"pinn_inst{inst}.json"
            if out_json.exists():
                continue
            print(f"[pinn_blind] {cid} inst={inst}")
            res = solve_instance_pinn(h5, inst)
            _atomic_write_json(out_json, res)


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if stage in ("all", "seeds"):
        run_seed_stability()
    if stage in ("all", "blind_v2"):
        run_blind_batch2()
    if stage in ("all", "pinn"):
        run_pinn_blind()
    print(f"Extended validation complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
