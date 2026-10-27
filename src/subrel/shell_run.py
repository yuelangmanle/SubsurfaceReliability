"""Train and evaluate the paired spectral-redistribution intervention."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import json

from .pinn_solve import solve_instance_pinn
from .shell import build_cell
from .train_grid import train_grid_model

ROOT = Path(__file__).resolve().parents[2]
CELL = (1.0, 0.05)
RHOS = [0.0, 0.2, 0.4]
PINN_RHOS = [0.0, 0.4]
INSTS = [288, 289, 290, 291]
DRY = os.environ.get("SUBREL_DRY", "0") == "1"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return len(json.loads(path.read_text())) > 0
    except Exception:
        return False


def main():
    s2, lam = CELL
    for rho in RHOS:
        tag = f"shell{int(rho*100):02d}"
        if rho == 0.0:
            h5 = ROOT / "data" / f"cell_s{s2:g}_l{lam:g}_n128.h5"
        else:
            h5 = build_cell(s2, lam, rho) if not DRY else None
        for fam in ("unet", "fno"):
            d = ROOT / "runs" / "shell" / f"{tag}_{fam}"
            if _valid(d / "metrics.json"):
                continue
            if DRY:
                log(f"PLAN {fam} {tag}"); continue
            log(f"train {fam} {tag}")
            train_grid_model(h5, dict(model=fam), d)
        if rho in PINN_RHOS:
            d = ROOT / "runs" / "shell" / f"{tag}_pinn"
            for inst in INSTS:
                if _valid(d / f"pinn_inst{inst}.json"):
                    continue
                if DRY:
                    log(f"PLAN pinn {tag} #{inst-288}"); continue
                log(f"pinn {tag} inst {inst}")
                solve_instance_pinn(h5, inst, dict(adam_iters=10000), d)
    log("SHELL DONE")


if __name__ == "__main__":
    main()
