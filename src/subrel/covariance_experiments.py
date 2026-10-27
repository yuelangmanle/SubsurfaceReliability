"""Covariance-family experiments with analytically matched integral scales.

Compares Gaussian, Matern nu=3/2, and Matern nu=1/2 log-conductivity fields
at fixed variance sigma^2 and matched 1D radial integral scale:
    L_gauss = lam * sqrt(pi / 2),  L_m05 = ell,  L_m15 = 2 * ell
so ell_m05 = lam * sqrt(pi / 2) and ell_m15 = lam * sqrt(pi / 2) / 2.

Evaluates U-Net, FNO, and PINN (on paired test realizations 288-291) and writes
metrics to runs/covariance/.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .data import generate_cell
from .pinn_solve import solve_instance_pinn
from .train_grid import train_grid_model

ROOT = Path(__file__).resolve().parents[2]
COVS = ["m15", "m05"]
CELLS = [(1.0, 0.05), (2.0, 0.03)]
INSTS = [288, 289, 290, 291]
DRY = os.environ.get("SUBREL_DRY", "0") == "1"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    log(f"covariance experiments start (dry={DRY})")
    for cov in COVS:
        for s2, lam in CELLS:
            cid = f"s{s2:g}_l{lam:g}_n128"
            h5 = ROOT / "data" / f"cell_{cid}_{cov}.h5"
            if not h5.exists():
                if DRY:
                    log(f"  PLAN datagen {cov} {cid}")
                else:
                    log(f"generate {cov} {cid}")
                    generate_cell(s2, lam, 128, 256, 32, 64, ROOT / "data", cov=cov)
            cdir = ROOT / "runs" / "covariance" / f"{cov}_{cid}"
            for fam in ("unet", "fno"):
                out = cdir / fam / "metrics.json"
                if out.exists():
                    continue
                if DRY:
                    log(f"  PLAN train {fam} {cov} {cid}")
                    continue
                log(f"train {fam} {cov} {cid}")
                train_grid_model(h5, dict(model=fam), cdir / fam)
            for inst in INSTS:
                out = cdir / "pinn" / f"pinn_inst{inst}.json"
                if out.exists():
                    continue
                if DRY:
                    log(f"  PLAN pinn {cov} {cid} #{inst - 288}")
                    continue
                log(f"pinn {cov} {cid} inst {inst}")
                solve_instance_pinn(h5, inst, dict(adam_iters=10000), cdir / "pinn")
    log("COVARIANCE INTERVENTIONS COMPLETE")


if __name__ == "__main__":
    main()
