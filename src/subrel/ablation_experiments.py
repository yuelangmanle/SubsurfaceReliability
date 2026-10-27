"""Controlled mechanistic ablations across benchmark regimes.

Experiment A -- Flux-regularized training: evaluates whether adding a
    conservative TPFA face-flux penalty to the training objective reduces
    interior velocity errors across smooth, transition, and rough regimes.
Experiment B -- Matched parameter capacity: compares a wide U-Net (730k
    parameters) against a narrow FNO (262k parameters) on identical media.
Experiment C -- PINN residual scaling: compares the unnormalized strong-form
    Darcy residual against the log-conductivity scaled formulation.
"""
from __future__ import annotations

import json
import json
import os
import time
from pathlib import Path

from .pinn_solve import solve_instance_pinn
from .train_grid import train_grid_model

ROOT = Path(__file__).resolve().parents[2]
DRY = os.environ.get("SUBREL_DRY", "0") == "1"

# Benchmark regimes ordered from roughest (4.0, 0.02) to smoothest (0.5, 0.20)
FLUX_CELLS = [(4.0, 0.02), (2.0, 0.03), (2.0, 0.08), (1.0, 0.05),
              (1.0, 0.12), (0.5, 0.08), (0.5, 0.2)]
# dimensionless trade-off after normalisation by the initial flux loss
FLUX_ALPHAS = [0.01, 0.1]
CAP_CELLS = [(2.0, 0.03), (1.0, 0.05), (0.5, 0.2)]
CAP_CFG = {"unet": dict(model="unet", width=40, depth=3),
           "fno": dict(model="fno", width=15, modes=12)}
RESID_CELLS = [(2.0, 0.03), (1.0, 0.05), (0.5, 0.2)]
INSTS = [288, 289, 290, 291]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _valid(path: Path) -> bool:
    """A result counts as complete only if the JSON parses and is non-empty:
    a truncated file left by a kill mid-write must be redone, not skipped."""
    if not path.exists():
        return False
    try:
        return len(json.loads(path.read_text())) > 0
    except Exception:
        return False


FAILED = []


def guarded(tag: str, fn, *a, **kw) -> None:
    """Run one unit; record outcome and continue execution."""
    try:
        fn(*a, **kw)
    except Exception as e:  # noqa: BLE001 -- isolate units, never abort the queue
        import traceback
        FAILED.append(tag)
        log(f"FAILED {tag}: {type(e).__name__}: {e}")
        (ROOT / "runs" / "ablation_failures.log").open("a").write(
            f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {tag} ===\n"
            + traceback.format_exc())


def h5(s2, lam):
    return ROOT / "data" / f"cell_s{s2:g}_l{lam:g}_n128.h5"


def queue_a():
    for s2, lam in FLUX_CELLS:
        cid = f"s{s2:g}_l{lam:g}_n128"
        for fam in ("unet", "fno"):
            for a in FLUX_ALPHAS:
                d = ROOT / "runs" / "fluxloss" / f"{fam}_a{a:g}_{cid}"
                if _valid(d / "metrics.json"):
                    continue
                if DRY:
                    log(f"PLAN fluxloss {fam} a={a} {cid}"); continue
                log(f"fluxloss {fam} a={a} {cid}")
                guarded(f"fluxloss {fam} a={a} {cid}", train_grid_model,
                        h5(s2, lam), dict(model=fam, alpha_flux=a), d)
    log("EXPERIMENT A COMPLETE")


def queue_b():
    for s2, lam in CAP_CELLS:
        cid = f"s{s2:g}_l{lam:g}_n128"
        for fam, cfg in CAP_CFG.items():
            d = ROOT / "runs" / "capacity" / f"{fam}_matched_{cid}"
            if _valid(d / "metrics.json"):
                continue
            if DRY:
                log(f"PLAN capacity {fam} {cid}"); continue
            log(f"capacity {fam} {cid}")
            guarded(f"capacity {fam} {cid}", train_grid_model, h5(s2, lam), cfg, d)
    log("EXPERIMENT B COMPLETE")


def queue_c():
    for s2, lam in RESID_CELLS:
        cid = f"s{s2:g}_l{lam:g}_n128"
        for inst in INSTS:
            d = ROOT / "runs" / "residform" / f"unnorm_{cid}"
            if _valid(d / f"pinn_inst{inst}.json"):
                continue
            if DRY:
                log(f"PLAN residform {cid} #{inst-288}"); continue
            log(f"residform {cid} inst {inst}")
            guarded(f"residform {cid} #{inst-288}", solve_instance_pinn,
                    h5(s2, lam), inst,
                    dict(adam_iters=10000, unnormalized_resid=True), d)
    log("EXPERIMENT C COMPLETE")


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "abc"
    if "a" in which: queue_a()
    if "b" in which: queue_b()
    if "c" in which: queue_c()
    log(f"ABLATION SUITE COMPLETE {which} | failed: {FAILED if FAILED else 'none'}")
