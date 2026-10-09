"""Reliability atlas scan driver.

Atlas grid: sigma2 x {0.5, 1, 2, 4} x lam x {0.02, 0.03, 0.05, 0.08, 0.12, 0.2}
at n=128. Per cell: 256 train / 32 val / 64 test FV instances, one U-Net,
one FNO, and PINN on 8 test instances. Every run writes its own JSON;
existence of the JSON marks the run complete (resume = re-invoke).
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from .data import generate_cell
from .pinn_solve import solve_instance_pinn
from .train_grid import train_grid_model

ROOT = Path(__file__).resolve().parents[2]
ATLAS = dict(
    sigma2=[0.5, 1.0, 2.0, 4.0],
    lam=[0.02, 0.03, 0.05, 0.08, 0.12, 0.2],
    n=128,
    n_train=256, n_val=32, n_test=64,
    pinn_instances=8,
)
GRID_CFG = dict(epochs=80, patience=15)   # identical everywhere: no per-cell tuning
PINN_CFG = dict(adam_iters=10000, lbfgs_iters=500)


def cell_id(sigma2: float, lam: float, n: int) -> str:
    return f"s{sigma2:g}_l{lam:g}_n{n}"


def claim(path: Path) -> bool:
    """Atomically claim a work item via an exclusive .running sentinel.

    O_EXCL creation means two processes racing for the same run cannot both
    win (the loser skips); a sentinel left by a crashed process expires after
    2 h so its work is retried on a later pass.
    """
    import os
    if path.exists():
        try:
            if time.time() - path.stat().st_mtime < 7200:
                return False
            path.unlink()   # stale sentinel: reclaim (loser of the race below loses)
        except OSError:
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def run_scan(stages: tuple[str, ...] = ("data", "grid", "pinn")) -> None:
    base = ROOT / "runs" / "atlas"
    base.mkdir(parents=True, exist_ok=True)
    log = base / "scan.log"

    def say(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log, "a") as fh:
            fh.write(line + "\n")

    import os
    shard = os.environ.get("SUBREL_SHARD", "0/1")   # "k/N": worker k of N
    ki, kn = (int(x) for x in shard.split("/"))
    n = ATLAS["n"]
    k = 0
    for lam in ATLAS["lam"]:
        for sigma2 in ATLAS["sigma2"]:
            k += 1
            if k % kn != ki:
                continue
            cid = cell_id(sigma2, lam, n)
            h5 = ROOT / "data" / f"cell_{cid}.h5"
            cell_dir = base / cid

            if "data" in stages and not h5.exists():
                t0 = time.time()
                generate_cell(sigma2, lam, n, ATLAS["n_train"], ATLAS["n_val"],
                              ATLAS["n_test"])
                say(f"{cid}: dataset done ({time.time()-t0:.0f}s)")
            if "grid" in stages:
                for model in ("unet", "fno"):
                    out = cell_dir / model
                    if (out / "metrics.json").exists():
                        continue
                    if not claim(out / "train.running"):
                        continue
                    try:
                        t0 = time.time()
                        r = train_grid_model(h5, {**GRID_CFG, "model": model}, out)
                        (out / "train.running").unlink(missing_ok=True)
                        say(f"{cid}/{model}: {r['train_seconds']:.0f}s "
                            f"head={r['median_head_err']:.3f} "
                            f"flux={r['median_flux_err']:.3f}")
                    except Exception:
                        (out / "train.running").unlink(missing_ok=True)
                        say(f"{cid}/{model}: FAILED\n{traceback.format_exc()}")
            if "pinn" in stages:
                # NB: pin is deliberately NOT named k -- shadowing the outer
                # cell counter scrambles the shard filter after the first cell
                for pin in range(ATLAS["pinn_instances"]):
                    inst = ATLAS["n_train"] + ATLAS["n_val"] + pin
                    out = cell_dir / "pinn"
                    if (out / f"pinn_inst{inst}.json").exists():   # writer uses the global instance index
                        continue
                    if not claim(out / f"pinn_inst{inst}.running"):
                        continue
                    try:
                        t0 = time.time()
                        r = solve_instance_pinn(h5, inst, PINN_CFG, out)
                        (out / f"pinn_inst{inst}.running").unlink(missing_ok=True)
                        say(f"{cid}/pinn#{pin}: {time.time()-t0:.0f}s "
                            f"head={r['head_err']:.3f} flux={r['flux_err']:.3f}")
                    except Exception:
                        (out / f"pinn_inst{inst}.running").unlink(missing_ok=True)
                        say(f"{cid}/pinn#{pin}: FAILED\n{traceback.format_exc()}")


def summarize() -> None:
    import pandas as pd

    rows = []
    for mp in (ROOT / "runs" / "atlas").glob("*/**/metrics.json"):
        r = json.loads(mp.read_text())
        rows.append(dict(cell=mp.parts[-3], model=r["model"],
                         sigma2=r["sigma2"], lam=r["lam"],
                         head=r.get("median_head_err", r.get("head_err")),
                         flux=r.get("median_flux_err", r.get("flux_err"))))
    for jp in (ROOT / "runs" / "atlas").glob("*/pinn/pinn_inst*.json"):
        r = json.loads(jp.read_text())
        rows.append(dict(cell=jp.parts[-3], model="pinn",
                         sigma2=r["sigma2"], lam=r["lam"],
                         head=r["head_err"], flux=r["flux_err"]))
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(ROOT / "runs" / "atlas" / "results.csv", index=False)
        print(df.groupby(["model", "lam", "sigma2"])[["head", "flux"]].mean()
                .round(3).to_string())


if __name__ == "__main__":
    import sys
    stages = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("data", "grid", "pinn")
    if stages == ("summary",):
        summarize()
    else:
        run_scan(stages)
