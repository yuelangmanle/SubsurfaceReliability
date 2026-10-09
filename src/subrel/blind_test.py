"""Pre-registered blind test of the screening score at new parameter values.

Protocol (executed strictly in this order):

0. FREEZE: descriptor (sigma^2/lambda for the U-Net, sigma^2 for the FNO) and
   threshold (Youden J on the full 24-regime atlas) are committed to
   runs/blind_test.json BEFORE any new cell is generated or trained.
1. TRAIN: eight parameter pairs that appear nowhere in the atlas grid (six
   interior interpolations, two mild extrapolations at sigma^2=6 and
   lambda=0.26) get their own field set and their own surrogate, trained with
   the atlas budget.  Idempotent: a cell/model whose metrics.json exists is
   skipped, so the run is safe to interrupt and restart.
2. TEST: the frozen rule predicts each cell's failure class from the analytic
   descriptor alone; the trained surrogate's measured median flux error
   decides the truth.  Reported per family against the trivial baseline.

Output: runs/blind_test.json (atomic, incremental).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from subrel.data import cell_seed, generate_cell  # noqa: E402
from subrel.train_grid import DEFAULTS, train_grid_model  # noqa: E402

OUT = ROOT / "runs" / "blind_test.json"
N = 128

# six interior interpolations + two mild extrapolations, all off the atlas grid
BLIND_CELLS = [
    (0.75, 0.04), (0.75, 0.16), (1.5, 0.025), (1.5, 0.09),
    (3.0, 0.065), (3.0, 0.16), (6.0, 0.065), (0.75, 0.26),
]

DESC = {"unet": "s2_over_lam", "fno": "sigma2"}


def atomic_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def youden_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    s, y = scores[order], labels[order]
    cands = [s[0] - 1.0]
    cands += [0.5 * (s[i] + s[i + 1]) for i in range(len(s) - 1) if s[i + 1] > s[i]]
    cands.append(s[-1] + 1.0)
    best_thr, best_j = cands[0], -2.0
    for t in cands:
        pred = s >= t
        j = float(pred[y == 1].mean()) - float(pred[y == 0].mean())
        if j > best_j:
            best_thr, best_j = float(t), j
    return best_thr


def freeze_calibration() -> dict:
    """Commit the frozen rule to disk before anything new is trained.

    The threshold is calibrated on each family's own regime labels across the
    24-regime atlas: sigma^2/lambda >= 3.3333 for the U-Net (J = 1.000) and
    sigma^2 >= 0.75 for the FNO (J = 0.9583)."""
    df = pd.read_csv(ROOT / "runs" / "atlas" / "instances_with_predictors.csv")
    thr = {}
    details = {}
    for m, d in DESC.items():
        fam = df[df.model == m]
        reg = (fam.groupby("cell")
                  .agg(flux_median=("flux_err", "median"), sigma2=("sigma2", "first"),
                       lam=("lam", "first"))
                  .reset_index())
        reg["score"] = reg.sigma2 / reg.lam if d == "s2_over_lam" else (
            reg.sigma2 / reg.lam ** 2 if d == "s2_over_lam2" else reg.sigma2)
        reg["y"] = (reg.flux_median > 1.0).astype(int)
        t = youden_threshold(reg.score.to_numpy(float), reg.y.to_numpy(int))
        thr[m] = t
        details[m] = {"regime_youden_J": round(float(
            np.mean([(s >= t) == bool(y) for s, y in zip(reg.score, reg.y)])), 4),
            "n_regimes": int(len(reg))}
    out = {
        "stage": "frozen", "frozen_thresholds": thr,
        "frozen_descriptors": DESC,
        "calibration": "per-family regime-level Youden calibration on the 24-regime atlas",
        "calibration_details": details,
        "blind_cells": BLIND_CELLS,
        "atlas_grid": {"sigma2": [0.5, 1.0, 2.0, 4.0],
                       "lam": [0.02, 0.03, 0.05, 0.08, 0.12, 0.2]},
        "note": "thresholds calibrated on the 24-regime atlas only; "
                "no blind cell contributed to any choice",
    }
    atomic_json(out, OUT)
    return out


def cell_id(sigma2: float, lam: float) -> str:
    return f"s{sigma2:g}_l{lam:g}_n128"


def train_surrogates() -> None:
    for sigma2, lam in BLIND_CELLS:
        cell = cell_id(sigma2, lam)
        h5 = ROOT / "data" / f"cell_{cell}.h5"
        if not h5.exists():
            t0 = time.time()
            generate_cell(sigma2, lam, N, 256, 32, 64)
            print(f"[gen] {cell} in {time.time() - t0:.1f}s", flush=True)
        for model in ("unet", "fno"):
            run_dir = ROOT / "runs" / "atlas_new" / cell / model
            if (run_dir / "metrics.json").exists():
                print(f"[skip] {cell}/{model} (trained)", flush=True)
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            print(f"[train] {cell}/{model} ...", flush=True)
            train_grid_model(h5, {**DEFAULTS, "model": model}, run_dir)
            print(f"[done] {cell}/{model} in {(time.time() - t0) / 60:.1f} min", flush=True)


def descriptor_value(model: str, sigma2: float, lam: float) -> float:
    """Per-model frozen descriptor score: each family is scored with the
    descriptor its threshold was calibrated on, never another family's."""
    desc = DESC[model]
    if desc == "s2_over_lam":
        return sigma2 / lam
    if desc == "s2_over_lam2":
        return sigma2 / lam**2
    if desc == "sigma2":
        return sigma2
    raise ValueError(f"unknown descriptor {desc!r} for {model}")


def evaluate_predictions() -> dict:
    with open(OUT) as f:
        out = json.load(f)
    thr = out["frozen_thresholds"]
    results = {}
    for sigma2, lam in BLIND_CELLS:
        cell = cell_id(sigma2, lam)
        preds = {m: bool(descriptor_value(m, sigma2, lam) >= thr[m])
                 for m in DESC}
        cell_res = {"sigma2": sigma2, "lam": lam,
                    "descriptor_values": {m: round(descriptor_value(m, sigma2, lam), 4)
                                          for m in DESC},
                    "predicted_fail": preds}
        for m in DESC:
            mp = ROOT / "runs" / "atlas_new" / cell / m / "metrics.json"
            if not mp.exists():
                cell_res[m] = None
                continue
            d = json.loads(mp.read_text())
            errs = [r["flux_err"] for r in d["per_inst"]]
            heads = [r["head_err"] for r in d["per_inst"]]
            cell_res[m] = {
                "median_flux_err": round(float(np.median(errs)), 4),
                "median_head_err": round(float(np.median(heads)), 4),
                "fail_rate": round(float(np.mean(np.array(errs) > 1.0)), 4),
                "n": len(errs),
                "train_seconds": d.get("train_seconds"),
            }
        results[cell] = cell_res
    out["stage"] = "tested"
    out["blind_results"] = results

    # summary per family: regime-level confusion on the 8 blind cells
    summary = {}
    for m in DESC:
        rows = [(c, r[m]) for c, r in results.items() if r[m] is not None]
        if len(rows) < len(BLIND_CELLS):
            summary[m] = {"status": f"incomplete ({len(rows)}/{len(BLIND_CELLS)} trained)"}
            continue
        y = np.array([1 if r["median_flux_err"] > 1.0 else 0 for _, r in rows])
        p = np.array([1 if results[c]["predicted_fail"][m] else 0 for c, _ in rows])
        tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
        acc = (tp + tn) / len(y)
        bal = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
        summary[m] = {
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "accuracy": round(acc, 4), "balanced_accuracy": round(bal, 4),
            "blind_fail_rate": round(float(y.mean()), 4),
            "per_cell_correct": {c: int(pp == yy) for (c, _), pp, yy in zip(rows, p, y)},
        }
    out["summary"] = summary
    atomic_json(out, OUT)
    return out


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("all", "freeze"):
        out = freeze_calibration()
        print("frozen:", out["frozen_thresholds"])
    if stage in ("all", "train"):
        train_surrogates()
    if stage in ("all", "test"):
        res = evaluate_predictions()
        print(json.dumps(res.get("summary", {}), indent=1))


if __name__ == "__main__":
    main()
