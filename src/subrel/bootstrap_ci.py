"""Regime-cluster bootstrap confidence intervals for instance-level ROC-AUC.

Resamples the 24 benchmark regimes with replacement 1,000 times, recomputes
the instance-level ROC-AUC of the roughness ratio sigma^2/lambda within each
resample for each solver family, and records the 2.5th and 97.5th percentiles
to runs/bootstrap_ci.json.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "bootstrap_ci.json"
N_BOOT = 1000


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sv = scores[order]
    ranks = np.empty(len(sv), dtype=float)
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    r_pos = ranks[labels == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    df = pd.read_csv(ROOT / "runs" / "atlas" / "instances_with_predictors.csv")
    df["flux_fail"] = df["flux_fail"].astype(int)
    # the paper's per-family descriptors (Section 4.1)
    df["score_unet"] = df.sigma2 / df.lam
    df["score_fno"] = df.sigma2
    df["score_pinn"] = df.sigma2 / df.lam**2
    out = {"n_boot": N_BOOT,
           "statistic": "instance-level AUC of the family's primary "
                        "descriptor (unet sigma^2/lambda, fno sigma^2, "
                        "pinn sigma^2/lambda^2)",
           "resample_unit": "regime (cell), 24 regimes with replacement",
           "seed": 12345}
    for model, col in (("unet", "score_unet"), ("fno", "score_fno"),
                       ("pinn", "score_pinn")):
        g = df[df.model == model]
        cells = sorted(g.cell.unique())
        rng = np.random.default_rng(12345)
        aucs = []
        for _ in range(N_BOOT):
            pick = rng.choice(len(cells), size=len(cells), replace=True)
            parts = [g[g.cell == cells[k]] for k in pick]
            sub = pd.concat(parts)
            aucs.append(rank_auc(sub[col].to_numpy(float),
                                 sub.flux_fail.to_numpy(int)))
        aucs = np.array(aucs)
        point = rank_auc(g[col].to_numpy(float), g.flux_fail.to_numpy(int))
        out[model] = {"point": round(point, 4),
                      "ci95": [round(float(np.quantile(aucs, 0.025)), 4),
                               round(float(np.quantile(aucs, 0.975)), 4)]}
        print(model, out[model])
    fd, tmp = tempfile.mkstemp(dir=str(OUT.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
