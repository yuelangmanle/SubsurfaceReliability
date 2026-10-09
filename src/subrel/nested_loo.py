"""Leave-one-regime-out cross-validation of pre-training screening descriptors.

Evaluates regime-level and instance-level classification accuracy when both
the heterogeneity descriptor and its classification threshold are calibrated
strictly inside each 23-regime training fold and tested on the held-out regime.
Also evaluates threshold sensitivity at flux-error cutoffs {0.5, 1.0, 2.0}.
Output is written to runs/nested_loo.json.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "runs" / "atlas" / "instances_with_predictors.csv"
OUT = ROOT / "runs" / "nested_loo.json"

# riskier-if-higher scores; a leading "-" flips the orientation
SPECTRAL = ["e_hi_4", "e_hi_6", "e_hi_8", "e_hi_12", "e_hi_16", "e_hi_24",
            "ek_hi_8", "ek_hi_16", "s2_over_lam", "slope", "sigma2", "-lam"]
ANALYTIC = ["sigma2", "-lam", "s2_over_lam", "s2_over_lam2"]
# primary analytical descriptor for each solver family
PRIMARY_DESC = {"unet": "s2_over_lam", "fno": "sigma2", "pinn": "s2_over_lam2"}


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC with tie handling; labels are 0/1 in arbitrary order."""
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
    r_pos = ranks[labels == 1].sum()  # ranks of the actual positives
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def oriented(df: pd.DataFrame, name: str) -> np.ndarray:
    if name.startswith("-"):
        return -df[name[1:]].to_numpy(float)
    return df[name].to_numpy(float)


def select_desc(train: pd.DataFrame, label: str, cands: list[str]) -> tuple[str, float]:
    """Descriptor with the best AUC on the calibration pool."""
    y = train[label].to_numpy(int)
    best, best_auc = None, -1.0
    for name in cands:
        a = rank_auc(oriented(train, name), y)
        if a == a and a > best_auc:  # NaN (one-class pool) never wins
            best, best_auc = name, a
    if best is None:  # calibration pool has a single class
        return cands[0], float("nan")
    return best, best_auc


def youden_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold maximising TPR - FPR (predict fail when score >= thr)."""
    order = np.argsort(scores)
    s, y = scores[order], labels[order]
    if y.sum() == 0 or (1 - y).sum() == 0:
        return float(s[-1] + 1.0)  # degenerate pool: everything predicted fail
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


def confusion(pred: np.ndarray, y: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    acc = (tp + tn) / max(len(y), 1)
    bal = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": round(acc, 4), "balanced_accuracy": round(bal, 4)}


def atomic_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def regime_frame(dfm: pd.DataFrame, cut: float) -> pd.DataFrame:
    reg = (dfm.groupby("cell")
              .agg(flux_median=("flux_err", "median"), sigma2=("sigma2", "first"),
                   lam=("lam", "first"))
              .reset_index())
    reg["s2_over_lam"] = reg.sigma2 / reg.lam
    reg["s2_over_lam2"] = reg.sigma2 / reg.lam**2
    reg["y"] = (reg.flux_median > cut).astype(int)
    return reg


def regime_loo(reg: pd.DataFrame, desc: str | None, cands: list[str] | None) -> dict:
    """Regime-level LOO. desc=None -> fully nested (re-pick per fold)."""
    folds, picks = [], []
    for held in sorted(reg.cell):
        tr, te = reg[reg.cell != held], reg[reg.cell == held]
        y_tr = tr.y.to_numpy(int)
        if desc is None:
            d, auc_tr = select_desc(tr, "y", cands)
        else:
            d, auc_tr = desc, rank_auc(oriented(tr, desc), y_tr)
        scored = len(set(y_tr)) == 2
        if scored:
            thr = youden_threshold(oriented(tr, d), y_tr)
            pred = int(oriented(te, d)[0] >= thr)
        else:
            pred = int(y_tr.mean() >= 0.5)  # degenerate: calibration majority
        folds.append({"held_out": held, "desc": d, "scored": scored,
                      "y": int(te.y.iloc[0]), "pred": pred,
                      "train_auc": round(auc_tr, 4) if scored else None})
        picks.append(d)
    sc = [f for f in folds if f["scored"]]
    correct = sum(f["pred"] == f["y"] for f in sc)
    base = sum(int(np.mean([g["y"] for g in sc]) >= 0.5) == f["y"] for f in sc)
    degen = [f for f in folds if not f["scored"]]
    return {
        "descriptor": desc or "nested",
        "n_scored_folds": len(sc),
        "scored_correct": correct,
        "majority_baseline_correct": base,
        "degenerate_folds": [{"held_out": f["held_out"], "y": f["y"],
                              "pred": f["pred"]} for f in degen],
        "picked_descriptors": {p: picks.count(p) for p in sorted(set(picks))},
        "folds": folds,
    }


def instance_loo(dfm: pd.DataFrame, desc: str | None, cands: list[str],
                 label: str = "flux_fail") -> dict:
    """Instance-level LOO with pooled calibration scores."""
    cells = sorted(dfm.cell.unique())
    folds, picks = [], []
    for held in cells:
        tr, te = dfm[dfm.cell != held], dfm[dfm.cell == held]
        y_tr = tr[label].to_numpy(int)
        if desc is None:
            d, auc_tr = select_desc(tr, label, cands)
        else:
            d = desc
            auc_tr = rank_auc(oriented(tr, d), y_tr)
        s_tr = oriented(tr, d)
        if len(set(y_tr)) == 2:
            thr = youden_threshold(s_tr, y_tr)
            pred = (oriented(te, d) >= thr).astype(int)
        else:
            pred = np.full(len(te), int(y_tr.mean() >= 0.5), dtype=int)
        maj = int(y_tr.mean() >= 0.5)
        base = np.full(len(te), maj, dtype=int)
        y_te = te[label].to_numpy(int)
        c = confusion(pred, y_te)
        folds.append({"held_out": held, "desc": d, "train_auc": round(auc_tr, 4), **c,
                      "baseline_acc": round(float((base == y_te).mean()), 4)})
        picks.append(d)
    tp = sum(f["tp"] for f in folds); fp = sum(f["fp"] for f in folds)
    fn = sum(f["fn"] for f in folds); tn = sum(f["tn"] for f in folds)
    n = max(tp + fp + fn + tn, 1)
    return {
        "descriptor": desc or "nested",
        "pooled_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "accuracy": round((tp + tn) / n, 4),
        "balanced_accuracy": round(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)), 4),
        "majority_baseline_accuracy": round(float(np.mean([f["baseline_acc"] for f in folds])), 4),
        "picked_descriptors": {p: picks.count(p) for p in sorted(set(picks))},
    }


def main() -> None:
    df = pd.read_csv(CSV)
    df["flux_fail"] = df["flux_fail"].astype(int)
    df["s2_over_lam2"] = df.sigma2 / df.lam**2
    out: dict = {"n_instances": int(len(df)), "n_cells": int(df.cell.nunique()),
                 "candidates": {"spectral": SPECTRAL, "analytic": ANALYTIC},
                 "primary_descriptor": PRIMARY_DESC}

    # full-grid AUC tables (both candidate sets, both levels)
    reg = regime_frame(df, 1.0)
    for model in ("unet", "fno", "pinn", "ALL"):
        sub = df if model == "ALL" else df[df.model == model]
        y = sub["flux_fail"].to_numpy(int)
        out.setdefault("full_grid_auc", {})[model] = {
            "instance": {c: round(rank_auc(oriented(sub, c), y), 4)
                         for c in sorted(set(SPECTRAL + ANALYTIC))},
        }
        rs = regime_frame(sub, 1.0)
        ry = rs.y.to_numpy(int)
        out["full_grid_auc"][model]["regime"] = {
            c: round(rank_auc(oriented(rs, c), ry), 4)
            for c in sorted(set(ANALYTIC + ["s2_over_lam"]))}
    atomic_json(out, OUT)

    # ---- regime-level LOO: frozen-descriptor and nested
    out["regime_loo"] = {}
    for model in ("unet", "fno", "pinn"):
        sub = df[df.model == model]
        rs = regime_frame(sub, 1.0)
        out["regime_loo"][model] = {
            "frozen": regime_loo(rs, PRIMARY_DESC[model], None),
            "nested": regime_loo(rs, None, ANALYTIC),
        }
        atomic_json(out, OUT)

    # ---- instance-level LOO: frozen / nested-analytic / nested-spectral
    out["instance_loo"] = {}
    for model in ("unet", "fno", "pinn"):
        sub = df[df.model == model]
        out["instance_loo"][model] = {
            "frozen": instance_loo(sub, PRIMARY_DESC[model], ANALYTIC),
            "nested_analytic": instance_loo(sub, None, ANALYTIC),
            "nested_spectral": instance_loo(sub, None, SPECTRAL),
        }
        atomic_json(out, OUT)

    # ---- threshold sensitivity with the primary descriptor frozen
    sens = {}
    for model in ("unet", "fno", "pinn"):
        sub = df[df.model == model]
        sens[model] = {"frozen_descriptor": PRIMARY_DESC[model]}
        for cut in (0.5, 1.0, 2.0):
            lab = (sub["flux_err"].to_numpy(float) > cut).astype(int)
            g2 = sub.assign(flux_fail=lab)
            inst = instance_loo(g2, PRIMARY_DESC[model], ANALYTIC)
            rs = regime_frame(g2, cut)
            reg_r = regime_loo(rs, PRIMARY_DESC[model], None)
            sens[model][f"cut_{cut:g}"] = {
                "fail_rate": round(float(lab.mean()), 4),
                "instance_acc": inst["accuracy"],
                "instance_bal": inst["balanced_accuracy"],
                "instance_baseline": inst["majority_baseline_accuracy"],
                "regime_scored_correct": reg_r["scored_correct"],
                "regime_n_scored": reg_r["n_scored_folds"],
                "regime_baseline": reg_r["majority_baseline_correct"],
            }
        atomic_json(out, OUT)
    out["threshold_sensitivity"] = sens

    atomic_json(out, OUT)
    print(f"wrote {OUT}\n")
    for model in ("unet", "fno", "pinn"):
        r = out["regime_loo"][model]
        fz, ne = r["frozen"], r["nested"]
        print(f"{model}: regime frozen {fz['scored_correct']}/{fz['n_scored_folds']} "
              f"(baseline {fz['majority_baseline_correct']}) | "
              f"nested {ne['scored_correct']}/{ne['n_scored_folds']} "
              f"(baseline {ne['majority_baseline_correct']}) "
              f"picks={ne['picked_descriptors']}")
        il = out["instance_loo"][model]
        for k in ("frozen", "nested_analytic", "nested_spectral"):
            v = il[k]
            print(f"   inst {k:16s} acc={v['accuracy']:.3f} bal={v['balanced_accuracy']:.3f} "
                  f"baseline={v['majority_baseline_accuracy']:.3f} conf={v['pooled_confusion']}")


if __name__ == "__main__":
    main()
