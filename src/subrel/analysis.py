"""Benchmark regime aggregation, spectral predictors, and ROC-AUC evaluation.

Primary failure criterion: interior Darcy-flux relative L2 error > 1.0
(where predicted velocity errors exceed those of a zero-flow baseline).
Secondary head criterion: interior hydraulic-head relative L2 error > 0.15.

Evaluates both analytical generating parameters (sigma^2, lambda,
sigma^2/lambda, sigma^2/lambda^2) and empirical spectral energy fractions
across all held-out test realizations.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FLUX_FAIL = 1.0
HEAD_FAIL = 0.15
K_CUTS = (4.0, 6.0, 8.0, 12.0, 16.0, 24.0)


def collect() -> pd.DataFrame:
    rows = []
    atlas = ROOT / "runs" / "atlas"
    for mp in atlas.glob("*/*/metrics.json"):
        r = json.loads(mp.read_text())
        model = r["model"]
        cell = mp.parts[-3]
        for p in r.get("per_inst", []):
            rows.append(dict(cell=cell, model=model, sigma2=r["sigma2"], lam=r["lam"],
                             inst=p["inst"], head_err=p["head_err"], flux_err=p["flux_err"]))
    for jp in atlas.glob("*/pinn/pinn_inst*.json"):
        r = json.loads(jp.read_text())
        rows.append(dict(cell=jp.parts[-3], model="pinn", sigma2=r["sigma2"], lam=r["lam"],
                         inst=r["inst"], head_err=r["head_err"], flux_err=r["flux_err"]))
    for jp in atlas.glob("*/pinn_rar/pinn_rar_inst*.json"):
        r = json.loads(jp.read_text())
        rows.append(dict(cell=jp.parts[-3], model="pinn+RAR", sigma2=r["sigma2"], lam=r["lam"],
                         inst=r["inst"], head_err=r["head_err"], flux_err=r["flux_err"]))
    df = pd.DataFrame(rows)
    if len(df):
        df["flux_fail"] = df.flux_err > FLUX_FAIL
        df["head_fail"] = df.head_err > HEAD_FAIL
    return df


@lru_cache(maxsize=None)
def _radial_energy(cell: str, inst: int, which: str = "H") -> tuple[np.ndarray, np.ndarray]:
    """(kr sorted, cumulative energy fraction) of field `which` ("H" solution or
    "Y" log-conductivity) of one instance."""
    h5 = ROOT / "data" / f"cell_{cell}.h5"
    with h5py.File(h5, "r") as f:
        h = f["H"][inst] if which == "H" else f["Y_cell"][inst]
    n = h.shape[0]
    spec = np.abs(np.fft.fft2(h - h.mean())) ** 2
    kx = np.fft.fftfreq(n, d=1.0 / n)
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    kr = np.sqrt(KX**2 + KY**2).ravel()
    e = spec.ravel()
    order = np.argsort(kr)
    kr, e = kr[order], e[order]
    cum = np.cumsum(e) / e.sum()
    return kr, cum


def e_hi(cell: str, inst: int, k_cut: float, which: str = "H") -> float:
    kr, cum = _radial_energy(cell, inst, which)
    j = np.searchsorted(kr, k_cut, side="right")
    return float(1.0 - cum[j - 1]) if j > 0 else 1.0


def solution_slope(cell: str, inst: int) -> float:
    from .metrics import spectral_slope
    h5 = ROOT / "data" / f"cell_{cell}.h5"
    with h5py.File(h5, "r") as f:
        return float(spectral_slope(f["H"][inst]))


def auc(scores, labels) -> float:
    """Rank-based AUC (Mann-Whitney) with tie handling."""
    s, l = np.asarray(scores, float), np.asarray(labels, bool)
    ok = np.isfinite(s)
    s, l = s[ok], l[ok]
    pos, neg = s[l], s[~l]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def attach_predictors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for kc in K_CUTS:
        out[f"e_hi_{kc:g}"] = [e_hi(c, int(i), kc) for c, i in zip(out.cell, out.inst)]
    # K-side predictors: computable BEFORE solving/training (practical use)
    for kc in (8.0, 16.0):
        out[f"ek_hi_{kc:g}"] = [e_hi(c, int(i), kc, "Y") for c, i in zip(out.cell, out.inst)]
    out["s2_over_lam"] = out.sigma2 / out.lam
    out["slope"] = [solution_slope(c, int(i)) for c, i in zip(out.cell, out.inst)]
    return out


def predictor_auc_table(pdf: pd.DataFrame, label: str = "flux_fail") -> pd.DataFrame:
    preds = ([f"e_hi_{kc:g}" for kc in K_CUTS] + ["ek_hi_8", "ek_hi_16",
              "sigma2", "lam", "s2_over_lam", "slope"])
    rows = []
    for m, g in list(pdf.groupby("model")) + [("ALL", pdf)]:
        row = {"model": m, "n": len(g), "fail_rate": float(g[label].mean())}
        for p in preds:
            sc = g[p] if p != "lam" else -g[p]       # short lam -> harder
            row[p] = auc(sc, g[label])
        rows.append(row)
    return pd.DataFrame(rows).set_index("model").round(3)


def phase_table(df: pd.DataFrame, value: str = "flux_err") -> pd.DataFrame:
    return (df.groupby(["model", "lam", "sigma2"])[value].median()
              .unstack("model").round(3))


if __name__ == "__main__":
    df = collect()
    if not len(df):
        print("no results yet")
        raise SystemExit
    print(f"{len(df)} instance-level results")
    print("\n=== median flux error by (lam, sigma2) x model ===")
    print(phase_table(df).to_string())
    print("\n=== failure rate (flux_err > 1) ===")
    print(df.groupby("model").flux_fail.agg(["mean", "sum", "count"]).round(3).to_string())
    csv_path = ROOT / "runs" / "atlas" / "instances_with_predictors.csv"
    sample_h5 = ROOT / "data" / f"cell_{df.cell.iloc[0]}.h5"
    if csv_path.exists() and (not sample_h5.exists() or len(pd.read_csv(csv_path)) == len(df)):
        pdf = pd.read_csv(csv_path)
    else:
        pdf = attach_predictors(df)
        pdf.to_csv(csv_path, index=False)
    print("\n=== predictor AUC for flux failure (higher = better) ===")
    print(predictor_auc_table(pdf).to_string())
