"""Aggregate multi-seed, out-of-grid blind, and extended control metrics into JSON.

Combines results from runs/seeds/, runs/atlas_v2/ (scored using the frozen
calibration thresholds from runs/blind_test.json), runs/pinn_blind/, and
runs/extended_controls.json into runs/extended_validation_summary.json.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "extended_validation_summary.json"


def atomic_json(obj, path: Path) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def main() -> None:
    out = {"stage": "complete"}
    seeds = {}
    for f in sorted((ROOT / "runs" / "seeds").glob("*/*_s*/metrics.json")):
        d = json.loads(f.read_text())
        cfg, seed = f.parent.name.rsplit("_s", 1)
        cell = f.parent.parent.name
        seeds.setdefault((cell, cfg), {})[int(seed)] = d["median_flux_err"]
    out["seeds"] = {}
    for (cell, cfg), sd in sorted(seeds.items()):
        if cfg == "fnoflux":
            s0 = json.loads((ROOT / "runs" / "fluxloss" / f"fno_a0.01_{cell}" /
                             "metrics.json").read_text())["median_flux_err"]
        else:
            s0 = json.loads((ROOT / "runs" / "atlas" / cell / cfg /
                             "metrics.json").read_text())["median_flux_err"]
        vals = {**{int(k): round(v, 3) for k, v in sd.items()},
                0: round(float(s0), 3)}
        out["seeds"][f"{cell}|{cfg}"] = {
            "values": vals,
            "class_stable": len(set(v > 1.0 for v in vals.values())) == 1}
    out["seeds_summary"] = {
        "n_configs": len(out["seeds"]),
        "n_class_unstable": sum(1 for v in out["seeds"].values()
                                if not v["class_stable"])}

    bt = json.loads((ROOT / "runs" / "blind_test.json").read_text())
    thr = bt["frozen_thresholds"]
    desc = bt["frozen_descriptors"]
    blind2 = {}
    for f in sorted((ROOT / "runs" / "atlas_v2").glob("*/*/metrics.json")):
        d = json.loads(f.read_text())
        model = f.parent.name
        s2, lam = d["sigma2"], d["lam"]
        score = s2 / lam if desc[model] == "s2_over_lam" else s2
        blind2.setdefault(model, []).append({
            "cell": f.parent.parent.name, "sigma2": s2, "lam": lam,
            "score": round(score, 4),
            "pred_fail": bool(score >= thr[model]),
            "median_flux_err": round(d["median_flux_err"], 4),
            "true_fail": bool(d["median_flux_err"] > 1.0)})
    out["blind2"] = {}
    for m, rows in blind2.items():
        y = [r["true_fail"] for r in rows]
        pr = [r["pred_fail"] for r in rows]
        conf = {"tp": sum(a and b for a, b in zip(pr, y)),
                "fp": sum(a and not b for a, b in zip(pr, y)),
                "fn": sum(not a and b for a, b in zip(pr, y)),
                "tn": sum(not a and not b for a, b in zip(pr, y))}
        c1 = bt["summary"][m]["confusion"]
        comb = {k: c1[k] + conf[k] for k in conf}
        acc = (comb["tp"] + comb["tn"]) / sum(comb.values())
        bal = 0.5 * (comb["tp"] / max(comb["tp"] + comb["fn"], 1)
                     + comb["tn"] / max(comb["tn"] + comb["fp"], 1))
        out["blind2"][m] = {"rows": rows, "confusion": conf,
                            "combined_confusion": comb,
                            "combined_accuracy": round(acc, 4),
                            "combined_balanced": round(bal, 4)}

    out["pinn_blind"] = []
    out["pinn_blind_summary"] = {}
    for cell_dir in sorted((ROOT / "runs" / "pinn_blind").glob("*")):
        fe, he = [], []
        for x in sorted(cell_dir.glob("pinn_inst*.json")):
            r = json.loads(x.read_text())
            fe.append(r["flux_err"])
            he.append(r["head_err"])
            out["pinn_blind"].append({
                "sigma2": r["sigma2"], "lam": r["lam"], "inst": r["inst"],
                "flux_err": round(r["flux_err"], 4),
                "head_err": round(r["head_err"], 4)})
        if fe:
            out["pinn_blind_summary"][cell_dir.name] = {
                "median_flux_err": round(float(np.median(fe)), 4),
                "median_head_err": round(float(np.median(he)), 4),
                "n": len(fe)}

    ext_ctrl = ROOT / "runs" / "extended_controls.json"
    if ext_ctrl.exists():
        out["extended_controls"] = json.loads(ext_ctrl.read_text())

    atomic_json(out, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
