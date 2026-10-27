"""Regression tests for the frozen screening rule's per-model descriptors.

Verifies family-specific pre-training roughness descriptor evaluation: each family is scored with
the descriptor its own threshold was calibrated on (U-Net: sigma^2/lambda,
FNO: sigma^2), even when the two formulas land on opposite sides of a
threshold.  Run:  cd src && ../.venv/bin/python -m subrel.tests.test_blind_descriptors
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from subrel.blind_test import DESC, descriptor_value  # noqa: E402


def test_per_model_descriptors() -> None:
    assert DESC["unet"] == "s2_over_lam"
    assert DESC["fno"] == "sigma2"
    # a cell where sigma^2/lambda and sigma^2 fall on OPPOSITE sides of the
    # other family's threshold: both values must still be computed from the
    # family's own descriptor
    s2, lam = 0.75, 0.26
    v_unet = descriptor_value("unet", s2, lam)
    v_fno = descriptor_value("fno", s2, lam)
    assert abs(v_unet - s2 / lam) < 1e-12, v_unet
    assert abs(v_fno - s2) < 1e-12, v_fno
    # opposite crossing: sigma^2 below the U-Net threshold while the ratio
    # is above it, and vice versa for another cell
    s2, lam = 0.4, 0.2  # ratio 2.0, sigma2 0.4
    assert descriptor_value("unet", s2, lam) > 0.75  # ratio side
    assert descriptor_value("fno", s2, lam) < 0.75   # variance side
    print("PASS per-model descriptors (unet sigma^2/lambda, fno sigma^2)")


def test_blind_json_records_both_descriptors() -> None:
    """Recompute the frozen-rule classification for all eight blind cells
    from the archived medians and thresholds; the stored confusion counts
    must match.  A missing archive is a FAILURE here, not a skip: this
    test verifies consistency with runs/blind_test.json."""
    import json
    import numpy as np
    root = Path(__file__).resolve().parents[3]
    p = root / "runs" / "blind_test.json"
    assert p.exists(), "runs/blind_test.json missing from the archive"
    d = json.loads(p.read_text())
    assert "blind_results" in d, "blind_results missing (test phase not run)"
    thr = d["frozen_thresholds"]
    for cell, r in d["blind_results"].items():
        assert "descriptor_values" in r, f"{cell}: per-model descriptors missing"
        assert abs(r["descriptor_values"]["unet"]
                   - r["sigma2"] / r["lam"]) < 1e-3, cell
        assert abs(r["descriptor_values"]["fno"] - r["sigma2"]) < 1e-3, cell
        for m in ("unet", "fno"):
            rec = r[m]
            assert rec is not None, f"{cell}/{m}: no trained-model record"
            pred = r["descriptor_values"][m] >= thr[m]
            assert pred == r["predicted_fail"][m], f"{cell}/{m}: prediction mismatch"
    # recompute the summary confusion from the per-cell records
    for m in ("unet", "fno"):
        tp = fp = fn = tn = 0
        for cell, r in d["blind_results"].items():
            y = 1 if r[m]["median_flux_err"] > 1.0 else 0
            p_hat = 1 if r["predicted_fail"][m] else 0
            tp += (y == 1 and p_hat == 1); fp += (y == 0 and p_hat == 1)
            fn += (y == 1 and p_hat == 0); tn += (y == 0 and p_hat == 0)
        got = d["summary"][m]["confusion"]
        assert got == {"tp": tp, "fp": fp, "fn": fn, "tn": tn}, f"{m}: confusion mismatch {got}"
    print(f"PASS blind_test.json: descriptors, predictions and confusions "
          f"verified for {len(d['blind_results'])} cells x 2 families")


if __name__ == "__main__":
    test_per_model_descriptors()
    test_blind_json_records_both_descriptors()
