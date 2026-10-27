"""Band-width test: does widening the FNO's effective band move the failure?

If flux failure is caused by medium energy above the solver's representable
band, then raising the FNO mode count from 12 to 24 and 32 should reduce the
flux error on cells whose Phi above the band is large, and leave cells with
little high-wavenumber energy unchanged. This is the direct test of the
band-mismatch reading that the covariance intervention supports.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from .train_grid import train_grid_model

ROOT = Path(__file__).resolve().parents[2]
CELLS = [(1.0, 0.05), (2.0, 0.03), (0.5, 0.2)]
MODES = [24, 32]
DRY = os.environ.get("SUBREL_DRY", "0") == "1"


def main() -> None:
    for s2, lam in CELLS:
        cid = f"s{s2:g}_l{lam:g}_n128"
        h5 = ROOT / "data" / f"cell_{cid}.h5"
        for mo in MODES:
            d = ROOT / "runs" / "modes" / f"fno_m{mo}_{cid}"
            if (d / "metrics.json").exists():
                continue
            if DRY:
                print(f"PLAN fno modes={mo} {cid}", flush=True)
                continue
            print(f"[{time.strftime('%H:%M:%S')}] fno modes={mo} {cid}", flush=True)
            train_grid_model(h5, dict(model="fno", modes=mo), d)
    print("MODES DONE", flush=True)


if __name__ == "__main__":
    main()
