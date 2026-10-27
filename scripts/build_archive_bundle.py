"""Assemble the complete data and checkpoint archive bundle with SHA-256 manifest.

Packages model checkpoints, HDF5 conductivity/solution datasets, environment
specifications, and SHA-256 checksums into a standalone archive directory.

Usage: python3 scripts/build_archive_bundle.py [output_dir]
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else
               ROOT.parent / "subsurface_reliability_archive_bundle")
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    # 1. model checkpoints required by the paper's predictions
    for pattern in ["runs/atlas/*_n128/*/model.pt",        # atlas surrogates
                    "runs/atlas_new/*_n128/*/model.pt",    # blind-cell surrogates
                    "runs/fluxloss/*/model.pt",            # flux-weighted FNOs
                    "runs/pinn_blind/*/*.pt"]:             # PINN blind checkpoints
        for f in ROOT.glob(pattern):
            rel = f.relative_to(ROOT)
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out / rel)
            manifest.append((str(rel), sha256(f)))
    # 2. HDF5 field archives
    for f in sorted(ROOT.glob("data/cell_*.h5")):
        rel = f.relative_to(ROOT)
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out / rel)
        manifest.append((str(rel), sha256(f)))
    # 3. environment lock
    env = ROOT / "environment.txt"
    if env.exists():
        shutil.copy2(env, out / "environment.txt")
        manifest.append(("environment.txt", sha256(env)))
    # 4. manifest
    lines = [f"{h}  {p}" for p, h in sorted(manifest)]
    (out / "manifest_sha256.txt").write_text("\n".join(lines) + "\n")
    n_pt = sum(1 for p, _ in manifest if p.endswith(".pt"))
    n_h5 = sum(1 for p, _ in manifest if p.endswith(".h5"))
    print(f"bundle: {out}\n  {n_pt} checkpoints, {n_h5} HDF5 fields, "
          f"{len(lines)} files hashed")


if __name__ == "__main__":
    main()
