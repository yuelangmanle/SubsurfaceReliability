r"""Figure 2: Spatial anatomy of hydraulic-head accuracy versus Darcy-flux error.

Displays three representative heterogeneity regimes (smooth: sigma^2=0.5, lam=0.20;
moderate: sigma^2=1.0, lam=0.05; rough: sigma^2=2.0, lam=0.02) across five columns:
  (a) Standardized log-conductivity field ln K / sigma (YlGnBu_r)
  (b) Reference finite-volume hydraulic head h (with equipotential contours)
  (c) U-Net predicted hydraulic head \hat{h} (with equipotential contours)
  (d) Normalized pointwise hydraulic-head error (\hat{h} - h) / max|h|
  (e) Log10 relative pointwise Darcy-flux error log10 e_q(x)
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .metrics import cell_centered_velocity
from .models import FNO2d, UNet
from .train_grid import make_inputs

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "figures"
CACHE_PATH = ROOT / "runs" / "fig2_cache.npz"


def _load_or_compute_rows(model_name: str, cells: list[tuple[float, float]]) -> list[dict]:
    can_compute = all(
        (ROOT / "data" / f"cell_s{s2:g}_l{lam:g}_n128.h5").exists()
        and (ROOT / "runs" / "atlas" / f"s{s2:g}_l{lam:g}_n128" / model_name / "model.pt").exists()
        for s2, lam in cells
    )
    if can_compute:
        rows = []
        for s2, lam in cells:
            cid = f"s{s2:g}_l{lam:g}_n128"
            mdir = ROOT / "runs" / "atlas" / cid / model_name
            meta = json.loads((mdir / "metrics.json").read_text())
            with h5py.File(ROOT / "data" / f"cell_{cid}.h5", "r") as f:
                n_tr = int(f.attrs["n_train"])
                n_va = int(f.attrs["n_val"])
                Yfull = f["Y_cell"][:]
                Ys, Hs = Yfull[n_tr + n_va:], f["H"][n_tr + n_va:]
            mcfg = meta["cfg"]
            model = (UNet(3, mcfg["width"], mcfg["depth"]) if mcfg["model"] == "unet"
                     else FNO2d(3, mcfg["width"], mcfg["modes"]))
            model.load_state_dict(torch.load(mdir / "model.pt", map_location="cpu"))
            model.eval()
            ymu = float(Yfull[:n_tr].mean())
            ystd = float(Yfull[:n_tr].std())
            meds = np.array([p["flux_err"] for p in meta["per_inst"]])
            k = int(np.argmin(np.abs(meds - np.median(meds))))
            with torch.no_grad():
                hp = model(torch.from_numpy(make_inputs(Ys[k][None], ymu, ystd))).numpy()[0, 0] * 0.5 + 0.5
            pi = meta["per_inst"][k]
            rows.append(dict(
                s2=float(s2), lam=float(lam),
                Y=Ys[k].astype(np.float32),
                H=Hs[k].astype(np.float32),
                Hp=hp.astype(np.float32),
                inst=int(pi["inst"]),
                eh=float(pi["head_err"]),
                eq=float(pi["flux_err"]),
            ))
        # Save compact cache for lightweight reproduction without .h5/.pt files
        save_dict = {}
        for idx, r in enumerate(rows):
            for key, val in r.items():
                save_dict[f"r{idx}_{key}"] = np.array(val)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE_PATH, n_rows=len(rows), **save_dict)
        return rows

    if CACHE_PATH.exists():
        z = np.load(CACHE_PATH)
        n_rows = int(z["n_rows"])
        rows = []
        for idx in range(n_rows):
            rows.append(dict(
                s2=float(z[f"r{idx}_s2"]),
                lam=float(z[f"r{idx}_lam"]),
                Y=z[f"r{idx}_Y"].astype(np.float64),
                H=z[f"r{idx}_H"].astype(np.float64),
                Hp=z[f"r{idx}_Hp"].astype(np.float64),
                inst=int(z[f"r{idx}_inst"]),
                eh=float(z[f"r{idx}_eh"]),
                eq=float(z[f"r{idx}_eq"]),
            ))
        return rows

    raise FileNotFoundError("Neither raw .h5/.pt files nor runs/fig2_cache.npz found.")


def make_figure2(model_name: str = "unet",
                 cells: list[tuple[float, float]] | None = None,
                 out_name: str = "fig2_anatomy_unet.png") -> Path | None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "axes.edgecolor": "#1E293B",
        "axes.linewidth": 0.95,
        "figure.dpi": 600,
        "savefig.dpi": 600,
    })

    if cells is None:
        cells = [(0.5, 0.2), (1.0, 0.05), (2.0, 0.02)]

    rows = _load_or_compute_rows(model_name, cells)
    qvmin, qvmax = -2.5, 2.5

    for r in rows:
        n_ = r["H"].shape[0]
        d_ = 1.0 / n_
        K_ = np.exp(r["Y"].astype(np.float64))
        H_ = r["H"].astype(np.float64)
        Hp_ = r["Hp"].astype(np.float64)
        vxt, vyt = cell_centered_velocity(K_, H_, d_)
        vxp, vyp = cell_centered_velocity(K_, Hp_, d_)
        qe = np.log10(np.sqrt((vxp - vxt) ** 2 + (vyp - vyt) ** 2) /
                      np.sqrt(vxt ** 2 + vyt ** 2).mean() + 1e-12)
        he = (Hp_ - H_) / max(np.abs(H_).max(), 1e-9)
        r["he"], r["qe"] = he, qe
        r["Y_std"] = r["Y"].astype(np.float64) / np.sqrt(r["s2"])

    fig = plt.figure(figsize=(14.8, 9.1), dpi=600, metadata={"Software": None})
    gs = fig.add_gridspec(
        4, 5,
        height_ratios=[1.0, 1.0, 1.0, 0.068],
        hspace=0.14, wspace=0.10,
    )

    col_titles = [
        r"(a) Standardized $\ln K / \sigma$",
        r"(b) Reference FV head $h$",
        r"(c) U-Net predicted head $\hat{h}$",
        r"(d) Head error $(\hat{h} - h)/\max|h|$",
        r"(e) Local flux error $\log_{10} e_q(\mathbf{x})$",
    ]
    regime_names = ["Smooth regime", "Moderate regime", "Rough regime"]

    for r_idx, (name, r_cur) in enumerate(zip(regime_names, rows)):
        row_lbl = (
            f"{name} (#{r_cur['inst']})\n"
            + rf"$\sigma^2={r_cur['s2']:g},\ \lambda={r_cur['lam']:g}$"
            + "\n"
            + rf"($E_h={r_cur['eh']*100:.1f}\%,\ E_q={r_cur['eq']:.2f}$)"
        )
        specs = [
            (r_cur["Y_std"], "YlGnBu_r", (-2.5, 2.5)),
            (r_cur["H"], "viridis", (0.0, 1.0)),
            (r_cur["Hp"], "viridis", (0.0, 1.0)),
            (r_cur["he"], "coolwarm", (-0.23, 0.23)),
            (r_cur["qe"], "magma", (qvmin, qvmax)),
        ]
        for c_idx in range(5):
            ax = fig.add_subplot(gs[r_idx, c_idx])
            img, cmap, vlim = specs[c_idx]
            ax.imshow(
                img, aspect="equal", interpolation="bilinear",
                cmap=cmap, vmin=vlim[0], vmax=vlim[1], origin="upper",
            )
            # Overlay equipotential contours on reference and predicted head columns
            if c_idx in (1, 2):
                cs = ax.contour(
                    img, levels=[0.2, 0.4, 0.6, 0.8],
                    colors="white", linewidths=0.95, alpha=0.88,
                )
                if hasattr(cs, "collections"):
                    for col in cs.collections:
                        col.set_path_effects([pe.Stroke(linewidth=1.8, foreground="#0F172A"), pe.Normal()])
                else:
                    cs.set_path_effects([pe.Stroke(linewidth=1.8, foreground="#0F172A"), pe.Normal()])

            # High-contrast summary badges in columns (d) and (e)
            if c_idx == 4:
                status_col = "#065F46" if r_cur["eq"] <= 1.0 else "#9F1239"
                status_bg = "#ECFDF5" if r_cur["eq"] <= 1.0 else "#FFF1F2"
                status_tag = "PASS" if r_cur["eq"] <= 1.0 else "FAIL"
                ax.text(
                    0.96, 0.05,
                    rf"$E_q = {r_cur['eq']:.2f}$ ({status_tag})",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.8, fontweight="bold", color=status_col,
                    bbox=dict(boxstyle="round,pad=0.20", fc=status_bg, ec=status_col, lw=0.85, alpha=0.94),
                )
            elif c_idx == 3:
                ax.text(
                    0.96, 0.05,
                    rf"$E_h = {r_cur['eh']*100:.1f}\%$",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.8, fontweight="bold", color="#1E293B",
                    bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="#94A3B8", lw=0.80, alpha=0.92),
                )

            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#1E293B")
                sp.set_linewidth(0.95)
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=9.4, fontweight="bold", color="#0F172A", pad=8)
            if c_idx == 0:
                ax.set_ylabel(row_lbl, fontsize=8.5, fontweight="bold", color="#0F172A", labelpad=10)

    # Bottom horizontal colorbars
    ax_cb_k = fig.add_subplot(gs[3, 0])
    cb_k = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=-2.5, vmax=2.5), cmap="YlGnBu_r"),
        cax=ax_cb_k, orientation="horizontal",
    )
    cb_k.set_label(r"Standardized $\ln K / \sigma$", fontsize=8.4, color="#0F172A")
    cb_k.set_ticks([-2, 0, 2])
    cb_k.ax.tick_params(labelsize=7.8)

    ax_cb_h = fig.add_subplot(gs[3, 1:3])
    cb_h = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap="viridis"),
        cax=ax_cb_h, orientation="horizontal",
    )
    cb_h.set_label(r"Hydraulic head $h,\ \hat{h}$ (white lines: equipotential contours $0.2\text{–}0.8$)", fontsize=8.4, color="#0F172A")
    cb_h.ax.tick_params(labelsize=7.8)

    ax_cb_he = fig.add_subplot(gs[3, 3])
    cb_he = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=-0.23, vmax=0.23), cmap="coolwarm"),
        cax=ax_cb_he, orientation="horizontal",
    )
    cb_he.set_label(r"Pointwise head error $(\hat{h} - h)/\max|h|$", fontsize=8.4, color="#0F172A")
    cb_he.set_ticks([-0.2, 0.0, 0.2])
    cb_he.ax.tick_params(labelsize=7.8)

    ax_cb_qe = fig.add_subplot(gs[3, 4])
    cb_qe = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=-2.5, vmax=2.5), cmap="magma"),
        cax=ax_cb_qe, orientation="horizontal",
    )
    cb_qe.set_label(r"Local flux error $\log_{10} e_q(\mathbf{x})$ ($0 \equiv e_q = 1$)", fontsize=8.4, color="#0F172A")
    cb_qe.set_ticks([-2, -1, 0, 1, 2])
    cb_qe.ax.tick_params(labelsize=7.8)

    out = FIGDIR / out_name
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=600, metadata={"Software": None})
    plt.close(fig)
    print("wrote", out)
    return out


if __name__ == "__main__":
    make_figure2()
