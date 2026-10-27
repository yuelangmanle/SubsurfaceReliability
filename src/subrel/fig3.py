"""Figure 3: Controlled boundary interventions versus matched optimization controls.

Panel (a): Full 64-realization raincloud distributions (half-violin KDE,
interquartile capsule, median marker, and jittered realization scatter) across
six threshold-adjacent amortized regimes comparing baseline training (80 epochs),
matched-budget MSE continuation (+30 epochs, beta=0), and soft TPFA volume-
conservation fine-tuning (+30 epochs, beta=0.3).
Panel (b): Paired horizontal Cleveland dumbbell trajectories across eight PINN
test realizations comparing uniform collocation against residual-adaptive
resampling (RAR) in a moderate transition regime (sigma^2=1.0, lam=0.03) and a
high-variance regime (sigma^2=4.0, lam=0.08).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "figures"
DISPLAY_NAME = {"unet": "U-Net", "fno": "FNO", "pinn": "PINN"}


def _draw_raincloud(ax, x_center: float, vals: np.ndarray, color: str, rng: np.random.Generator, width: float = 0.12):
    vals = np.asarray(vals, dtype=float)
    q25, med, q75 = np.percentile(vals, [25, 50, 75])
    vmin, vmax = np.percentile(vals, [2, 98])
    if vmax > vmin + 1e-6:
        kde = gaussian_kde(vals, bw_method=0.38)
        y_grid = np.linspace(vmin, vmax, 80)
        dens = kde(y_grid)
        dens = dens / max(dens.max(), 1e-12) * width
        ax.fill_betweenx(
            y_grid, x_center, x_center + dens,
            facecolor=color, edgecolor=color, alpha=0.42, lw=0.9, zorder=3,
        )
    # Jittered realization scatter on left side of center
    jit = rng.uniform(-width * 0.68, -width * 0.12, size=len(vals))
    ax.scatter(
        x_center + jit, vals,
        s=7.5, color=color, alpha=0.45, edgecolors="none", zorder=3,
    )
    # IQR capsule + whiskers
    ax.plot([x_center, x_center], [vmin, vmax], color=color, lw=1.0, alpha=0.75, zorder=4)
    ax.plot([x_center, x_center], [q25, q75], color=color, lw=3.6, solid_capstyle="round", zorder=4)
    ax.scatter(
        [x_center], [med],
        s=24, facecolors="white", edgecolors=color, linewidths=1.5, zorder=5,
    )
    return med, q75, vmax


def make_figure3(out_name: str = "fig3_repairs.png") -> Path | None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "axes.edgecolor": "#1E293B",
        "axes.linewidth": 0.95,
        "figure.dpi": 600,
        "savefig.dpi": 600,
    })

    consft = []
    for d in sorted((ROOT / "runs" / "repairs").glob("consft2_*")):
        tag = d.name.split("consft2_")[1]
        fam, cid = tag.split("_")[0], tag.split("_", 1)[1]
        s2 = float(cid[1:].split("_l")[0])
        lam = float(cid.split("_l")[1].split("_n")[0])
        base = json.loads((ROOT / "runs/atlas" / cid / fam / "metrics.json").read_text())
        mse = json.loads((ROOT / f"runs/repairs/mse2_{fam}_{cid}" / "metrics.json").read_text())
        m = json.loads((d / "metrics.json").read_text())
        consft.append(dict(
            model=fam, sigma2=s2, lam=lam,
            base=base["median_flux_err"],
            mse=mse["median_flux_err"],
            fixed=m["median_flux_err"],
            base_arr=np.array([p["flux_err"] for p in base["per_inst"]], dtype=float),
            mse_arr=np.array([p["flux_err"] for p in mse["per_inst"]], dtype=float),
            fixed_arr=np.array([p["flux_err"] for p in m["per_inst"]], dtype=float),
        ))

    rar = []
    for d in sorted((ROOT / "runs" / "repairs").glob("rarstd_*")):
        cid = d.name.split("rarstd_")[1]
        s2 = float(cid[1:].split("_l")[0])
        lam = float(cid.split("_l")[1].split("_n")[0])
        for jp in sorted(d.glob("pinn_rarstd_inst*.json")):
            r = json.loads(jp.read_text())
            b = json.loads((ROOT / "runs/atlas" / cid / "pinn" /
                            f"pinn_inst{r['inst']}.json").read_text())
            rar.append(dict(
                sigma2=s2, lam=lam, inst=int(r["inst"]),
                base=float(b["flux_err"]), fixed=float(r["flux_err"]),
            ))
    if not consft and not rar:
        print("no repair results yet")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.4), dpi=600, constrained_layout=True)
    rng = np.random.default_rng(42)

    # Panel (a): 64-realization raincloud distributions
    ax = axes[0]
    ax.axhspan(0.0, 1.0, color="#ECFDF5", alpha=0.55, zorder=0)
    ax.axhspan(1.0, 2.72, color="#FFF1F2", alpha=0.35, zorder=0)
    ax.grid(axis="y", ls="-", lw=0.55, color="#CBD5E1", alpha=0.75, zorder=1)
    ax.axvline(2.5, color="#64748B", lw=1.1, ls=":", zorder=2)
    ax.axhline(1.0, color="#BE123C", lw=1.6, ls="--", zorder=2)

    xs = np.arange(len(consft))
    offsets = [-0.27, 0.0, 0.27]
    colors = ["#475569", "#D97706", "#0284C7"]
    txt_colors = ["#1E293B", "#92400E", "#0369A1"]
    keys = ["base_arr", "mse_arr", "fixed_arr"]
    mkeys = ["base", "mse", "fixed"]

    for i, c in enumerate(consft):
        for off, col, tcol, key, mkey in zip(offsets, colors, txt_colors, keys, mkeys):
            med, q75, vmax = _draw_raincloud(ax, xs[i] + off, c[key], col, rng, width=0.115)
            ax.annotate(
                f"{c[mkey]:.2f}",
                (xs[i] + off, q75),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=6.7, fontweight="bold", color=tcol,
                bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.92),
                zorder=6,
            )

    ax.text(
        1.25, 2.16, "FNO threshold-adjacent regimes",
        fontsize=8.0, fontweight="bold", color="#334155", ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.82),
        zorder=6,
    )
    ax.text(
        3.75, 2.60, "U-Net threshold-adjacent regimes",
        fontsize=8.0, fontweight="bold", color="#334155", ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.82),
        zorder=6,
    )
    ax.text(
        2.55, 1.96,
        "Matched MSE continuation (orange) equals or beats\nsoft divergence penalty (blue) on 4 of 6 regimes",
        fontsize=7.5, color="#1E293B", fontweight="bold", ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.22", fc="#F8FAFC", ec="#94A3B8", lw=0.85, alpha=0.95),
        zorder=6,
    )

    ax.set_ylim(0.28, 2.70)
    ax.set_xlim(-0.55, len(consft) - 0.42)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"$\\sigma^2={c['sigma2']:g},\\ \\lambda={c['lam']:g}$\n({DISPLAY_NAME.get(c['model'], c['model'])})"
         for c in consft],
        fontsize=8.0,
    )
    ax.set_ylabel(r"Realization Darcy-flux error $E_q$ (rel. $L_2$)", fontsize=9.4, color="#0F172A")
    ax.set_title(
        "(a) Amortized surrogates: 64-realization raincloud distributions across interventions",
        fontsize=9.6, fontweight="bold", color="#0F172A", loc="left", pad=8,
    )
    leg_a = [
        Patch(facecolor="#475569", edgecolor="#475569", alpha=0.65, label="Baseline head-only (80 epochs)"),
        Patch(facecolor="#D97706", edgecolor="#D97706", alpha=0.65, label=r"Matched MSE continuation (+30 epochs, $\beta=0$)"),
        Patch(facecolor="#0284C7", edgecolor="#0284C7", alpha=0.65, label=r"Soft TPFA divergence (+30 epochs, $\beta=0.3$)"),
    ]
    ax.legend(handles=leg_a, fontsize=7.5, frameon=True, facecolor="white", edgecolor="#CBD5E1", loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (b): Paired horizontal Cleveland dumbbell trajectories
    ax = axes[1]
    ax.axvspan(0.0, 1.0, color="#ECFDF5", alpha=0.55, zorder=0)
    ax.axvspan(1.0, 6.6, color="#FFF1F2", alpha=0.35, zorder=0)
    ax.grid(axis="x", ls="-", lw=0.55, color="#CBD5E1", alpha=0.75, zorder=1)
    ax.axhline(3.5, color="#64748B", lw=1.1, ls=":", zorder=2)
    ax.axvline(1.0, color="#BE123C", lw=1.6, ls="--", zorder=2)

    ypos = np.arange(len(rar))[::-1]  # Top-to-bottom: s1_l0.03 first, then s4_l0.08
    for y, r in zip(ypos, rar):
        b_val, f_val = r["base"], r["fixed"]
        improved = f_val <= b_val
        seg_col = "#059669" if improved else "#BE123C"
        end_col = "#059669" if improved else "#BE123C"
        pct = (f_val - b_val) / b_val * 100.0
        sign = "+" if pct > 0 else ""

        ax.plot([b_val, f_val], [y, y], color=seg_col, lw=2.5, alpha=0.85, zorder=3)
        ax.scatter([b_val], [y], s=58, facecolors="white", edgecolors="#475569", linewidths=1.8, zorder=4)
        ax.scatter([f_val], [y], s=62, facecolors=end_col, edgecolors="white", linewidths=0.9, zorder=5)

        # Position labels cleanly away from the x = 1.0 dashed line
        if improved:
            # Shallow transition regime (b_val ~ 1.05..1.20, f_val ~ 0.90..1.06)
            # Put baseline value to the right of the open circle, and RAR value to the left of the green dot
            ax.annotate(
                f"{b_val:.2f}",
                (b_val, y),
                xytext=(7, 0), textcoords="offset points",
                ha="left", va="center", fontsize=7.4, color="#475569", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.85),
                zorder=6,
            )
            ax.annotate(
                f"{f_val:.2f} ({sign}{pct:.0f}%)",
                (min(f_val, 0.96), y),
                xytext=(-7, 0), textcoords="offset points",
                ha="right", va="center", fontsize=7.4, color="#065F46", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.92),
                zorder=6,
            )
        else:
            # Degraded high-variance cases
            ax.annotate(
                f"{b_val:.2f}",
                (b_val, y),
                xytext=(-7, 0), textcoords="offset points",
                ha="right", va="center", fontsize=7.4, color="#475569", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.85),
                zorder=6,
            )
            ax.annotate(
                f"{f_val:.2f} ({sign}{pct:.0f}%)",
                (f_val, y),
                xytext=(7, 0), textcoords="offset points",
                ha="left", va="center", fontsize=7.4, color="#9F1239", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.92),
                zorder=6,
            )

    # Properly sized callout boxes using bbox on ax.text so text never overflows borders
    ax.text(
        4.35, 6.25,
        "Shallow transition boundary\n" + r"($\sigma^2=1.0,\ \lambda=0.03$): RAR stabilizes 4/4",
        ha="center", va="center", fontsize=7.8, color="#065F46", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", fc="#ECFDF5", ec="#059669", lw=0.95, alpha=0.96),
        zorder=6,
    )
    ax.text(
        4.35, 2.55,
        "High-variance rough regime\n" + r"($\sigma^2=4.0,\ \lambda=0.08$): RAR degrades 3/4",
        ha="center", va="center", fontsize=7.8, color="#9F1239", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", fc="#FFF1F2", ec="#E11D48", lw=0.95, alpha=0.96),
        zorder=6,
    )

    ax.set_xlim(-0.28, 6.55)
    ax.set_ylim(-0.6, len(rar) - 0.3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(
        [f"$\\sigma^2={r['sigma2']:g},\\ \\lambda={r['lam']:g}$ (Inst #{i%4+1})"
         for i, r in enumerate(rar)],
        fontsize=8.0,
    )
    ax.set_xlabel(r"Instance Darcy-flux error $E_q$ (rel. $L_2$; dashed line: $E_q = 1.0$)", fontsize=9.4, color="#0F172A")
    ax.set_title(
        "(b) Per-instance PINN: paired uniform collocation vs. RAR trajectories",
        fontsize=9.6, fontweight="bold", color="#0F172A", loc="left", pad=8,
    )
    leg_b = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="#475569", markeredgewidth=1.8, markersize=7.0, label="Uniform collocation PINN"),
        Line2D([0], [0], marker="o", color="#059669", lw=2.2, markerfacecolor="#059669", markeredgecolor="white", markersize=7.2, label="RAR improved trajectory"),
        Line2D([0], [0], marker="o", color="#BE123C", lw=2.2, markerfacecolor="#BE123C", markeredgecolor="white", markersize=7.2, label="RAR degraded trajectory"),
    ]
    ax.legend(handles=leg_b, fontsize=7.5, frameon=True, facecolor="white", edgecolor="#CBD5E1", loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = FIGDIR / out_name
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=600, metadata={"Software": None})
    plt.close(fig)
    print("wrote", out)
    return out


if __name__ == "__main__":
    make_figure3()
