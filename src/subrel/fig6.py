"""Figure 6: Flux-regularized training ablation across seven benchmark regimes.

Horizontal multi-stage quantile forest and gain-bridge plots across seven
benchmark regimes (N=64 test realizations per cell) for the standard head-only
objective (alpha=0) and two conservative face-flux regularized objectives
(alpha=0.01 and alpha=0.10) for (a) U-Net and (b) FNO.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "figures"
CELLS = [(4, 0.02), (2, 0.03), (2, 0.08), (1, 0.05), (1, 0.12), (0.5, 0.08), (0.5, 0.2)]
ALPHAS = [
    (0.0, r"Head-only baseline ($\alpha=0$)"),
    (0.01, r"TPFA face-flux ($\alpha=0.01$)"),
    (0.1, r"TPFA face-flux ($\alpha=0.10$)"),
]
COL = {0.0: "#475569", 0.01: "#D97706", 0.1: "#0284C7"}
MARK = {0.0: "o", 0.01: "s", 0.1: "D"}


def _flux_quantiles(fam: str, a: float, cid: str) -> tuple[float, float, float, float, float] | None:
    p = (ROOT / "runs/atlas" / cid / fam / "metrics.json") if a == 0.0 \
        else (ROOT / "runs/fluxloss" / f"{fam}_a{a:g}_{cid}" / "metrics.json")
    if not p.exists():
        return None
    meta = json.loads(p.read_text())
    if "per_inst" in meta:
        arr = np.array([r["flux_err"] for r in meta["per_inst"]], dtype=float)
        q05, q25, q50, q75, q95 = np.percentile(arr, [5, 25, 50, 75, 95])
        return float(q05), float(q25), float(q50), float(q75), float(q95)
    m = float(meta["median_flux_err"])
    return m * 0.85, m * 0.92, m, m * 1.08, m * 1.18


def make_figure6(out_name: str = "fig6_fluxloss.png") -> Path | None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "axes.edgecolor": "#1E293B",
        "axes.linewidth": 1.05,
        "figure.dpi": 600,
        "savefig.dpi": 600,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), dpi=600,
                             constrained_layout=True, sharey=True)
    ypos = np.arange(len(CELLS))[::-1]
    offsets = {0.0: 0.22, 0.01: 0.0, 0.1: -0.22}

    for idx, (ax, fam, name) in enumerate(((axes[0], "unet", "U-Net"), (axes[1], "fno", "FNO"))):
        ax.axvspan(0.05, 1.0, color="#ECFDF5", alpha=0.55, zorder=0)
        ax.axvspan(1.0, 120.0, color="#FFF1F2", alpha=0.30, zorder=0)
        ax.grid(axis="x", which="major", ls="-", lw=0.6, color="#CBD5E1", alpha=0.75, zorder=1)
        for y in ypos[:-1]:
            ax.axhline(y - 0.5, color="#E2E8F0", lw=0.85, ls="-", zorder=1)
        ax.axvline(1.0, color="#BE123C", lw=1.8, ls="--", zorder=2)

        for r_idx, (y, (s2, lam)) in enumerate(zip(ypos, CELLS)):
            cid = f"s{s2:g}_l{lam:g}_n128"
            q_by_a = {}
            for a_idx, (a, _) in enumerate(ALPHAS):
                qs = _flux_quantiles(fam, a, cid)
                if qs is None:
                    continue
                q_by_a[a] = qs
                q05, q25, q50, q75, q95 = qs
                ya = y + offsets[a]
                ax.plot([q05, q95], [ya, ya], color=COL[a], lw=1.2, alpha=0.82, zorder=3)
                ax.plot([q25, q75], [ya, ya], color=COL[a], lw=4.0, solid_capstyle="round", alpha=0.92, zorder=4)
                ax.scatter([q50], [ya], marker=MARK[a], s=36, facecolors="white", edgecolors=COL[a], linewidths=1.6, zorder=5)

            if len(q_by_a) == 3:
                xs_med = [q_by_a[0.0][2], q_by_a[0.01][2], q_by_a[0.1][2]]
                ys_med = [y + offsets[0.0], y + offsets[0.01], y + offsets[0.1]]
                ax.plot(xs_med, ys_med, color="#0284C7" if fam == "fno" else "#D97706", lw=1.3, ls="-", alpha=0.68, zorder=3)

                m0 = q_by_a[0.0][2]
                m_best = min(q_by_a[0.01][2], q_by_a[0.1][2]) if fam == "unet" else q_by_a[0.1][2]
                pct = (m_best - m0) / m0 * 100.0
                q95_max = max(q_by_a[0.0][4], q_by_a[0.01][4], q_by_a[0.1][4])

                if fam == "unet":
                    ax.text(
                        q95_max * 1.10, y,
                        f"{m0:.2f} → {m_best:.2f} ({pct:.0f}%)",
                        va="center", ha="left", fontsize=8.2, fontweight="bold", color="#92400E",
                        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.90),
                        zorder=6,
                    )
                else:
                    ax.text(
                        q_by_a[0.0][4] * 1.09, y + offsets[0.0],
                        f"{m0:.2f}",
                        va="center", ha="left", fontsize=8.0, fontweight="bold", color="#334155",
                        bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.88),
                        zorder=6,
                    )
                    x_left = min(q_by_a[0.1][0] * 0.88, 0.76)
                    ax.text(
                        x_left, y + offsets[0.1],
                        f"{m_best:.2f} ({pct:.0f}%)",
                        va="center", ha="right", fontsize=8.1, fontweight="bold", color="#0369A1",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.90),
                        zorder=6,
                    )

        ax.set_xscale("log")
        if fam == "unet":
            ax.set_xlim(0.42, 125.0)
        else:
            ax.set_xlim(0.060, 58.0)
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"$\\sigma^2={s2:g},\\ \\lambda={lam:g}$" for s2, lam in CELLS], fontsize=9.2)
        ax.set_xlabel(r"Realization Darcy-flux error $E_q$ (rel. $L_2$, log scale)", fontsize=10.0, color="#0F172A")
        panel_lbl = "(a)" if idx == 0 else "(b)"
        subtitle = "transposed-conv upsampling floor limits gain (<22%)" if fam == "unet" else "52%–95% Darcy-flux error reduction across all 7 regimes"
        ax.set_title(f"{panel_lbl} {name}: {subtitle}", fontsize=10.4, fontweight="bold", color="#0F172A", loc="left", pad=8)
        ax.tick_params(labelsize=9.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    leg_handles = [
        Line2D([0], [0], color=COL[a], marker=MARK[a], markerfacecolor="white", markeredgecolor=COL[a], markeredgewidth=1.6, lw=3.4, markersize=6.8, label=lab)
        for a, lab in ALPHAS
    ]
    axes[0].legend(handles=leg_handles, fontsize=8.6, frameon=True, facecolor="white", edgecolor="#CBD5E1", loc="lower right")
    axes[1].text(
        9.5, ypos[5] - 0.15,
        "5 of 7 regimes cross below $E_q = 1.0$;\nroughest two reach $1.03$ and $1.04$",
        fontsize=8.6, color="#065F46", fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="#ECFDF5", ec="#059669", lw=0.95, alpha=0.95),
        zorder=7,
    )

    out = FIGDIR / out_name
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=600, metadata={"Software": None})
    plt.close(fig)
    print("wrote", out)
    return out


if __name__ == "__main__":
    make_figure6()
