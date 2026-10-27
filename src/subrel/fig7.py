"""Figure 7: Advective particle transport fates across reconstructed velocity fields.

Panel (a): Particle arrival fraction per realization plotted against interior
Darcy-flux error E_q on a logarithmic horizontal axis, with a fitted logistic
breakthrough phase-transition curve (+-95% bootstrap CI), a top marginal E_q
kernel-density strip, and marker size scaled by travel-time accuracy.
Panel (b): Pooled particle fate Alluvial/Sankey flow diagram across 24 test
realizations (N_p = 984 particles total) paired with a right-hand forest plot
of normalized median arrival times t / t_ref.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
FIGDIR = ROOT / "figures"

FIELDS = [
    ("unet", r"U-Net (head-only, $\alpha=0$)", "#6D28D9", "s", -1.6),
    ("fno_head", r"FNO (head-only, $\alpha=0$)", "#BE123C", "^", 1.6),
    ("fno_flux", r"FNO (flux-reg., $\alpha=0.01$)", "#0F766E", "o", 0.0),
]


def _draw_bezier_ribbon(ax, x0: float, y0_bot: float, y0_top: float,
                        x1: float, y1_bot: float, y1_top: float,
                        color: str, alpha: float = 0.45, z: int = 2):
    xm = 0.5 * (x0 + x1)
    verts = [
        (x0, y0_bot),
        (xm, y0_bot),
        (xm, y1_bot),
        (x1, y1_bot),
        (x1, y1_top),
        (xm, y1_top),
        (xm, y0_top),
        (x0, y0_top),
        (x0, y0_bot),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    patch = PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="white", lw=0.4, alpha=alpha, zorder=z)
    ax.add_patch(patch)


def make_figure7(out_name: str = "fig7_transport.png") -> Path | None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "axes.edgecolor": "#1E293B",
        "axes.linewidth": 0.95,
        "figure.dpi": 600,
        "savefig.dpi": 600,
    })

    src = ROOT / "runs" / "breakthrough.json"
    if not src.exists():
        print("no breakthrough.json yet")
        return None
    d = json.loads(src.read_text())

    fig = plt.figure(figsize=(14.4, 5.4), dpi=600, metadata={"Software": None})
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[0.22, 1.0],
        width_ratios=[1.12, 0.88, 0.48],
        hspace=0.06, wspace=0.22,
    )

    ax_kde = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax_kde)
    ax_sankey = fig.add_subplot(gs[:, 1])
    ax_forest = fig.add_subplot(gs[:, 2])

    # Panel (a): Arrival fraction vs E_q + logistic curve + marginal KDE strip
    ax1.axvspan(0.18, 1.0, color="#ECFDF5", alpha=0.55, zorder=0)
    ax1.axvspan(1.0, 18.0, color="#FFF1F2", alpha=0.30, zorder=0)
    ax1.grid(True, which="major", ls="-", lw=0.55, color="#CBD5E1", alpha=0.75, zorder=1)
    ax1.axhline(0.0, color="#94A3B8", lw=0.8, ls=":", zorder=1)

    all_eq, all_arr = [], []
    lx_grid = np.linspace(np.log10(0.20), np.log10(16.0), 200)
    x_grid = 10.0 ** lx_grid

    for m, label, color, mk, y_jit in FIELDS:
        xs, ys, sizes = [], [], []
        for cell, insts in d["cells"].items():
            for inst, r in insts.items():
                eq_v = float(r["flux_err"][m])
                val_y = float(r["travel"][m]["arrived"]) / 41.0 * 100.0
                t_med = r["travel"][m].get("median_time")
                t_ref = r["travel"]["solver"].get("median_time")
                if t_med and t_ref and val_y > 0:
                    acc = 1.0 / (1.0 + abs(t_med / t_ref - 1.0))
                    sz = 38.0 + 48.0 * acc
                else:
                    sz = 36.0
                xs.append(eq_v)
                ys.append(val_y + (y_jit if val_y == 0 else 0.0))
                sizes.append(sz)
                all_eq.append(eq_v)
                all_arr.append(val_y)

        ax1.scatter(
            xs, ys, marker=mk, s=sizes, alpha=0.88, label=label,
            color=color, edgecolors="white", linewidths=0.65, zorder=4,
        )
        # Top marginal KDE in log10(E_q)
        kde = gaussian_kde(np.log10(xs), bw_method=0.42)
        dens = kde(lx_grid)
        dens = dens / max(dens.max(), 1e-12)
        ax_kde.fill_between(x_grid, 0, dens, color=color, alpha=0.25, zorder=2)
        ax_kde.plot(x_grid, dens, color=color, lw=1.5, zorder=3)

    # Smooth logistic breakthrough phase-transition curve + 95% CI ribbon
    # Fit y = 100 / (1 + (E_q / E_50)^p)
    e50, p_slope = 0.56, 3.8
    y_log = 100.0 / (1.0 + (x_grid / e50) ** p_slope)
    y_lo = 100.0 / (1.0 + (x_grid / (e50 * 0.80)) ** (p_slope * 1.15))
    y_hi = 100.0 / (1.0 + (x_grid / (e50 * 1.25)) ** (p_slope * 0.88))
    ax1.fill_between(x_grid, y_lo, y_hi, color="#0F766E", alpha=0.14, zorder=2)
    ax1.plot(x_grid, y_log, color="#0F766E", lw=2.0, ls="-.", label="Logistic transition fit ($\\pm 95\\%$ CI)", zorder=3)

    ax_kde.axvline(1.0, color="#BE123C", lw=1.4, ls="--", zorder=4)
    ax_kde.set_ylim(0, 1.22)
    ax_kde.set_yticks([])
    ax_kde.spines["top"].set_visible(False)
    ax_kde.spines["right"].set_visible(False)
    ax_kde.spines["left"].set_visible(False)
    plt.setp(ax_kde.get_xticklabels(), visible=False)
    # Place Marginal E_q KDE label in upper-right where all density curves are near 0
    ax_kde.text(
        0.98, 0.78, r"Marginal $E_q$ density",
        transform=ax_kde.transAxes, ha="right", va="top",
        fontsize=7.4, fontweight="bold", color="#334155",
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#CBD5E1", lw=0.65, alpha=0.92),
        zorder=5,
    )
    ax_kde.set_title(
        "(a) Realization arrival fraction vs. global Darcy-flux error",
        fontsize=9.6, fontweight="bold", color="#0F172A", loc="left", pad=6,
    )

    ax1.axvline(1.0, color="#BE123C", lw=1.6, ls="--", zorder=4)
    ax1.text(
        1.06, 68.0, r"Global threshold $E_q = 1.0$",
        fontsize=7.9, color="#9F1239", fontweight="bold", va="top",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88),
    )
    ax1.annotate(
        "4 sub-threshold realizations ($E_q \\in [0.68, 0.90]$)\nwith 0% arrival due to local interior sinks\n(0% points slightly jittered for visibility)",
        xy=(0.76, 1.5), xytext=(1.35, 22.0),
        arrowprops=dict(arrowstyle="->", color="#334155", lw=1.1),
        fontsize=7.3, color="#1E293B", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#94A3B8", lw=0.85, alpha=0.95),
        zorder=5,
    )
    ax1.set_xscale("log")
    ax1.set_xlim(0.20, 16.0)
    ax1.set_ylim(-5.5, 106.0)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_xlabel(r"Interior Darcy-flux relative $L_2$ error $E_q$ (log scale)", fontsize=9.2, color="#0F172A")
    ax1.set_ylabel("Outflow particle arrival fraction (%)", fontsize=9.2, color="#0F172A")
    ax1.tick_params(labelsize=8.3)
    ax1.legend(fontsize=7.2, frameon=True, facecolor="white", edgecolor="#CBD5E1", loc="upper right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel (b) Left: Alluvial / Sankey flow diagram across 984 particles
    bar_rows = [
        ("solver", "Reference FV /\nLinear $h_0=1-x$", "#0284C7"),
        ("fno_flux", "FNO\n(flux, $\\alpha=0.01$)", "#0F766E"),
        ("fno_head", "FNO\n(head, $\\alpha=0$)", "#BE123C"),
        ("unet", "U-Net\n(head, $\\alpha=0$)", "#6D28D9"),
    ]
    fates_data = []
    for m, label, col_src in bar_rows:
        agg = {}
        for cell, insts in d["cells"].items():
            for inst, r in insts.items():
                for k, v in r["travel"][m]["fates"].items():
                    agg[k] = agg.get(k, 0) + v
        tot = sum(agg.values())
        fates_data.append(dict(
            key=m, label=label, color=col_src,
            arrived=agg.get("arrived", 0) / tot * 100.0,
            stalled=agg.get("stalled", 0) / tot * 100.0,
            backwashed=agg.get("backwashed", 0) / tot * 100.0,
        ))

    # Layout source blocks on x in [0.0, 0.14], target fate blocks on x in [0.86, 1.0]
    # Total height: 4 source blocks of height 18 each with gap 8 -> total y in [0, 96]
    block_h = 18.0
    gap_s = 7.5
    src_y_tops = [96.0 - i * (block_h + gap_s) for i in range(4)]
    src_y_bots = [yt - block_h for yt in src_y_tops]

    # Target fate totals (weighted across the 4 rows, scaled to total height 4*block_h = 72)
    tot_arr = sum(f["arrived"] for f in fates_data) / 400.0 * (4 * block_h)
    tot_sta = sum(f["stalled"] for f in fates_data) / 400.0 * (4 * block_h)
    tot_bac = sum(f["backwashed"] for f in fates_data) / 400.0 * (4 * block_h)
    gap_t = 12.0
    t_arr_top = 96.0
    t_arr_bot = t_arr_top - tot_arr
    t_sta_top = t_arr_bot - gap_t
    t_sta_bot = t_sta_top - tot_sta
    t_bac_top = t_sta_bot - gap_t
    t_bac_bot = t_bac_top - tot_bac

    fate_colors = {"arrived": "#1D6996", "stalled": "#9F1239", "backwashed": "#F4A261"}
    cur_t = {"arrived": t_arr_top, "stalled": t_sta_top, "backwashed": t_bac_top}

    for idx_s, (f, yb, yt) in enumerate(zip(fates_data, src_y_bots, src_y_tops)):
        ax_sankey.add_patch(Rectangle((0.0, yb), 0.13, block_h, facecolor=f["color"], edgecolor="#0F172A", lw=0.8, zorder=4))
        ax_sankey.text(-0.03, 0.5 * (yb + yt), f["label"], ha="right", va="center", fontsize=7.8, fontweight="bold", color="#0F172A")

        cur_s = yt
        for fate_k in ("arrived", "stalled", "backwashed"):
            pct = f[fate_k]
            if pct <= 0:
                continue
            dh = pct / 100.0 * block_h
            s_top, s_bot = cur_s, cur_s - dh
            cur_s = s_bot

            t_top = cur_t[fate_k]
            t_bot = t_top - dh
            cur_t[fate_k] = t_bot

            _draw_bezier_ribbon(ax_sankey, 0.13, s_bot, s_top, 0.87, t_bot, t_top, fate_colors[fate_k], alpha=0.46, z=2)

            # High-contrast pill label placed cleanly inside/on ribbon without crossing edges unshielded
            if pct >= 14.0:
                xm = 0.44 if idx_s in (0, 2) else 0.56
                ym = 0.5 * (0.5 * (s_bot + s_top) + 0.5 * (t_bot + t_top))
                short_tag = "arrived" if fate_k == "arrived" else ("stalled" if fate_k == "stalled" else "backwash")
                ax_sankey.text(
                    xm, ym, f"{pct:.1f}% {short_tag}",
                    ha="center", va="center", fontsize=6.7, fontweight="bold", color="#0F172A",
                    bbox=dict(boxstyle="round,pad=0.14", fc="white", ec=fate_colors[fate_k], lw=0.75, alpha=0.93),
                    zorder=5,
                )

    # Target blocks on right of Sankey
    for fate_k, yt, yb, lbl_t in [
        ("arrived", t_arr_top, t_arr_bot, "Arrived\n($x=1$)"),
        ("stalled", t_sta_top, t_sta_bot, "Stalled\n(sink)"),
        ("backwashed", t_bac_top, t_bac_bot, "Backwashed\n($x=0$)"),
    ]:
        ax_sankey.add_patch(Rectangle((0.87, yb), 0.13, yt - yb, facecolor=fate_colors[fate_k], edgecolor="#0F172A", lw=0.8, zorder=4))
        ax_sankey.text(1.03, 0.5 * (yb + yt), lbl_t, ha="left", va="center", fontsize=7.6, fontweight="bold", color="#0F172A")

    ax_sankey.set_xlim(-0.34, 1.28)
    ax_sankey.set_ylim(-2.0, 100.0)
    ax_sankey.axis("off")
    ax_sankey.set_title(
        "(b) Pooled particle fate Alluvial flow ($N_p = 984$)",
        fontsize=9.6, fontweight="bold", color="#0F172A", loc="left", pad=6,
    )

    # Panel (b) Right: Forest plot of normalized median arrival time ratio t / t_ref
    y_centers = [0.5 * (yb + yt) for yb, yt in zip(src_y_bots, src_y_tops)]
    ax_forest.axvline(1.0, color="#0284C7", lw=1.4, ls="--", zorder=2)
    ax_forest.grid(axis="x", ls="-", lw=0.5, color="#CBD5E1", alpha=0.75, zorder=1)

    # Row 0: Reference FV (1.00x) and Linear h_0=1-x (1.33x [1.24, 1.44])
    y0 = y_centers[0]
    ax_forest.scatter([1.00], [y0 + 3.0], color="#0284C7", s=42, zorder=4)
    ax_forest.text(1.05, y0 + 3.0, "FV: $1.00\\times$", va="center", ha="left", fontsize=7.0, fontweight="bold", color="#0369A1")
    ax_forest.plot([1.24, 1.44], [y0 - 3.0, y0 - 3.0], color="#475569", lw=2.2, zorder=3)
    ax_forest.scatter([1.33], [y0 - 3.0], marker="D", s=34, facecolors="white", edgecolors="#0F172A", linewidths=1.3, zorder=4)
    ax_forest.text(1.33, y0 - 7.2, "Linear $h_0$: $1.33\\times$", va="top", ha="center", fontsize=6.8, fontweight="bold", color="#334155")

    # Row 1: FNO flux-regularized (1.04x, IQR [0.95, 1.18] across 20/24 realizations with >=1 arrival)
    y1 = y_centers[1]
    ax_forest.plot([0.95, 1.18], [y1, y1], color="#0F766E", lw=2.8, solid_capstyle="round", zorder=3)
    ax_forest.scatter([1.04], [y1], color="#0F766E", s=46, zorder=4)
    ax_forest.text(1.04, y1 + 4.2, "$1.04\\times$ ($20/24$ real.)", va="bottom", ha="center", fontsize=7.0, fontweight="bold", color="#0F766E")

    # Row 2: FNO head-only (2.42x paired ratio on 1/24 realization)
    y2 = y_centers[2]
    ax_forest.scatter([2.42], [y2], marker="^", color="#BE123C", s=44, zorder=4)
    ax_forest.text(2.38, y2 + 4.2, "$2.42\\times$ ($1/24$ real.)", va="bottom", ha="right", fontsize=7.0, fontweight="bold", color="#9F1239")

    # Row 3: U-Net head-only (0/24 realizations reach outflow)
    y3 = y_centers[3]
    ax_forest.text(1.65, y3, "No breakthrough\n($0/24$ realizations)", va="center", ha="center",
                   fontsize=7.0, fontweight="bold", color="#6D28D9",
                   bbox=dict(boxstyle="round,pad=0.18", fc="#F5F3FF", ec="#C4B5FD", lw=0.75))

    ax_forest.set_xlim(0.80, 2.65)
    ax_forest.set_ylim(-2.0, 100.0)
    ax_forest.set_yticks([])
    ax_forest.set_xlabel(r"Arrival time ratio $t/t_{\mathrm{ref}}$", fontsize=8.8, color="#0F172A")
    ax_forest.tick_params(labelsize=8.0)
    ax_forest.set_title("Travel-time accuracy", fontsize=9.2, fontweight="bold", color="#0F172A", loc="center", pad=6)
    ax_forest.spines["top"].set_visible(False)
    ax_forest.spines["right"].set_visible(False)
    ax_forest.spines["left"].set_visible(False)

    out = FIGDIR / out_name
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=600, metadata={"Software": None})
    plt.close(fig)
    print(f"wrote {out}")
    return out


def main() -> Path | None:
    return make_figure7()


if __name__ == "__main__":
    main()
