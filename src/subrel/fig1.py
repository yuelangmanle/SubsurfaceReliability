"""Figure 1: Benchmark failure phase diagrams and head-versus-flux decoupling.

Top panel (a): Interior hydraulic-head relative L2 error E_h versus interior
Darcy-flux relative L2 error E_q across 3,264 held-out test evaluations,
plotted with distinct markers per architecture family, colored by the
roughness ratio sigma^2/lambda, with 90% log-space covariance hulls, an
analytical k^2 derivative-gain reference, and the analytical linear head
profile h_0(x) = 1 - x (open white diamonds).
Bottom panels (b)-(d): Per-family 4x6 regime heatmaps over the (lambda, sigma^2)
parameter space with protective pill annotations and E_q = 1.0 reliability contours.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from .analysis import ROOT, collect

FIGDIR = ROOT / "figures"
DISPLAY_NAME = {"unet": "U-Net", "fno": "FNO", "pinn": "PINN"}
MARKERS = {"unet": "o", "fno": "s", "pinn": "^"}
HULL_COLORS = {"unet": "#D55E00", "fno": "#0072B2", "pinn": "#009E73"}


def _style_rc():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "axes.edgecolor": "#1E293B",
        "axes.linewidth": 1.05,
        "xtick.color": "#1E293B",
        "ytick.color": "#1E293B",
        "figure.dpi": 600,
        "savefig.dpi": 600,
    })


def _draw_log_ellipse(ax, x_vals: np.ndarray, y_vals: np.ndarray, color: str, z: int = 3):
    lx = np.log10(np.clip(x_vals, 1e-6, None))
    ly = np.log10(np.clip(y_vals, 1e-6, None))
    cov = np.cov(lx, ly)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    r = 2.146
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
    transform = vecs @ np.diag(r * np.sqrt(np.maximum(vals, 1e-9)))
    pts = transform @ circle
    ex = 10.0 ** (lx.mean() + pts[0])
    ey = 10.0 ** (ly.mean() + pts[1])
    ax.plot(ex, ey, ls="--", lw=1.5, color=color, alpha=0.85, zorder=z)


def _phase_panel(ax, sub, title, subtitle, lams, s2s):
    grid = np.full((len(s2s), len(lams)), np.nan)
    for (lam, s2), g in sub.groupby(["lam", "sigma2"]):
        i, j = s2s.index(s2), lams.index(lam)
        grid[i, j] = np.median(g.flux_err)

    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap="magma_r",
        norm=LogNorm(vmin=0.008, vmax=30.0),
        extent=[-0.5, len(lams) - 0.5, -0.5, len(s2s) - 0.5],
        zorder=1,
    )

    for x in np.arange(-0.5, len(lams), 1.0):
        ax.axvline(x, color="white", lw=1.15, alpha=0.80, zorder=2)
    for y in np.arange(-0.5, len(s2s), 1.0):
        ax.axhline(y, color="white", lw=1.15, alpha=0.80, zorder=2)

    has_fail = np.nansum(grid > 1.0) > 0
    has_pass = np.nansum(grid <= 1.0) > 0
    if has_fail and has_pass:
        X, Y = np.meshgrid(np.arange(len(lams)), np.arange(len(s2s)))
        cs = ax.contour(
            X, Y, np.log10(grid),
            levels=[0.0],
            colors=["#00F5D4"],
            linewidths=2.6,
            linestyles="-",
            zorder=3,
        )
        if hasattr(cs, "collections"):
            for col in cs.collections:
                col.set_path_effects([pe.Stroke(linewidth=4.0, foreground="#0B132B"), pe.Normal()])
        else:
            cs.set_path_effects([pe.Stroke(linewidth=4.0, foreground="#0B132B"), pe.Normal()])

    for i in range(len(s2s)):
        for j in range(len(lams)):
            val = grid[i, j]
            if not np.isfinite(val):
                continue
            txt = f"{val:.2f}" if val < 10.0 else f"{val:.1f}"
            if val < 0.01:
                txt = f"{val:.3f}"
            is_fail = val > 1.0
            if val >= 0.42:
                color = "white"
                pill_fc = "#0F172A"
                pill_alpha = 0.52
            else:
                color = "#0F172A"
                pill_fc = "white"
                pill_alpha = 0.78
            weight = "bold" if not is_fail else "normal"
            ax.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=8.6, color=color, fontweight=weight,
                bbox=dict(boxstyle="round,pad=0.15", fc=pill_fc, ec="none", alpha=pill_alpha),
                zorder=5,
            )

    ax.set_xticks(range(len(lams)))
    ax.set_xticklabels([f"{l:g}" for l in lams], fontsize=9.2)
    ax.set_yticks(range(len(s2s)))
    ax.set_yticklabels([f"{s:g}" for s in s2s], fontsize=9.2)
    ax.set_title(
        f"{title}\n",
        fontsize=10.6, fontweight="bold", color="#0F172A", loc="left", pad=3,
    )
    ax.text(
        0.0, 1.018, subtitle,
        transform=ax.transAxes, fontsize=8.8, color="#475569", fontweight="normal",
    )
    ax.set_xlabel(r"Dimensionless correlation length $\lambda/L$", fontsize=9.8, color="#0F172A")
    ax.set_ylabel(r"Log-conductivity variance $\sigma^2$", fontsize=9.8, color="#0F172A")
    return im


def make_figure1(out_name: str = "fig1_phase.png", k_cut: float = 8.0) -> Path | None:
    _style_rc()
    df = collect()
    df = df[df.model.isin(["unet", "fno", "pinn"])].copy()
    if not len(df):
        print("no results yet")
        return None
    FIGDIR.mkdir(exist_ok=True)
    lams = sorted(df.lam.unique())
    s2s = sorted(df.sigma2.unique())

    fig = plt.figure(figsize=(13.6, 9.2), dpi=600)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.32, 1.0], hspace=0.40, wspace=0.25)

    # Top panel (a)
    ax = fig.add_subplot(gs[0, :])
    pdf = df
    rough = pdf.sigma2 / pdf.lam
    norm_r = LogNorm(vmin=float(rough.min()), vmax=float(rough.max()))

    ax.axhspan(1.0, 500.0, color="#FFF1F2", alpha=0.55, zorder=0)
    ax.axhspan(1e-4, 1.0, color="#ECFDF5", alpha=0.55, zorder=0)
    ax.axvspan(0.035, 0.085, color="#E2E8F0", alpha=0.48, zorder=0)
    ax.grid(True, which="major", ls="-", lw=0.6, color="#CBD5E1", alpha=0.75, zorder=1)

    sc = None
    for fam, marker, sz, al in [("unet", "o", 22, 0.76), ("fno", "s", 21, 0.76), ("pinn", "^", 28, 0.90)]:
        sub = pdf[pdf.model == fam]
        sc = ax.scatter(
            sub.head_err, sub.flux_err,
            c=sub.sigma2 / sub.lam,
            cmap="viridis",
            norm=norm_r,
            marker=marker,
            s=sz,
            alpha=al,
            edgecolors="none",
            zorder=3,
        )
        _draw_log_ellipse(ax, sub.head_err.to_numpy(), sub.flux_err.to_numpy(), HULL_COLORS[fam], z=3)

    x_ref = np.array([1.2e-4, 2.2e-3])
    y_ref = 16.0 * (x_ref / 2.2e-3) ** 1.0
    ax.plot(x_ref, y_ref, ls="-.", lw=1.5, color="#334155", alpha=0.88, zorder=4)
    ax.text(
        1.5e-4, 2.8,
        r"$E_q \propto (k_{\mathrm{eff}} L)\, E_h$ spectral derivative gain",
        fontsize=8.6, color="#1E293B", fontweight="bold", rotation=23,
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.88),
        zorder=5,
    )

    ax.axhline(1.0, color="#BE123C", lw=2.0, ls="--", zorder=4)
    ax.axvline(0.15, color="#64748B", lw=1.1, ls=":", alpha=0.80, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(4e-5, 1.45)
    ax.set_ylim(5e-4, 450.0)
    ax.set_xlabel(r"Interior hydraulic-head relative $L_2$ error $E_h$", fontsize=10.5, color="#0F172A")
    ax.set_ylabel(r"Interior Darcy-flux relative $L_2$ error $E_q$", fontsize=10.5, color="#0F172A")
    ax.tick_params(labelsize=9.4)

    ax.text(
        7.0e-5, 1.35,
        r"Zero-velocity threshold $E_q = 1.0$ (worse than $\hat{\mathbf{q}} = \mathbf{0}$)",
        fontsize=8.8, color="#9F1239", fontweight="bold", va="bottom",
        bbox=dict(boxstyle="round,pad=0.18", fc="#FFF1F2", ec="#FDA4AF", lw=0.85, alpha=0.95),
        zorder=6,
    )
    ax.text(
        7.0e-5, 0.44,
        r"Skillful velocity reconstruction zone ($E_q \leq 1.0$)",
        fontsize=8.8, color="#065F46", fontweight="bold", va="top",
        bbox=dict(boxstyle="round,pad=0.18", fc="#ECFDF5", ec="#6EE7B7", lw=0.85, alpha=0.95),
        zorder=6,
    )

    ax.text(
        0.055, 340.0,
        "Amortized head-error band\n" + r"($E_h \approx 3.5\%\text{–}8.5\%$ across $100\times$ flux spread)",
        fontsize=8.5, color="#1E293B", fontweight="bold", ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#94A3B8", lw=0.9, alpha=0.95),
        zorder=6,
    )

    badges = [
        ("U-Net ($N=1,536$)\n" + r"$91.4\% > 1.0,\ \mathrm{med}\ E_q = 2.75$", (0.46, 62.0), "#D55E00"),
        ("FNO ($N=1,536$)\n" + r"$75.3\% > 1.0,\ \mathrm{med}\ E_q = 1.71$", (0.0085, 36.0), "#0072B2"),
        ("PINN ($N=192$)\n" + r"$44.3\% > 1.0,\ \mathrm{med}\ E_q = 0.90$", (0.38, 0.045), "#009E73"),
    ]
    for label_txt, (hx, fy), ec_col in badges:
        ax.annotate(
            label_txt, (hx, fy),
            fontsize=8.6, color="#0F172A", fontweight="bold", ha="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=ec_col, lw=1.1, alpha=0.95),
        )

    tb = json.loads((ROOT / "runs" / "repairs" / "trivial_baseline.json").read_text())
    ax.scatter(
        [t["head_err"] for t in tb], [t["flux_err"] for t in tb],
        marker="D", s=54, facecolors="white", edgecolors="#0F172A",
        linewidths=1.5, zorder=6,
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2A9D8F", markersize=7.2, label="U-Net ($N = 1,536$ evaluations; 90% hull)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#264653", markersize=7.2, label="FNO ($N = 1,536$ evaluations; 90% hull)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#E9C46A", markeredgecolor="#264653", markersize=7.6, label="PINN ($N = 192$ evaluations; 90% hull)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="white", markeredgecolor="#0F172A", markeredgewidth=1.5, markersize=7.0, label=r"Linear head baseline $h_0(x) = 1 - x$ (24 medians)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8.6, frameon=True, facecolor="white", edgecolor="#CBD5E1", loc="lower right")
    ax.set_title(
        "(a) Decoupling between interior hydraulic-head error and Darcy-flux error across 3,264 held-out evaluations",
        fontsize=11.2, fontweight="bold", color="#0F172A", loc="left", pad=8,
    )
    cb2 = fig.colorbar(sc, ax=ax, fraction=0.024, pad=0.012)
    cb2.set_label(r"Medium roughness ratio $\sigma^2/\lambda$", fontsize=9.6, color="#0F172A")
    cb2.ax.tick_params(labelsize=9.0)

    order = [
        ("unet", "(b) U-Net (convolutional encoder)", r"Fails in $23/24$ regimes ($\sigma^2/\lambda$ boundary)"),
        ("fno", "(c) FNO (spectral operator, $m=12$)", r"Fails in $17/24$ regimes ($\sigma^2$ boundary)"),
        ("pinn", "(d) PINN (coordinate PDE optimization)", r"Fails in $10/24$ regimes ($\sigma^2/\lambda^2$ boundary)"),
    ]
    axes = [fig.add_subplot(gs[1, k]) for k in range(3)]
    im = None
    for ax_sub, (m, title, sub_txt) in zip(axes, order):
        sub = df[df.model == m]
        im = _phase_panel(ax_sub, sub, title, sub_txt, lams, s2s)
    if im is not None:
        cb = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.018)
        cb.set_label(r"Regime median Darcy-flux error $E_q$ (log scale; cyan contour: $E_q = 1.0$)", fontsize=9.4, color="#0F172A")
        cb.ax.tick_params(labelsize=9.0)

    out = FIGDIR / out_name
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=600, metadata={"Software": None})
    plt.close(fig)
    print("wrote", out, f"({len(df)} instances)")
    return out


if __name__ == "__main__":
    make_figure1()
