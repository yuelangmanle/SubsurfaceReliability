"""Figure 5: Covariance-family intervention across matched spectral regimes.

Log-conductivity variance is fixed to sigma^2 and the 1D radial integral scale
is matched analytically (L_int = lam*sqrt(pi/2)) across Gaussian, Matern nu=3/2,
and Matern nu=1/2 covariance models. The horizontal axis reports the measured
fraction Phi of log-conductivity variance carried above k = 8 cycles per domain.
Includes 25th-75th percentile interquartile fan ribbons, vertical marginal
half-violin KDEs, and top-left relative Delta E_q lollipop insets.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from .spectral_phi import cell_phi

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "figures"
CELLS = [("s1_l0.05_n128", 1.0, 0.05), ("s2_l0.03_n128", 2.0, 0.03)]
COVS = ["gauss", "m15", "m05"]
COV_NAMES = {"gauss": "Gaussian", "m15": r"Matérn $\nu=3/2$", "m05": r"Matérn $\nu=1/2$"}
INSTS = [288, 289, 290, 291]
COL = {"unet": "#D55E00", "fno": "#0072B2", "pinn": "#009E73"}
MARK = {"unet": "o", "fno": "s", "pinn": "^"}
LAB = {"unet": "U-Net (amortized)", "fno": "FNO (amortized, $m=12$)", "pinn": "PINN (per-instance)"}


def _h5(cid: str, cov: str) -> Path:
    sfx = "" if cov == "gauss" else f"_{cov}"
    return ROOT / "data" / f"cell_{cid}{sfx}.h5"


def _flux_dist(cid: str, cov: str, fam: str) -> np.ndarray | None:
    if cov == "gauss":
        d = ROOT / "runs/atlas" / cid / fam
    else:
        d = ROOT / "runs/covariance" / f"{cov}_{cid}" / fam
        if not d.exists():
            d = ROOT / "runs/cov" / f"{cov}_{cid}" / fam
    if fam == "pinn":
        xs = [json.loads((d / f"pinn_inst{i}.json").read_text())["flux_err"]
              for i in INSTS if (d / f"pinn_inst{i}.json").exists()]
        return np.array(xs, dtype=float) if len(xs) == len(INSTS) else None
    p = d / "metrics.json"
    if not p.exists():
        return None
    meta = json.loads(p.read_text())
    if "per_inst" in meta:
        return np.array([ r["flux_err"] for r in meta["per_inst"] ], dtype=float)
    return np.array([meta["median_flux_err"]], dtype=float)


def _draw_half_violin(ax, x_val: float, vals: np.ndarray, color: str, width_x: float = 1.65):
    if len(vals) < 4:
        return
    vmin, vmax = np.percentile(vals, [5, 95])
    if vmax <= vmin + 1e-6:
        return
    ly = np.log10(np.clip(vals, 1e-4, None))
    kde = gaussian_kde(ly, bw_method=0.45)
    ly_grid = np.linspace(np.log10(vmin), np.log10(vmax), 60)
    dens = kde(ly_grid)
    dens = dens / max(dens.max(), 1e-12) * width_x
    y_grid = 10.0 ** ly_grid
    ax.fill_betweenx(
        y_grid, x_val, x_val + dens,
        facecolor=color, edgecolor=color, alpha=0.22, lw=0.7, zorder=2,
    )


def make_figure5(out_name: str = "fig5_covariance.png") -> Path | None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "axes.edgecolor": "#1E293B",
        "axes.linewidth": 0.95,
        "figure.dpi": 600,
        "savefig.dpi": 600,
    })

    cache_path = ROOT / "runs" / "fig5_cache.npz"
    cache_data = dict(np.load(cache_path)) if cache_path.exists() else {}
    cache_updated = False

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), dpi=600, constrained_layout=True)
    for idx, (ax, (cid, s2, lam)) in enumerate(zip(axes, CELLS)):
        ax.axhspan(0.5, 1.0, color="#ECFDF5", alpha=0.55, zorder=0)
        ax.axhspan(1.0, 20.0, color="#FFF1F2", alpha=0.30, zorder=0)
        ax.grid(True, which="major", ls="-", lw=0.55, color="#CBD5E1", alpha=0.75, zorder=1)
        if f"{cid}_phi_mu" in cache_data and not _h5(cid, "gauss").exists():
            phi_mu = cache_data[f"{cid}_phi_mu"]
            phi_sd = cache_data[f"{cid}_phi_sd"]
            Ls = list(cache_data[f"{cid}_Ls"])
        else:
            phi_mu, phi_sd, Ls = [], [], []
            for cov in COVS:
                p = _h5(cid, cov)
                if not p.exists():
                    phi_mu.append(np.nan); phi_sd.append(0); Ls.append(np.nan); continue
                phis, L = cell_phi(p)
                phi_mu.append(100 * phis.mean()); phi_sd.append(100 * phis.std())
                Ls.append(L.mean())
            phi_mu = np.array(phi_mu); phi_sd = np.array(phi_sd)
            cache_data[f"{cid}_phi_mu"] = phi_mu
            cache_data[f"{cid}_phi_sd"] = phi_sd
            cache_data[f"{cid}_Ls"] = np.array(Ls)
            cache_updated = True

        for x_val in phi_mu:
            ax.axvline(x_val, color="#94A3B8", lw=0.95, ls=":", zorder=2)

        pct_by_fam = {}
        for fam in ("unet", "fno", "pinn"):
            dists = [_flux_dist(cid, cov, fam) for cov in COVS]
            if any(d is None for d in dists):
                continue
            ys = np.array([np.median(d) for d in dists])
            q25 = np.array([np.percentile(d, 25) for d in dists])
            q75 = np.array([np.percentile(d, 75) for d in dists])

            # IQR fan ribbon + marginal half-violin KDEs
            ax.fill_between(phi_mu, q25, q75, color=COL[fam], alpha=0.16, zorder=2)
            for x_v, d_arr in zip(phi_mu, dists):
                _draw_half_violin(ax, x_v, d_arr, COL[fam], width_x=1.55)

            ax.errorbar(
                phi_mu, ys, xerr=phi_sd,
                fmt=MARK[fam] + "-", color=COL[fam],
                lw=2.2, ms=7.0, elinewidth=1.05, capsize=2.8, alpha=0.95,
                label=LAB[fam], zorder=4,
            )
            for k_idx, (x, y) in enumerate(zip(phi_mu, ys)):
                # Carefully offset labels so none touch E_q = 1.0, insets, or neighboring points
                dx, dy = 0, 9
                if idx == 0 and fam == "pinn" and k_idx == 0:
                    dx, dy = -12, 10  # Keep 1.03 well above the E_q = 1.0 dashed line
                elif idx == 0 and fam == "unet" and k_idx == 1:
                    dx, dy = 6, -15   # Keep 2.50 cleanly below the orange curve and right of inset
                elif idx == 1 and fam == "pinn" and k_idx == 2:
                    dx, dy = 0, -16
                elif idx == 1 and fam == "fno" and k_idx == 2:
                    dx, dy = 0, 10
                ax.annotate(
                    f"{y:.2f}", (x, y),
                    textcoords="offset points", xytext=(dx, dy),
                    ha="center", fontsize=7.7, fontweight="bold", color=COL[fam],
                    bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.90),
                    zorder=6,
                )
            pct = (ys[-1] - ys[0]) / ys[0] * 100.0
            pct_by_fam[fam] = pct

        ax.axhline(1.0, color="#BE123C", lw=1.6, ls="--", zorder=3)
        ax.set_yscale("log")
        if idx == 0:
            ax.set_ylim(0.78, 5.2)
            ax.set_xlim(-2.0, 56.0)
        else:
            ax.set_ylim(0.78, 14.5)
            ax.set_xlim(12.0, 74.0)

        ax.set_xlabel(r"Measured high-wavenumber energy fraction $\Phi(k \geq 8)$ (%)", fontsize=9.3, color="#0F172A")
        panel_lbl = "(a)" if idx == 0 else "(b)"
        l_mean = float(np.nanmean(Ls))
        l_min, l_max = float(np.nanmin(Ls)), float(np.nanmax(Ls))
        ax.set_title(
            f"{panel_lbl} Fixed variance $\\sigma^2 = {s2:g}$, scale $\\lambda = {lam:g}$\n"
            rf"Matched $L_{{\mathrm{{int}}}} \approx {l_mean:.3f}$ (range ${l_min:.3f}\text{{–}}{l_max:.3f}$ across families)",
            fontsize=9.4, fontweight="bold", color="#0F172A", loc="left", pad=8,
        )
        ax.tick_params(labelsize=8.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        y_top = ax.get_ylim()[1]
        for c_i, (x_val, cov_key) in enumerate(zip(phi_mu, COVS)):
            # Stagger middle pill slightly in Panel (b) where Phi values (33.9, 39.8, 44.8) are close
            y_mult = 0.78 if (idx == 1 and c_i == 1) else 0.93
            ax.text(
                x_val, y_top * y_mult, COV_NAMES[cov_key],
                ha="center", va="top", fontsize=7.3, color="#1E293B", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc="#F8FAFC", ec="#CBD5E1", lw=0.80, alpha=0.95),
                zorder=6,
            )

        # Lollipop inset showing relative Delta E_q (%) from Gaussian to Matern nu=1/2
        if pct_by_fam:
            inset_bounds = [0.03, 0.56, 0.22, 0.23] if idx == 0 else [0.56, 0.08, 0.40, 0.24]
            ax_in = ax.inset_axes(inset_bounds)
            ax_in.set_facecolor("#F8FAFC")
            fams_order = ["unet", "fno", "pinn"]
            y_in = np.arange(len(fams_order))[::-1]
            for yi, f_key in zip(y_in, fams_order):
                val_p = pct_by_fam.get(f_key, 0.0)
                ax_in.plot([0, val_p], [yi, yi], color=COL[f_key], lw=2.0, zorder=3)
                ax_in.scatter([val_p], [yi], color=COL[f_key], s=26, zorder=4)
                sign = "+" if val_p >= 0 else ""
                x_txt = max(val_p, 0.0) + 4.0
                ax_in.text(
                    x_txt, yi, f"{sign}{val_p:.0f}%",
                    va="center", ha="left", fontsize=6.5, fontweight="bold", color=COL[f_key],
                )
            ax_in.set_yticks(y_in)
            ax_in.set_yticklabels(["U-Net", "FNO", "PINN"], fontsize=6.5, fontweight="bold")
            ax_in.set_xlim(-10 if idx == 1 else 0, 98 if idx == 0 else 108)
            ax_in.set_ylim(-0.6, 2.6)
            ax_in.set_xticks([])
            ax_in.set_title(r"$\Delta E_q$ (Gauss $\to \nu=1/2$)", fontsize=6.7, fontweight="bold", color="#0F172A", pad=2)
            for sp in ("top", "right", "bottom"):
                ax_in.spines[sp].set_visible(False)

    if cache_updated:
        np.savez_compressed(cache_path, **cache_data)

    axes[0].set_ylabel(r"Median Darcy-flux error $E_q$ (rel. $L_2$, log scale; shaded: IQR)", fontsize=9.3, color="#0F172A")
    axes[0].legend(fontsize=7.7, frameon=True, facecolor="white", edgecolor="#CBD5E1", loc="lower right", bbox_to_anchor=(0.98, 0.05))

    out = FIGDIR / out_name
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=600, metadata={"Software": None})
    plt.close(fig)
    print("wrote", out)
    return out


if __name__ == "__main__":
    make_figure5()
