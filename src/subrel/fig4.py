"""Figure 4: Radial spectral decomposition of hydraulic-head prediction errors.

Top row (a)-(c): Ratio of radial head-error spectral energy to true solution
spectral energy E_err(k) / E_truth(k) across three representative regimes,
shaded into macro-advection (k < 8), spectral transition (8 <= k < 12), and
unresolved k^2 derivative-amplification (k >= 12) bands with +-1 SD realization
ribbons and an analytical k^2 Darcy gain reference triangle.
Bottom row (d)-(f): Absolute radial error energy spectra E_err(k) versus
wavenumber k (cycles per domain), showing the sharp m = 12 mode truncation of
FNO and high-wavenumber error concentration in U-Net versus rapid decay in PINN.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .metrics import radial_spectrum
from .models import PINN, FNO2d, UNet
from .train_grid import make_inputs

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "figures"
CACHE_PATH = ROOT / "runs" / "fig4_cache.npz"
CELLS = [
    (0.5, 0.2, "Smooth regime"),
    (1.0, 0.05, "Moderate regime"),
    (4.0, 0.02, "Rough regime"),
]
N_INST = 4
COL = {"unet": "#D55E00", "fno": "#0072B2", "pinn": "#009E73"}


def _spec(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k, e = radial_spectrum(field, n_bins=40)
    ok = e > 0
    return k[ok], e[ok]


def _unet_pred(model_dir: Path, meta: dict, Y: np.ndarray, ymu: float, ystd: float):
    model = UNet(3, meta["cfg"]["width"], meta["cfg"]["depth"])
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(make_inputs(Y[None], ymu, ystd))).numpy()[0, 0]
    return out * 0.5 + 0.5


def _fno_pred(model_dir: Path, meta: dict, Y: np.ndarray, ymu: float, ystd: float):
    model = FNO2d(3, meta["cfg"]["width"], meta["cfg"]["modes"])
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(make_inputs(Y[None], ymu, ystd))).numpy()[0, 0]
    return out * 0.5 + 0.5


def _pinn_pred(model_dir: Path, inst: int, n: int):
    model = PINN(64, 4)
    model.load_state_dict(torch.load(model_dir / f"pinn_inst{inst}.pt", map_location="cpu"))
    model.eval()
    x = (np.arange(n) + 0.5) / n
    Xc, Yc = np.meshgrid(x, x, indexing="xy")
    pts = torch.from_numpy(np.stack([Xc.reshape(-1), Yc.reshape(-1)], 1).astype(np.float32))
    with torch.no_grad():
        return model(pts).reshape(n, n).numpy()


def _load_or_compute_spectra() -> list[dict]:
    can_compute = all(
        (ROOT / "data" / f"cell_s{s2:g}_l{lam:g}_n128.h5").exists()
        and (ROOT / "runs/atlas" / f"s{s2:g}_l{lam:g}_n128" / "unet" / "model.pt").exists()
        and (ROOT / "runs/atlas" / f"s{s2:g}_l{lam:g}_n128" / "fno" / "model.pt").exists()
        and (ROOT / "runs/atlas" / f"s{s2:g}_l{lam:g}_n128" / "pinn" / "pinn_inst288.pt").exists()
        for s2, lam, _ in CELLS
    )
    if can_compute:
        cols_data = []
        save_dict = {}
        for col, (s2, lam, rname) in enumerate(CELLS):
            cid = f"s{s2:g}_l{lam:g}_n128"
            with h5py.File(ROOT / "data" / f"cell_{cid}.h5", "r") as f:
                n = int(f.attrs["n"])
                Yfull = f["Y_cell"][:]
                Hs = f["H"][288:288 + N_INST]
            Ys = Yfull[288:288 + N_INST]
            umeta = json.loads((ROOT / "runs/atlas" / cid / "unet" / "metrics.json").read_text())
            fmeta = json.loads((ROOT / "runs/atlas" / cid / "fno" / "metrics.json").read_text())
            ymu, ystd = float(Yfull[:256].mean()), float(Yfull[:256].std())
            xg = (np.arange(n) + 0.5) / n
            bg = 1.0 - xg[None, :]

            def curve_stats(get_pred):
                r_list, a_list = [], []
                kt_ref = None
                for i in range(N_INST):
                    hp = get_pred(i)
                    ke, ee = _spec(hp - Hs[i])
                    kt, et = _spec(Hs[i] - bg)
                    kt_ref = kt
                    r_interp = np.interp(kt, ke, ee)
                    a_list.append(r_interp)
                    rat = r_interp / np.maximum(et, 1e-30)
                    rat[et < 1e-8 * et.max()] = np.nan
                    r_list.append(rat)
                r_arr = np.array(r_list, dtype=float)
                a_arr = np.array(a_list, dtype=float)
                # Geometric mean and 1 SD in log10 space for smooth log-scale ribbons
                lr = np.log10(np.clip(r_arr, 1e-20, None))
                la = np.log10(np.clip(a_arr, 1e-20, None))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    r_mean = np.nanmean(r_arr, axis=0)
                    r_lo = 10.0 ** (np.nanmean(lr, axis=0) - np.nanstd(lr, axis=0))
                    r_hi = 10.0 ** (np.nanmean(lr, axis=0) + np.nanstd(lr, axis=0))
                    a_mean = np.nanmean(a_arr, axis=0)
                    a_lo = 10.0 ** (np.nanmean(la, axis=0) - np.nanstd(la, axis=0))
                    a_hi = 10.0 ** (np.nanmean(la, axis=0) + np.nanstd(la, axis=0))
                return kt_ref, r_mean, r_lo, r_hi, a_mean, a_lo, a_hi

            ku, ru, ru_lo, ru_hi, au, au_lo, au_hi = curve_stats(
                lambda i: _unet_pred(ROOT / "runs/atlas" / cid / "unet", umeta, Ys[i], ymu, ystd)
            )
            kf, rf, rf_lo, rf_hi, af, af_lo, af_hi = curve_stats(
                lambda i: _fno_pred(ROOT / "runs/atlas" / cid / "fno", fmeta, Ys[i], ymu, ystd)
            )
            kp, rp, rp_lo, rp_hi, ap, ap_lo, ap_hi = curve_stats(
                lambda i: _pinn_pred(ROOT / "runs/atlas" / cid / "pinn", 288 + i, n)
            )
            specs = [_spec(Ys[i]) for i in range(N_INST)]
            kmed = specs[0][0]
            emed_abs = np.mean([sp[1] for sp in specs], axis=0)
            emed = emed_abs / emed_abs.max()

            entry = dict(
                ku=ku, ru=ru, ru_lo=ru_lo, ru_hi=ru_hi, au=au, au_lo=au_lo, au_hi=au_hi,
                kf=kf, rf=rf, rf_lo=rf_lo, rf_hi=rf_hi, af=af, af_lo=af_lo, af_hi=af_hi,
                kp=kp, rp=rp, rp_lo=rp_lo, rp_hi=rp_hi, ap=ap, ap_lo=ap_lo, ap_hi=ap_hi,
                kmed=kmed, emed=emed, emed_abs=emed_abs,
            )
            cols_data.append(entry)
            for k_name, v_arr in entry.items():
                save_dict[f"c{col}_{k_name}"] = np.asarray(v_arr, dtype=np.float64)

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE_PATH, n_cols=len(CELLS), **save_dict)
        return cols_data

    if CACHE_PATH.exists():
        z = np.load(CACHE_PATH)
        cols_data = []
        keys = [
            "ku", "ru", "ru_lo", "ru_hi", "au", "au_lo", "au_hi",
            "kf", "rf", "rf_lo", "rf_hi", "af", "af_lo", "af_hi",
            "kp", "rp", "rp_lo", "rp_hi", "ap", "ap_lo", "ap_hi",
            "kmed", "emed", "emed_abs",
        ]
        for col in range(int(z["n_cols"])):
            cols_data.append({k_name: z[f"c{col}_{k_name}"] for k_name in keys})
        return cols_data

    raise FileNotFoundError("Neither raw .h5/.pt files nor runs/fig4_cache.npz found.")


def make_figure4(out_name: str = "fig4_spectral.png") -> Path | None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "axes.edgecolor": "#1E293B",
        "axes.linewidth": 0.95,
        "figure.dpi": 600,
        "savefig.dpi": 600,
    })

    cols_data = _load_or_compute_spectra()
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.9), dpi=600, constrained_layout=True)
    top_tags = ["(a)", "(b)", "(c)"]
    bot_tags = ["(d)", "(e)", "(f)"]

    for col, ((s2, lam, rname), d) in enumerate(zip(CELLS, cols_data)):
        ax_r = axes[0, col]
        ax_a = axes[1, col]

        # Three wavenumber bands: k < 8, 8 <= k < 12, k >= 12
        for ax_cur in (ax_r, ax_a):
            ax_cur.axvspan(0.9, 8.0, color="#F1F5F9", alpha=0.65, zorder=0)
            ax_cur.axvspan(8.0, 12.0, color="#FEF3C7", alpha=0.50, zorder=0)
            ax_cur.axvspan(12.0, 95.0, color="#FFF1F2", alpha=0.52, zorder=0)
            ax_cur.axvline(8.0, color="#64748B", lw=1.0, ls=":", alpha=0.85, zorder=2)
            ax_cur.axvline(12.0, color="#0072B2", lw=1.15, ls="--", alpha=0.85, zorder=2)
            ax_cur.grid(True, which="major", ls="-", lw=0.5, color="#CBD5E1", alpha=0.72, zorder=1)
            ax_cur.spines["top"].set_visible(False)
            ax_cur.spines["right"].set_visible(False)

        # Band header tags on top row
        ax_r.text(2.8, 420.0, r"Macro ($k<8$)", fontsize=6.8, color="#475569", fontweight="bold", ha="center", va="top")
        ax_r.text(9.8, 420.0, r"$8\text{–}12$", fontsize=6.5, color="#92400E", fontweight="bold", ha="center", va="top")
        ax_r.text(28.5, 420.0, r"Unresolved $k^2$ band ($k\geq 12$)", fontsize=6.8, color="#9F1239", fontweight="bold", ha="center", va="top")

        # Plot relative error spectra + +-1 SD ribbons
        for k_arr, mean_arr, lo_arr, hi_arr, c_hex, lbl in [
            (d["ku"], d["ru"], d["ru_lo"], d["ru_hi"], COL["unet"], "U-Net (amortized)"),
            (d["kf"], d["rf"], d["rf_lo"], d["rf_hi"], COL["fno"], "FNO (amortized, $m=12$)"),
            (d["kp"], d["rp"], d["rp_lo"], d["rp_hi"], COL["pinn"], "PINN (per-instance)"),
        ]:
            ax_r.fill_between(k_arr, lo_arr, hi_arr, color=c_hex, alpha=0.18, zorder=3)
            ax_r.loglog(k_arr, mean_arr, color=c_hex, lw=2.2, label=lbl, zorder=4)

        ax_r.loglog(d["kmed"], d["emed"], color="#475569", lw=1.35, ls=":", label=r"Normalized $\ln K$ spectrum", zorder=3)
        ax_r.set_ylim(1e-6, 8e2)
        ax_r.set_xlim(0.9, 95.0)
        ax_r.axhline(1.0, color="#BE123C", lw=1.2, ls="--", alpha=0.85, zorder=2)

        # Analytical k^2 Darcy gain reference triangle placed cleanly in upper-right (y in [5, 28])
        k1, k2 = 22.0, 52.0
        y1 = 4.5
        y2 = y1 * (k2 / k1) ** 2
        ax_r.plot([k1, k2, k2, k1], [y1, y1, y2, y1], color="#BE123C", lw=1.15, ls="-", zorder=5)
        ax_r.text(
            34.0, 2.2, r"Slope $\propto k^2$ (Darcy gain)",
            fontsize=6.9, color="#9F1239", fontweight="bold", ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.85),
            zorder=6,
        )

        if col == 0:
            ax_r.text(
                14.5, 1.8e-5,
                r"PINN smooth error: $10^{-5}\text{–}10^{-4}$",
                fontsize=7.1, color="#065F46", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc="#ECFDF5", ec="#6EE7B7", lw=0.75, alpha=0.94),
                zorder=6,
            )

        ax_r.set_title(
            f"{top_tags[col]} {rname} ($\\sigma^2={s2:g},\\ \\lambda={lam:g}$)\n"
            r"Relative head-error spectrum $E_{\mathrm{err}}(k) / E_{\mathrm{truth}}(k)$",
            fontsize=9.3, fontweight="bold", color="#0F172A", loc="left", pad=6,
        )
        if col == 0:
            ax_r.set_ylabel(r"Relative error energy $E_{\mathrm{err}}(k)/E_{\mathrm{truth}}(k)$", fontsize=9.2, color="#0F172A")
        ax_r.tick_params(labelsize=8.3)

        # Bottom row: absolute error spectra + +-1 SD ribbons
        for k_arr, mean_arr, lo_arr, hi_arr, c_hex, lbl in [
            (d["ku"], d["au"], d["au_lo"], d["au_hi"], COL["unet"], "U-Net"),
            (d["kf"], d["af"], d["af_lo"], d["af_hi"], COL["fno"], "FNO ($m=12$)"),
            (d["kp"], d["ap"], d["ap_lo"], d["ap_hi"], COL["pinn"], "PINN"),
        ]:
            ax_a.fill_between(k_arr, lo_arr, hi_arr, color=c_hex, alpha=0.18, zorder=3)
            ax_a.loglog(k_arr, mean_arr, color=c_hex, lw=2.2, label=lbl, zorder=4)

        ax_a.loglog(d["kmed"], d["emed_abs"], color="#475569", lw=1.35, ls=":", label=r"$\ln K$ spectrum $E_Y(k)$", zorder=3)
        ax_a.set_xlim(0.9, 95.0)
        ax_a.set_ylim(1e-16, 1e3)
        ax_a.set_title(
            f"{bot_tags[col]} {rname} ($\\sigma^2={s2:g},\\ \\lambda={lam:g}$)\n"
            r"Absolute head-error spectrum $E_{\mathrm{err}}(k)$ vs. $E_Y(k)$",
            fontsize=9.3, fontweight="bold", color="#0F172A", loc="left", pad=6,
        )
        ax_a.set_xlabel(r"Radial wavenumber $k$ (cycles per domain)", fontsize=9.2, color="#0F172A")
        if col == 0:
            ax_a.set_ylabel(r"Absolute spectral energy $E_{\mathrm{err}}(k)$", fontsize=9.2, color="#0F172A")
        ax_a.tick_params(labelsize=8.3)
        ax_a.text(
            13.5, 3e-15, r"FNO mode cutoff ($m=12$)",
            fontsize=7.3, color="#0072B2", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85),
            zorder=5,
        )

    axes[0, 0].legend(fontsize=7.4, frameon=True, facecolor="white", edgecolor="#CBD5E1", loc="lower left")
    axes[1, 0].legend(fontsize=7.4, frameon=True, facecolor="white", edgecolor="#CBD5E1", loc="lower left")

    out = FIGDIR / out_name
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=600, metadata={"Software": None})
    plt.close(fig)
    print("wrote", out)
    return out


if __name__ == "__main__":
    make_figure4()
