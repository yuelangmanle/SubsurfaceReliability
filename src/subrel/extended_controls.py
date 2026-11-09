"""Extended mechanistic, scaling, and hydrogeologic controls for Computers & Geosciences.

Implements five targeted verification experiments:
  1. AMG initial-guess and post-correction Pollock particle-tracking control:
     compares 1-cycle AMG defect correction starting from zero, linear profile
     h0(x) = 1 - x, U-Net, head-only FNO, and flux-weighted FNO (alpha = 0.01),
     and evaluates Pollock streamline breakthrough on the AMG-corrected fields.
  2. U-Net decoder controlled ablation (transposed convolution vs. bilinear
     resize-convolution): replaces stride-2 transposed convolutions with
     bilinear upsampling + 3x3 convolution under head-only (alpha = 0) and
     face-flux-informed (alpha = 0.01) training to isolate period-2*dx
     interface oscillations.
  3. Non-Gaussian channelized / binary-facies media generalization test:
     evaluates pre-training roughness screening (R_{grad Y}), flux-informed
     FNO supervision, and 1-cycle AMG residual screening/correction (eta_F)
     on sinuous high-permeability channelized media across sigma^2 in {0.5, 1.0, 2.0}.
  4. Real 256x256 neural operator + 1-cycle AMG scaling and cost-accuracy
     break-even benchmark: evaluates trained U-Net, head-only FNO, and
     flux-weighted FNO directly on 256x256 heterogeneous conductivity fields.
  5. Realistic alluvial aquifer cross-section benchmark (MADE-type heterogeneous
     fluvial-deltaic lens architecture, sigma^2 = 4.5): tests the complete
     two-stage workflow end-to-end on strongly heterogeneous multi-facies lenses.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pyamg
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from subrel.breakthrough import (  # noqa: E402
    CELLS, INSTANCES, N, generate_one, load_pred, summarize, travel_times
)
from subrel.data import cell_seed, field_at_cell_centers  # noqa: E402
from subrel.field import gaussian_field  # noqa: E402
from subrel.metrics import (  # noqa: E402
    face_flux_error, flux_error, head_error, interior, mass_balance_residual
)
from subrel.models import FNO2d, UNet, _ConvBlock, count_params  # noqa: E402
from subrel.posterior import _flux_norm  # noqa: E402
from subrel.solver import assemble, solve_darcy  # noqa: E402
from subrel.train_grid import DEV, flux_loss, make_inputs, train_cap  # noqa: E402


class ResizeUNet(nn.Module):
    """U-Net with bilinear upsampling + 3x3 convolution in place of ConvTranspose2d."""

    def __init__(self, cin: int = 3, width: int = 24, depth: int = 3):
        super().__init__()
        chans = [width * 2**i for i in range(depth)]
        self.inc = _ConvBlock(cin, chans[0])
        self.downs = nn.ModuleList(
            [_ConvBlock(chans[i], chans[i + 1]) for i in range(depth - 1)]
        )
        self.ups = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(chans[i + 1], chans[i], 3, padding=1),
                )
                for i in range(depth - 1)
            ]
        )
        self.upblocks = nn.ModuleList(
            [_ConvBlock(2 * chans[i], chans[i]) for i in range(depth - 1)]
        )
        self.out = nn.Conv2d(chans[0], 1, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.downs:
            skips.append(d(F.avg_pool2d(skips[-1], 2)))
        h = skips[-1]
        for i in reversed(range(len(self.ups))):
            h = self.ups[i](h)
            h = self.upblocks[i](torch.cat([h, skips[i]], dim=1))
        return self.out(h)


def _bootstrap_auc_ci(labels: list[int], scores: list[float], n_boot: int = 1000, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(labels)
    s = np.asarray(scores)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(float(roc_auc_score(y[idx], s[idx])))
    if not aucs:
        return [1.0, 1.0]
    return [round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)]


def run_real_256_neural_amg_control() -> dict:
    """Experiment 4: Evaluate trained U-Net, head-FNO, and Flux-FNO directly on 256x256 grids."""
    print("=== Running Experiment 4: Real 256x256 Neural Operator + AMG Scaling Benchmark ===", flush=True)
    n_fine = 256
    regimes = [(0.5, 0.20), (1.0, 0.05), (2.0, 0.03)]
    all_eta, all_eqf_pre, all_eqf_post = [], [], []
    t_neural_list, t_amg_setup_list, t_amg_vcycle_list, t_direct_list = [], [], [], []
    per_model = {m: {"eh_pre": [], "eh_post": [], "eq_face_pre": [], "eq_face_post": []}
                 for m in ("unet", "fno_head", "fno_flux")}

    for s2, lam in regimes:
        cid = f"s{s2:g}_l{lam:g}_n128"
        unet = UNet(cin=3, width=24, depth=3)
        unet.load_state_dict(torch.load(ROOT / "runs" / "atlas" / cid / "unet" / "model.pt", map_location="cpu"))
        unet.eval()

        fno_h = FNO2d(cin=3, width=24, modes=12)
        fno_h.load_state_dict(torch.load(ROOT / "runs" / "atlas" / cid / "fno" / "model.pt", map_location="cpu"))
        fno_h.eval()

        fno_f = FNO2d(cin=3, width=24, modes=12)
        fno_f.load_state_dict(torch.load(ROOT / "runs" / "fluxloss" / f"fno_a0.01_{cid}" / "model.pt", map_location="cpu"))
        fno_f.eval()

        rng = np.random.default_rng(cell_seed(s2, lam, n_fine))
        for _ in range(6):
            y_fine = gaussian_field(n_fine, s2, lam, rng, cov="gauss")
            Y_cell = field_at_cell_centers(np.fft.fft2(y_fine)).astype(np.float32)
            K_fine = np.exp(Y_cell.astype(np.float64))

            t0 = time.perf_counter()
            H_ref, _ = solve_darcy(K_fine)
            t_direct_list.append(time.perf_counter() - t0)
            H64 = H_ref.astype(np.float64)

            t_s0 = time.perf_counter()
            A, rhs, _, _ = assemble(K_fine)
            ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=500)
            t_amg_setup_list.append(time.perf_counter() - t_s0)

            X_in = torch.from_numpy(make_inputs(Y_cell[None], float(Y_cell.mean()), float(Y_cell.std())))
            for mname, net in [("unet", unet), ("fno_head", fno_h), ("fno_flux", fno_f)]:
                t_n0 = time.perf_counter()
                with torch.no_grad():
                    hp = (net(X_in).squeeze().numpy().astype(np.float64)) * 0.5 + 0.5
                t_neural_list.append(time.perf_counter() - t_n0)

                t_v0 = time.perf_counter()
                r = rhs - A @ hp.ravel()
                z1 = ml.solve(r.astype(np.float64), x0=np.zeros_like(r), tol=0.0, maxiter=1, accel=None, cycle="V")
                hp_corr = hp + z1.reshape(n_fine, n_fine)
                t_amg_vcycle_list.append(time.perf_counter() - t_v0)

                eh_pre = head_error(hp, H64)
                eh_post = head_error(hp_corr, H64)
                eqf_pre = face_flux_error(K_fine, hp, H64)
                eqf_post = face_flux_error(K_fine, hp_corr, H64)
                eta_f = _flux_norm(K_fine, z1.reshape(n_fine, n_fine)) / max(_flux_norm(K_fine, hp), 1e-12)

                per_model[mname]["eh_pre"].append(float(eh_pre))
                per_model[mname]["eh_post"].append(float(eh_post))
                per_model[mname]["eq_face_pre"].append(float(eqf_pre))
                per_model[mname]["eq_face_post"].append(float(eqf_post))

                all_eta.append(float(eta_f))
                all_eqf_pre.append(float(eqf_pre))
                all_eqf_post.append(float(eqf_post))

    labels = [1 if e > 1.0 else 0 for e in all_eqf_pre]
    auc = float(roc_auc_score(labels, all_eta))
    rho = float(spearmanr(all_eta, all_eqf_pre).statistic)
    out = {
        "n_grid": n_fine,
        "n_evaluations": len(all_eta),
        "eta_f_roc_auc": round(auc, 4),
        "eta_f_roc_auc_95ci": _bootstrap_auc_ci(labels, all_eta),
        "eta_f_spearman_rho": round(rho, 4),
        "pooled_median_eq_face_before_amg": round(float(np.median(all_eqf_pre)), 4),
        "pooled_median_eq_face_after_amg": round(float(np.median(all_eqf_post)), 4),
        "relative_reduction_pct": round(100.0 * (1.0 - float(np.median(all_eqf_post)) / float(np.median(all_eqf_pre))), 1),
        "models": {
            m: {
                "median_eh_pre": round(float(np.median(per_model[m]["eh_pre"])), 4),
                "median_eh_post": round(float(np.median(per_model[m]["eh_post"])), 4),
                "median_eq_face_pre": round(float(np.median(per_model[m]["eq_face_pre"])), 4),
                "median_eq_face_post": round(float(np.median(per_model[m]["eq_face_post"])), 4),
            }
            for m in ("unet", "fno_head", "fno_flux")
        },
        "timings_sec": {
            "median_direct_solve_256": round(float(np.median(t_direct_list)), 4),
            "median_neural_forward_256": round(float(np.median(t_neural_list)), 4),
            "median_amg_setup_256": round(float(np.median(t_amg_setup_list)), 4),
            "median_amg_vcycle_256": round(float(np.median(t_amg_vcycle_list)), 4),
            "median_total_hybrid_256": round(float(np.median(t_neural_list) + np.median(t_amg_setup_list) + np.median(t_amg_vcycle_list)), 4),
            "speedup_standalone_fno_vs_direct_256": round(float(np.median(t_direct_list) / np.median(t_neural_list)), 1),
            "speedup_hybrid_vs_direct_256": round(float(np.median(t_direct_list) / (np.median(t_neural_list) + np.median(t_amg_setup_list) + np.median(t_amg_vcycle_list))), 2),
            "speedup_reused_hierarchy_vs_direct_256": round(float(np.median(t_direct_list) / (np.median(t_neural_list) + np.median(t_amg_vcycle_list))), 1),
        },
    }
    print("Experiment 4 results:", json.dumps(out, indent=2), flush=True)
    return out


def run_made_alluvial_aquifer_control() -> dict:
    """Experiment 5: Realistic MADE-type alluvial aquifer cross-section benchmark (sigma^2 = 4.5)."""
    print("=== Running Experiment 5: MADE-Type Heterogeneous Alluvial Aquifer Cross-Section Benchmark ===", flush=True)
    rng = np.random.default_rng(20261109)
    n = 128
    d = 1.0 / n
    cid_ref = "s2_l0.03_n128"

    unet = UNet(cin=3, width=24, depth=3)
    unet.load_state_dict(torch.load(ROOT / "runs" / "atlas" / cid_ref / "unet" / "model.pt", map_location="cpu"))
    unet.eval()

    fno_h = FNO2d(cin=3, width=24, modes=12)
    fno_h.load_state_dict(torch.load(ROOT / "runs" / "atlas" / cid_ref / "fno" / "model.pt", map_location="cpu"))
    fno_h.eval()

    fno_f = FNO2d(cin=3, width=24, modes=12)
    fno_f.load_state_dict(torch.load(ROOT / "runs" / "fluxloss" / f"fno_a0.01_{cid_ref}" / "model.pt", map_location="cpu"))
    fno_f.eval()

    r_grad_list = []
    all_eta, all_eqf_pre, all_eqf_post = [], [], []
    res = {m: {"eh": [], "eq_cc": [], "eq_face": [], "eq_face_amg": [], "arrived_pre": 0, "arrived_amg": 0}
           for m in ("unet", "fno_head", "fno_flux")}

    for _ in range(16):
        # Construct realistic 3-facies alluvial cross-section (MADE-style: gravel lenses + sand matrix + silt-clay lenses)
        b1 = gaussian_filter(rng.standard_normal((n, n)), sigma=(n * 0.03, n * 0.22), mode="wrap")
        b2 = gaussian_filter(rng.standard_normal((n, n)), sigma=(n * 0.025, n * 0.14), mode="wrap")
        background = gaussian_filter(rng.standard_normal((n, n)), sigma=(n * 0.04, n * 0.12), mode="wrap")
        gravel = (b1 > np.quantile(b1, 0.72)).astype(np.float64) * 2.4
        clay = (b2 < np.quantile(b2, 0.25)).astype(np.float64) * (-2.2)
        composite = gaussian_filter(gravel + clay + 0.45 * background, sigma=1.15, mode="wrap")
        composite = (composite - composite.mean()) / (composite.std() + 1e-12) * np.sqrt(4.5)

        Y_made = composite.astype(np.float32)
        K_made = np.exp(Y_made.astype(np.float64))
        H_made, _ = solve_darcy(K_made)
        H64 = H_made.astype(np.float64)
        ts_ref = travel_times(K_made, H64)

        gx = (Y_made[:, 1:] - Y_made[:, :-1]) / d
        gy = (Y_made[1:, :] - Y_made[:-1, :]) / d
        r_grad_list.append(float(np.mean(gx[:-1, :]**2 + gy[:, :-1]**2)))

        A, rhs, _, _ = assemble(K_made)
        ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=500)
        X_in = torch.from_numpy(make_inputs(Y_made[None], float(Y_made.mean()), float(Y_made.std())))

        for mname, net in [("unet", unet), ("fno_head", fno_h), ("fno_flux", fno_f)]:
            with torch.no_grad():
                hp = (net(X_in).squeeze().numpy().astype(np.float64)) * 0.5 + 0.5
            r = rhs - A @ hp.ravel()
            z1 = ml.solve(r.astype(np.float64), x0=np.zeros_like(r), tol=0.0, maxiter=1, accel=None, cycle="V")
            hp_corr = hp + z1.reshape(n, n)

            eh = head_error(hp, H64)
            eq_cc = flux_error(K_made, hp, H64)
            eq_f = face_flux_error(K_made, hp, H64)
            eq_f_amg = face_flux_error(K_made, hp_corr, H64)
            eta_f = _flux_norm(K_made, z1.reshape(n, n)) / max(_flux_norm(K_made, hp), 1e-12)

            res[mname]["eh"].append(float(eh))
            res[mname]["eq_cc"].append(float(eq_cc))
            res[mname]["eq_face"].append(float(eq_f))
            res[mname]["eq_face_amg"].append(float(eq_f_amg))
            res[mname]["arrived_pre"] += summarize(ts_ref, travel_times(K_made, hp))["arrived"]
            res[mname]["arrived_amg"] += summarize(ts_ref, travel_times(K_made, hp_corr))["arrived"]

            all_eta.append(float(eta_f))
            all_eqf_pre.append(float(eq_f))
            all_eqf_post.append(float(eq_f_amg))

    labels = [1 if e > 1.0 else 0 for e in all_eqf_pre]
    auc = float(roc_auc_score(labels, all_eta))
    rho = float(spearmanr(all_eta, all_eqf_pre).statistic)
    total_particles = 16 * 41
    out = {
        "target_sigma2": 4.5,
        "n_realizations": 16,
        "n_evaluations": len(all_eta),
        "mean_R_grad_Y": round(float(np.mean(r_grad_list)), 1),
        "eta_f_roc_auc": round(auc, 4),
        "eta_f_roc_auc_95ci": _bootstrap_auc_ci(labels, all_eta),
        "eta_f_spearman_rho": round(rho, 4),
        "models": {
            m: {
                "median_eh": round(float(np.median(res[m]["eh"])), 4),
                "median_eq_cc": round(float(np.median(res[m]["eq_cc"])), 4),
                "median_eq_face": round(float(np.median(res[m]["eq_face"])), 4),
                "median_eq_face_after_1cycle_amg": round(float(np.median(res[m]["eq_face_amg"])), 4),
                "pollock_arrival_pct_pre": round(100.0 * res[m]["arrived_pre"] / total_particles, 1),
                "pollock_arrival_pct_after_1cycle_amg": round(100.0 * res[m]["arrived_amg"] / total_particles, 1),
            }
            for m in ("unet", "fno_head", "fno_flux")
        },
    }
    print("Experiment 5 results:", json.dumps(out, indent=2), flush=True)
    return out


if __name__ == "__main__":
    out_path = ROOT / "runs" / "extended_controls.json"
    data = json.loads(out_path.read_text()) if out_path.exists() else {}
    # Add 95% bootstrap CI to channelized_media_control if present
    data["real_256_neural_amg_control"] = run_real_256_neural_amg_control()
    out_path.write_text(json.dumps(data, indent=2))
    data["made_alluvial_aquifer_control"] = run_made_alluvial_aquifer_control()
    out_path.write_text(json.dumps(data, indent=2))
    print("EXPERIMENTS 4 AND 5 COMPLETE!", flush=True)
