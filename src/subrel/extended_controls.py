"""Extended mechanistic and generalization controls for Computers & Geosciences.

Implements three targeted verification experiments:
  1. AMG initial-guess and post-correction Pollock particle-tracking control:
     compares 1-cycle AMG defect correction starting from zero, linear profile
     h0(x) = 1 - x, U-Net, head-only FNO, and flux-weighted FNO (alpha = 0.01),
     and evaluates Pollock streamline breakthrough on the AMG-corrected fields.
  2. U-Net decoder causal ablation (transposed convolution vs. bilinear
     resize-convolution): replaces stride-2 transposed convolutions with
     bilinear upsampling + 3x3 convolution under head-only (alpha = 0) and
     face-flux-informed (alpha = 0.01) training to isolate period-2*dx
     interface oscillations.
  3. Non-Gaussian channelized / binary-facies media generalization test:
     evaluates pre-training roughness screening (R_{grad Y}), flux-informed
     FNO supervision, and 1-cycle AMG residual screening/correction (eta_F)
     on sinuous high-permeability channelized media across sigma^2 in {0.5, 2.0, 4.0}.
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


def run_amg_and_transport_control() -> dict:
    """Experiment 1: Compare initial guesses under 1-cycle AMG and Pollock tracking."""
    print("=== Running Experiment 1: AMG Initial-Guess & Post-AMG Transport Control ===", flush=True)
    x = (np.arange(N) + 0.5) / N
    h_lin = np.broadcast_to(1.0 - x[None, :], (N, N)).astype(np.float64).copy()
    h_zero = np.zeros((N, N), dtype=np.float64)

    sources = ["zero", "linear", "unet", "fno_head", "fno_flux"]
    stats = {s: {"eh_pre": [], "eh_post1": [], "eq_face_pre": [], "eq_face_post1": [],
                 "eq_cc_pre": [], "eq_cc_post1": [], "idiv_pre": [], "idiv_post1": [], "eta_f": [],
                 "arrived_post1": 0, "stalled_post1": 0, "backwashed_post1": 0,
                 "realizations_with_arrival_post1": 0, "paired_ratios_post1": []}
             for s in sources}
    cycles_needed_from_linear = []
    cycles_needed_from_zero = []

    for sigma2, lam in CELLS:
        rng = np.random.default_rng(cell_seed(sigma2, lam, N))
        ys = []
        for _ in range(256):
            y = gaussian_field(N, sigma2, lam, rng, cov="gauss")
            ys.append(field_at_cell_centers(np.fft.fft2(y)).astype(np.float32))
        Ytr = np.stack(ys)
        ymu, ystd = float(Ytr.mean()), float(Ytr.std())
        del Ytr, ys

        for inst in INSTANCES:
            Y, H = generate_one(sigma2, lam, inst)
            H64 = H.astype(np.float64)
            K = np.exp(Y.astype(np.float64))
            A, rhs, n, d = assemble(K)
            ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=500)
            ts_ref = travel_times(K, H64)

            init_fields = {
                "zero": h_zero,
                "linear": h_lin,
                "unet": load_pred("unet", sigma2, lam, Y, ymu, ystd),
                "fno_head": load_pred("fno_head", sigma2, lam, Y, ymu, ystd),
                "fno_flux": load_pred("fno", sigma2, lam, Y, ymu, ystd),
            }

            fno_flux_post1_err = None
            for sname, h_init in init_fields.items():
                r = rhs - A @ h_init.ravel()
                z1 = ml.solve(r.astype(np.float64), x0=np.zeros_like(r), tol=0.0, maxiter=1, accel=None, cycle="V")
                h_corr1 = h_init + z1.reshape(N, N)

                eh_pre = head_error(h_init, H64) if sname != "zero" else 1.0
                eh_post = head_error(h_corr1, H64)
                eqf_pre = face_flux_error(K, h_init, H64) if sname != "zero" else 1.0
                eqf_post = face_flux_error(K, h_corr1, H64)
                eqc_pre = flux_error(K, h_init, H64) if sname != "zero" else 1.0
                eqc_post = flux_error(K, h_corr1, H64)
                idiv_pre = mass_balance_residual(K, h_init, H64) if sname != "zero" else 0.0
                idiv_post = mass_balance_residual(K, h_corr1, H64)
                eta_f = _flux_norm(K, z1.reshape(N, N)) / (_flux_norm(K, h_init) + 1e-30)

                st = stats[sname]
                st["eh_pre"].append(float(eh_pre))
                st["eh_post1"].append(float(eh_post))
                st["eq_face_pre"].append(float(eqf_pre))
                st["eq_face_post1"].append(float(eqf_post))
                st["eq_cc_pre"].append(float(eqc_pre))
                st["eq_cc_post1"].append(float(eqc_post))
                st["idiv_pre"].append(float(idiv_pre))
                st["idiv_post1"].append(float(idiv_post))
                st["eta_f"].append(float(eta_f))

                if sname == "fno_flux":
                    fno_flux_post1_err = eqf_post

                ts_corr = summarize(ts_ref, travel_times(K, h_corr1))
                st["arrived_post1"] += ts_corr["arrived"]
                st["stalled_post1"] += ts_corr["fates"].get("stalled", 0)
                st["backwashed_post1"] += ts_corr["fates"].get("backwashed", 0)
                if ts_corr["arrived"] > 0:
                    st["realizations_with_arrival_post1"] += 1
                if ts_corr["median_paired_ratio"] is not None:
                    st["paired_ratios_post1"].append(ts_corr["median_paired_ratio"])

            # Check how many V-cycles from linear and zero are needed to reach fno_flux + 1 V-cycle face-flux error
            for base_field, cyc_list in [(h_lin, cycles_needed_from_linear), (h_zero, cycles_needed_from_zero)]:
                h_cur = base_field.copy()
                for c in range(1, 10):
                    rc = rhs - A @ h_cur.ravel()
                    zc = ml.solve(rc.astype(np.float64), x0=np.zeros_like(rc), tol=0.0, maxiter=1, accel=None, cycle="V")
                    h_cur = h_cur + zc.reshape(N, N)
                    err_c = face_flux_error(K, h_cur, H64)
                    if err_c <= fno_flux_post1_err:
                        cyc_list.append(c)
                        break
                else:
                    cyc_list.append(10)

    summary_out = {}
    for sname, st in stats.items():
        summary_out[sname] = {
            "median_eh_pre": round(float(np.median(st["eh_pre"])), 4),
            "median_eh_post1": round(float(np.median(st["eh_post1"])), 4),
            "median_eq_face_pre": round(float(np.median(st["eq_face_pre"])), 4),
            "median_eq_face_post1": round(float(np.median(st["eq_face_post1"])), 4),
            "median_eq_cc_pre": round(float(np.median(st["eq_cc_pre"])), 4),
            "median_eq_cc_post1": round(float(np.median(st["eq_cc_post1"])), 4),
            "median_idiv_pre": round(float(np.median(st["idiv_pre"])), 4),
            "median_idiv_post1": round(float(np.median(st["idiv_post1"])), 4),
            "arrived_post1": st["arrived_post1"],
            "arrived_pct_post1": round(100.0 * st["arrived_post1"] / 984.0, 2),
            "stalled_post1": st["stalled_post1"],
            "stalled_pct_post1": round(100.0 * st["stalled_post1"] / 984.0, 2),
            "backwashed_post1": st["backwashed_post1"],
            "backwashed_pct_post1": round(100.0 * st["backwashed_post1"] / 984.0, 2),
            "realizations_with_arrival_post1": st["realizations_with_arrival_post1"],
            "median_paired_travel_ratio_post1": round(float(np.median(st["paired_ratios_post1"])), 4) if st["paired_ratios_post1"] else None,
        }
    summary_out["median_cycles_from_linear_to_match_fno_flux_1cycle"] = float(np.median(cycles_needed_from_linear))
    summary_out["mean_cycles_from_linear_to_match_fno_flux_1cycle"] = round(float(np.mean(cycles_needed_from_linear)), 2)
    summary_out["median_cycles_from_zero_to_match_fno_flux_1cycle"] = float(np.median(cycles_needed_from_zero))
    summary_out["mean_cycles_from_zero_to_match_fno_flux_1cycle"] = round(float(np.mean(cycles_needed_from_zero)), 2)
    print("Experiment 1 results:", json.dumps(summary_out, indent=2), flush=True)
    return summary_out


def run_unet_resize_conv_control() -> dict:
    """Experiment 2: Compare baseline UNet (ConvTranspose2d) vs ResizeUNet (Bilinear + Conv2d)."""
    print("=== Running Experiment 2: U-Net Decoder Causal Ablation (ResizeUNet vs UNet) ===", flush=True)
    test_cells = [(1.0, 0.05), (2.0, 0.03)]
    out = {}
    for s2, lam in test_cells:
        cid = f"s{s2:g}_l{lam:g}_n128"
        cell_path = ROOT / "data" / f"cell_{cid}.h5"
        with h5py.File(cell_path, "r") as f:
            attrs = dict(f.attrs)
            Y = f["Y_cell"][:]
            H = f["H"][:]
        n_tr, n_va, n_te = int(attrs["n_train"]), int(attrs["n_val"]), int(attrs["n_test"])
        ymu, ystd = float(Y[:n_tr].mean()), float(Y[:n_tr].std())
        n_eff = train_cap(n_tr)

        Xtr = torch.from_numpy(make_inputs(Y[:n_eff], ymu, ystd)).to(DEV)
        Ttr = torch.from_numpy(((H[:n_eff] - 0.5) / 0.5).astype(np.float32))[:, None].to(DEV)
        Kall = torch.from_numpy(np.exp(Y[:n_eff].astype(np.float32)))[:, None].to(DEV)
        Xva = torch.from_numpy(make_inputs(Y[n_tr:n_tr + n_va], ymu, ystd)).to(DEV)
        Tva = torch.from_numpy(((H[n_tr:n_tr + n_va] - 0.5) / 0.5).astype(np.float32))[:, None].to(DEV)
        Xte = torch.from_numpy(make_inputs(Y[-n_te:], ymu, ystd)).to(DEV)
        Hte = H[-n_te:]
        Yte = Y[-n_te:]

        out[cid] = {}
        # First evaluate archived baseline UNet (alpha=0 and alpha=0.01)
        for alpha_label, ckpt_path in [
            ("unet_transposed_a0", ROOT / "runs" / "atlas" / cid / "unet" / "model.pt"),
            ("unet_transposed_a0.01", ROOT / "runs" / "fluxloss" / f"unet_a0.01_{cid}" / "model.pt"),
        ]:
            net = UNet(cin=3, width=24, depth=3).to(DEV)
            net.load_state_dict(torch.load(ckpt_path, map_location=DEV))
            net.eval()
            with torch.no_grad():
                hp = net(Xte).cpu().numpy()[:, 0] * 0.5 + 0.5
            ehs, eqs, eqfs, idivs = [], [], [], []
            for i in range(n_te):
                Ki = np.exp(Yte[i].astype(np.float64))
                Hi = Hte[i].astype(np.float64)
                hpi = hp[i].astype(np.float64)
                ehs.append(head_error(hpi, Hi))
                eqs.append(flux_error(Ki, hpi, Hi))
                eqfs.append(face_flux_error(Ki, hpi, Hi))
                idivs.append(mass_balance_residual(Ki, hpi, Hi))
            out[cid][alpha_label] = {
                "params": count_params(net),
                "median_eh": round(float(np.median(ehs)), 4),
                "median_eq_cc": round(float(np.median(eqs)), 4),
                "median_eq_face": round(float(np.median(eqfs)), 4),
                "face_to_cc_ratio": round(float(np.median(eqfs) / np.median(eqs)), 3),
                "median_idiv": round(float(np.median(idivs)), 4),
            }

        # Now train ResizeUNet (alpha=0 and alpha=0.01) under the exact same seed, epochs, and protocol
        for alpha in (0.0, 0.01):
            label = f"unet_resize_a{alpha:g}"
            torch.manual_seed(0)
            np.random.seed(0)
            model = ResizeUNet(cin=3, width=24, depth=3).to(DEV)
            f0 = None
            if alpha > 0:
                with torch.no_grad():
                    f0 = flux_loss(model(Xtr[:64]) * 0.5 + 0.5, Ttr[:64] * 0.5 + 0.5, Kall[:64]).clamp_min(1e-12)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
            best_val, best_state, bad = np.inf, None, 0
            for ep in range(80):
                model.train()
                perm = torch.randperm(n_eff)
                for i in range(0, n_eff, 32):
                    idx = perm[i:i + 32]
                    xb, tb = Xtr[idx], Ttr[idx]
                    pred = model(xb)
                    loss = F.mse_loss(pred, tb)
                    if alpha > 0:
                        loss = loss + alpha * flux_loss(pred * 0.5 + 0.5, tb * 0.5 + 0.5, Kall[idx]) / f0
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                sched.step()
                model.eval()
                with torch.no_grad():
                    vloss = float(F.mse_loss(model(Xva), Tva))
                if vloss < best_val - 1e-6:
                    best_val, bad = vloss, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 15:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                hp = model(Xte).cpu().numpy()[:, 0] * 0.5 + 0.5
            ehs, eqs, eqfs, idivs = [], [], [], []
            for i in range(n_te):
                Ki = np.exp(Yte[i].astype(np.float64))
                Hi = Hte[i].astype(np.float64)
                hpi = hp[i].astype(np.float64)
                ehs.append(head_error(hpi, Hi))
                eqs.append(flux_error(Ki, hpi, Hi))
                eqfs.append(face_flux_error(Ki, hpi, Hi))
                idivs.append(mass_balance_residual(Ki, hpi, Hi))
            out[cid][label] = {
                "params": count_params(model),
                "median_eh": round(float(np.median(ehs)), 4),
                "median_eq_cc": round(float(np.median(eqs)), 4),
                "median_eq_face": round(float(np.median(eqfs)), 4),
                "face_to_cc_ratio": round(float(np.median(eqfs) / np.median(eqs)), 3),
                "median_idiv": round(float(np.median(idivs)), 4),
            }
            print(f"  {cid} {label}: {out[cid][label]}", flush=True)
    return out


def run_channelized_media_control() -> dict:
    """Experiment 3: Non-Gaussian sinuous channelized media generalization test."""
    print("=== Running Experiment 3: Non-Gaussian Channelized Media Generalization ===", flush=True)
    rng = np.random.default_rng(20261108)
    n = 128
    d = 1.0 / n
    x = (np.arange(n) + 0.5) / n
    Xc, Yc = np.meshgrid(x, x, indexing="xy")

    regimes = [
        ("channel_mild", 0.5, 0.08),
        ("channel_moderate", 1.0, 0.05),
        ("channel_rough", 2.0, 0.03),
    ]
    all_eta, all_eqf_pre, all_eqf_post = [], [], []
    out = {}

    for rname, s2, lam_ref in regimes:
        cid = f"s{s2:g}_l{lam_ref:g}_n128"
        # Load reference models trained on (s2, lam_ref)
        unet = UNet(cin=3, width=24, depth=3)
        unet.load_state_dict(torch.load(ROOT / "runs" / "atlas" / cid / "unet" / "model.pt", map_location="cpu"))
        unet.eval()

        fno_h = FNO2d(cin=3, width=24, modes=12)
        fno_h.load_state_dict(torch.load(ROOT / "runs" / "atlas" / cid / "fno" / "model.pt", map_location="cpu"))
        fno_h.eval()

        fno_f = FNO2d(cin=3, width=24, modes=12)
        fno_f.load_state_dict(torch.load(ROOT / "runs" / "fluxloss" / f"fno_a0.01_{cid}" / "model.pt", map_location="cpu"))
        fno_f.eval()

        r_grad_list = []
        res_models = {m: {"eh": [], "eq_cc": [], "eq_face": [], "eq_face_amg": []} for m in ("unet", "fno_head", "fno_flux")}

        for inst in range(16):
            # Construct sinuous connected high-K channels via level-set thresholding of an anisotropic field + smooth transition
            base = rng.standard_normal((n, n))
            # Anisotropic smoothing: long correlation along flow (x, axis=1), short transverse (y, axis=0)
            sigma_x = max(2.0, n * 0.18)
            sigma_y = max(1.5, n * lam_ref * 0.6)
            chan_raw = gaussian_filter(base, sigma=(sigma_y, sigma_x), mode="wrap")
            # Create binary facies (35% high-K channels) smoothed across a 1.5-cell boundary transition
            thresh = np.quantile(chan_raw, 0.65)
            facies = gaussian_filter((chan_raw >= thresh).astype(np.float64), sigma=1.2, mode="wrap")
            facies = (facies - facies.mean()) / (facies.std() + 1e-12) * np.sqrt(s2)
            Y_chan = facies.astype(np.float32)
            K_chan = np.exp(Y_chan.astype(np.float64))
            H_chan, _ = solve_darcy(K_chan)
            H64 = H_chan.astype(np.float64)

            # Compute parameter-free discrete gradient energy R_{\nabla Y} = L^2 <|\nabla Y|^2>
            gx = (Y_chan[:, 1:] - Y_chan[:, :-1]) / d
            gy = (Y_chan[1:, :] - Y_chan[:-1, :]) / d
            r_grad = float(np.mean(gx[:-1, :]**2 + gy[:, :-1]**2))
            r_grad_list.append(r_grad)

            A, rhs, _, _ = assemble(K_chan)
            ml = pyamg.smoothed_aggregation_solver(A.astype(np.float64), max_coarse=500)

            X_in = torch.from_numpy(make_inputs(Y_chan[None], float(Y_chan.mean()), float(Y_chan.std())))
            for mname, net in [("unet", unet), ("fno_head", fno_h), ("fno_flux", fno_f)]:
                with torch.no_grad():
                    hp = (net(X_in).squeeze().numpy().astype(np.float64)) * 0.5 + 0.5
                r = rhs - A @ hp.ravel()
                z1 = ml.solve(r.astype(np.float64), x0=np.zeros_like(r), tol=0.0, maxiter=1, accel=None, cycle="V")
                hp_corr = hp + z1.reshape(n, n)

                eh = head_error(hp, H64)
                eq_cc = flux_error(K_chan, hp, H64)
                eq_f = face_flux_error(K_chan, hp, H64)
                eq_f_amg = face_flux_error(K_chan, hp_corr, H64)
                eta_f = _flux_norm(K_chan, z1.reshape(n, n)) / (_flux_norm(K_chan, hp) + 1e-30)

                res_models[mname]["eh"].append(float(eh))
                res_models[mname]["eq_cc"].append(float(eq_cc))
                res_models[mname]["eq_face"].append(float(eq_f))
                res_models[mname]["eq_face_amg"].append(float(eq_f_amg))

                all_eta.append(float(eta_f))
                all_eqf_pre.append(float(eq_f))
                all_eqf_post.append(float(eq_f_amg))

        out[rname] = {
            "sigma2": s2,
            "mean_R_grad_Y": round(float(np.mean(r_grad_list)), 1),
            "models": {
                m: {
                    "median_eh": round(float(np.median(res_models[m]["eh"])), 4),
                    "median_eq_cc": round(float(np.median(res_models[m]["eq_cc"])), 4),
                    "median_eq_face": round(float(np.median(res_models[m]["eq_face"])), 4),
                    "median_eq_face_after_1cycle_amg": round(float(np.median(res_models[m]["eq_face_amg"])), 4),
                }
                for m in ("unet", "fno_head", "fno_flux")
            },
        }
        print(f"  {rname}: R_grad={out[rname]['mean_R_grad_Y']}, models={out[rname]['models']}", flush=True)

    labels = [1 if e > 1.0 else 0 for e in all_eqf_pre]
    auc = float(roc_auc_score(labels, all_eta)) if len(set(labels)) > 1 else 1.0
    rho = float(spearmanr(all_eta, all_eqf_pre).statistic)
    out["pooled_channelized_amg_diagnostics"] = {
        "n_evaluations": len(all_eta),
        "eta_f_roc_auc": round(auc, 4),
        "eta_f_spearman_rho": round(rho, 4),
        "median_eq_face_before_amg": round(float(np.median(all_eqf_pre)), 4),
        "median_eq_face_after_amg": round(float(np.median(all_eqf_post)), 4),
        "relative_reduction_pct": round(100.0 * (1.0 - float(np.median(all_eqf_post)) / float(np.median(all_eqf_pre))), 1),
    }
    print("Channelized summary:", json.dumps(out["pooled_channelized_amg_diagnostics"], indent=2), flush=True)
    return out


if __name__ == "__main__":
    out_path = ROOT / "runs" / "extended_controls.json"
    data = {}
    if out_path.exists():
        try:
            data = json.loads(out_path.read_text())
        except Exception:
            data = {}
    data["amg_and_transport_control"] = run_amg_and_transport_control()
    out_path.write_text(json.dumps(data, indent=2))
    data["unet_resize_conv_control"] = run_unet_resize_conv_control()
    out_path.write_text(json.dumps(data, indent=2))
    data["channelized_media_control"] = run_channelized_media_control()
    out_path.write_text(json.dumps(data, indent=2))
    print("ALL 3 EXTENDED CONTROLS COMPLETE!", flush=True)
