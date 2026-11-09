# SubsurfaceReliability v1.0 — User Guide

This document describes the inputs, outputs, command-line options, data structures, and expected behavior of the `SubsurfaceReliability v1.0` software package accompanying the *Computers & Geosciences* article *"Flux blindness in neural solvers for heterogeneous subsurface flow: Spectral mechanisms, pre-training screening, and flux-informed learning"*.

## 1. System Requirements and Installation

- **Operating System:** macOS, Linux, or Windows (64-bit)
- **Python Version:** Python $\ge 3.10$ (tested on Python 3.12)
- **Core Dependencies:** `numpy >= 2.0`, `scipy >= 1.14`, `torch >= 2.4`, `pyamg >= 5.2`, `pandas >= 2.2`, `matplotlib >= 3.9`, `h5py >= 3.11`, `scikit-learn >= 1.5`

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy torch pyamg pandas matplotlib h5py scikit-learn
export PYTHONPATH=src
```

## 2. Core Workflow Modules (`src/subrel/`)

### 2.1 Stage 1: Pre-Training Medium Roughness Screening (`subrel.measured_roughness`, `subrel.analysis`)
- **Input:** Discretized log-conductivity array `Y` of shape `(N, N)` or `(B, N, N)` where hydraulic conductivity is $K(\mathbf{x}) = \exp(Y(\mathbf{x}))$.
- **Parameters:**
  - `sigma2` (float, optional): Log-conductivity variance $\sigma^2$.
  - `lam` (float, optional): Dimensionless correlation length $\lambda / L$.
- **Outputs:**
  - Parameter-free discrete gradient energy $\mathcal{R}_{\nabla Y} = L^2 \langle |\nabla Y|^2 \rangle$.
  - High-wavenumber spectral energy fraction $\Phi(k_c = 8) = E_Y(k \ge 8) / E_{Y,\text{tot}}$.
  - Binary screening recommendation indicating whether head-only $L_2$ training is sufficient ($\sigma^2/\lambda < 3.33$ for U-Net, $\sigma^2 < 0.75$ for FNO, $\mathcal{R}_{\nabla Y} < 400$ for PINN) or whether conservative TPFA face-flux supervision ($\alpha \in [0.01, 0.10]$) is required.

### 2.2 Verified TPFA Finite-Volume Reference Solver (`subrel.solver`)
- **Function:** `solve_darcy(K)` and `assemble(K)`
- **Input:** 2D positive conductivity array `K` of shape `(N, N)` on the unit square $\Omega = [0, 1]^2$.
- **Boundary Conditions:** Constant head $h(0, y) = 1$ (left inflow), $h(1, y) = 0$ (right outflow), and no-flow $\partial_y h(x, 0) = \partial_y h(x, 1) = 0$ (top and bottom).
- **Output:**
  - `H`: Exact discrete hydraulic head array of shape `(N, N)`.
  - `(qx, qy)`: Cell-centered specific discharge components.
  - `A, rhs`: Sparse symmetric positive-definite TPFA stiffness matrix $A_K$ and right-hand-side vector $\mathbf{b}$ with harmonic interface transmissibilities $T_{i+1/2,j} = 2K_{i,j}K_{i+1,j}/(K_{i,j}+K_{i+1,j})$.

### 2.3 Neural Solver Architectures and Flux-Informed Training (`subrel.models`, `subrel.train_grid`)
- **Supported Architectures:**
  - `UNet(cin=3, width=24, depth=3)`: Baseline convolutional encoder–decoder with stride-2 transposed convolutions.
  - `ResizeUNet(cin=3, width=24, depth=3)`: Controlled decoder variant using bilinear upsampling followed by $3\times 3$ convolution (`subrel.extended_controls`).
  - `FNO2d(cin=3, width=24, modes=12)`: 2D Fourier Neural Operator retaining $m_1 = m_2 = 12$ spectral modes.
  - `PINN`: Coordinate MLP ($4 \times 64$ $\tanh$ layers) with hard Dirichlet trial ansatz $\hat{h}(x,y) = (1-x) + x(1-x)\mathcal{N}_\theta(x,y)$.
- **Input Tensor Format (U-Net / FNO):** `[Y_std, x, y]` of shape `(B, 3, N, N)`, where `Y_std = (Y - mean) / std`.
- **Training Objective:**
  $$\mathcal{L}(\theta) = \mathcal{L}_h + \frac{\alpha}{f_0}\mathcal{L}_{q,\text{face}}$$
  where $\alpha = 0$ corresponds to standard head-only MSE training and $\alpha \in \{0.01, 0.10\}$ activates normalized conservative TPFA face-flux supervision.

### 2.4 Stage 2: Online 1-Cycle AMG Residual Screening and Defect Correction (`subrel.posterior`)
- **Input:** Conductivity array `K` of shape `(N, N)` and neural head prediction `h_pred` of shape `(N, N)`.
- **Procedure:**
  1. Assemble discrete TPFA residual $r = b - A_K \hat{h}$.
  2. Execute a single smoothed-aggregation algebraic multigrid (AMG) V-cycle $z_1 \approx A_K^{-1} r$ (`maxiter=1, cycle="V"`).
  3. Compute the reference-free dual-norm velocity error indicator:
     $$\eta_F = \frac{\|\mathbf{q}_f(z_1)\|_{2,\Omega_{\text{int}}}}{\max(\|\mathbf{q}_f(\hat{h})\|_{2,\Omega_{\text{int}}}, 10^{-12})}$$
  4. Apply the one-step defect correction $\hat{h}_{\text{corr}} = \hat{h} + z_1$ and reconstruct conservative TPFA face fluxes $\mathbf{q}_f(\hat{h}_{\text{corr}})$.
- **Expected Behavior:** $\eta_F > 0.35$ reliably flags velocity failure ($E_{q,\text{face}} > 1.0$) with $\text{ROC-AUC} \ge 0.997$, and $\hat{h}_{\text{corr}}$ reduces median interior face-flux error by $90\%\text{--}93\%$ while restoring $>94\%$ Pollock advective particle arrival.

### 2.5 Semi-Analytical Pollock Streamline Particle Tracking (`subrel.breakthrough`)
- **Function:** `travel_times(K, H)` and `summarize(ts_ref, ts_pred)`
- **Input:** Conductivity field `K` and head field `H` of shape `(N, N)`.
- **Output:** Per-particle fate (`arrived`, `stalled`, or `backwashed`), breakthrough fraction, and paired arrival travel-time ratio $t / t_{\text{ref}}$ across $N_p = 41$ particles released along the inflow plane $x = 2\Delta x$.

## 3. Command-Line Usage and Expected Outputs

| Task | Command | Expected Runtime & Output |
| :--- | :--- | :--- |
| Run 4-gate solver verification & unit tests | `python3 -m subrel.verify && python3 src/subrel/tests/test_blind_descriptors.py` | $\sim 5\text{ s}$; prints `PASS` on MMS 2nd-order convergence, linearity, field stats, and mass conservation |
| Run quickstart end-to-end tutorial | `python3 tutorials/quickstart_workflow.py` | $\sim 3\text{ s}$; runs Stage 1 roughness screening, neural/surrogate evaluation, Stage 2 AMG correction, and Pollock tracking on a test field |
| Reproduce all summary tables & Figures 1–7 | `bash scripts/reproduce_figures_and_tables.sh` | $\sim 15\text{ s}$; prints atlas tables, screening AUCs, and writes 600 dpi `figures/fig1_phase.png` .. `fig7_transport.png` |
| Run extended controls (AMG warm-start, `ResizeUNet`, channelized, $256^2$, MADE) | `python3 src/subrel/extended_controls.py` | $\sim 3\text{ min}$; writes `runs/extended_controls.json` |
