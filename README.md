# SubsurfaceReliability

**Flux blindness in neural solvers for heterogeneous subsurface flow: Spectral mechanisms, pre-training screening, and flux-informed learning**

Open-source two-stage diagnostic framework, benchmark suite, and hybrid neural–multigrid reproducibility workflow accompanying the manuscript submitted to *Computers & Geosciences*. This software package evaluates three canonical neural partial differential equation (PDE) solver families—convolutional U-Net, Fourier Neural Operator (FNO), and Physics-Informed Neural Network (PINN)—across 24 regimes of lognormal hydraulic conductivity heterogeneity ($\sigma^2 \in \{0.5, 1.0, 2.0, 4.0\} \times \lambda \in \{0.02, 0.03, 0.05, 0.08, 0.12, 0.20\}$), 12 prospective off-grid validation regimes, and three non-Gaussian channelized facies regimes, benchmarked against a manufactured-solution-verified two-point flux approximation (TPFA) finite-volume reference solver.

## Software Architecture and Repository Structure

```text
src/subrel/
  field.py                    Band-limited spectral generator for Gaussian and Matérn log-K fields
  solver.py                   Cell-centered TPFA finite-volume Darcy solver
  verify.py                   Four-gate solver and field verification suite (MMS, linearity, statistics, conservation)
  data.py                     Per-regime dataset generation (Fourier spectra, cell-center K, reference heads)
  models.py                   Baseline U-Net, FNO2d, and coordinate PINN architectures
  metrics.py                  Interior relative L2 head and Darcy-flux error operators
  train_grid.py               Amortized surrogate trainer (uniform hyperparameters across regimes)
  pinn_solve.py               Per-instance PINN solver (Adam + L-BFGS)
  scan.py                     Benchmark driver across the 24 heterogeneity regimes
  analysis.py                 Regime phase tables, roughness descriptors, and ROC-AUC evaluation
  nested_loo.py               Nested leave-one-regime-out cross-validation of screening descriptors
  blind_test.py               Prospective off-grid blind validation protocol
  bootstrap_ci.py             Cell-cluster bootstrap confidence intervals for screening AUCs
  measured_roughness.py       Empirical conductivity gradient-energy descriptor evaluation
  recompute_dual.py           Deterministic test-slice regeneration and vector-norm sensitivity checks
  grid_check.py               Grid-refinement sensitivity analysis (128x128 vs. 256x256)
  spectral_phi.py             High-wavenumber spectral energy fraction and radial integral scale
  covariance_experiments.py   Controlled Gaussian vs. Matérn (nu=3/2, 1/2) covariance interventions
  shell.py / shell_run.py     Phase-preserving Fourier-shell energy redistribution experiments
  modes_test.py               FNO spectral mode-cutoff sensitivity experiments (12, 24, 32 modes)
  repairs.py                  TPFA conservation fine-tuning and residual-adaptive resampling (RAR)
  repair_experiments.py       Data-budget doubling, conservation fine-tuning, and linear-profile baselines
  boundary_interventions.py   Matched-budget MSE continuation controls and boundary RAR evaluations
  ablation_experiments.py     Direct flux-loss training, matched-capacity controls, and PINN residual scaling
  breakthrough.py             Semi-analytic 2D Pollock streamline tracking for advective transport
  transport_robustness.py     Grid-convergence and injection-density robustness checks for particle tracking
  frc.py                      Error-covariance orthogonality and amplification analysis
  posterior.py                Single-cycle algebraic multigrid (AMG) residual indicator and defect correction
  posterior_scaling.py        Multi-resolution runtime and discrimination scaling (128^2, 256^2, 512^2)
  extended_controls.py        AMG initial-guess/transport control, ResizeUNet ablation, and channelized media test
  fig1.py .. fig7.py          Scripts reproducing Figures 1–7 from archived run records
  tests/                      Unit tests verifying blind-test descriptor rules and confusion matrices

scripts/
  run_benchmark_atlas.sh          Runs verification gates and executes the 24-regime benchmark scan
  run_mechanistic_ablations.py    Executes multi-seed, off-grid blind Batch 2, and PINN blind experiments
  summarize_extended_validation.py Aggregates multi-seed, off-grid blind, and extended control results
  reproduce_figures_and_tables.sh Recomputes summary tables, runs verification tests, and generates Figures 1–7
  build_archive_bundle.py         Packages HDF5 fields and model checkpoints with SHA-256 manifest

runs/                         Archived per-realization and per-regime JSON/CSV/NPZ evaluation records
figures/                      Publication figures (600 dpi PNG, including 3D phase and regularization surfaces)
```

## Quick Start and One-Command Reproduction

To run the four-gate numerical verification suite, unit tests, summary tables, and regenerate Figures 1–7 from the archived evaluation records (no `.h5` or `.pt` files required, completes in ~15 seconds on CPU):

```bash
export PYTHONPATH=src

# 1. Run ground-truth verification gates and unit tests
python3 -m subrel.verify
python3 src/subrel/tests/test_blind_descriptors.py

# 2. Print benchmark summary tables and regenerate Figures 1–7
bash scripts/reproduce_figures_and_tables.sh
```

To regenerate the full training benchmark and extended mechanistic controls from scratch:

```bash
bash scripts/run_benchmark_atlas.sh
python3 scripts/run_mechanistic_ablations.py
python3 -m subrel.extended_controls
python3 scripts/summarize_extended_validation.py
```

## Methodological Guarantees

1. **Unified physical medium definition:** Every conductivity field $Y = \ln K$ is defined by its exact band-limited Fourier spectrum. The finite-volume reference evaluates exact spectral values at cell centers, while the PINN queries bilinear interpolants of the exact spectral $(Y, \partial_x Y, \partial_y Y)$ fields on a $4N \times 4N$ grid at collocation points.
2. **Uniform baseline protocols:** Hyperparameters within each solver family are held fixed across all 24 benchmark regimes and 12 off-grid validation regimes.
3. **Consistent interior evaluation:** All head and Darcy-flux error metrics are evaluated on the interior domain (excluding a 2-cell boundary margin) using identical discrete operators across reference and predicted fields.
4. **Deterministic reproducibility:** Field generation and training seeds derive deterministically from regime parameters $(\sigma^2, \lambda, n)$.

## License

MIT License. See `LICENSE` and `CITATION.cff` for details.
