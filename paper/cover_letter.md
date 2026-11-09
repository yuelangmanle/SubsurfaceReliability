November 8, 2026

To the Editors-in-Chief
*Computers & Geosciences*

Dear Editors,

We are pleased to submit our original research article, **"Flux blindness in neural solvers for heterogeneous subsurface flow: Spectral mechanisms, pre-training screening, and flux-informed learning,"** by Zijie Deng, Zhenyu Liao, and Jiyang Yu, for consideration for publication in *Computers & Geosciences*.

Deep-learning surrogates for subsurface flow (such as U-Nets, Fourier Neural Operators, and PINNs) are routinely trained and benchmarked using relative L2 errors on hydraulic head or pressure. However, groundwater remediation, solute transport, and geological carbon storage depend directly on the derived Darcy velocity field **q = -K ∇h**. Rather than presenting a routine application of existing architectures, our manuscript introduces an open-source, reliability-aware two-stage computational workflow (`SubsurfaceReliability v1.0`) that answers a fundamental question in geoscientific computing: **Why do neural PDE solvers with low hydraulic-head error (E_h < 6%) fail in Darcy velocity and streamline transport across heterogeneous porous media, and how can practitioners screen, prevent, and correct this failure?**

Our work makes three integrated contributions at the interface of scientific computing and hydrogeology:

1. **Spectral Mechanism and Controlled Decoder Diagnosis:** Across 3,264 held-out evaluations spanning 24 lognormal regimes, 12 unseen off-grid regimes, and non-Gaussian channelized and MADE-type alluvial lens media, we show that wavenumber-squared (k²) derivative weighting across conductivity interfaces drives pervasive velocity failure (75.3%–91.4% of head-trained FNO and U-Net predictions exceed the error of a zero-velocity guess). Controlled Matérn covariance interventions and a bilinear resize-convolution decoder ablation (`ResizeUNet`) isolate the respective roles of medium spectral tails and stride-2 transposed-convolution interface artifacts.
2. **Stage 1 Pre-Training Roughness Screening and Conservative Face-Flux Learning:** We establish that medium roughness invariants (σ²/λ, σ², and parameter-free discrete gradient energy R_∇Y) screen regime suitability before training any network (instance ROC-AUC 0.915–0.955; off-grid balanced accuracy 1.000 for U-Net and 100% sensitivity for FNO). We further show that direct supervision on two-point flux-approximation (TPFA) harmonic face fluxes overcomes the fourth-order conditioning bottleneck of unsupervised divergence penalties, cutting FNO velocity error by 52%–95% across Gaussian regimes and by 57%–73% zero-shot on non-Gaussian channelized and alluvial media.
3. **Stage 2 Reference-Free AMG Residual Screening and Transport Recovery:** During online inference, a single algebraic multigrid (AMG) V-cycle yields a reference-free dual-norm residual indicator η_F (ROC-AUC 0.991–1.000 across 128², 256², and non-Gaussian media), cuts median face-flux error by 89%–93% as an O(N²) one-step defect correction (2.4×–4.4× more accurate than one AMG cycle from linear or zero initial guesses), and raises Pollock advective particle breakthrough from 2.2% to 94.0% (median travel-time ratio 1.00).

In full compliance with *Computers & Geosciences* reproducibility standards, the open-source Python package (`SubsurfaceReliability v1.0`, MIT License), user guide (`USER_GUIDE.md`), quickstart tutorial (`tutorials/quickstart_workflow.py`), automated verification suite, evaluation archives, and all 409 trained model checkpoints (via GitHub Releases) are publicly available at https://github.com/yuelangmanle/SubsurfaceReliability.

This manuscript is original, has not been published previously, and is not under consideration elsewhere. All authors have approved the submission and declare no competing interests.

Sincerely,

**Zijie Deng** (Corresponding Author)  
Jiangxi Provincial Key Laboratory of Water Resources Allocation and Efficient Utilization  
Jiangxi University of Water Resources and Electric Power  
No. 289 Tianxiang Avenue, Nanchang, 330099, China  
Email: 3335196397vip@gmail.com | ORCID: 0009-0002-4824-9764  
On behalf of co-authors Zhenyu Liao and Jiyang Yu
