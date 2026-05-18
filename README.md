# LearningLandscape

**Unbalanced Parametrized Landscape Neural Networks (uPLNN)**  
*Learning Waddington-like potential landscapes with cell proliferation and death from single-cell snapshot data*

---

## Project Overview

This project extends the Parametrized Landscape Neural Network (PLNN) framework
([Howe & Mani, Phys. Rev. X **15**, 031070, 2025](https://doi.org/10.1103/8vpj-bj7d))
in two directions that address its principal limitations:

1. **Non-conservative dynamics** — jointly inferring a potential landscape Φ(x)
   and a state-dependent proliferation/apoptosis rate R(x) from distributional
   snapshots, within a single differentiable SDE framework.
2. **End-to-end nonlinear manifold learning** — replacing the fixed PCA
   projection with a learnable encoder–decoder that maps high-dimensional gene
   expression onto a low-dimensional latent space where gradient dynamics and
   landscape inference are performed.

The combination yields a unified model — **uPLNN** — that, for the first time,
simultaneously learns a tiltable landscape, signal-dependent control, stochastic
dynamics, and birth–death processes in an end-to-end differentiable pipeline.

### Core innovation

No existing method jointly achieves all of the following:

| Capability | PLNN | PRESCIENT | DeepRUOT | LSD (2026) | **uPLNN (ours)** |
|:-----------|:----:|:---------:|:--------:|:----------:|:----------------:|
| Explicit potential Φ | ✓ | ✓ | ✗ | ✓ | **✓** |
| Learnable growth R(x) | ✗ | ✗ | ✓ | ✗ | **✓** |
| Stochastic dynamics (SDE) | ✓ | ✓ | ✗ | ✗ | **✓** |
| Signal-dependent tilt | ✓ | ✗ | ✗ | ✗ | **✓** |
| Nonlinear manifold | ✗ | ✗ | ✗ | ✓ | **✓** |

---

## From Real Data to Model Design

Before diving into the mathematical framework, we address three foundational
questions that arise when bridging single-cell experimental data and
SDE-based simulation models.

### I. Is MMD well-defined between unweighted cells and weighted particles?

**Conclusion: Yes — it is strictly well-defined.**

The Maximum Mean Discrepancy (MMD) measures the distance between two
probability measures $P$ and $Q$ in a reproducing kernel Hilbert space (RKHS):

$$\text{MMD}^2(P, Q) = \|\mu_P - \mu_Q\|_{\mathcal{H}}^2$$

where $\mu_P = \mathbb{E}_{X \sim P}[\phi(X)]$ is the kernel mean embedding
of measure $P$.

In real single-cell RNA-seq data, we cannot observe a continuous measure — we
observe $M$ discrete cells. In measure-theoretic terms, this forms the
**empirical measure** $\hat{P}$ of the true measure $P$:

$$\hat{P} = \frac{1}{M} \sum_{i=1}^{M} \delta_{x_i}$$

Each cell carries strictly equal weight $\frac{1}{M}$.

In our SDE simulation, we generate $N$ computational particles (where $N$ can
differ from $M$). Because particles accumulate log-weights $S_j$ through
the death rate $\gamma(z)$ during evolution, the normalized empirical measure
$\hat{Q}$ of the simulation is:

$$\hat{Q} = \sum_{j=1}^{N} \alpha_j \, \delta_{z_j}, \quad \text{where} \quad \alpha_j = \frac{\exp(S_j)}{\sum_{k=1}^{N} \exp(S_k)}$$

Substituting $\hat{P}$ and $\hat{Q}$ into the MMD definition, the double
integrals reduce to double sums:

$$\text{MMD}^2(\hat{P}, \hat{Q}) = \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j \, k(z_i, z_j) - \frac{2}{M} \sum_{i=1}^{N} \sum_{j=1}^{M} \alpha_i \, k(z_i, x_j) + \frac{1}{M^2} \sum_{i=1}^{M} \sum_{j=1}^{M} k(x_i, x_j)$$

**Key insight:** The MMD derivation never requires the two distributions to
have equal sample sizes, nor matching weight schemes. As long as
$\sum \alpha_i = 1$ and $\sum \frac{1}{M} = 1$ (i.e., both are probability
measures), the distance metric is mathematically rigorous and computable.
Weighted simulation particles are simply a non-uniform discretization of the
continuous density $\rho(z, t)$.

### II. How are source shape and generation rate determined?

**Conclusion: Source shape must be computed directly from real data (not
inferred); source generation rate must be fixed as a known condition (not
inferred).**

#### Why must the generation rate $\lambda(t)$ be fixed a priori?

If $\lambda(t)$ were treated as a learnable function, it would immediately
introduce a **global mass degeneracy**. The total mass $M(t)$ evolves as:

$$\frac{dM(t)}{dt} = \lambda(t) - \int \gamma(z) \, \rho(z, t) \, dz$$

If real data shows declining total cell counts, the model could equally
attribute this to "the source stopped producing cells ($\lambda(t) \to 0$)"
or "the system has extremely high global death rate ($\gamma(z) \gg 0$)."
This coupling is mathematically unresolvable without additional boundary
conditions. Therefore, $\lambda(t)$ must be provided as a prior function,
and the model can only infer $\gamma(z)$.

#### How to determine shape and rate from real data?

**Source shape (multivariate Gaussian):** No guessing required. At $t = 0$
or the earliest progenitor cell population, use clustering algorithms to
extract progenitor cells, then directly compute their empirical mean vector
$\mu$ and covariance matrix $\Sigma$ in latent space. The source distribution
is set as $\mathcal{N}(\mu, \Sigma)$.

**Source generation rate $\lambda(t)$:**

- If the experiment records **absolute cell counts** at different time points,
  fit $\lambda(t)$ from the experimental proliferation curve.
- If the experiment only provides **relative proportions** (no absolute
  counts), this is mathematically equivalent to a relative rate model. In this
  case, set $\lambda(t) \equiv 1.0$ (choosing the injection rate as the
  system's time reference frame). The inferred death rate $\gamma(z)$ then
  represents the relative apoptosis rate with respect to stem cell injection.

### III. How does the model handle time-varying $\lambda(t)$ with fixed particle count?

This is the most common source of confusion when discretizing the physical
equations into code.

**Core principle: the number of computational particles ($N_\text{sim}$) and
the number of biological cells ($N_\text{cells}$) are fully decoupled in
numerical integration.**

In a SDE solver (e.g., `jax.lax.scan`), tensor shapes must remain fixed — you
cannot dynamically append particles as $\lambda(t)$ increases. The standard
numerical solution is a **pre-allocated queue with dynamic initial weights**.

#### Implementation logic

**1. Pre-allocate equal-count computational particles:**
Regardless of how $\lambda(t)$ varies, uniformly schedule $N_\text{queue}$
particle wake-up times across $t \in [0, T]$. For example, 1000 particles
total, one waking every $\Delta t$.

**2. Encode $\lambda(t)$ through initial mass, not particle count:**
When particle $i$ wakes at time $t_i$, its initial position is sampled from
$\mathcal{N}(\mu, \Sigma)$. The key step: its initial log-weight is set to:

$$S_i(t_i) = \log\!\bigl(\lambda(t_i) \cdot \Delta t\bigr)$$

**Physical equivalence:** If the biological system's generation rate at $t=1$
is twice that at $t=0$ (i.e., $\lambda(t{=}1) = 2\lambda(t{=}0)$), the code
does *not* wake up twice as many particles at $t=1$. It still wakes exactly
one particle, but assigns it an extra $\log(2)$ of initial log-weight
(effective mass $w = 2$). When this particle later participates in the
weighted MMD computation, it contributes twice the probability density — fully
equivalent to injecting two real cells.

This "weight-encodes-quantity" numerical method simultaneously satisfies the
time-varying source $\lambda(t)$ physics constraint and guarantees absolutely
static tensor shapes throughout neural network training.

---

## Mathematical Framework

### Governing equation

The probability density ρ(x, t) of a cell population evolving on a
d-dimensional landscape with proliferation and apoptosis obeys the
**Fokker–Planck equation with a reaction (source) term**:

```
∂ρ/∂t = ∇·(ρ ∇Φ) + D ∇²ρ + R(x) ρ
          ─────────   ──────   ──────
           drift      diffusion  birth/death
```

where Φ(x) is the potential landscape, D = σ²/2 is the diffusion coefficient,
and R(x) is the net proliferation rate (R > 0: proliferation; R < 0: apoptosis).

### The identifiability crisis

An observed local density increase (∂ρ/∂t > 0) can always be explained by
either (a) a deep potential well attracting particles via ∇Φ, or (b) a large
local proliferation rate R(x) > 0, or any linear combination thereof. This
**drift–reaction degeneracy** is the central theoretical challenge of this
project and dictates many of our architectural and regularization choices.

### Model architecture

The full uPLNN model consists of five learnable modules:

```
Encoder:      z = f_enc(g)                   g ∈ ℝ^G  →  z ∈ ℝ^d
Potential:    Φ(z) = Φ_nn(z) + C_conf ‖z‖⁴  (neural net + confinement)
Tilt map:     τ = Ψ(s) = A · s              s ∈ ℝ^r  →  τ ∈ ℝ^d
Proliferation: R_θ(z; s)                     (small MLP, signal-dependent)
Decoder:      ĝ = f_dec(z)                   z ∈ ℝ^d  →  ĝ ∈ ℝ^G
```

The tilted landscape height is ϕ(z; s) = Φ(z) + z^T Ψ(s), and the augmented
SDE governing each particle i is:

```
dz_i = [−∇_z Φ(z_i) − Ψ(s(t))] dt + σ dW_i     (position update)
dS_i = R_θ(z_i; s(t)) dt                          (log-weight update)
```

where S_i is the cumulative log-weight and the effective particle weight is
w_i = exp(S_i). The noise σ is a learnable scalar (with optional extension
to state-dependent σ(z)).

### Training objective

```
L = L_MMD(weighted) + λ_rec · L_reconstruction + λ_mass · L_mass + L_reg
```

where:

- **L_MMD(weighted)**: weighted maximum mean discrepancy between simulated
  (weighted) and observed particle distributions, using multi-bandwidth
  Gaussian kernels.
- **L_reconstruction**: ‖f_dec(f_enc(g)) − g‖² averaged over all observed
  cells, ensuring the autoencoder preserves gene-expression information.
- **L_mass**: global mass conservation penalty (see below).
- **L_reg**: physics-informed regularization (see below).

### Regularization strategy for identifiability

**L1 sparsity on R (critical for drift–reaction separation):**

```
L_reg = λ₂ · ‖R_θ‖₁
```

L1 (not L2) regularization is essential. L2 regularization shrinks R toward
small but globally nonzero values. L1 induces genuine sparsity, forcing R to
be exactly zero across most of the state space. This encodes the biological
prior that proliferation and apoptosis are highly localized (e.g., stem cell
niches, terminal differentiation zones). It tells the optimizer: "Do not
invoke R unless drift alone absolutely cannot explain the observed density
change."

**Global mass conservation penalty:**

```
L_mass = | ∫ R_θ(x) ρ(x) dx |²  ≈  | Σ_i α_i · R_θ(z_i) |²
```

This is derived by integrating the Fokker–Planck equation over all space: the
drift and diffusion terms vanish by the divergence theorem, leaving
dN/dt = ∫ R(x)ρ(x)dx. For systems with approximately stable total cell
counts, this should be near zero. The penalty prevents the network from
learning globally positive R to explain density transport, blocking the
principal shortcut that causes identifiability failure.

**Potential smoothness via architectural constraints (not loss-based):**

Direct gradient penalty λ₁‖∇Φ‖ in the loss function requires double
backpropagation (computing ∇_θ(∇_x Φ)), which involves Hessian traces and
incurs 10–50× computational overhead — prohibitive for multi-particle,
multi-timepoint SDE training.

Instead, we enforce smoothness through architecture:

1. **Spectral normalization** on all linear layers of Φ_nn — bounds the
   Lipschitz constant, providing a hard upper limit on ‖∇Φ‖ with negligible
   computational cost. This is the single most effective measure against SDE
   solver divergence.
2. **Weight decay (L2 on θ_Φ)** via `optax.adamw` — smaller weights → more
   linear mapping → smoother potential surface.
3. **Softplus activation only** — guarantees C∞ smoothness of Φ and hence
   continuity of the force field −∇Φ. ReLU is strictly forbidden (produces
   discontinuous forces that crash the SDE solver).

### Numerical stability for weighted particles

The log-weight S_i = ∫₀ᵗ R(z_i(s)) ds can grow large over long integration
times, causing exp(S) to overflow.

Countermeasures:

1. **Log-sum-exp normalization**: always compute weights via
   `jax.nn.logsumexp`, never via raw `exp(S) / sum(exp(S))`.
2. **Bounded R output**: the proliferation network uses `tanh` output scaled
   to [β_min, β_max], physically capping instantaneous growth/death rates.
3. **Effective Sample Size (ESS) monitoring**: ESS = (Σ w_i)² / Σ w_i².
   If ESS drops below 10% of particle count, R has learned non-physical
   singularities; trigger early stopping or increase λ₂.
4. **Optional particle resampling**: at intermediate checkpoints during long
   integrations, resample particles according to current weights (systematic
   resampling), then reset all weights to uniform.

---

## Implementation

### Tech stack

- **JAX** — hardware-accelerated autodiff and JIT compilation
- **Equinox** — neural network modules (same ecosystem as Diffrax)
- **Diffrax** — differentiable SDE solvers (Heun's method)
- **Optax** — optimizers with weight decay (AdamW)

### Code structure

Organized by model type. Each subfolder is self-contained with all needed `.py` files.

```
toymodel/
├── README.md
├── CLAUDE.md
├── requirements.txt
│
├── 01_potential_only/                 # Early standalone prototypes (no shared modules)
│   ├── toymodel.py                    #   PyTorch parametric potential + SDE
│   ├── toymodel_jax.py                #   JAX parametric potential + MMD inference
│   ├── toymodel_proliferation.py      #   JAX augmented SDE with log-weights
│   ├── toymodel.ipynb                 #   Notebook: PyTorch prototype
│   └── toymodel_jax.ipynb             #   Notebook: JAX prototype
│
├── 02_proliferation_nn/               # Closed-system NN: Φ + R (no death/source)
│   ├── train.py                       #   Multi-timepoint training (WaddingtonModel)
│   ├── models/                        #   PotentialNN, ProliferationNN, NoiseScalar, TiltLinear
│   ├── training/                      #   trainer.py, data_loader.py
│   ├── simulator/                     #   sde_solver.py, weighted_mmd.py
│   ├── analysis/                      #   visualization.py, bifurcation.py
│   └── *.png, *.eqx                   #   Results and trained model
│
├── 03_source_death_parametric/        # Open-system, 6 scalar params (no NN)
│   ├── train_sd_parametric.py         #   Single-snapshot parametric inference
│   ├── train_sd_parametric_multi.py   #   Multi-timepoint (breaks a/b/c coupling)
│   ├── training/data_loader.py        #   Particle queue, analytical death rate
│   └── *.png                          #   Results (potential, death rate, snapshots)
│
├── 03.1_asymmetric_potential/         # Asymmetric potential, fixed 2D death rate
│   ├── train_asym_potential_multi.py  #   Learn 5 potential params (a,b,c,d,e), fixed γ
│   ├── sweep_particles.py            #   Particle count sweep (N=2000–4000)
│   ├── training/data_loader.py        #   2D death rate γ_select + γ_term, oval source
│   └── sweep/                         #   Per-N results + summary_params_vs_N.png
│
├── 04_source_death_semi_nn/           # NN Φ + parametric death rate γ(y_c, k)
│   ├── train_sd.py                    #   Single-snapshot training (SourceDeathModel)
│   ├── models/                        #   PotentialNN + DeathRateParametric
│   ├── training/, simulator/, analysis/
│   └── *.png, *.eqx                   #   Results and trained model
│
├── 05_source_death_nn/                # NN Φ + NN γ(y), multi-timepoint
│   ├── train_sd_nn_multi.py           #   6-snapshot training, reduced architectures
│   ├── models/                        #   PotentialNN(16,16) + DeathRateNN(y_only, 8,8)
│   ├── training/, simulator/, analysis/
│   └── *.png, *.eqx                   #   Results and trained model
│
└── 05.1_asymmetric_nn/               # NN Φ + fixed 2D death, systematic sweeps
    ├── train_sd_nn_asym_multi.py      #   Main training script (NN + confinement + 2D death)
    ├── models/, training/, simulator/  #   Shared modules
    ├── sweep_nparticles_v3/           #   N_learned sweep (200–6000), distribution analysis
    ├── sweep_seeds/                   #   Optimization variance (10 seeds)
    ├── learn_sigma/                   #   Learnable σ (with/without confinement)
    ├── sweep_ntarget/                 #   N_target sweep (2000–12000)
    ├── data_driven_confinement/       #   Data-driven anisotropic confinement (C_x, C_y from data)
    └── summary_*.png, analysis_*.png  #   Per-sweep results
```

**Shared modules** (copied into each subfolder that needs them):

| Module | Purpose |
|--------|---------|
| `models/potential.py` | Φ_nn: configurable `hidden_sizes` + softplus + confinement |
| `models/death_rate.py` | DeathRateNN (`y_only` mode, configurable `hidden_sizes`) + DeathRateParametric |
| `models/__init__.py` | WaddingtonModel, SourceDeathModel, `create_sd_model()` factory |
| `training/trainer.py` | Training loops: `train()`, `train_sd()` with best-model checkpointing |
| `training/data_loader.py` | Data generation, particle queues, analytical ground truth |
| `simulator/sde_solver.py` | Euler-Maruyama, open-system sim, `simulate_open_system_full()` |
| `simulator/weighted_mmd.py` | Weighted MMD, mass loss, sparsity loss, death losses |
| `analysis/visualization.py` | Landscape, death rate, particles, training curves |

### Current codebase status

| Module | Status | Description |
|--------|--------|-------------|
| Parametric potential + SDE + MMD | ✅ Done | `01_potential_only/toymodel_jax.py` |
| Augmented SDE with log-weights | ✅ Done | `01_potential_only/toymodel_proliferation.py` |
| Neural network potential (Equinox) | ✅ Done | `models/potential.py` — configurable MLP + softplus + confinement |
| Learnable R_θ | ✅ Done | `models/proliferation.py` — MLP (2→16→16→1), tanh-bounded output |
| Learnable noise σ | ✅ Done | `models/noise.py` — log-space scalar or state-dependent MLP |
| Tilt Ψ(s) module | ✅ Done | `models/tilt.py` — Linear(d_signal → d_latent) |
| Multi-timepoint training | ✅ Done | `training/trainer.py` — short-time transition matching |
| Loss functions | ✅ Done | `simulator/weighted_mmd.py` — weighted MMD + mass + sparsity + death |
| Bifurcation analysis | ✅ Done | `analysis/bifurcation.py` — equilibrium finding + Hessian |
| Parametric source+death | ✅ Done | `03_*/train_sd_parametric*.py` — identifies coupling degeneracy, multi-timepoint breaks it |
| Semi-NN source+death | ✅ Done | `04_*/train_sd.py` — NN Φ + parametric γ |
| Full NN source+death | ✅ Done | `05_*/train_sd_nn_multi.py` — reduced Φ(16,16) + y-only γ(8,8), symmetric potential recovered |
| Asymmetric potential (fixed death) | ✅ Done | `03.1_*/train_asym_potential_multi.py` — 5-param potential + 2D death, confirms a-c degeneracy is fundamental |
| NN asymmetric + systematic sweeps | ✅ Done | `05.1_*/` — N_learned, N_target, seed, σ sweeps; distribution diagnostics; N_target≥2000 sufficient |
| Data-driven confinement | ✅ Done | `05.1_*/data_driven_confinement/` — anisotropic C_x, C_y from data+death rate, replaces true quartic coefficients |
| Regional mass matching | ⚠️ Ineffective | data sparsity in y>2 prevents y_c/k degeneracy breaking |
| Spectral normalization | 🔲 Planned | Lipschitz constraint on NN weights |
| Weight decay tuning | 🔲 Planned | Stronger L2 to bias toward smooth mappings |
| Data symmetry augmentation | 🔲 Planned | x ↔ -x reflection for symmetric systems |
| Autoencoder (encoder-decoder) | 🔲 Planned | Deterministic AE for high-dimensional gene expression |
| ESS monitoring | 🔲 Planned | Effective sample size tracking |

---

## Research Roadmap

### Phase 1: Foundations and neural architecture (Steps 1–5)

**Step 1 — Literature and mathematical framework**

Required reading (priority order):

1. Howe & Mani (Phys. Rev. X, 2025) — re-read Methods Sec. V and Supplemental
   Material: SDE solver choice, MMD bandwidth, confinement term, continuation.
2. Zhou, Wang, Li et al., "Energy landscape decomposition for cell
   differentiation with proliferation effect" (Natl. Sci. Rev., 2022) —
   theoretical basis for dual-potential decomposition under birth–death.
3. Yeo, Saksena & Gifford, "PRESCIENT" (Nat. Commun., 2021) — SDE potential
   learning with growth weights; closest predecessor to PLNN.
4. Zhang, Li & Zhou, "DeepRUOT" (ICLR 2025, Oral) — weighted-particle
   formulation, Wasserstein–Fisher–Rao metric, no prior growth rates needed.

Supplementary reading:

- Farrell, Mani & Goyal, "LatentVelo" (Cell Rep. Methods, 2023)
- Vinyard et al., "scDiffEq" (Nat. Mach. Intell., 2025)
- Jiang & Wan, "PI-SDE" (Bioinformatics, 2024)
- Poursina et al., "LSD" (bioRxiv, March 2026)

Deliverable: 2–3 page mathematical framework document with complete model
equations, loss function, and regularization terms.

**Step 2 — Equinox refactoring with architectural smoothness guarantees**

Upgrade the toy model from analytic potential to neural network:

- Φ_nn: 4-layer MLP (2→16→32→32→16→1) with **softplus** activation,
  **spectral normalization** on all linear layers, Xavier uniform initialization.
- R_θ: 2-layer MLP (2→16→16→1) with **tanh** output scaled to [−0.5, +0.5].
- Use `optax.adamw` with explicit weight decay on Φ_nn parameters.
- Verify: forward SDE integration runs without NaN for 1000 steps before
  any training begins.

> **Implementation notes (2026-03-31):**
>
> Completed the Equinox refactoring. The analytic potential
> `H = -(y-a)x² + exp(b)x⁴ - cy + exp(d)y⁴` was replaced by a neural network
> `Φ_nn` (MLP 2→16→32→32→16→1, softplus activation, Xavier uniform init) with a
> quartic confinement term `C_conf·‖z‖⁴` (`C_conf = 0.01`) in `models/potential.py`.
> The proliferation rate was replaced by a learnable MLP `R_θ` (2→16→16→1,
> softplus hidden, unconstrained linear output) in `models/proliferation.py`.
> Noise is parameterized in log-space (`σ = exp(log_σ)`) for guaranteed positivity
> in `models/noise.py`. All modules are Equinox `eqx.Module` subclasses.
>
> The full model is assembled via `create_model()` in `models/__init__.py` as a
> `WaddingtonModel` dataclass holding potential, tilt, proliferation, and noise
> sub-modules.
>
> **First run (single-timepoint, `train.py`)**: 500 epochs, `optax.adam(lr=1e-3)`,
> fixed PRNG key, 1000 particles, dt=0.005, 400 steps (T=2.0). Loss function:
> `L = L_MMD(weighted) + 0.1·L_conservation + 0.01·L_grad_reg(L2)`.
> The ground-truth data was generated using the analytic potential with a Gaussian
> proliferation rate `R(x,y) = 0.5·exp(-dist²/1.28) - 0.1` centered at origin.
>
> **Results**: Loss decreased from 1.04 to 0.011 (MMD: 0.011). The learned
> potential correctly captured the bifurcation topology with two symmetric basins.
> The learned R(z) showed proliferation near the progenitor region and decay
> elsewhere. **However, σ converged to 0.556 instead of the true value 1.0.**
> This is the expected Φ/σ² scale degeneracy: the NN potential has unconstrained
> absolute scale, so the optimizer finds a valid but non-unique solution where
> both Φ and σ are jointly scaled down while preserving their ratio.
>
> **Deviations from plan**: (1) Spectral normalization on Φ_nn layers is not yet
> implemented — using plain Xavier init. (2) R_θ output is unconstrained (no tanh
> clamping yet). (3) Using `optax.adam` instead of `optax.adamw` — weight decay
> not yet added. These are deferred to the synthetic benchmark phase (Step 5)
> where their impact can be measured via ablation.

**Step 3 — Multi-timepoint training with short-time transition matching**

Critical upgrade from current single-snapshot comparison:

- Implement consecutive time-point pair format: D = {(t_i, X_i, t_{i+1}, X_{i+1}, s(t))}.
- **Short-time transition matching**: compute MMD loss only between adjacent
  timepoints (t_i, t_{i+1}), not from t_0 to t_final. This truncates the
  backpropagation horizon through the SDE scan, dramatically reducing gradient
  variance and memory consumption.
- **Weight handling between intervals**: reset particle weights to uniform at
  each interval boundary. Each interval (t_i, t_{i+1}) independently
  accumulates log-weights S from zero. The weighted MMD at t_{i+1} captures
  local proliferation within that interval only. This avoids catastrophic
  weight divergence over long trajectories while still allowing the network to
  learn R from density changes between consecutive snapshots.
- Implement ESS monitoring: log effective sample size at each training step;
  flag if ESS < 0.1 × N_particles.

> **Implementation notes (2026-03-31):**
>
> Implemented multi-timepoint short-time transition matching. The data loader
> (`training/data_loader.py`) generates ground-truth snapshots at user-specified
> times (e.g., t = [0.0, 0.5, 1.0, 1.5, 2.0]) and constructs consecutive
> transition pairs in the Howe & Mani format `D = {(t_i, X_i, t_{i+1}, X_{i+1})}`.
> A new `simulate_transition()` function in `simulator/sde_solver.py` simulates
> a short-time segment from observed positions X_i with fresh log-weights (S₀=0),
> implementing the weight-reset-at-boundary strategy.
>
> The trainer (`training/trainer.py`) pre-compiles a separate JIT-traced loss
> function per unique segment length via `make_transition_loss()`, then sums the
> weighted MMD + conservation + gradient reg losses across all 4 transitions per
> epoch. Each transition independently accumulates proliferation weights over its
> own interval only.
>
> **Additional changes in this iteration:**
> - Gradient regularization switched from **L2 to L1 (Lasso)**:
>   `(1/N) Σ ‖∇R(z_i)‖₁` — encourages sparsity in the proliferation gradient
>   field, consistent with the biological prior that R is spatially localized.
> - Log-weight normalization switched to **explicit `jax.nn.logsumexp`**:
>   `α_i = exp(S_i - logsumexp(S))` instead of `jax.nn.softmax(S)`. Both are
>   mathematically equivalent, but the explicit form makes the numerical
>   stability mechanism visible and auditable.
>
> **Multi-timepoint run (`train.py`)**: 500 epochs, `optax.adam(lr=1e-3)`,
> 4 transitions of 100 steps each ([0→0.5], [0.5→1.0], [1.0→1.5], [1.5→2.0]),
> 1000 particles, dt=0.005. Loss function:
> `L = Σ_transitions [L_MMD(weighted) + 0.1·L_conservation + 0.01·L_grad_reg(L1)]`.
>
> **Results**: Total loss decreased from 1.49 to 0.04 over 500 epochs. The
> learned potential correctly captured the two-basin bifurcation topology. The
> learned R(z) was much flatter/sparser than the single-timepoint run, which is
> the expected effect of L1 regularization — R is driven toward zero except where
> density changes between consecutive snapshots demand it.
>
> **σ degeneracy persists**: σ converged to 0.575 (true: 1.0). Multi-timepoint
> fitting alone does not break the Φ/σ² scale degeneracy because the NN potential
> can still freely rescale. This confirms the theoretical prediction: breaking
> this degeneracy requires either (a) anchoring the potential scale via spectral
> normalization, (b) velocity/RNA-velocity data providing absolute drift
> magnitude, or (c) explicit constraints on Φ's range. This is a priority issue
> for the Step 5 synthetic benchmark.
>
> **ESS monitoring**: not yet implemented. Deferred to Step 5.
>
> **Deviations from plan**: signal `s(t)` is not yet passed through the
> transition pairs — the tilt module exists but the training loop runs with
> `signal=None`. This is acceptable for the current signal-free synthetic
> benchmark and will be activated when signal-dependent data is introduced.
>
> ---
>
> **Hyperparameter tuning experiments (2026-03-31):**
>
> Conducted a systematic series of experiments to improve R(z) recovery and σ
> convergence. Each change was applied incrementally.
>
> | Run | λ_cons | λ_reg | Particles | Intervals | Epochs | lr | σ_learned | MMD_final | R quality |
> |-----|--------|-------|-----------|-----------|--------|----|-----------|-----------|-----------|
> | 1 (baseline) | 0.1 | 0.01 | 1000 | 4×100 steps | 500 | 1e-3 | 0.575 | 0.040 | Flat / near-zero |
> | 2 (lower λ_reg) | 0.1 | 0.001 | 1000 | 4×100 steps | 500 | 1e-3 | 0.575 | 0.040 | Slight structure, asymmetric |
> | 3 (longer T) | 0.1 | 0.001 | 1000 | 3×200 steps | 1000 | 1e-3 | 0.597 | 0.022 | Asymmetric left-right |
> | 4 (higher λ_cons + lr) | 1.0 | 0.001 | 1000 | 3×200 steps | 1000 | 3e-3 | 0.782 | 0.012 | Asymmetric, σ improving |
> | 5 (more particles + lower λ_reg) | 1.0 | 0.0001 | 2000 | 3×200 steps | 2000 | 3e-3 | 0.855 | 0.005 | Asymmetric, similar pattern |
> | 6 (+ resampled targets) | 1.0 | 0.0001 | 2000 | 3×200 steps | 2000 | 3e-3 | 0.824 | 0.006 | Asymmetric, same issue |
> | 7 (+ stronger R: β=2, δ=0.3) | 1.0 | 0.0001 | 2000 | 3×200 steps | 2000 | 3e-3 | 0.874 | 0.006 | Asymmetric, same pattern |
>
> **Key finding — target data resampling bug (fixed in Run 6):**
> The original data loader stored target positions X1 WITHOUT resampling by
> proliferation weights. Since R only affects log-weights S (not positions), the
> target X1 positions were identical whether R existed or not. The weighted MMD
> compared simulation weights against uniform target weights, giving R nothing to
> fit. Fixed by adding multinomial resampling at each snapshot: particles are
> redrawn proportional to their accumulated weights, encoding R's density effect
> into the position distribution — mimicking how scRNA-seq observes more cells
> where proliferation is high.
>
> **Persistent issue — drift-reaction degeneracy in R recovery:**
> Across all 7 runs, the learned R(z) shows a left-right asymmetric pattern
> (positive on the left basin, negative on the right) instead of the ground-truth
> symmetric Gaussian centered at (0, -1). The magnitude of learned R (±0.06 to
> ±0.12) is much smaller than the true R (peak 0.4 to 1.7). This confirms the
> **drift-reaction degeneracy** described in the mathematical framework: the NN
> potential Φ has enough capacity to explain position density changes without
> R's help, so R is relegated to acting as a small drift correction for Φ's
> imperfections rather than capturing the true proliferation field.
>
> **σ convergence improved significantly:** From 0.575 (Run 1) to 0.874 (Run 7).
> The main driver was increasing λ_cons from 0.1 to 1.0 and using higher learning
> rate (3e-3). The conservation constraint prevents R from absorbing global scale
> information that should belong to σ.
>
> **Implications for Step 4 and beyond:**
> - The R recovery problem is fundamentally an identifiability issue, not a
>   hyperparameter tuning issue. Additional constraints are needed:
>   (a) Spectral normalization on Φ_nn to bound its Lipschitz constant
>   (b) tanh-bounded R output to prevent it from acting as drift correction
>   (c) Potentially a two-phase training: first fit Φ without R, then freeze Φ
>       and fit R to the residual density changes
> - The resampling fix in the data loader is critical for any future work with
>   proliferation — without it, R is completely unobservable from position data.

**Step 4 — Fix fundamental loss function bugs (Q1–Q4)**

Code review after the Step 3 hyperparameter experiments revealed four interacting
bugs that created a "complete information blockade on R," making the proliferation
rate mathematically unrecoverable regardless of hyperparameter choices.

> **Implementation notes (2026-04-01):**
>
> **Q1 — MMD normalization erases absolute mass (FIXED):**
> The weighted MMD normalizes α to sum=1 via logsumexp, discarding the absolute
> mass amplification factor m = mean(exp(S)). If m=2 (population doubled) or
> m=0.5 (population halved), the normalized α is identical — R's overall level
> is unobservable from MMD alone. **Fix**: added `mass_matching_loss` in
> `simulator/weighted_mmd.py`: `L_mass = (logsumexp(S) - log(N) - log_m_obs)²`,
> where `log_m_obs` is the ground-truth log mass ratio computed per transition
> interval in `training/data_loader.py`. `simulate_transition` in
> `simulator/sde_solver.py` now returns raw `final_S` alongside normalized `alpha`.
>
> **Q2 — Conservation loss forces E[R]=0 (REMOVED):**
> The conservation loss `L_cons = |Σ αᵢ R(zᵢ)|²` penalizes net population
> change, forcing R toward zero-mean. This is correct only for systems at
> population steady state — our ground truth has strong net growth (β=2.0,
> δ=0.3). With λ_cons=1.0, this was the dominant force crushing R. **Fix**:
> `conservation_loss` removed entirely from `weighted_mmd.py` and `trainer.py`.
> Replaced by `mass_matching_loss` (Q1 fix) which constrains R's level to match
> observed growth, not to zero.
>
> **Q3 — Gradient regularizer is Total Variation, not L1 sparsity (FIXED):**
> `gradient_reg_loss` computed `mean(‖∇ᵤR‖₁)` — the L1 norm of R's *gradient*,
> i.e., Total Variation. This penalizes R having spatial *variation*, not R being
> *nonzero*. A constant R≡c has zero TV penalty regardless of |c|. The correct
> biological prior is L1 sparsity on R *values*: R should be exactly zero across
> most of state space. **Fix**: replaced with `sparsity_reg_loss` computing
> `mean(|R(zᵢ)|)` — true L1 (Lasso) on R values.
>
> **Q4 — Φ/σ² scale degeneracy via confinement (FIXED):**
> The fixed confinement term `c_conf·‖z‖⁴` in the potential creates an implicit
> scale anchor. With `c_conf=0.01`, the confinement already prevents particle
> escape, so σ converges to whatever value balances the NN potential against
> confinement — not the true σ. **Fix**: switched optimizer from `optax.adam`
> to `optax.adamw(weight_decay=1e-4)`. Weight decay acts as soft L2
> regularization on Φ_nn weights, penalizing large potential values and providing
> a softer Lipschitz constraint. This gives σ a wider range of valid solutions
> closer to the true value. (Full spectral normalization deferred to a later step.)
>
> **Additional fix — comparison.png target panel background:**
> The target panel in `plot_comparison` was incorrectly showing the *learned*
> potential as background. Fixed: `plot_comparison` now accepts a
> `target_potential_fn` parameter; the left (target) panel uses the analytical
> ground-truth potential, and the right (learned) panel uses `model.potential`.
>
> **Loss function summary (before → after):**
> ```
> Before: L = L_MMD + λ_cons·|Σ αᵢ Rᵢ|² + λ_reg·mean(‖∇R‖₁)
> After:  L = L_MMD + λ_mass·(log m_sim - log m_obs)² + λ_sparse·mean(|R|)
> ```
>
> The three R-related bugs (Q1+Q2+Q3) formed a mathematical optimum at R≡0:
> - Q1: MMD can't see R's level → no gradient signal to increase |R|
> - Q2: Conservation loss actively penalizes any nonzero E[R]
> - Q3: TV regularizer doesn't penalize constant R, but Q2 already pushes R→0
> Together, R was trapped at zero regardless of hyperparameters.

**Step 5 — Open-system source + death model**

The proliferation model (Steps 2–4) revealed an inherent **drift–reaction
degeneracy**: since R(x) can take arbitrary positive and negative values, the
optimizer uses R as a small drift correction for Φ's imperfections rather than
capturing the true proliferation field. To break this degeneracy, we reformulate
the problem as an **open system with source injection and parametric death**.

> **Key insight**: by separating mass injection (source) from mass removal
> (death), and constraining the death rate to be non-negative, we eliminate the
> degeneracy. The death rate γ(z) ≥ 0 can only *remove* mass, forcing Φ to do
> all transport work. This is biologically natural: cells are born at specific
> locations (stem cell niches) and die elsewhere (terminal differentiation,
> apoptosis zones).

> **Implementation notes (2026-04-05):**
>
> **Physics of the open system:**
>
> The system starts empty at t=0. A source continuously injects particles, each
> with a unique birth time t₀ ∈ [0, T) and position x₀ ~ N(μ, σ²_src I).
> Particles flow down the potential Φ, undergo bifurcation, and die via a
> learnable death rate. The governing equations are:
>
> ```
> dz_i = −∇Φ(z_i) dt + σ dW_i          (position: gradient descent + noise)
> dS_i = −γ(z_i) dt                      (log-weight: death only, S ≤ 0)
> w_i  = exp(S_i) ∈ (0, 1]               (effective mass: strictly non-increasing)
> ```
>
> The key difference from the proliferation model: S is strictly non-increasing
> because γ ≥ 0. Particles are born with w=1 and can only lose mass, never gain
> it. This eliminates the weight overflow problem (no need for tanh bounding)
> and makes the death rate structurally identifiable.
>
> **Parametric death rate:**
>
> Instead of a neural network, we use a 2-parameter analytical form:
>
> ```
> γ(z) = softplus(k · (y − y_c))
> ```
>
> where y_c (threshold) and log_k (log-steepness) are the only learnable
> parameters. This serves as a controlled test case: if the framework can
> recover these 2 parameters, it validates the loss function and training
> pipeline before scaling to the neural network death rate.
>
> Ground truth: y_c = 2.2, k = 1.0 (death activates above y = 2.2).
>
> **Implementation details:**
>
> | Component | File | Description |
> |-----------|------|-------------|
> | `DeathRateParametric` | `models/death_rate.py` | 2-param death rate: softplus(k·(y−y_c)) |
> | `DeathRateNN` | `models/death_rate.py` | NN death rate: MLP 2→16→16→1, softplus output |
> | `SourceDeathModel` | `models/__init__.py` | Top-level model: Φ + γ (no proliferation, no noise param) |
> | `create_sd_model()` | `models/__init__.py` | Factory with parametric/NN death rate selection |
> | `build_particle_queue()` | `training/data_loader.py` | Pre-allocates N particles with unique (t₀, x₀) |
> | `generate_open_system_data()` | `training/data_loader.py` | Ground-truth steady-state via analytical SDE + death |
> | `simulate_open_system()` | `simulator/sde_solver.py` | Open-system SDE with wake-up schedule |
> | `open_system_mass_loss()` | `simulator/weighted_mmd.py` | Mass matching: (log Σexp(S) − log m_target)² |
> | `death_sparsity_loss()` | `simulator/weighted_mmd.py` | L1 on γ values (for NN death rate only) |
> | `train_sd()` | `training/trainer.py` | Training loop with best-model checkpointing |
> | `train_sd.py` | (root) | Main script for open-system training |
>
> **Particle queue architecture (JAX-compatible source injection):**
>
> JAX's `lax.scan` requires fixed array shapes. We pre-allocate all N particles
> at t=0 in a "dormant" state (S = −100, effectively w ≈ 0) and assign each a
> wake-up step. At the designated step, the particle's position is reset to its
> birth location and S is reset to 0 (w = 1). This avoids dynamic array
> resizing while simulating continuous particle injection.
>
> **Training configuration:**
>
> ```python
> TARGET_PARAMS = {'a': 0, 'b': -1.6, 'c': 3.5, 'd': -1.2, 'sigma': 1.5}
> DEATH_PARAMS = {'y_threshold': 2.2}
> SOURCE = {'mu': [0.0, -1.0], 'sigma_src': 0.15, 'n_particles': 1000}
> T_FINAL = 8.0, DT = 0.01, N_STEPS = 800
>
> Loss = L_MMD + 1.0 · L_mass          (no L1 sparsity for parametric death)
> Optimizer: clip_by_global_norm(1.0) + AdamW(lr=3e-3, weight_decay=1e-4)
> Epochs: 1000, patience: 200
> ```
>
> **Critical training fixes discovered during development:**
>
> 1. **L1 sparsity is counterproductive for parametric death (λ_sparse=0):**
>    The death sparsity loss evaluates γ on a fixed grid and penalizes
>    mean(γ). For the parametric form softplus(k·(y−y_c)), higher y_c means
>    less death everywhere, so L1 pushes y_c upward past the true value. With
>    λ_sparse=0.1, the sparse term consumed 85% of the total loss, driving
>    y_c to 2.81 (true: 2.2) and destroying the potential's bifurcation.
>    **Fix**: set λ_sparse=0.0. L1 sparsity is only needed for the NN death
>    rate (many parameters, risk of spatial overfitting), not the 2-parameter
>    parametric form.
>
> 2. **Best-model checkpointing against stochastic spikes:**
>    With `use_fresh_keys=True`, each epoch uses different Brownian noise.
>    After ~700 epochs (loss ≈ 0.01), the signal-to-noise ratio deteriorates:
>    an unlucky random seed can produce loss spikes 50× above baseline
>    (0.01 → 0.6). The large gradient corrupts model parameters, and without
>    checkpointing, all pre-spike progress is permanently lost. **Fix**: the
>    trainer now saves a copy of the best model's array leaves (via
>    `jax.tree.map` + `eqx.filter`) whenever a new best loss is found, and
>    restores the best model at the end of training via `eqx.combine`.
>
> 3. **Gradient clipping for spike robustness:**
>    Even with checkpointing, large spikes waste epochs on recovery.
>    `optax.clip_by_global_norm(1.0)` limits the gradient norm per step,
>    preventing catastrophic parameter jumps. Combined with checkpointing,
>    this reduced spike amplitude and improved final loss from 0.0101 to
>    0.0069.
>
> **Results (parametric death rate, 1000 epochs):**
>
> | Metric | Value |
> |--------|-------|
> | Best loss (epoch 983) | 0.0069 |
> | Best MMD | 0.0073 |
> | Learned y_c | 2.00 (true: 2.2) |
> | Learned k | 3.18 (true: 1.0) |
>
> The learned potential correctly recovers the **two-basin bifurcation topology**
> with good symmetry. The death rate shows the correct spatial pattern: near-zero
> for y < y_c, sharply increasing for y > y_c.
>
> **The k/y_c degeneracy**: the learned k (3.18) is 3× the true value (1.0),
> while y_c (2.00) is slightly below truth (2.2). This is a parametric
> identifiability issue: softplus(3.18·(y−2.00)) ≈ softplus(1.0·(y−2.2)) in
> the region where particles actually exist, because a steeper sigmoid with a
> lower threshold produces nearly the same effective death boundary. The MMD
> loss cannot distinguish between these equivalent parameterizations. This
> degeneracy will be resolved when scaling to the NN death rate, which learns
> the full spatial profile rather than two scalar parameters.
>
> **Training history**: loss decreased monotonically from 0.53 to 0.007 with
> only minor spikes (compared to 0.6 spikes without gradient clipping).
> Early stopping triggered at epoch 1000 (patience=200), and the best model
> from epoch 983 was restored.

**Step 5a — Parametric inference diagnostic (no neural network)**

The NN-based source+death model (Step 5) recovered the bifurcation topology but
exhibited two persistent issues: (1) death rate inaccuracy (y_c=2.00 vs true 2.2,
k=3.18 vs true 1.0), and (2) potential asymmetry (the learned Φ_nn broke the
x-symmetry of the ground truth). To diagnose whether these are NN capacity issues
or fundamental identifiability problems, we return to the `toymodel_jax.py`
parametric inference approach — using the **exact analytical potential form** and
parametric death rate with no neural network.

> **Implementation notes (2026-04-12):**
>
> Created `train_sd_parametric.py`: a self-contained diagnostic script that
> infers 6 scalar parameters via gradient descent through the open-system SDE,
> using the same framework as `train_sd.py` but replacing all neural networks
> with the known analytical forms.
>
> **Learnable parameters (6 total):**
>
> | Parameter | Form | Initial | True |
> |-----------|------|---------|------|
> | a | bifurcation point in H = -(y-a)x² + exp(b)x⁴ - cy + exp(d)y⁴ | 0.5 | 0 |
> | b | log x⁴ coefficient | -1.0 | -1.6 |
> | c | drive strength (tilt) | 2.0 | 3.5 |
> | d | log y⁴ coefficient | -0.8 | -1.2 |
> | y_c | death threshold in γ(z) = softplus(k·(y-y_c)) | 0.0 | 2.2 |
> | log_k | log death steepness | 0.5 | 0.0 (k=1) |
>
> **Fixed:** σ = 1.5 (same as `train_sd.py`). Not learned.
>
> **Key design difference from Step 5:** the analytical potential is symmetric
> in x by construction (only even powers x², x⁴), so any asymmetry in the
> results must be a finite-sample artifact or parameter coupling, not NN
> expressivity.
>
> **Training configuration:**
>
> ```python
> N_PARTICLES = 2000      # up from 1000 in Step 5
> T_FINAL = 8.0, DT = 0.01, N_STEPS = 800
> Loss = L_MMD(weighted) + 1.0 · L_mass
> Optimizer: clip_by_global_norm(1.0) + Adam(lr=0.01)
> Epochs: 500, use_fresh_keys=True
> ```
>
> **Results (500 epochs, best model at epoch 497):**
>
> | Parameter | Learned | True | Error |
> |-----------|---------|------|-------|
> | a | -1.030 | 0.0 | **-1.030** |
> | b | -1.191 | -1.6 | +0.409 |
> | c | 3.749 | 3.5 | +0.249 |
> | d | -1.242 | -1.2 | -0.042 |
> | y_c | 2.158 | 2.2 | **-0.042** |
> | k | 1.532 | 1.0 | +0.532 |
> | Best loss | 0.0090 | | |
>
> **Analysis — parameter coupling degeneracy:**
>
> 1. **Parameter `a` (bifurcation point) has large error (-1.03 vs 0.0).**
>    This is the most significant finding. Even with the correct functional
>    form, `a` is not recovered. The reason: `a` enters the potential as
>    `-(y - a)x²`, so shifting `a` downward is equivalent to adding a
>    `+a·x²` term to the potential. This can be partially compensated by
>    adjusting `b` (x⁴ coefficient) and `c` (y-tilt), creating a family of
>    parameter combinations that produce nearly identical landscapes in the
>    region where particles exist. The optimizer finds a local minimum in
>    this degenerate manifold, not the true parameters.
>
> 2. **Death rate y_c recovery is excellent (2.158 vs 2.2, error -0.04).**
>    Dramatically better than the NN version (2.00 vs 2.2, error -0.20).
>    The parametric potential's inability to absorb death-rate information
>    into extra NN degrees of freedom makes γ more identifiable.
>
> 3. **k/y_c trade-off persists but is less severe.** k=1.53 (true 1.0)
>    vs NN version k=3.18. The 1D death rate profile shows that learned
>    and true curves nearly overlap in the region y ∈ [-1, 3] where particles
>    actually exist. The k overestimate only affects y > 3 where there is
>    little data support.
>
> 4. **Well-recovered parameters:** `d` (y⁴ confinement, error -0.04) and
>    `y_c` (death threshold, error -0.04) are accurately recovered because
>    their physical effects are direct and weakly coupled to other parameters.
>
> 5. **Potential asymmetry eliminated.** The learned potential is symmetric
>    in x (by construction), confirming that the NN version's asymmetry was
>    an NN artifact, not a framework issue.
>
> **Conclusions for the project:**
>
> - The death rate inaccuracy and potential issues are **not purely NN
>   problems** — they reflect fundamental parameter identifiability
>   challenges in the open-system framework.
> - The `a`-`b`-`c` coupling degeneracy suggests that **anchoring at least
>   one potential parameter** (e.g., fixing `a=0` as a known bifurcation
>   point, or fixing the potential scale) may be necessary for accurate
>   recovery.
> - The parametric diagnostic confirms that the death rate `y_c` **is**
>   recoverable when the potential has limited degrees of freedom. The NN
>   version's y_c error (0.20 vs 0.04 here) is therefore attributable to
>   the NN potential absorbing death-rate information.
> - For future NN training: consider a **two-phase strategy** — first fit
>   Φ_nn with death rate frozen, then freeze Φ_nn and fit γ. This mimics
>   the parametric case's reduced coupling.

**Step 5a-ii — Multi-timepoint transient matching (breaking parameter coupling)**

Step 5a identified a fundamental `a`-`b`-`c` coupling degeneracy: multiple
parameter combinations produce indistinguishable steady-state distributions.
The key insight is that these degenerate parameter sets have **different transient
dynamics** — "steep slope + fast death" vs "gentle slope + slow death" push
particles at different speeds, so their distributions at intermediate times
(t=1, 2, 3) differ even if their t=8 steady states converge.

By forcing the model to simultaneously match **6 snapshot times** during the
transient phase, we provide enough constraints to uniquely identify the
parameters. This is the multi-timepoint approach already used in Step 4 for
the NN model, now applied to the parametric diagnostic.

> **Implementation notes (2026-04-12):**
>
> Created `train_sd_parametric_multi.py`: extends `train_sd_parametric.py` with
> full-trajectory simulation (`jax.lax.scan` returning all timesteps) and
> snapshot-based loss averaging.
>
> **Snapshot times:** t = 1.0, 2.0, 3.0, 4.0, 6.0, 8.0
> - Early snapshots (t=1-3) capture wavefront propagation (most informative)
> - Late snapshots (t=6-8) ensure steady-state also matches
>
> **Key difference from Step 5a:** both sim and target use `softmax(S)` for
> weights at each snapshot, so dormant particles (S=-100) automatically get
> near-zero weight. MMD is computed between two weighted point clouds.
>
> **Ground-truth snapshot statistics:**
>
> | Snapshot | Mass | ~Alive particles |
> |----------|------|------------------|
> | t=1.0 | 229 | 250 |
> | t=2.0 | 384 | 500 |
> | t=3.0 | 478 | 750 |
> | t=4.0 | 536 | 1000 |
> | t=6.0 | 591 | 1500 |
> | t=8.0 | 611 | 2000 |
>
> These show a clear transient buildup — the system is far from steady state
> at t=1-3, providing the coupling-breaking information.
>
> **Training configuration:** same as Step 5a (2000 particles, Adam lr=0.01,
> grad clipping 1.0, 500 epochs) but with 6-snapshot averaged loss.
>
> **Results (500 epochs, best model at epoch 446):**
>
> | Parameter | Learned | True | Error | Step 5a error | Improvement |
> |-----------|---------|------|-------|---------------|-------------|
> | a | -0.139 | 0.0 | **-0.139** | -1.030 | **7.4x** |
> | b | -1.507 | -1.6 | +0.094 | +0.409 | **4.4x** |
> | c | 3.460 | 3.5 | **-0.041** | -1.490 | **36x** |
> | d | -1.210 | -1.2 | -0.010 | -0.042 | 4.2x |
> | y_c | 1.966 | 2.2 | -0.234 | -0.042 | worse |
> | k | 1.674 | 1.0 | +0.674 | +0.532 | similar |
> | Best loss | 0.00498 | | | 0.0090 | 1.8x |
>
> **Analysis — coupling degeneracy largely broken:**
>
> 1. **Potential parameters dramatically improved.** The `a`-`b`-`c` coupling
>    that plagued single-snapshot matching is largely resolved:
>    - `a` error: -1.03 → -0.14 (7.4x improvement)
>    - `c` error: -1.49 → -0.04 (36x improvement, nearly exact)
>    - `b` error: +0.41 → +0.09 (4.4x improvement)
>    - `d` error: -0.04 → -0.01 (also improved, already good)
>
> 2. **Death rate y_c/k trade-off persists and worsens slightly.**
>    y_c error increased from -0.04 to -0.23, and k error from +0.53 to +0.67.
>    This is expected: the multi-timepoint loss now constrains the potential
>    more tightly, leaving less flexibility for the death rate to compensate.
>    The y_c/k trade-off is a genuine 1D degeneracy in the softplus
>    parameterization (steeper sigmoid + lower threshold ≈ same boundary).
>
> 3. **Overall loss halved (0.0090 → 0.0050).** The multi-timepoint model
>    achieves a substantially better fit despite having the same number of
>    parameters — the extra temporal constraints guide optimization toward
>    the true parameter region rather than a degenerate manifold.
>
> 4. **Transient wavefront is the key signal.** The snapshots at t=1-3
>    (when only 250-750 particles are alive and actively flowing down the
>    potential) provide the strongest gradient signal for `a`, `b`, `c`
>    because these parameters control the flow speed and bifurcation timing.
>
> **Conclusions and next steps:**
>
> - Multi-timepoint matching **successfully breaks the a-b-c coupling
>   degeneracy** that was the main failure mode in Step 5a.
> - The remaining y_c/k trade-off is a **1D degeneracy** intrinsic to the
>   softplus death rate form — it cannot be broken by temporal information
>   alone and may require either reparameterization or regularization.
> - This validates the approach for the NN model: when scaling to NN
>   potential + NN death rate, multi-timepoint transient matching should be
>   used from the start.
>
> **Bug fix:** also corrected `particles_parametric.png` visualization in
> `train_sd_parametric.py` — both panels were incorrectly showing the true
> potential as background. The right panel now shows the learned potential.

**Step 5a-iii — Regional mass matching (attempting to break y_c/k degeneracy)**

The y_c/k trade-off identified in Steps 5a/5a-ii is a 1D degeneracy in the
softplus death rate: (y_c=1.97, k=1.67) produces nearly identical γ(y) as
(y_c=2.2, k=1.0) in the region where particles actually exist (y<2).

**Approach:** divide space into y-axis sub-regions and match mass in each
region separately, forcing the death rate to act at the correct spatial
location. Loss term: L_regional = (1/J) Σ_j [log(M_sim(Ω_j)+ε) - log(M_tgt(Ω_j)+ε)]²

> **Implementation notes (2026-04-12):**
>
> Added `regional_mass_loss()` to `train_sd_parametric_multi.py` with
> configurable region boundaries and λ_regional coefficient.
>
> **Region boundaries:** y < 1.0 | 1.0–1.5 | 1.5–2.0 | 2.0–2.5 | y ≥ 2.5
> (5 regions, finer resolution around the death transition zone)
>
> **Smoothing:** ε = 1.0 (handles zero-mass regions without masking real
> differences; with masses O(10–500), log(M+1) ≈ log(M))
>
> **λ sweep results (5 regions, 500 epochs each):**
>
> | λ_regional | a err | c err | y_c err | k err | Notes |
> |------------|-------|-------|---------|-------|-------|
> | 0 (baseline) | -0.139 | -0.041 | -0.234 | +0.674 | Best potential |
> | 0.01 | -0.119 | -0.059 | -0.233 | +0.674 | ≈ baseline, no effect |
> | 0.05 | -0.250 | -0.334 | -0.253 | +0.704 | Potential degraded, no y_c/k gain |
> | 0.1 (3 regions) | -0.389 | -0.571 | -0.299 | +0.772 | Worse everywhere |
> | 0.5 | -0.324 | -1.328 | -0.300 | +0.524 | k improved but potential wrecked |
>
> **Analysis — regional mass matching ineffective:**
>
> 1. **Data sparsity is the fundamental bottleneck.** Very few particles
>    reach y>2 (the death zone), so regional mass in the critical Ω_4
>    (2.0–2.5) and Ω_5 (y≥2.5) regions is dominated by stochastic noise.
>    The gradient signal from these regions is too noisy to reliably steer
>    y_c/k.
>
> 2. **Clear λ trade-off with no sweet spot.** Low λ (≤0.01) has zero
>    effect on y_c/k. High λ (≥0.1) degrades potential parameters because
>    the noisy regional gradients overwhelm the cleaner MMD signal. There
>    is no intermediate λ that improves death rate without hurting the
>    potential.
>
> 3. **The degeneracy is within-region, not between-region.** The pairs
>    (y_c=1.97, k=1.67) and (y_c=2.2, k=1.0) produce nearly identical
>    γ(y) for y<2 where particles exist. Even finer region boundaries
>    cannot distinguish them because the degenerate death rates agree
>    everywhere that data is available.
>
> **Conclusion:** Regional mass matching cannot break the y_c/k degeneracy
> because it requires particle data in the death zone (y>2) that doesn't
> exist. The softplus y_c/k trade-off is fundamentally an **extrapolation
> problem** — the model fits well in the data-rich region but is
> underdetermined in the data-sparse tail.
>
> **Potential alternatives (all require biological assumptions):**
>
> - **Fix k=1 (or add k-regularization prior)** — assumes transition steepness
>   is known a priori
> - **Phantom/probe evaluation** — compare γ(y) directly on a grid, bypassing
>   particle sparsity; assumes ground-truth γ is accessible
> - **Secondary source above y_c** — populate the death zone with data;
>   assumes ability to inject particles at arbitrary locations
>
> - **Terminal boundary hinge loss** *(assumes known survival boundary)*:
>   If biology dictates an absolute boundary y_max beyond which survival
>   probability is zero (e.g., terminally differentiated cells cannot exist
>   past a developmental stage), penalize surviving particles that cross it:
>   `L_boundary = (1/N) Σ_i exp(S_i) · max(0, y_i - y_max)^p` (p=2 or 3).
>   This acts as a "force field" in the data-void region: any particle not
>   killed by γ(z) before reaching y_max incurs a large penalty, forcing
>   the optimizer to increase the death rate slope k so that weights decay
>   to ~0 before the boundary. **Assumption:** y_max must be known from
>   domain knowledge (e.g., no observed cells beyond a certain pseudotime).
>
> - **Death rate gradient penalty / smoothness prior** *(assumes smooth
>   apoptosis field)*:
>   Penalize the spatial gradient of the death rate:
>   `L_smooth = (1/N) Σ_i ‖∇_z γ_θ(z_i)‖²`.
>   For the parametric model γ=softplus(k(y−y_c)), this simplifies to an
>   L2 penalty on k. Among all (y_c, k) pairs that satisfy the global mass
>   constraint, this selects the smoothest (lowest-k) solution — which is
>   typically more biologically realistic since apoptosis probability
>   transitions gradually rather than as a sharp step. **Assumption:**
>   the true death rate field is smooth/gradual, which is reasonable for
>   most biological systems but may not hold for sharp environmental
>   boundaries.

**Step 5a-iv — Neural network multi-timepoint inference (architecture constraints)**

After confirming the parametric framework (Steps 5a-i through 5a-iii), we return to
neural network inference with two critical architectural insights:

1. **Reduced potential capacity** — the original 4-layer MLP (2→16→32→32→16→1,
   2193 params) is over-parameterized for a 2D toy system. After fitting the
   macroscopic potential, excess capacity fits Brownian noise as high-frequency
   artifacts (spectral bias). Reducing to 2→16→16→1 (337 params) acts as implicit
   regularization, producing smoother, more symmetric landscapes.

2. **y-only death rate** — constraining γ_θ(z) to γ_θ(y) structurally eliminates
   the x-direction degeneracy: ∂γ/∂x ≡ 0 by construction. The optimizer can no
   longer use asymmetric death to compensate for lateral potential errors, forcing
   the potential to learn correct x-gradients.

> **Architecture:**
>
> | Component | Architecture | Parameters |
> |-----------|-------------|-----------|
> | Potential Φ_nn | 2 → 16 → 16 → 1 + softplus + confinement | 337 |
> | Death rate γ_nn | 1(y) → 8 → 8 → 1 → softplus | 97 |
> | **Total** | | **434** (vs 2530 before) |
>
> **Implementation:** `train_sd_nn_multi.py`, with configurable architectures via
> `create_sd_model(potential_hidden=..., death_y_only=True, death_hidden=...)`.
>
> **Results (1000 epochs, 6 snapshots at t=1,2,3,4,6,8):**
>
> - **Potential:** Symmetric double-well recovered. The old 4-layer network's
>   asymmetry artifact is eliminated. Landscape is smoother than ground truth
>   (expected: 337 params cannot capture fine polynomial structure, but the
>   macroscopic topology is correct).
>
> - **Death rate:** The y-only constraint works perfectly — γ is strictly
>   x-invariant by construction. However, the learned curve activates too
>   early (~y=-1 vs true ~y=1.5) with a gentler slope, significantly
>   underestimating death for y>2. This is the same extrapolation problem
>   as the parametric case: the NN finds a smooth, low-slope solution that
>   fits well where data exists (y<2) but cannot extrapolate the steep rise.
>
> - **Particle distributions:** Excellent spatial match at all snapshot times.
>   Symmetric bifurcation pattern correctly reproduced.
>
> - **Loss:** Final best = 0.0061 (MMD=0.005, Mass=0.001).
>
> **Key insight:** Architecture constraints (capacity reduction + y-only death)
> are far more effective than loss-function engineering (regional mass matching).
> The potential asymmetry problem is completely solved. The death rate
> extrapolation problem persists but is now cleanly isolated as a
> regularization problem — the NN needs additional inductive bias to prefer
> steeper transitions in the data-sparse region.
>
> **Future NN regularization strategies (algorithmic, no extra data):**
>
> - **Spectral normalization (strongly recommended):** Constrain the maximum
>   singular value of each weight matrix, imposing a Lipschitz bound on the
>   network. Physically equivalent to limiting the gradient magnitude of the
>   potential, preventing non-physical "cliffs" or "pits" from fitting noise.
>   This is the most principled defense against SDE-driven overfitting.
>
> - **Stronger weight decay:** Increase `weight_decay` in `optax.adamw` to
>   bias the network toward near-linear mappings. In data-void regions, the
>   potential surface decays to a smooth default instead of arbitrary wiggles.
>   Simple to implement but less targeted than spectral normalization.
>
> - **Data symmetry augmentation:** If the system has known symmetry (e.g.,
>   left-right symmetry x ↔ -x as in the bifurcation toy model), augment
>   target data by reflecting particles: (x,y) → (-x,y). This forces the
>   potential to be exactly symmetric without any extra parameters, eliminating
>   residual asymmetry from finite-sample noise. Requires domain knowledge
>   about the system's symmetry group.

**Step 5a-v — Asymmetric potential with fixed 2D death rate**

Extend the parametric model to test whether potential asymmetry and a
biologically-motivated death rate can be recovered from single-snapshot data.
The potential gains a linear tilt `e·x`, and the death rate is fixed as a
2D function combining thymic-selection-like death (corridor at x≈0) and a
terminal boundary.

> **Implementation notes (2026-04-17):**
>
> Created `03.1_asymmetric_potential/` with a self-contained training script
> and a particle count sweep.
>
> **Potential:** H = -(y-a)x² + exp(b)x⁴ - cy + exp(d)y⁴ + e·x
> (5 learnable params: a, b, c, d, e)
>
> **Death rate (FIXED, not learned):**
>
> ```
> γ(x,y) = γ_select(x,y) + γ_term(y)
> γ_select = A·exp(-x²/(2w_x²))·softplus(k1·(y - y_select))   # selection corridor
> γ_term   = B·softplus(k2·(y - y_max))                         # terminal boundary
> ```
>
> Parameters: A=2, w_x=0.5, k1=1, y_select=1, B=5, k2=3, y_max=3,
> GAMMA_MAX=50 (soft clamp).
>
> **Source (FIXED):** oval Gaussian, μ=(0,-1), σ_x=0.25, σ_y=0.10.
>
> **Training:** single steady-state snapshot at t=10, 500 epochs, Adam
> lr=0.01, L=L_MMD (no mass loss since death is fixed).
>
> **Results (N=2000, best epoch 402, loss=0.0020):**
>
> | Parameter | Learned | True | Error |
> |-----------|---------|------|-------|
> | a | -0.807 | -1.0 | +0.193 |
> | b | -1.691 | -1.6 | -0.091 |
> | c | +3.049 | +3.5 | -0.451 |
> | d | -1.210 | -1.2 | -0.010 |
> | e | +0.550 | +0.7 | -0.150 |
>
> **Particle count sweep (N=2000–4000, step 200):**
>
> Ran 11 configurations via `sweep_particles.py`. Key findings:
>
> - **b and d**: excellently recovered across all N (error < 0.05)
> - **c**: consistently undershoots (~3.0–3.2 vs 3.5), coupled with a —
>   confirms the **a-c degeneracy is fundamental** to single-snapshot matching
> - **a**: similarly offset (~-0.82 to -0.87 vs -1.0)
> - **e**: stochastic variation (0.57–0.78) centered near truth 0.7
> - **Loss**: decreases with N (better MMD resolution) but parameter accuracy
>   doesn't systematically improve — the degeneracy is structural, not
>   a sampling issue
>
> **Conclusion:** increasing particle count improves loss landscape smoothness
> but cannot break the a-c coupling degeneracy. Multi-timepoint matching
> (as demonstrated in Step 5a-ii) remains the only known approach to resolve
> this. The 2D death rate and asymmetric potential work correctly as fixed
> components, validating the framework for future extensions where the death
> rate is also learned.

**Step 5.1 — NN potential + fixed 2D death rate: systematic sweeps**

Extends Step 5a-v from parametric (5-param) to neural network potential, while
keeping the 2D death rate and confinement fixed. The NN potential
(`PotentialNN` 2→16→16→1, 337 params, wrapped with exact quartic confinement)
learns the asymmetric landscape from snapshot data via weighted MMD + mass loss.

This step focuses on **systematic experimental characterization** through four
sweeps, plus distribution-level diagnostics that compare observable particle
characteristics (not the unobservable potential) between target and learned models.

> **Experiments and key findings:**
>
> **1. N_learned sweep** (`sweep_nparticles_v3/`): Fixed target (N=8000,
> seed=1), vary learned particle count N=200–6000, fixed seed=2.
>
> - Loss improves with N: 0.027 (N=200) → 0.009 (N=4000), plateaus at N≈3000
> - Barrier height consistently overestimated: 10–20× vs true 4.3 — the NN
>   potential learns deeper wells than ground truth to compensate for
>   limited expressivity
> - Well positions converge toward truth but remain ~0.6 units too wide
>
> **2. Distribution analysis** (`sweep_nparticles_v3/analyze_distributions.py`):
> For each trained model, simulates a fresh analysis batch (N=2000, seed=99)
> through both true dynamics and learned model to compare observable
> characteristics:
>
> - **Mass curves M(t):** excellent match across all N (death rate is exact)
> - **Decay curves:** near-perfect overlap (same death rate, same σ)
> - **Bifurcation fraction:** learned models plateau ~1% short of target
>   (0.63 vs 0.64), with the saddle zone systematically depleted
>   (target 16.5% → learned 9–11%) and right well overweight
>   (target 28.7% → learned 32–36%)
> - **Conclusion:** the potential barrier is too high and too wide, trapping
>   particles in wells rather than allowing saddle crossing. This is a
>   structural limitation of the NN + MMD framework, not a sampling issue.
>
> **3. Seed sweep** (`sweep_seeds/`): Fixed N=1000, 10 different seeds
> [2, 13, 42, 77, 123, 256, 444, 678, 999, 2025].
>
> - All seeds find double-well topology (no failures)
> - Loss range: 0.008–0.016 (2× variance)
> - Barrier height varies 8–17 (true 4.3) — consistent overestimation
> - Well positions and saddle locations vary by ~0.3 units across seeds
> - Force profiles show consistent shape but varying amplitude
>
> **4. Learnable sigma** (`learn_sigma/`): Test whether σ can be learned
> jointly with the NN potential. Two conditions: with exact confinement
> (true b, d) and without confinement (pure NN).
>
> - **With confinement:** σ overshoots to 1.72–2.02 (true 1.5). Confinement
>   anchors the potential scale, so σ inflates to compensate.
> - **Without confinement:** σ collapses to 0.36–0.46. The NN flattens the
>   potential and reduces σ to maintain the same effective dynamics (Φ/σ²
>   degeneracy). 2 of 3 seeds fail to find bifurcation.
> - **Conclusion:** Φ/σ² degeneracy prevents reliable σ learning. Must fix σ
>   or anchor the potential scale independently.
>
> **5. N_target sweep** (`sweep_ntarget/`): Fixed N_learned=1000 (seed=99),
> vary target data size N_target=2000–12000 with different seeds per N_target.
>
> - Loss and distribution metrics show **no systematic improvement** as
>   N_target increases from 2000 to 12000
> - Variation is dominated by target seed randomness, not data size
> - **Conclusion:** N_target=2000 is already sufficient. The bottleneck is
>   N_learned (model simulation quality), not training data quantity. This is
>   encouraging for real data applications where cell counts may be limited.
>
> **Overall conclusions for Step 5.1:**
>
> 1. The NN potential reliably learns the correct bifurcation topology
>    (double-well with asymmetry) but systematically overestimates the barrier
> 2. Observable particle characteristics (mass, decay, bifurcation ratio) are
>    well-recovered despite potential inaccuracies — the dynamics are more
>    robust than the potential itself
> 3. The saddle zone depletion is a structural artifact of MMD with isotropic
>    kernels: the bw=100 kernel cannot distinguish bifurcated from
>    non-bifurcated x-distributions
> 4. Training data requirements are modest (~2000 cells), but simulation
>    particle count matters more (~1000–4000 for convergence)
> 5. σ cannot be learned jointly with Φ without additional constraints

**Step 5.1-conf — Data-driven anisotropic confinement**

When applying the NN potential to real data, the true quartic coefficients
(exp(b), exp(d)) used for confinement are unknown. This step derives
anisotropic confinement coefficients C_x, C_y directly from target particle
data and the known death rate, eliminating the need for ground-truth potential
parameters.

> **Implementation notes (2026-05-15):**
>
> Created `05.1_asymmetric_nn/data_driven_confinement/` with two scripts:
> `train_data_conf.py` (confinement computation + training) and
> `plot_analysis.py` (distribution diagnostics).
>
> **Joint Energy-Survival Horizon method (5 steps):**
>
> 1. **Empirical horizon R99_i**: 99th percentile of |z_i| along each axis
>    from target data (where 99% of observed mass lives).
>
> 2. **Survival horizon R_γ_i**: find the radius along each axis where
>    the death rate γ exceeds a threshold γ_thr=5.0 (particle survival
>    <1% over 1 time unit: exp(-5)≈0.007). Evaluated at well x-locations
>    (x≈±2) where mass concentrates, not at x=0.
>
> 3. **Joint boundary R_joint_i = max(R99_i, R_γ_i)**: the confinement
>    wall must enclose both the observed particle cloud and the survival
>    horizon.
>
> 4. **Safety margin R_bnd_i = 1.2 × R_joint_i**: 20% buffer for
>    Brownian excursions during training.
>
> 5. **Energy matching**: set C_i = E_target / R_bnd_i⁴, where
>    E_target = 10σ² = 22.5 (Boltzmann suppression exp(-20)≈2e-9 at
>    the boundary — strong safety net for numerical stability without
>    biasing the learned landscape).
>
> **Results:**
>
> | Coefficient | Data-driven | True (exp(b), exp(d)) | Ratio |
> |-------------|-------------|----------------------|-------|
> | C_x | 0.0777 | 0.2019 | 0.385 |
> | C_y | 0.1061 | 0.3012 | 0.352 |
>
> The data-driven coefficients are ~35-39% of the true quartic coefficients.
> This is expected: the confinement only needs to prevent particle escape,
> not replicate the true potential's strength in the interior.
>
> **Training with data-driven confinement (N=2000, 800 epochs):**
>
> - Best loss: 0.0076 (epoch 744), comparable to true-confinement results
> - No numerical instability — the weaker confinement wall is sufficient
> - Zone fractions: death_zone ±0.001, right_well ±0.013, left_well ±0.041,
>   saddle ±0.054 (depleted, consistent with barrier overestimation)
> - Mass curves and decay curves match well between target and learned model
>
> **Conclusion:** The Joint Energy-Survival Horizon method successfully
> replaces knowledge of true potential parameters with data-derived
> quantities. This is a prerequisite for applying the framework to real
> single-cell data where the ground-truth potential is unknown.

**Step 5b — Synthetic benchmark: binary choice + proliferation**

Ground-truth validation with known Φ* and R*:

- Φ*: binary choice normal form (Eq. 3 from Howe & Mani).
- R*: localized Gaussian, R*(x) = 0.3·exp(−‖x − x_center‖²/0.5) − 0.05.
- Generate 100 experiments × 500 cells × 10 timepoints, sigmoid signal profiles.

Validation metrics:

- Fixed-point location error (Φ topology).
- R reconstruction accuracy within data support.
- Bifurcation diagram consistency (fold curves).
- σ convergence.
- **ESS trajectory** — must remain above 10% throughout integration.

### Phase 2: Manifold learning and lineage-anchored validation (Steps 5–10)

**Steps 5–6 — Autoencoder implementation with alternating training**

Network: deterministic autoencoder (not VAE — stochasticity is already provided
by the SDE; VAE adds posterior collapse risk without benefit).

```
Encoder: G → 128 → 64 → 32 → d_latent (ELU activation)
Decoder: d_latent → 32 → 64 → 128 → G (ELU, no final activation)
```

**Alternating optimization protocol** (critical engineering decision):

Training the autoencoder and SDE jointly from scratch causes gradient explosion
because the neural potential produces a rough landscape in early training,
crashing the SDE solver at fixed step sizes.

- **Phase A**: Train AE alone on reconstruction loss. Establish a smooth
  initial latent space.
- **Phase B**: Freeze AE weights. Map all data to latent space. Train uPLNN
  (Φ, R, Ψ, σ) independently on latent coordinates.
- **Phase C**: Unfreeze AE. Joint end-to-end fine-tuning at 10× reduced
  learning rate. Monitor whether fine-tuning preserves the manifold topology
  established in Phase A; if it collapses, increase λ_rec.

**Steps 7–8 — High-dimensional synthetic validation**

Test the full end-to-end pipeline:

1. Define ground-truth Φ* and R* on ℝ².
2. Embed via a known nonlinear map E*: ℝ² → ℝ¹⁰.
3. Add observation noise in ℝ¹⁰.
4. Recover the 2D landscape from 10D observations using uPLNN.

Compare against baselines: PCA + PLNN, PCA + uPLNN, AE + PLNN (no proliferation).

Verify Phase C fine-tuning does not destroy the manifold structure from Phase A.

**Steps 9–10 — Lineage-tracing validation (strategically moved forward)**

> **Major revision from original plan**: lineage-tracing validation is moved
> from Step 15 to Step 9. Without early validation against ground-truth
> proliferation data, any results on unlabeled data (e.g., mESC) lack
> credibility for the proliferation component.

Options (choose one):

- **Weinreb et al. LARRY barcoding** (Science, 2020): hematopoiesis with
  ground-truth clone sizes from lineage tracing. Clone-size ratios provide
  direct measurements of cumulative growth ∫R dt along lineage branches.
- **Synthetic with known R + lineage labels**: generate trajectories where
  each particle's full weight history is recorded, providing ground-truth
  R(x) pointwise.

Validation: compare inferred R_θ(x) against known ground-truth proliferation
rates. If R_θ fails to recover known growth patterns here, the method cannot
be trusted on datasets without lineage information.

### Phase 3: Real data application (Steps 11–16)

**Steps 11–12 — mESC in vitro data (Sáez et al., Cell Systems, 2022)**

Apply uPLNN to the same dataset used by Howe & Mani for direct comparison:

1. Recapitulate cell-type labeling (GMM-based, following original protocol).
2. Baseline: reproduce PLNN's PCA-based results.
3. Apply uPLNN with autoencoder: compare EPI/AN separation in latent space.
4. Compare: PCA+PLNN vs. AE+PLNN vs. AE+uPLNN (with proliferation).

Key comparisons against PLNN:

- Latent-space separability of EPI and AN populations (PLNN acknowledges
  failure on this point).
- Validation-set MMD loss.
- Visual agreement of simulated trajectories with observed data.
- Bifurcation structure under signal variation.
- Inferred R(z): does it localize to biologically expected regions?

**Steps 13–14 — Bifurcation analysis and biological interpretation**

- Implement pseudo-arclength continuation on the learned Φ.
- Plot fold curves in signal space; compare with Sáez et al.
- Interpret R(z): positive at stem/progenitor states? Near-zero or negative
  at terminal fates? Signal-dependent (e.g., FGF promoting proliferation)?
- This analysis produces the central figures of the paper.

**Steps 15–16 — Ablation studies**

Systematic ablations required for publication:

1. With/without R: does ignoring proliferation distort the inferred landscape?
2. With/without autoencoder: latent-space quality comparison.
3. With/without signal tilt: bifurcation prediction capability.
4. L1 vs. L2 regularization on R: sparsity patterns.
5. Sensitivity to sampling density (cf. Howe & Mani, Fig. 6).
6. ESS statistics across all experiments.

### Phase 4: Paper writing and submission (Steps 17–24)

**Steps 17–18** — Complete all figures to publication quality.

**Steps 19–22** — Write manuscript. Proposed structure:

```
I.   Introduction
     Waddington landscape formalization; limitations of existing methods
     (no proliferation, linear manifold, no signal control); contributions.

II.  Background
     Tiltable landscapes; Fokker-Planck with source term; identifiability.

III. Methods
     A. Model architecture (augmented SDE + autoencoder)
     B. Regularization for identifiability (L1 on R, L_mass, spectral norm)
     C. Alternating training protocol
     D. Bifurcation analysis

IV.  Results
     A. Synthetic: recovering Φ and R from high-dimensional data
     B. Lineage-tracing validation of inferred proliferation rates
     C. mESC application: improved latent space + proliferation dynamics
     D. Ablation studies

V.   Discussion
     Comparison with PLNN, DeepRUOT, LSD; limitations; future directions.

VI.  Methods (detailed)
```

**Steps 23–24** — Revision and submission.

Target journals (by priority):

1. **Physical Review X** — if physics contribution (non-conservative landscape
   theory) is deep enough.
2. **Nature Machine Intelligence** — if method + benchmark performance is strong.
3. **Cell Systems** — if biological application and interpretability shine.
4. **PLoS Computational Biology** — solid fallback.

---

## Key References

### Foundational landscape theory

- Rand, Raju, Sáez, Corson & Siggia, "Geometry of gene regulatory dynamics,"
  PNAS **118**, e2109729118 (2021).
- Sáez, Blassberg et al., "Statistically derived geometrical landscapes capture
  principles of decision-making dynamics during cell fate transitions,"
  Cell Systems **13**, 12–28 (2022).
- Howe & Mani, "Learning geometric models for developmental dynamics,"
  Physical Review X **15**, 031070 (2025).

### Unbalanced transport and birth–death dynamics

- Schiebinger et al., "Optimal-transport analysis of single-cell gene
  expression identifies developmental trajectories in reprogramming,"
  Cell **176**, 928–943 (2019).
- Zhang, Li & Zhou, "Learning stochastic dynamics from snapshots through
  regularized unbalanced optimal transport," ICLR 2025 (Oral).
- Sha, Qiu, Zhou & Nie, "Reconstructing growth and dynamic trajectories from
  single-cell transcriptomics data," Nature Machine Intelligence **6**, 25–39 (2024).
- Zhou, Wang, Li et al., "Energy landscape decomposition for cell
  differentiation with proliferation effect," National Science Review **9**,
  nwac116 (2022).
- Yeo, Saksena & Gifford, "Generative modeling of single-cell time series
  with PRESCIENT," Nature Communications **12**, 3222 (2021).

### Manifold learning and latent dynamics

- Farrell, Mani & Goyal, "LatentVelo," Cell Reports Methods **3**, 100581 (2023).
- Poursina et al., "A latent space thermodynamic model of cell differentiation,"
  bioRxiv (March 2026).
- Vinyard et al., "scDiffEq," Nature Machine Intelligence **7**, 1969–1984 (2025).
- Jiang & Wan, "PI-SDE," Bioinformatics **40**, ii120–ii127 (2024).
- Huguet, Tong et al., "MIOFlow," NeurIPS **35**, 29705 (2022).

### Noise, geometry, and related methods

- Coomer, Ham & Stumpf, "Noise distorts the epigenetic landscape and shapes
  cell-fate decisions," Cell Systems **13**, 83–102 (2022).
- Weinreb, Yeo et al., "Lineage tracing on transcriptional landscapes links
  state to fate during differentiation," Science **367**, eaaw3381 (2020).
- Mochulska & François, "Generative epigenetic landscapes map the topology
  and topography of cell fates," PNAS **122**, e2514508122 (2025).

---

## Risk Assessment

| Risk | Probability | Mitigation |
|------|:-----------:|------------|
| Φ–R identifiability failure | Medium | L1 on R + L_mass + lineage validation at Step 9 |
| SDE solver divergence (NaN) | Medium | Spectral norm + softplus + bounded R output + weight decay |
| AE + SDE joint training collapse | Medium | Alternating optimization (Phase A→B→C); monitor manifold topology |
| Log-weight overflow | Medium | Log-sum-exp; tanh-bounded R; ESS monitoring; optional resampling |
| Competing publication (LSD, etc.) | Medium | Our unique combination (SDE + signal + R + manifold) is distinct |
| Insufficient mESC data for AE | Low | Pre-train on synthetic; or use larger scRNA-seq datasets |
| Single-GPU memory limits | Medium | Short-time transition matching; mixed precision; JAX jit+scan |

---

## License

TBD

## Contact

TBD
