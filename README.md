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

```
LearningLandscape/
├── README.md
├── train.py                        # [DONE] Main entry point: multi-timepoint training
├── toymodel_proliferation.py       # [DONE] Augmented SDE with log-weight (analytic)
├── toymodel_jax.ipynb              # [DONE] Basic SDE landscape + MMD inference
├── models/
│   ├── __init__.py                 # [DONE] WaddingtonModel + create_model() factory
│   ├── potential.py                # [DONE] Φ_nn: MLP (2→16→32→32→16→1) + softplus + confinement
│   ├── tilt.py                     # [DONE] Ψ: Linear(d_signal → d_latent)
│   ├── proliferation.py            # [DONE] R_θ: MLP (2→16→16→1), unconstrained output
│   ├── noise.py                    # [DONE] σ: log-space scalar or state-dependent MLP
│   └── autoencoder.py              # [PLANNED] Deterministic encoder-decoder
├── simulator/
│   ├── __init__.py                 # [DONE]
│   ├── sde_solver.py               # [DONE] Euler-Maruyama with logsumexp + transition sim
│   ├── weighted_mmd.py             # [DONE] Weighted MMD + L_mass matching + L1 sparsity on R
│   └── resampling.py               # [PLANNED] Systematic resampling for weight degeneracy
├── training/
│   ├── __init__.py                 # [DONE]
│   ├── trainer.py                  # [DONE] Multi-timepoint transition matching + early stopping
│   └── data_loader.py              # [DONE] Multi-timepoint + single-timepoint generators
├── analysis/
│   ├── __init__.py                 # [DONE]
│   ├── bifurcation.py              # [DONE] Equilibrium finding + Hessian classification
│   ├── visualization.py            # [DONE] Landscape, particles, training curves, comparison
│   └── ess_monitor.py              # [PLANNED] Effective sample size tracking
├── experiments/
│   ├── synthetic/                   # [PLANNED] Ground-truth benchmark experiments
│   └── mesc/                        # [PLANNED] Mouse ESC in vitro application
└── notebooks/                       # [PLANNED] Analysis notebooks
```

### Current codebase status

| Module | Status | Description |
|--------|--------|-------------|
| Parametric potential + SDE + MMD | ✅ Done | `toymodel_jax.ipynb` |
| Augmented SDE with log-weights + weighted MMD | ✅ Done | `toymodel_proliferation.py` |
| Neural network potential (Equinox) | ✅ Done | `models/potential.py` — MLP (2→16→32→32→16→1) + softplus + confinement |
| Learnable R_θ | ✅ Done | `models/proliferation.py` — MLP (2→16→16→1), unconstrained output |
| Learnable noise σ | ✅ Done | `models/noise.py` — log-space scalar or state-dependent MLP |
| Tilt Ψ(s) module | ✅ Done | `models/tilt.py` — Linear(d_signal → d_latent), not yet tested with signal data |
| Multi-timepoint training | ✅ Done | `training/trainer.py` — short-time transition matching over consecutive pairs |
| L1 sparsity on R + mass matching loss + logsumexp | ✅ Done | `simulator/weighted_mmd.py`, `simulator/sde_solver.py` |
| Target data resampling by weights | ✅ Done | `training/data_loader.py` — multinomial resampling at snapshots |
| Bifurcation analysis | ✅ Done | `analysis/bifurcation.py` — equilibrium finding + Hessian classification |
| Signal-dependent training | 🔲 Planned | Tilt module exists but training loop not yet signal-aware |
| Autoencoder (encoder-decoder) | 🔲 Planned | Deterministic AE for high-dimensional gene expression |
| Spectral normalization on Φ_nn | 🔲 Planned | Currently using plain Xavier init without spectral norm |
| tanh-bounded R_θ output | 🔲 Planned | Currently unconstrained; needs clamping to [β_min, β_max] |
| ESS monitoring | 🔲 Planned | Not yet implemented |

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

**Step 5 — Synthetic benchmark: binary choice + proliferation**

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
