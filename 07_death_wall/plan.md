# Model 07 Experiment Plan — Improving Death-Wall Learning

## Problem

Without y⁴ confinement, particles take a one-way trip (source → death wall) instead of bouncing around wells. Each particle provides far less information about the potential than in the confined model. Result: even with exact x⁴ confinement provided, (16,16) NN + σ=1.0 + N=3000/6000 + 2000 epochs gives poor results.

**Root cause:** transient dynamics (one-way transit) vs. ergodic dynamics (recirculating in wells). Each particle "samples" the potential along one narrow path then dies, instead of exploring the full landscape.

## Implementation Note

PotentialNN's built-in `c_conf` adds `c_conf * (x² + y²)²`, which includes unwanted y⁴ and x²y² cross terms. All experiments below must use a **custom x⁴-only wrapper**:

```python
class PotentialWithX4(eqx.Module):
    nn: PotentialNN
    c_x4: float = eqx.field(static=True)  # exp(b) = exp(-1.6) ≈ 0.2019

    def __call__(self, z):
        return self.nn(z) + self.c_x4 * z[0]**4
```

Use `PotentialNN(c_conf=0.0)` inside the wrapper so the NN learns only the residual.

## Baseline (known bad)

| Setting | Value |
|---------|-------|
| NN | (16,16), 337 params |
| Confinement | exact x⁴ (exp(b) ≈ 0.2019) |
| σ | 1.0 |
| N_learned / N_target | 3000 / 6000 |
| Epochs | 2000 |
| Patience | 800 |
| LR | 3e-3 |
| MMD bandwidths | (0.005, 0.05, 0.5, 5.0) |
| LAM_MASS | 5.0 |

## Step 1 — Increase σ to 1.5

**Rationale:** Most direct fix. Higher noise pushes more particles to large |x| during transit, giving the MMD loss signal about the x-confinement walls. Currently particles start at σ_x=0.05 and mostly stay near x=0.

| Change | Value |
|--------|-------|
| SIGMA | 1.5 (both ground truth and learned) |
| Everything else | Same as baseline |

**What to look for:** Better 1D slice fits at y=0.5 and y=1.5. Force field arrows at |x|>2 should be stronger. Loss should drop below baseline.

**Risk:** Slightly noisier MMD gradients, but σ=1.5 is still moderate.

## Step 2 — Increase σ to 2.0

**Only if** Step 1 shows improvement but results are still insufficient.

| Change | Value |
|--------|-------|
| SIGMA | 2.0 |
| Everything else | Same as Step 1 |

**Risk:** If too high, diffusion dominates and the distribution becomes near-Gaussian regardless of Φ. Compare the target distribution shape — if it looks round/featureless, σ is too high.

## Step 3 — Increase particle count

**Only if** σ tuning alone isn't enough.

| Change | Value |
|--------|-------|
| N_target | 12000 |
| N_learned | 6000 |
| σ | Best from Steps 1–2 |
| Everything else | Same |

**Rationale:** More particles = better tail coverage. The x⁴ walls are probed by outlier particles at large |x|; doubling N gives ~40% more outliers beyond 2σ.

**Cost:** ~4× MMD compute per epoch (N² kernel). Expect ~4× longer training.

## Step 4 — Increase epochs to 4000

**Only if** loss is still decreasing at epoch 2000 (check training curve).

| Change | Value |
|--------|-------|
| N_EPOCHS | 4000 |
| PATIENCE | 1500 |
| Everything else | Best from above |

**Skip if:** Loss has clearly plateaued before epoch 1500 — more epochs won't break through a signal-limited plateau.

## Step 5 — NN capacity (16,32,32,16)

**Only if** Steps 1–4 show the NN is underfitting (training loss still high, not just noisy).

| Change | Value |
|--------|-------|
| hidden_sizes | (16, 32, 32, 16) |
| Everything else | Best from above |

**Rationale:** With x⁴ provided, the NN only needs to learn `-(y-a)x² - cy + ex` — smooth, low-degree functions. (16,16) should be enough. Only try this if the residual error pattern suggests underfitting (systematic bias in slices, not random noise).

## Step 6 — MMD bandwidth tuning

**Fine-tuning step.** Only after σ and N are settled.

| Change | Value |
|--------|-------|
| BANDWIDTHS | (0.001, 0.01, 0.1, 1.0, 5.0) |
| Everything else | Best from above |

**Rationale:** In the death-wall model, particles spread over a wider channel than the tight wells of the confined model. Shifting bandwidths toward smaller scales may help resolve finer spatial structure in the transit channel.

## Execution

Run each step on a GPU machine. Compare results via:
1. Training loss curve (total, MMD, mass)
2. Landscape 1D slices at y = -0.5, 0.5, 1.5
3. Force field comparison (true vs learned)
4. Snapshot distributions at t=0.5, 1.0, 2.0, 3.0

Only proceed to the next step if the current one is insufficient.
