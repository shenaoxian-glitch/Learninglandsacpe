# CLAUDE.md — Project Context for Claude Code

## Project overview

uPLNN (unified Proliferation Landscape Neural Network): learning Waddington
potential landscapes from single-cell snapshot data with birth/death dynamics.

**Stack:** JAX + Equinox + Optax, Python. No PyTorch.

**Core physics:** SDE `dz = -∇Φ dt + σ dW` with log-weight `dS = (R - γ) dt`.
Particles flow down potential Φ, proliferate (R), and die (γ).

## Project structure

Organized by model type (each subfolder is self-contained with all needed .py files):

| Folder | Model type | Key script(s) |
|--------|-----------|---------------|
| `01_potential_only/` | Early standalone prototypes (torch/JAX) | `toymodel.py`, `toymodel_jax.py`, `toymodel_proliferation.py` |
| `02_proliferation_nn/` | Closed-system NN (Φ + R) | `train.py` |
| `03_source_death_parametric/` | Open-system, 6 scalar params | `train_sd_parametric.py`, `train_sd_parametric_multi.py` |
| `04_source_death_semi_nn/` | NN Φ + parametric death rate | `train_sd.py` |
| `05_source_death_nn/` | NN Φ + NN γ(y), multi-timepoint | `train_sd_nn_multi.py` |

Shared modules (copied into each subfolder that needs them):

| Module | Purpose |
|--------|---------|
| `models/potential.py` | Φ_nn: configurable hidden sizes + softplus + confinement |
| `models/death_rate.py` | DeathRateNN (y_only mode) + DeathRateParametric |
| `models/__init__.py` | WaddingtonModel, SourceDeathModel, factories |
| `training/trainer.py` | Training loops: `train()`, `train_sd()` |
| `training/data_loader.py` | Data generation, particle queues |
| `simulator/sde_solver.py` | Euler-Maruyama, open-system simulation + full trajectory |
| `simulator/weighted_mmd.py` | MMD loss, mass loss, sparsity loss |

## Ground-truth parameters

```python
TARGET_PARAMS = {'a': 0, 'b': -1.6, 'c': 3.5, 'd': -1.2, 'sigma': 1.5}
# H = -(y-a)x² + exp(b)x⁴ - cy + exp(d)y⁴
DEATH_PARAMS = {'y_threshold': 2.2}  # γ = softplus(k·(y - y_c)), k=1
```

## Current progress

Completed through Step 5a-iv (NN multi-timepoint inference).
See README.md "Research Roadmap" for full history and next steps.

Key findings:
- Single-snapshot matching has a-b-c parameter coupling degeneracy
- Multi-timepoint transient matching breaks this (7-36x error reduction)
- Death rate y_c/k trade-off is intrinsic to softplus form
- Regional mass matching cannot break y_c/k: data sparsity in y>2 death zone
- Architecture constraints >> loss-function engineering:
  - Reduced Φ capacity (2→16→16→1, 337 params) eliminates asymmetry artifacts
  - y-only death rate (γ(y) not γ(x,y)) eliminates x-direction degeneracy
  - Death rate extrapolation (slope too shallow in y>2) remains unsolved
- Next: spectral normalization, weight decay, data symmetry augmentation

## Conventions

- New scripts should not modify existing files
- Always preserve backward compatibility with prior steps
- Training scripts save figures to project root as `*_{suffix}.png`
- Models saved as `trained_model_{suffix}.eqx`

## Running scripts

```bash
python train_sd_parametric_multi.py > log.txt 2>&1  # redirect output
```
