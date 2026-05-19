#!/usr/bin/env python
"""
analyze_distributions.py — Distribution analysis for Model 6.1

Loads the trained model and compares target vs learned distributions:
  - Mass curves M(t)
  - x/y marginals
  - Bifurcation fractions
  - Per-particle weight decay curves
  - Sample trajectories
  - Zone weight fractions

Usage:
    python analyze_distributions.py > log_analysis.txt 2>&1
"""
import matplotlib
matplotlib.use('Agg')

import os
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from models import SourceDeathModel
from models.potential import PotentialNN
from training.data_loader import (
    build_particle_queue, analytical_death_rate_2d, analytical_potential,
)
from simulator.sde_solver import simulate_open_system_full


# ==========================================================================
# Constants (must match training script)
# ==========================================================================

TARGET_PARAMS = {
    'a': -1.0, 'b': -1.6, 'c': 3.5, 'd': -1.2,
    'e': 0.7, 'sigma': 1.5,
}
DEATH_PARAMS_2D = {
    'A': 2.0, 'w_x': 0.5, 'k1': 1.0, 'y_select': 1.0,
    'B': 5.0, 'k2': 3.0, 'y_max': 3.0,
}

N_LEARNED = 3000
N_TARGET = 8000
DT = 0.01
T_FINAL = 10.0
N_STEPS = int(T_FINAL / DT)
SIGMA = 1.5

TRUE_MU_X = 0.0
TRUE_MU_Y = -1.0
TRUE_SIGMA_X = 0.25
TRUE_SIGMA_Y = 0.10

# Analysis
N_ANALYSIS = 3000
SEED_ANALYSIS = 777
N_TRAJ_PLOT = 30

ZONES = {
    'left_well':   lambda x, y: x < -1.0,
    'saddle':      lambda x, y: (x >= -1.0) & (x <= 1.0),
    'right_well':  lambda x, y: x > 1.0,
    'source':      lambda x, y: y < 0.0,
    'death_zone':  lambda x, y: y > 2.2,
}

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "init_mux_0.0")
os.makedirs(SCRIPT_DIR, exist_ok=True)
SUFFIX = "mux_N3000_Ntgt8000_t10_ep2000"


# ==========================================================================
# Model components (must match training)
# ==========================================================================

class FixedDeathRate2D(eqx.Module):
    A: float = eqx.field(static=True)
    w_x: float = eqx.field(static=True)
    k1: float = eqx.field(static=True)
    y_select: float = eqx.field(static=True)
    B: float = eqx.field(static=True)
    k2: float = eqx.field(static=True)
    y_max: float = eqx.field(static=True)

    def __call__(self, z):
        x, y = z[0], z[1]
        gamma_select = (
            self.A * jnp.exp(-x**2 / (2 * self.w_x**2))
            * jax.nn.softplus(self.k1 * (y - self.y_select))
        )
        gamma_term = self.B * jax.nn.softplus(self.k2 * (y - self.y_max))
        return gamma_select + gamma_term


class PotentialWithConfinement(eqx.Module):
    nn: PotentialNN
    b: float = eqx.field(static=True)
    d: float = eqx.field(static=True)

    def __call__(self, z):
        x, y = z[0], z[1]
        conf = jnp.exp(self.b) * x**4 + jnp.exp(self.d) * y**4
        return self.nn(z) + conf


class SourceDeathModelLearnMuX(eqx.Module):
    potential: eqx.Module
    death_rate: eqx.Module
    mu_x: jax.Array
    mu_y: float = eqx.field(static=True)
    sigma_x: float = eqx.field(static=True)
    sigma_y: float = eqx.field(static=True)


def make_z_init(model, eps):
    x = model.mu_x + eps[:, 0] * model.sigma_x
    y = model.mu_y + eps[:, 1] * model.sigma_y
    return jnp.column_stack([x, y])


# ==========================================================================
# Ground-truth simulation
# ==========================================================================

_force_fn = jax.vmap(
    jax.grad(analytical_potential, argnums=(0, 1)),
    in_axes=(0, 0, None),
)


def simulate_ground_truth_full(z_init, wake_schedule, n_steps, dt, key):
    N = z_init.shape[0]
    S0 = jnp.full(N, -100.0)
    step_keys = jax.random.split(key, n_steps)

    def scan_body(carry, inputs):
        px, py, S = carry
        step_key, step_idx = inputs
        is_waking = (wake_schedule == step_idx)
        S = jnp.where(is_waking, 0.0, S)
        px = jnp.where(is_waking, z_init[:, 0], px)
        py = jnp.where(is_waking, z_init[:, 1], py)
        gx, gy = _force_fn(px, py, TARGET_PARAMS)
        fx, fy = -gx, -gy
        kx, ky = jax.random.split(step_key)
        nx = jax.random.normal(kx, px.shape) * jnp.sqrt(dt)
        ny = jax.random.normal(ky, py.shape) * jnp.sqrt(dt)
        new_px = px + fx * dt + SIGMA * nx
        new_py = py + fy * dt + SIGMA * ny
        gamma = analytical_death_rate_2d(px, py, DEATH_PARAMS_2D)
        new_S = S - gamma * dt
        return (new_px, new_py, new_S), (new_px, new_py, new_S)

    step_indices = jnp.arange(n_steps)
    _, (all_px, all_py, all_S) = jax.lax.scan(
        scan_body, (z_init[:, 0], z_init[:, 1], S0),
        (step_keys, step_indices)
    )
    return all_px, all_py, all_S


# ==========================================================================
# Helper functions
# ==========================================================================

def compute_mass_curve(all_S):
    return np.array(jnp.sum(jnp.exp(all_S), axis=1))


def compute_bifurcation_fraction(z_final, S_final, x_threshold=0.0):
    weights = jnp.exp(S_final)
    total = jnp.sum(weights) + 1e-12
    frac_left = float(jnp.sum(weights * (z_final[:, 0] < x_threshold)) / total)
    return frac_left, 1.0 - frac_left


def compute_weighted_marginal(z_final, S_final, dim, bins, range_lim):
    weights = np.array(jnp.exp(S_final))
    values = np.array(z_final[:, dim])
    hist, bin_edges = np.histogram(values, bins=bins, range=range_lim,
                                    weights=weights, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return hist, bin_centers


def compute_zone_fractions(z_final, S_final, zones):
    x = np.array(z_final[:, 0])
    y = np.array(z_final[:, 1])
    weights = np.array(jnp.exp(S_final))
    total = weights.sum() + 1e-12
    fracs = {}
    for name, mask_fn in zones.items():
        mask = mask_fn(x, y)
        fracs[name] = float(weights[mask].sum() / total)
    return fracs


def compute_decay_curves(all_S, wake_schedule, dt):
    all_S = np.array(all_S)
    wake = np.array(wake_schedule)
    n_steps, N = all_S.shape
    max_life = n_steps
    aligned = np.full((N, max_life), np.nan)
    for i in range(N):
        birth = int(wake[i])
        if birth >= n_steps:
            continue
        life_len = n_steps - birth
        aligned[i, :life_len] = np.exp(all_S[birth:, i])
    tau = np.arange(max_life) * dt
    with np.errstate(all='ignore'):
        mean_curve = np.nanmean(aligned, axis=0)
        median_curve = np.nanmedian(aligned, axis=0)
        lo = np.nanpercentile(aligned, 10, axis=0)
        hi = np.nanpercentile(aligned, 90, axis=0)
    valid_count = np.sum(~np.isnan(aligned), axis=0)
    cutoff = np.searchsorted(-valid_count, -int(0.1 * N))
    if cutoff < 10:
        cutoff = max_life
    return tau[:cutoff], mean_curve[:cutoff], lo[:cutoff], hi[:cutoff], \
        median_curve[:cutoff]


# ==========================================================================
# Load trained model
# ==========================================================================

print("=" * 65)
print("Model 6.1 — Distribution analysis")
print("=" * 65)

model_path = os.path.join(SCRIPT_DIR, f"trained_model_{SUFFIX}.eqx")
print(f"Loading model from {model_path} ...")

# Create skeleton
key = jax.random.PRNGKey(4)
key, model_key = jax.random.split(key)
# Must use same key split sequence as training to get same skeleton structure
_k1, _k2, _k3 = jax.random.split(jax.random.PRNGKey(4), 3)
_, model_key_skel = jax.random.split(_k2)

phi_nn = PotentialNN(model_key_skel, d_latent=2, c_conf=0.0,
                     hidden_sizes=(16, 16))
potential = PotentialWithConfinement(
    nn=phi_nn, b=TARGET_PARAMS['b'], d=TARGET_PARAMS['d'],
)
fixed_death = FixedDeathRate2D(**DEATH_PARAMS_2D)
skeleton = SourceDeathModelLearnMuX(
    potential=potential,
    death_rate=fixed_death,
    mu_x=jnp.array(0.5),
    mu_y=TRUE_MU_Y,
    sigma_x=TRUE_SIGMA_X,
    sigma_y=TRUE_SIGMA_Y,
)
model = eqx.tree_deserialise_leaves(model_path, skeleton)
print(f"  mu_x = {float(model.mu_x):.4f} (true = {TRUE_MU_X})")


# ==========================================================================
# Generate analysis batches
# ==========================================================================

print(f"\nSimulating analysis batch (seed={SEED_ANALYSIS}, N={N_ANALYSIS}) ...")

ana_master = jax.random.PRNGKey(SEED_ANALYSIS)
ana_queue_key, ana_sim_key = jax.random.split(ana_master)

# Target analysis batch (true source position)
ana_sources_tgt = [
    {'mu': jnp.array([TRUE_MU_X, TRUE_MU_Y]),
     'sigma_x': TRUE_SIGMA_X, 'sigma_y': TRUE_SIGMA_Y,
     'n_particles': N_ANALYSIS},
]
ana_z_init_tgt, ana_wake_tgt = build_particle_queue(
    sources=ana_sources_tgt, n_particles=N_ANALYSIS,
    t_final=T_FINAL, dt=DT, key=ana_queue_key,
)

# Simulate target through true dynamics
print("  Simulating target ...")
ana_tgt_px, ana_tgt_py, ana_tgt_S = simulate_ground_truth_full(
    ana_z_init_tgt, ana_wake_tgt, N_STEPS, DT, ana_sim_key
)
ana_tgt_z = jnp.stack([ana_tgt_px, ana_tgt_py], axis=-1)

# Learned analysis batch (learned source position via reparameterization)
ana_eps_key = jax.random.PRNGKey(SEED_ANALYSIS + 100)
ana_eps = jax.random.normal(ana_eps_key, (N_ANALYSIS, 2))

# Wake schedule for learned (same structure)
ana_birth_times = jnp.linspace(0, T_FINAL, N_ANALYSIS, endpoint=False)
ana_wake_lrn = (ana_birth_times / DT).astype(int)
ana_wake_lrn = jnp.clip(ana_wake_lrn, 0, N_STEPS - 1)

# Build z_init from learned mu_x
ana_z_init_lrn = make_z_init(model, ana_eps)

# Simulate learned
print("  Simulating learned ...")
ana_lrn_z, ana_lrn_S = simulate_open_system_full(
    model, ana_z_init_lrn, ana_wake_lrn, SIGMA, N_STEPS, DT, ana_sim_key
)

final_step = N_STEPS - 1
time_axis = np.arange(1, N_STEPS + 1) * DT

# Also simulate with TRUE source position through learned potential
# (to separate source-position error from potential error)
ana_z_init_true_src = jnp.column_stack([
    TRUE_MU_X + ana_eps[:, 0] * TRUE_SIGMA_X,
    TRUE_MU_Y + ana_eps[:, 1] * TRUE_SIGMA_Y,
])
print("  Simulating learned potential + true source ...")
ana_truesrc_z, ana_truesrc_S = simulate_open_system_full(
    model, ana_z_init_true_src, ana_wake_lrn, SIGMA, N_STEPS, DT, ana_sim_key
)


# ==========================================================================
# Compute stats
# ==========================================================================

# Target
tgt_z_final = ana_tgt_z[final_step]
tgt_S_final = ana_tgt_S[final_step]
tgt_mass_curve = compute_mass_curve(ana_tgt_S)
tgt_frac_left, tgt_frac_right = compute_bifurcation_fraction(tgt_z_final, tgt_S_final)
tgt_zone_fracs = compute_zone_fractions(tgt_z_final, tgt_S_final, ZONES)
tgt_decay_tau, tgt_decay_mean, tgt_decay_lo, tgt_decay_hi, _ = \
    compute_decay_curves(ana_tgt_S, ana_wake_tgt, DT)
x_hist_tgt, x_bins = compute_weighted_marginal(
    np.array(tgt_z_final), np.array(tgt_S_final), dim=0, bins=60, range_lim=(-4, 4))
y_hist_tgt, y_bins = compute_weighted_marginal(
    np.array(tgt_z_final), np.array(tgt_S_final), dim=1, bins=60, range_lim=(-2, 4))

# Learned (with learned mu_x)
lrn_z_final = ana_lrn_z[final_step]
lrn_S_final = ana_lrn_S[final_step]
lrn_mass_curve = compute_mass_curve(ana_lrn_S)
lrn_frac_left, lrn_frac_right = compute_bifurcation_fraction(lrn_z_final, lrn_S_final)
lrn_zone_fracs = compute_zone_fractions(lrn_z_final, lrn_S_final, ZONES)
lrn_decay_tau, lrn_decay_mean, lrn_decay_lo, lrn_decay_hi, _ = \
    compute_decay_curves(ana_lrn_S, ana_wake_lrn, DT)
x_hist_lrn, _ = compute_weighted_marginal(
    np.array(lrn_z_final), np.array(lrn_S_final), dim=0, bins=60, range_lim=(-4, 4))
y_hist_lrn, _ = compute_weighted_marginal(
    np.array(lrn_z_final), np.array(lrn_S_final), dim=1, bins=60, range_lim=(-2, 4))

# Learned potential + true source (counterfactual)
ts_z_final = ana_truesrc_z[final_step]
ts_S_final = ana_truesrc_S[final_step]
ts_mass_curve = compute_mass_curve(ana_truesrc_S)
ts_frac_left, ts_frac_right = compute_bifurcation_fraction(ts_z_final, ts_S_final)
x_hist_ts, _ = compute_weighted_marginal(
    np.array(ts_z_final), np.array(ts_S_final), dim=0, bins=60, range_lim=(-4, 4))
y_hist_ts, _ = compute_weighted_marginal(
    np.array(ts_z_final), np.array(ts_S_final), dim=1, bins=60, range_lim=(-2, 4))

print(f"\n  Target:  frac_left={tgt_frac_left:.3f}, mass={tgt_mass_curve[-1]:.1f}")
print(f"  Learned: frac_left={lrn_frac_left:.3f}, mass={lrn_mass_curve[-1]:.1f} "
      f"(mu_x={float(model.mu_x):.3f})")
print(f"  Lrn+TrueSrc: frac_left={ts_frac_left:.3f}, mass={ts_mass_curve[-1]:.1f}")
print(f"\n  Target zones:  {tgt_zone_fracs}")
print(f"  Learned zones: {lrn_zone_fracs}")


# ==========================================================================
# PLOT 1: Mass curves
# ==========================================================================

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(time_axis, tgt_mass_curve, 'b-', lw=2, label='Target (true dynamics)')
ax.plot(time_axis, lrn_mass_curve, 'r--', lw=2, label=f'Learned (mu_x={float(model.mu_x):.3f})')
ax.plot(time_axis, ts_mass_curve, 'g:', lw=2, label='Learned pot + true source')
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('$M(t) = \\sum e^S$', fontsize=12)
ax.set_title(f'Total mass: target vs learned (T={T_FINAL})', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, f"analysis_mass_curves_{SUFFIX}.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved analysis_mass_curves.png")


# ==========================================================================
# PLOT 2: Marginals (x and y)
# ==========================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(x_bins, x_hist_tgt, 'b-', lw=2, label='Target')
axes[0].plot(x_bins, x_hist_lrn, 'r--', lw=2,
             label=f'Learned (mu_x={float(model.mu_x):.3f})')
axes[0].plot(x_bins, x_hist_ts, 'g:', lw=2, label='Lrn pot + true src')
axes[0].axvline(TRUE_MU_X, color='blue', ls=':', alpha=0.5, label=f'True mu_x={TRUE_MU_X}')
axes[0].axvline(float(model.mu_x), color='red', ls=':', alpha=0.5,
                label=f'Learned mu_x={float(model.mu_x):.3f}')
axes[0].set_xlabel('x', fontsize=12)
axes[0].set_ylabel('Weighted density', fontsize=12)
axes[0].set_title(f'x-marginal at t={T_FINAL}', fontsize=13)
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

axes[1].plot(y_bins, y_hist_tgt, 'b-', lw=2, label='Target')
axes[1].plot(y_bins, y_hist_lrn, 'r--', lw=2, label='Learned')
axes[1].plot(y_bins, y_hist_ts, 'g:', lw=2, label='Lrn pot + true src')
axes[1].set_xlabel('y', fontsize=12)
axes[1].set_ylabel('Weighted density', fontsize=12)
axes[1].set_title(f'y-marginal at t={T_FINAL}', fontsize=13)
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, f"analysis_marginals_{SUFFIX}.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved analysis_marginals.png")


# ==========================================================================
# PLOT 3: Bifurcation
# ==========================================================================

fig, ax = plt.subplots(figsize=(8, 5))
labels = ['Target', f'Learned\n(mu_x={float(model.mu_x):.3f})', 'Lrn pot\n+true src']
fl_vals = [tgt_frac_left, lrn_frac_left, ts_frac_left]
fr_vals = [tgt_frac_right, lrn_frac_right, ts_frac_right]

x_pos = np.arange(len(labels))
w = 0.35
ax.bar(x_pos - w/2, fl_vals, w, label='Left (x<0)', color='tab:blue', alpha=0.8)
ax.bar(x_pos + w/2, fr_vals, w, label='Right (x>0)', color='tab:orange', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylabel('Weight fraction', fontsize=12)
ax.set_title('Bifurcation: weight fraction left vs right', fontsize=13)
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1)

for i, (fl, fr) in enumerate(zip(fl_vals, fr_vals)):
    ax.text(i - w/2, fl + 0.02, f'{fl:.3f}', ha='center', fontsize=9)
    ax.text(i + w/2, fr + 0.02, f'{fr:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, f"analysis_bifurcation_{SUFFIX}.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved analysis_bifurcation.png")


# ==========================================================================
# PLOT 4: Decay curves
# ==========================================================================

fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(tgt_decay_tau, tgt_decay_lo, tgt_decay_hi,
                color='blue', alpha=0.15, label='Target 10-90%')
ax.plot(tgt_decay_tau, tgt_decay_mean, 'b-', lw=2, label='Target mean')
ax.fill_between(lrn_decay_tau, lrn_decay_lo, lrn_decay_hi,
                color='red', alpha=0.15, label='Learned 10-90%')
ax.plot(lrn_decay_tau, lrn_decay_mean, 'r-', lw=2, label='Learned mean')
ax.set_xlabel('Time since birth', fontsize=12)
ax.set_ylabel('Weight $e^S$', fontsize=12)
ax.set_title('Per-particle weight decay (aligned by birth time)', fontsize=13)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, f"analysis_decay_curves_{SUFFIX}.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved analysis_decay_curves.png")


# ==========================================================================
# PLOT 5: Sample trajectories (target vs learned, side by side)
# ==========================================================================

rng = np.random.RandomState(42)
early_mask_tgt = np.array(ana_wake_tgt) < N_STEPS // 3
early_idx_tgt = np.where(early_mask_tgt)[0]
traj_idx_tgt = rng.choice(early_idx_tgt,
                           size=min(N_TRAJ_PLOT, len(early_idx_tgt)),
                           replace=False)
traj_idx_tgt.sort()

early_mask_lrn = np.array(ana_wake_lrn) < N_STEPS // 3
early_idx_lrn = np.where(early_mask_lrn)[0]
traj_idx_lrn = rng.choice(early_idx_lrn,
                           size=min(N_TRAJ_PLOT, len(early_idx_lrn)),
                           replace=False)
traj_idx_lrn.sort()

ana_tgt_z_np = np.array(ana_tgt_z)
ana_tgt_S_np = np.array(ana_tgt_S)
wake_tgt_np = np.array(ana_wake_tgt)
ana_lrn_z_np = np.array(ana_lrn_z)
ana_lrn_S_np = np.array(ana_lrn_S)
wake_lrn_np = np.array(ana_wake_lrn)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for ax, z_np, S_np, wake_np, traj_idx, title, mu_x_val in [
    (axes[0], ana_tgt_z_np, ana_tgt_S_np, wake_tgt_np, traj_idx_tgt,
     f'Target (mu_x={TRUE_MU_X})', TRUE_MU_X),
    (axes[1], ana_lrn_z_np, ana_lrn_S_np, wake_lrn_np, traj_idx_lrn,
     f'Learned (mu_x={float(model.mu_x):.3f})', float(model.mu_x)),
]:
    for idx in traj_idx:
        birth = int(wake_np[idx])
        if birth >= N_STEPS:
            continue
        traj_x = z_np[birth:, idx, 0]
        traj_y = z_np[birth:, idx, 1]
        traj_w = np.exp(S_np[birth:, idx])
        points = np.column_stack([traj_x, traj_y]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='coolwarm',
                            norm=plt.Normalize(0, 1))
        lc.set_array(traj_w[:-1])
        lc.set_linewidth(0.8)
        ax.add_collection(lc)
    ax.plot(mu_x_val, TRUE_MU_Y, 'w*', markersize=15,
            markeredgecolor='k', markeredgewidth=1, zorder=10)
    ax.set_xlim(-4, 4); ax.set_ylim(-2, 3.5)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(title, fontsize=13)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(0, 1))
sm.set_array([])
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
fig.colorbar(sm, cax=cbar_ax, label='Weight $e^S$')
plt.suptitle(f'Sample trajectories ({N_TRAJ_PLOT} particles, colored by weight)',
             fontsize=14, y=1.01)
plt.savefig(os.path.join(SCRIPT_DIR, f"analysis_trajectories_{SUFFIX}.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved analysis_trajectories.png")


# ==========================================================================
# PLOT 6: Zone weight fractions
# ==========================================================================

zone_names = list(ZONES.keys())
zone_colors = ['tab:blue', 'tab:gray', 'tab:orange', 'tab:green', 'tab:red']

fig, ax = plt.subplots(figsize=(10, 6))
labels = ['Target', f'Learned (mu_x={float(model.mu_x):.3f})']
all_zone_data = [tgt_zone_fracs, lrn_zone_fracs]

x_pos = np.arange(len(labels))
bar_width = 0.6
for il, (label, zfracs) in enumerate(zip(labels, all_zone_data)):
    bottom = 0.0
    for iz, zname in enumerate(zone_names):
        val = zfracs[zname]
        ax.bar(x_pos[il], val, bar_width, bottom=bottom,
               color=zone_colors[iz], alpha=0.7,
               label=zname if il == 0 else None)
        if val > 0.03:
            ax.text(x_pos[il], bottom + val/2, f'{val:.3f}',
                    ha='center', va='center', fontsize=8)
        bottom += val

ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('Weight fraction', fontsize=12)
ax.set_title('Zone weight fractions: target vs learned', fontsize=13)
ax.legend(fontsize=9, loc='upper right')
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, f"analysis_zones_{SUFFIX}.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved analysis_zones.png")


# ==========================================================================
# PLOT 7: Summary — 2D scatter at final time
# ==========================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Shared potential contours
x_grid = jnp.linspace(-4, 4, 200)
y_grid = jnp.linspace(-2, 3.5, 200)
Xg, Yg = jnp.meshgrid(x_grid, y_grid)
Z_true = jax.vmap(jax.vmap(
    lambda x, y: analytical_potential(x, y, TARGET_PARAMS)
))(Xg, Yg)
ref_true = float(analytical_potential(0.0, 0.0, TARGET_PARAMS))
Z_true = Z_true - ref_true
pot_vmin = float(jnp.min(Z_true))
pot_vmax = min(float(jnp.max(Z_true)), 30.0)
pot_levels = np.linspace(pot_vmin, pot_vmax, 31)

for ax, z_f, S_f, title, mu_x_val in [
    (axes[0], tgt_z_final, tgt_S_final, 'Target', TRUE_MU_X),
    (axes[1], lrn_z_final, lrn_S_final,
     f'Learned (mu_x={float(model.mu_x):.3f})', float(model.mu_x)),
    (axes[2], ts_z_final, ts_S_final, 'Lrn pot + true src', TRUE_MU_X),
]:
    w = np.array(jnp.clip(jnp.exp(S_f), 0.0, 1.0))
    alive = w > 1e-6
    ax.contourf(np.array(Xg), np.array(Yg), np.array(Z_true),
                levels=pot_levels, cmap='viridis', alpha=0.3)
    sc = ax.scatter(np.array(z_f[alive, 0]), np.array(z_f[alive, 1]),
                    c=w[alive], cmap='coolwarm', s=6, alpha=0.5,
                    vmin=0, vmax=1, edgecolors='none')
    ax.plot(mu_x_val, TRUE_MU_Y, 'w*', markersize=12,
            markeredgecolor='k', markeredgewidth=0.8, zorder=10)
    ax.set_xlim(-4, 4); ax.set_ylim(-2, 3.5)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(title, fontsize=13)

fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
fig.colorbar(sc, cax=cbar_ax, label=r'$w = e^S$')
plt.suptitle(f'Final distribution at t={T_FINAL}', fontsize=14, y=1.01)
plt.savefig(os.path.join(SCRIPT_DIR, f"analysis_scatter_{SUFFIX}.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved analysis_scatter.png")


# ==========================================================================
# Summary
# ==========================================================================

x_rmse = float(np.sqrt(np.mean((x_hist_lrn - x_hist_tgt)**2)))
y_rmse = float(np.sqrt(np.mean((y_hist_lrn - y_hist_tgt)**2)))
x_rmse_ts = float(np.sqrt(np.mean((x_hist_ts - x_hist_tgt)**2)))

print(f"\n{'='*65}")
print(f"Summary")
print(f"{'='*65}")
print(f"  mu_x: {float(model.mu_x):.4f} (true={TRUE_MU_X}, error={abs(float(model.mu_x)-TRUE_MU_X):.4f})")
print(f"  Bifurcation: target={tgt_frac_left:.3f}L, "
      f"learned={lrn_frac_left:.3f}L, lrn+truesrc={ts_frac_left:.3f}L")
print(f"  Mass: target={tgt_mass_curve[-1]:.1f}, "
      f"learned={lrn_mass_curve[-1]:.1f}, lrn+truesrc={ts_mass_curve[-1]:.1f}")
print(f"  x-marginal RMSE: learned={x_rmse:.4f}, lrn+truesrc={x_rmse_ts:.4f}")
print(f"  y-marginal RMSE: learned={y_rmse:.4f}")
print(f"\n  Zone fractions:")
print(f"    {'Zone':>12s}  {'Target':>8s}  {'Learned':>8s}  {'Error':>8s}")
for zname in zone_names:
    t = tgt_zone_fracs[zname]
    l = lrn_zone_fracs[zname]
    print(f"    {zname:>12s}  {t:>8.3f}  {l:>8.3f}  {abs(l-t):>8.3f}")

print(f"\nAll analysis plots saved to {SCRIPT_DIR}/")
