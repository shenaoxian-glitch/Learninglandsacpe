#!/usr/bin/env python
"""
analyze_distributions.py — Compare particle-level observables: target vs learned

Retrains each model from sweep_nparticles_v3 (deterministic seeds → same models),
then simulates a fresh analysis batch through both target and learned models
to compare observable distribution characteristics:

  1. Mass curve M(t)                  — total alive mass over time
  2. Bifurcation fraction             — weight fraction at x<0 vs x>0
  3. Weighted spatial marginals       — 1D x- and y-histograms
  4. Per-particle decay curves        — weight decay aligned by birth time
  5. Sample trajectories              — x-y paths for individual particles
  6. Zone weight fractions            — weight in spatial zones

Design:
  - Analysis batch: N_analysis=2000, seed=99 (same init + noise for target & learned)
  - This isolates the effect of the learned potential on particle distribution

Usage:
    python sweep_nparticles_v3/analyze_distributions.py > sweep_nparticles_v3/log_analysis.txt 2>&1
"""
import matplotlib
matplotlib.use('Agg')

import os
import sys
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import SourceDeathModel
from models.potential import PotentialNN
from training.data_loader import (
    build_particle_queue, analytical_death_rate_2d, analytical_potential,
)
from simulator.sde_solver import simulate_open_system_full


# ======================================================================
# Constants (same as sweep_nparticles_v3)
# ======================================================================

TARGET_PARAMS = {
    'a': -1.0, 'b': -1.6, 'c': 3.5, 'd': -1.2,
    'e': 0.7, 'sigma': 1.5,
}
DEATH_PARAMS_2D = {
    'A': 2.0, 'w_x': 0.5, 'k1': 1.0, 'y_select': 1.0,
    'B': 5.0, 'k2': 3.0, 'y_max': 3.0,
}

DT = 0.01
T_FINAL = 6.0
N_STEPS = int(T_FINAL / DT)
SIGMA = 1.5
LAM_MASS = 5.0
N_EPOCHS = 800
PATIENCE = 300

N_TARGET = 8000
SEED_TARGET = 1
SEED_LEARNED = 2
N_LEARNED_LIST = [200, 400, 600, 1000, 1500, 2000, 3000, 4000, 6000]

# Analysis settings
N_ANALYSIS = 2000
SEED_ANALYSIS = 99
N_TRAJ_PLOT = 30           # number of trajectories to plot
SELECT_N = [200, 1000, 4000]  # N values for detailed per-particle plots

# Spatial zones for weight fraction analysis
ZONES = {
    'left_well':   lambda x, y: x < -1.0,
    'saddle':      lambda x, y: (x >= -1.0) & (x <= 1.0),
    'right_well':  lambda x, y: x > 1.0,
    'source':      lambda x, y: y < 0.0,
    'death_zone':  lambda x, y: y > 2.2,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ======================================================================
# Model components (same as sweep)
# ======================================================================

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


# ======================================================================
# Ground-truth simulation
# ======================================================================

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


# ======================================================================
# Loss functions (same as sweep)
# ======================================================================

def mass_loss(S, target_mass):
    M_sim = jnp.sum(jnp.exp(S))
    return (M_sim - target_mass)**2 / (target_mass**2 + 1e-8)


def weighted_mmd_both(x_sim, alpha_sim, x_obs, alpha_obs,
                      bandwidths=(0.01, 1.0, 100.0)):
    def k_matrix(a, b):
        diff = a[:, None, :] - b[None, :, :]
        dist_sq = jnp.sum(diff**2, axis=-1)
        K = jnp.zeros_like(dist_sq)
        for bw in bandwidths:
            K = K + jnp.exp(-dist_sq / (2 * bw))
        return K
    K_xx = k_matrix(x_sim, x_sim)
    K_yy = k_matrix(x_obs, x_obs)
    K_xy = k_matrix(x_sim, x_obs)
    term_xx = jnp.sum(alpha_sim[:, None] * alpha_sim[None, :] * K_xx)
    term_yy = jnp.sum(alpha_obs[:, None] * alpha_obs[None, :] * K_yy)
    term_xy = jnp.sum(alpha_sim[:, None] * alpha_obs[None, :] * K_xy)
    return term_xx + term_yy - 2 * term_xy


def make_objective(n_steps, dt, sigma, lam_mass):
    def objective(model, z_init, wake_schedule, tgt_z, tgt_S, tgt_mass, key):
        all_z, all_S = simulate_open_system_full(
            model, z_init, wake_schedule, sigma, n_steps, dt, key
        )
        sim_z = all_z[-1]
        sim_S = all_S[-1]
        sim_alpha = jax.nn.softmax(sim_S)
        tgt_alpha = jax.nn.softmax(tgt_S)
        l_mmd = weighted_mmd_both(sim_z, sim_alpha, tgt_z, tgt_alpha)
        l_mass = mass_loss(sim_S, tgt_mass)
        total = l_mmd + lam_mass * l_mass
        return total, (l_mmd, l_mass)
    return objective


def make_train_step(optimizer, objective_fn):
    @eqx.filter_jit
    def step(model, opt_state, z_init, wake_schedule,
             tgt_z, tgt_S, tgt_mass, key):
        def loss_fn(model):
            return objective_fn(
                model, z_init, wake_schedule, tgt_z, tgt_S, tgt_mass, key
            )
        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss, aux
    return step


# ======================================================================
# Retrain model (deterministic → same as sweep)
# ======================================================================

def retrain_model(n_learned, tgt_z, tgt_S, tgt_mass):
    print(f"\n  Retraining N={n_learned} ...")
    learn_master = jax.random.PRNGKey(SEED_LEARNED)
    queue_key, model_key, opt_key = jax.random.split(learn_master, 3)

    sources = [
        {'mu': jnp.array([0.0, -1.0]), 'sigma_x': 0.25, 'sigma_y': 0.10,
         'n_particles': n_learned},
    ]
    z_init, wake_schedule = build_particle_queue(
        sources=sources, n_particles=n_learned,
        t_final=T_FINAL, dt=DT, key=queue_key,
    )

    phi_nn = PotentialNN(model_key, d_latent=2, c_conf=0.0,
                         hidden_sizes=(16, 16))
    potential = PotentialWithConfinement(
        nn=phi_nn, b=TARGET_PARAMS['b'], d=TARGET_PARAMS['d'],
    )
    fixed_death = FixedDeathRate2D(**DEATH_PARAMS_2D)
    model = SourceDeathModel(potential=potential, death_rate=fixed_death)

    scaled_tgt_mass = tgt_mass * (n_learned / N_TARGET)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=3e-3, weight_decay=1e-4),
    )
    objective_fn = make_objective(N_STEPS, DT, SIGMA, LAM_MASS)
    step_fn = make_train_step(optimizer, objective_fn)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    best_loss = float('inf')
    best_epoch = 0
    best_model_leaves = None
    wait = 0

    for epoch in range(N_EPOCHS):
        opt_key, sim_key = jax.random.split(opt_key)
        model, opt_state, loss, (l_mmd, l_mass) = step_fn(
            model, opt_state, z_init, wake_schedule,
            tgt_z, tgt_S, scaled_tgt_mass, sim_key,
        )
        loss_val = float(loss)

        if epoch % 200 == 0:
            print(f"    Epoch {epoch:4d}: Loss={loss_val:.5f}")

        if loss_val < best_loss - 1e-6:
            best_loss = loss_val
            best_epoch = epoch
            best_model_leaves = jax.tree.map(
                lambda x: x.copy() if hasattr(x, 'copy') else x,
                eqx.filter(model, eqx.is_array)
            )
            wait = 0
        else:
            wait += 1
        if wait >= PATIENCE:
            print(f"    Early stopping at epoch {epoch}")
            break

    if best_model_leaves is not None:
        model = eqx.combine(best_model_leaves,
                             eqx.filter(model, lambda x: not eqx.is_array(x)))
    print(f"    Best epoch {best_epoch}, loss={best_loss:.6f}")
    return model


# ======================================================================
# Analysis functions
# ======================================================================

def compute_mass_curve(all_S):
    """M(t) = Σ exp(S_i(t)). Shape: (n_steps,)"""
    return np.array(jnp.sum(jnp.exp(all_S), axis=1))


def compute_bifurcation_fraction(z_final, S_final, x_threshold=0.0):
    """Weighted fraction at x < threshold vs x > threshold."""
    weights = jnp.exp(S_final)
    total = jnp.sum(weights) + 1e-12
    frac_left = float(jnp.sum(weights * (z_final[:, 0] < x_threshold)) / total)
    return frac_left, 1.0 - frac_left


def compute_weighted_marginal(z_final, S_final, dim, bins, range_lim):
    """Weighted 1D histogram."""
    weights = np.array(jnp.exp(S_final))
    values = np.array(z_final[:, dim])
    hist, bin_edges = np.histogram(values, bins=bins, range=range_lim,
                                    weights=weights, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return hist, bin_centers


def compute_zone_fractions(z_final, S_final, zones):
    """Weighted fraction of mass in each spatial zone."""
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
    """
    Per-particle weight decay aligned by birth time.

    Returns:
        tau: time-since-birth axis (n_tau,)
        mean_curve: mean weight at each tau
        lo, hi: 10th and 90th percentile envelope
        median_curve: median weight
    """
    all_S = np.array(all_S)                    # (n_steps, N)
    wake = np.array(wake_schedule)             # (N,)
    n_steps, N = all_S.shape

    # Max possible lifetime
    max_life = n_steps
    aligned = np.full((N, max_life), np.nan)

    for i in range(N):
        birth = int(wake[i])
        if birth >= n_steps:
            continue
        life_len = n_steps - birth
        # weight from birth onward
        aligned[i, :life_len] = np.exp(all_S[birth:, i])

    tau = np.arange(max_life) * dt

    # Compute stats ignoring NaN (particles not yet born)
    with np.errstate(all='ignore'):
        mean_curve = np.nanmean(aligned, axis=0)
        median_curve = np.nanmedian(aligned, axis=0)
        lo = np.nanpercentile(aligned, 10, axis=0)
        hi = np.nanpercentile(aligned, 90, axis=0)

    # Trim to where we have at least 10% of particles
    valid_count = np.sum(~np.isnan(aligned), axis=0)
    cutoff = np.searchsorted(-valid_count, -int(0.1 * N))
    if cutoff < 10:
        cutoff = max_life

    return tau[:cutoff], mean_curve[:cutoff], lo[:cutoff], hi[:cutoff], \
        median_curve[:cutoff]


# ======================================================================
# Main
# ======================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("Distribution analysis: target vs learned models")
    print("=" * 65)
    print(f"  N_analysis = {N_ANALYSIS}, seed = {SEED_ANALYSIS}")
    print(f"  N_learned = {N_LEARNED_LIST}")
    print(f"  Detailed plots for N = {SELECT_N}")
    print("=" * 65)

    # ==================================================================
    # 1. Generate training target (N=8000, seed=1)
    # ==================================================================
    print("\nGenerating training target (seed=1, N=8000) ...")
    tgt_master = jax.random.PRNGKey(SEED_TARGET)
    tgt_queue_key, tgt_sim_key = jax.random.split(tgt_master)
    tgt_z_init_train, tgt_wake_train = build_particle_queue(
        sources=[{'mu': jnp.array([0.0, -1.0]), 'sigma_x': 0.25,
                  'sigma_y': 0.10, 'n_particles': N_TARGET}],
        n_particles=N_TARGET, t_final=T_FINAL, dt=DT, key=tgt_queue_key,
    )
    tgt_all_px, tgt_all_py, tgt_all_S_train = simulate_ground_truth_full(
        tgt_z_init_train, tgt_wake_train, N_STEPS, DT, tgt_sim_key
    )
    final_step = N_STEPS - 1
    tgt_z = jnp.column_stack([tgt_all_px[final_step], tgt_all_py[final_step]])
    tgt_S = tgt_all_S_train[final_step]
    tgt_mass = float(jnp.sum(jnp.exp(tgt_S)))
    print(f"  Training target: mass={tgt_mass:.1f}")

    # ==================================================================
    # 2. Generate analysis batch (N=2000, seed=99)
    # ==================================================================
    print(f"\nGenerating analysis batch (seed={SEED_ANALYSIS}, "
          f"N={N_ANALYSIS}) ...")
    ana_master = jax.random.PRNGKey(SEED_ANALYSIS)
    ana_queue_key, ana_sim_key = jax.random.split(ana_master)
    ana_sources = [
        {'mu': jnp.array([0.0, -1.0]), 'sigma_x': 0.25, 'sigma_y': 0.10,
         'n_particles': N_ANALYSIS},
    ]
    ana_z_init, ana_wake = build_particle_queue(
        sources=ana_sources, n_particles=N_ANALYSIS,
        t_final=T_FINAL, dt=DT, key=ana_queue_key,
    )

    # Target analysis trajectory (full)
    print("  Simulating analysis batch through target ...")
    ana_tgt_px, ana_tgt_py, ana_tgt_S = simulate_ground_truth_full(
        ana_z_init, ana_wake, N_STEPS, DT, ana_sim_key
    )
    ana_tgt_z = jnp.stack([ana_tgt_px, ana_tgt_py], axis=-1)  # (n_steps, N, 2)

    # Target stats
    time_axis = np.arange(1, N_STEPS + 1) * DT
    tgt_mass_curve = compute_mass_curve(ana_tgt_S)
    tgt_z_final = ana_tgt_z[final_step]
    tgt_S_final = ana_tgt_S[final_step]
    tgt_frac_left, tgt_frac_right = compute_bifurcation_fraction(
        tgt_z_final, tgt_S_final)
    tgt_zone_fracs = compute_zone_fractions(tgt_z_final, tgt_S_final, ZONES)
    tgt_decay_tau, tgt_decay_mean, tgt_decay_lo, tgt_decay_hi, tgt_decay_med = \
        compute_decay_curves(ana_tgt_S, ana_wake, DT)

    x_hist_tgt, x_bins = compute_weighted_marginal(
        np.array(tgt_z_final), np.array(tgt_S_final),
        dim=0, bins=60, range_lim=(-4, 4))
    y_hist_tgt, y_bins = compute_weighted_marginal(
        np.array(tgt_z_final), np.array(tgt_S_final),
        dim=1, bins=60, range_lim=(-2, 4))

    print(f"  Target: frac_left={tgt_frac_left:.3f}, "
          f"final_mass={tgt_mass_curve[-1]:.1f}")
    print(f"  Target zones: {tgt_zone_fracs}")

    # ==================================================================
    # 3. Retrain + analyze each N
    # ==================================================================
    all_stats = {}

    for n_learned in N_LEARNED_LIST:
        print(f"\n{'='*50}")
        print(f"  N_learned = {n_learned}")
        print(f"{'='*50}")

        model = retrain_model(n_learned, tgt_z, tgt_S, tgt_mass)

        # Simulate analysis batch through learned model
        print(f"    Simulating analysis batch (N={N_ANALYSIS}) ...")
        ana_lrn_z, ana_lrn_S = simulate_open_system_full(
            model, ana_z_init, ana_wake, SIGMA, N_STEPS, DT, ana_sim_key
        )

        lrn_z_final = ana_lrn_z[final_step]
        lrn_S_final = ana_lrn_S[final_step]
        lrn_mass_curve = compute_mass_curve(ana_lrn_S)
        lrn_frac_left, lrn_frac_right = compute_bifurcation_fraction(
            lrn_z_final, lrn_S_final)
        lrn_zone_fracs = compute_zone_fractions(lrn_z_final, lrn_S_final, ZONES)

        x_hist_lrn, _ = compute_weighted_marginal(
            np.array(lrn_z_final), np.array(lrn_S_final),
            dim=0, bins=60, range_lim=(-4, 4))
        y_hist_lrn, _ = compute_weighted_marginal(
            np.array(lrn_z_final), np.array(lrn_S_final),
            dim=1, bins=60, range_lim=(-2, 4))

        stats = {
            'mass_curve': lrn_mass_curve,
            'frac_left': lrn_frac_left,
            'frac_right': lrn_frac_right,
            'final_mass': float(lrn_mass_curve[-1]),
            'zone_fracs': lrn_zone_fracs,
            'x_hist': x_hist_lrn,
            'y_hist': y_hist_lrn,
        }

        # Detailed analysis for selected N values
        if n_learned in SELECT_N:
            # Decay curves
            lrn_tau, lrn_decay_mean, lrn_decay_lo, lrn_decay_hi, lrn_decay_med = \
                compute_decay_curves(ana_lrn_S, ana_wake, DT)
            stats['decay_tau'] = lrn_tau
            stats['decay_mean'] = lrn_decay_mean
            stats['decay_lo'] = lrn_decay_lo
            stats['decay_hi'] = lrn_decay_hi
            stats['decay_med'] = lrn_decay_med

            # Save full trajectories for trajectory plots
            stats['all_z'] = np.array(ana_lrn_z)  # (n_steps, N, 2)
            stats['all_S'] = np.array(ana_lrn_S)  # (n_steps, N)

        print(f"    Bifurcation: {lrn_frac_left:.3f}L / {lrn_frac_right:.3f}R "
              f"(target: {tgt_frac_left:.3f}/{tgt_frac_right:.3f})")
        print(f"    Final mass: {lrn_mass_curve[-1]:.1f} "
              f"(target: {tgt_mass_curve[-1]:.1f})")
        print(f"    Zones: {lrn_zone_fracs}")

        all_stats[n_learned] = stats

    # ==================================================================
    # Save compact data for replotting
    # ==================================================================
    save_dict = {
        'time_axis': time_axis,
        'n_learned_list': np.array(N_LEARNED_LIST),
        'tgt_mass_curve': tgt_mass_curve,
        'tgt_z_final': np.array(tgt_z_final),
        'tgt_S_final': np.array(tgt_S_final),
        'tgt_frac_left': np.array(tgt_frac_left),
        'x_bins': x_bins, 'y_bins': y_bins,
        'x_hist_tgt': x_hist_tgt, 'y_hist_tgt': y_hist_tgt,
        'tgt_decay_tau': tgt_decay_tau,
        'tgt_decay_mean': tgt_decay_mean,
        'tgt_decay_lo': tgt_decay_lo,
        'tgt_decay_hi': tgt_decay_hi,
    }
    for n, stats in all_stats.items():
        save_dict[f'mass_curve_N{n}'] = stats['mass_curve']
        save_dict[f'frac_left_N{n}'] = np.array(stats['frac_left'])
        save_dict[f'x_hist_N{n}'] = stats['x_hist']
        save_dict[f'y_hist_N{n}'] = stats['y_hist']
        if 'decay_mean' in stats:
            save_dict[f'decay_tau_N{n}'] = stats['decay_tau']
            save_dict[f'decay_mean_N{n}'] = stats['decay_mean']
            save_dict[f'decay_lo_N{n}'] = stats['decay_lo']
            save_dict[f'decay_hi_N{n}'] = stats['decay_hi']
    np.savez(os.path.join(SCRIPT_DIR, "analysis_data.npz"), **save_dict)
    print("\nSaved analysis_data.npz")

    # ==================================================================
    # PLOT 1: Mass curves (3x3 grid)
    # ==================================================================
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for i, n_learned in enumerate(N_LEARNED_LIST):
        ax = axes.ravel()[i]
        ax.plot(time_axis, tgt_mass_curve, 'b-', linewidth=2,
                label='Target', alpha=0.8)
        ax.plot(time_axis, all_stats[n_learned]['mass_curve'], 'r--',
                linewidth=2, label=f'Learned (N={n_learned})', alpha=0.8)
        ax.set_xlabel('Time')
        ax.set_ylabel('$M(t)$')
        ax.set_title(f'N = {n_learned}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Total mass $M(t) = \\sum e^{S_i}$: target vs learned',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "analysis_mass_curves.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved analysis_mass_curves.png")

    # ==================================================================
    # PLOT 2: Bifurcation fraction vs N
    # ==================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n_arr = N_LEARNED_LIST
    lrn_fl = [all_stats[n]['frac_left'] for n in n_arr]
    lrn_fr = [all_stats[n]['frac_right'] for n in n_arr]

    x_pos = np.arange(len(n_arr))
    w = 0.35
    axes[0].bar(x_pos - w/2, lrn_fl, w, label='Learned (left)',
                color='tab:blue', alpha=0.8)
    axes[0].bar(x_pos + w/2, lrn_fr, w, label='Learned (right)',
                color='tab:orange', alpha=0.8)
    axes[0].axhline(tgt_frac_left, color='blue', ls='--', lw=2,
                    label=f'Target left={tgt_frac_left:.3f}')
    axes[0].axhline(tgt_frac_right, color='orange', ls='--', lw=2,
                    label=f'Target right={tgt_frac_right:.3f}')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(n_arr, rotation=45)
    axes[0].set_xlabel('N (training particles)')
    axes[0].set_ylabel('Weight fraction')
    axes[0].set_title('Bifurcation: weight at x<0 vs x>0')
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim(0, 1)

    frac_errors = [abs(f - tgt_frac_left) for f in lrn_fl]
    axes[1].plot(n_arr, frac_errors, 'ro-', lw=2, ms=8)
    axes[1].set_xlabel('N (training particles)')
    axes[1].set_ylabel('|frac_left error|')
    axes[1].set_title('Bifurcation fraction error vs N')
    axes[1].set_xscale('log'); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "analysis_bifurcation.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved analysis_bifurcation.png")

    # ==================================================================
    # PLOT 3: Weighted marginals for selected N
    # ==================================================================
    fig, axes = plt.subplots(2, len(SELECT_N), figsize=(6*len(SELECT_N), 10))
    for col, n_sel in enumerate(SELECT_N):
        axes[0, col].plot(x_bins, x_hist_tgt, 'b-', lw=2, label='Target')
        axes[0, col].plot(x_bins, all_stats[n_sel]['x_hist'], 'r--', lw=2,
                          label=f'Learned (N={n_sel})')
        axes[0, col].set_xlabel('x'); axes[0, col].set_ylabel('Weighted density')
        axes[0, col].set_title(f'x-marginal (N_train={n_sel})')
        axes[0, col].legend(fontsize=9); axes[0, col].grid(True, alpha=0.3)

        axes[1, col].plot(y_bins, y_hist_tgt, 'b-', lw=2, label='Target')
        axes[1, col].plot(y_bins, all_stats[n_sel]['y_hist'], 'r--', lw=2,
                          label=f'Learned (N={n_sel})')
        axes[1, col].set_xlabel('y'); axes[1, col].set_ylabel('Weighted density')
        axes[1, col].set_title(f'y-marginal (N_train={n_sel})')
        axes[1, col].legend(fontsize=9); axes[1, col].grid(True, alpha=0.3)
    plt.suptitle('Weighted spatial marginals at t=6.0', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "analysis_marginals.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved analysis_marginals.png")

    # ==================================================================
    # PLOT 4: Per-particle decay curves (birth-aligned)
    # ==================================================================
    fig, axes = plt.subplots(1, len(SELECT_N), figsize=(6*len(SELECT_N), 5))
    for col, n_sel in enumerate(SELECT_N):
        ax = axes[col]
        s = all_stats[n_sel]
        # Target
        ax.fill_between(tgt_decay_tau, tgt_decay_lo, tgt_decay_hi,
                        color='blue', alpha=0.15, label='Target 10-90%')
        ax.plot(tgt_decay_tau, tgt_decay_mean, 'b-', lw=2,
                label='Target mean')
        ax.plot(tgt_decay_tau, tgt_decay_med, 'b:', lw=1.5,
                label='Target median')
        # Learned
        ax.fill_between(s['decay_tau'], s['decay_lo'], s['decay_hi'],
                        color='red', alpha=0.15, label='Learned 10-90%')
        ax.plot(s['decay_tau'], s['decay_mean'], 'r-', lw=2,
                label='Learned mean')
        ax.plot(s['decay_tau'], s['decay_med'], 'r:', lw=1.5,
                label='Learned median')
        ax.set_xlabel('Time since birth')
        ax.set_ylabel('Weight $e^S$')
        ax.set_title(f'Particle decay (N_train={n_sel})')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    plt.suptitle('Per-particle weight decay (aligned by birth time)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "analysis_decay_curves.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved analysis_decay_curves.png")

    # ==================================================================
    # PLOT 5: Sample trajectories (x-y) for selected N
    # ==================================================================
    # Pick particles that are born early enough to have visible paths
    rng = np.random.RandomState(42)
    early_mask = np.array(ana_wake) < N_STEPS // 3  # born in first 1/3
    early_idx = np.where(early_mask)[0]
    traj_idx = rng.choice(early_idx, size=min(N_TRAJ_PLOT, len(early_idx)),
                          replace=False)
    traj_idx.sort()

    ana_tgt_z_np = np.array(ana_tgt_z)  # (n_steps, N, 2)
    ana_tgt_S_np = np.array(ana_tgt_S)  # (n_steps, N)
    wake_np = np.array(ana_wake)

    fig, axes = plt.subplots(2, len(SELECT_N), figsize=(6*len(SELECT_N), 12))

    for col, n_sel in enumerate(SELECT_N):
        lrn_z_np = all_stats[n_sel]['all_z']
        lrn_S_np = all_stats[n_sel]['all_S']

        for row, (z_arr, S_arr, label) in enumerate([
            (ana_tgt_z_np, ana_tgt_S_np, 'Target'),
            (lrn_z_np, lrn_S_np, f'Learned N={n_sel}'),
        ]):
            ax = axes[row, col]
            for idx in traj_idx:
                birth = int(wake_np[idx])
                if birth >= N_STEPS:
                    continue
                traj_x = z_arr[birth:, idx, 0]
                traj_y = z_arr[birth:, idx, 1]
                traj_w = np.exp(S_arr[birth:, idx])

                # Color by weight (fading as particle dies)
                points = np.column_stack([traj_x, traj_y]).reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap='coolwarm',
                                    norm=plt.Normalize(0, 1))
                lc.set_array(traj_w[:-1])
                lc.set_linewidth(0.8)
                ax.add_collection(lc)

            ax.set_xlim(-4, 4); ax.set_ylim(-2, 3.5)
            ax.set_xlabel('x'); ax.set_ylabel('y')
            ax.set_title(f'{label}')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label='Weight $e^S$')
    plt.suptitle(f'Sample trajectories ({N_TRAJ_PLOT} particles, '
                 f'colored by weight)', fontsize=14, y=1.01)
    plt.savefig(os.path.join(SCRIPT_DIR, "analysis_trajectories.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved analysis_trajectories.png")

    # ==================================================================
    # PLOT 6: Zone weight fractions vs N
    # ==================================================================
    zone_names = list(ZONES.keys())
    n_zones = len(zone_names)
    zone_colors = ['tab:blue', 'tab:gray', 'tab:orange', 'tab:green', 'tab:red']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Stacked bar: target vs learned for each zone
    x_pos = np.arange(len(N_LEARNED_LIST))
    bar_width = 0.7

    # Target reference (horizontal bands)
    ax = axes[0]
    bottom = np.zeros(len(N_LEARNED_LIST))
    for iz, zname in enumerate(zone_names):
        vals = [all_stats[n]['zone_fracs'][zname] for n in N_LEARNED_LIST]
        ax.bar(x_pos, vals, bar_width, bottom=bottom, color=zone_colors[iz],
               alpha=0.7, label=zname)
        bottom += np.array(vals)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(N_LEARNED_LIST, rotation=45)
    ax.set_xlabel('N (training particles)')
    ax.set_ylabel('Weight fraction')
    ax.set_title('Learned: zone weight fractions')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2, axis='y')

    # Per-zone error vs N
    ax = axes[1]
    for iz, zname in enumerate(zone_names):
        errors = [abs(all_stats[n]['zone_fracs'][zname] - tgt_zone_fracs[zname])
                  for n in N_LEARNED_LIST]
        ax.plot(N_LEARNED_LIST, errors, 'o-', color=zone_colors[iz],
                lw=2, ms=6, label=zname)
    ax.set_xlabel('N (training particles)')
    ax.set_ylabel('|zone fraction error|')
    ax.set_title('Zone fraction error vs N')
    ax.set_xscale('log')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Add target reference text
    tgt_str = "Target: " + ", ".join(
        f"{k}={v:.3f}" for k, v in tgt_zone_fracs.items())
    fig.text(0.5, -0.02, tgt_str, ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "analysis_zones.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved analysis_zones.png")

    # ==================================================================
    # PLOT 7: Summary metrics vs N
    # ==================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    mass_errors = [abs(all_stats[n]['final_mass'] - tgt_mass_curve[-1])
                   / tgt_mass_curve[-1] for n in N_LEARNED_LIST]
    axes[0].plot(N_LEARNED_LIST, mass_errors, 'ko-', lw=2, ms=8)
    axes[0].set_xlabel('N'); axes[0].set_ylabel('Relative mass error')
    axes[0].set_title('Final mass error vs N')
    axes[0].set_xscale('log'); axes[0].grid(True, alpha=0.3)

    axes[1].plot(N_LEARNED_LIST, frac_errors, 'ro-', lw=2, ms=8)
    axes[1].set_xlabel('N'); axes[1].set_ylabel('Bifurcation fraction error')
    axes[1].set_title('Bifurcation error vs N')
    axes[1].set_xscale('log'); axes[1].grid(True, alpha=0.3)

    x_marg_err = [np.sqrt(np.mean((all_stats[n]['x_hist'] - x_hist_tgt)**2))
                  for n in N_LEARNED_LIST]
    axes[2].plot(N_LEARNED_LIST, x_marg_err, 'bs-', lw=2, ms=8)
    axes[2].set_xlabel('N'); axes[2].set_ylabel('x-marginal RMSE')
    axes[2].set_title('x-marginal error vs N')
    axes[2].set_xscale('log'); axes[2].grid(True, alpha=0.3)

    plt.suptitle('Summary: distribution errors vs N', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "analysis_summary.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved analysis_summary.png")

    # ==================================================================
    # Print summary table
    # ==================================================================
    print(f"\n{'='*80}")
    print(f"Target: frac_left={tgt_frac_left:.3f}, "
          f"final_mass={tgt_mass_curve[-1]:.1f}")
    zone_str = "  ".join(f"{k}={v:.3f}" for k, v in tgt_zone_fracs.items())
    print(f"Target zones: {zone_str}")
    print(f"{'='*80}")
    print(f"{'N':>6}  {'frac_L':>7}  {'frac_R':>7}  {'mass':>7}  "
          f"{'left_w':>7}  {'saddle':>7}  {'right_w':>7}  "
          f"{'source':>7}  {'death':>7}")
    print("-" * 80)
    for n in N_LEARNED_LIST:
        s = all_stats[n]
        zf = s['zone_fracs']
        print(f"{n:>6}  {s['frac_left']:>7.3f}  {s['frac_right']:>7.3f}  "
              f"{s['final_mass']:>7.1f}  "
              f"{zf['left_well']:>7.3f}  {zf['saddle']:>7.3f}  "
              f"{zf['right_well']:>7.3f}  {zf['source']:>7.3f}  "
              f"{zf['death_zone']:>7.3f}")

    print(f"\nAll plots saved to {SCRIPT_DIR}/analysis_*.png")
