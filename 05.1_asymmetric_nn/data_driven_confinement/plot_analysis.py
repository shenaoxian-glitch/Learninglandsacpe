#!/usr/bin/env python
"""
plot_analysis.py — Distribution diagnostics for data-driven confinement model.

Runs a fresh analysis batch through both ground-truth and learned models,
then produces:
  1. Mass curves M(t): target vs learned
  2. Weighted marginals (x and y) at final time
  3. Per-particle weight decay curves (aligned by birth time)
  4. Zone weight fractions (bar chart + table)
  5. 2D weighted density comparison (target vs learned)

Usage:
    cd 05.1_asymmetric_nn
    python data_driven_confinement/plot_analysis.py
"""
import matplotlib
matplotlib.use('Agg')

import os
import sys
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import SourceDeathModel
from models.potential import PotentialNN
from training.data_loader import (
    build_particle_queue, analytical_death_rate_2d, analytical_potential,
)
from simulator.sde_solver import simulate_open_system_full

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================================================================
# Constants (must match train_data_conf.py)
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

# Analysis batch — independent seed, larger than training for smoother stats
N_ANALYSIS = 2000
SEED_ANALYSIS = 777

SOURCES = [
    {'mu': jnp.array([0.0, -1.0]), 'sigma_x': 0.25, 'sigma_y': 0.10,
     'n_particles': N_ANALYSIS},
]

ZONES = {
    'left_well':   lambda x, y: x < -1.0,
    'saddle':      lambda x, y: (x >= -1.0) & (x <= 1.0),
    'right_well':  lambda x, y: x > 1.0,
    'source':      lambda x, y: y < 0.0,
    'death_zone':  lambda x, y: y > 2.2,
}

# Confinement coefficients (from train_data_conf.py)
C_X = 0.07765374132916351
C_Y = 0.10606128401967316


# ======================================================================
# Model components (must match train_data_conf.py)
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


class PotentialWithAnisotropicConfinement(eqx.Module):
    nn: PotentialNN
    c_x: float = eqx.field(static=True)
    c_y: float = eqx.field(static=True)

    def __call__(self, z):
        x, y = z[0], z[1]
        conf = self.c_x * x**4 + self.c_y * y**4
        return self.nn(z) + conf


# ======================================================================
# Analysis utilities
# ======================================================================

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


def compute_weighted_hist2d(z_final, S_final, x_bins, y_bins):
    weights = np.array(jnp.exp(S_final))
    x = np.array(z_final[:, 0])
    y = np.array(z_final[:, 1])
    H, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=weights)
    # Normalize to density
    dx = x_bins[1] - x_bins[0]
    dy = y_bins[1] - y_bins[0]
    H = H / (weights.sum() * dx * dy + 1e-12)
    return H.T  # (ny, nx) for imshow/contourf


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
# Main
# ======================================================================

if __name__ == '__main__':

    print("=" * 65)
    print("Distribution analysis: data-driven confinement model")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Load trained model
    # ------------------------------------------------------------------
    phi_nn = PotentialNN(jax.random.PRNGKey(0), d_latent=2, c_conf=0.0,
                         hidden_sizes=(16, 16))
    potential = PotentialWithAnisotropicConfinement(nn=phi_nn, c_x=C_X, c_y=C_Y)
    fixed_death = FixedDeathRate2D(**DEATH_PARAMS_2D)
    model = SourceDeathModel(potential=potential, death_rate=fixed_death)

    model_path = os.path.join(SCRIPT_DIR, "trained_model_data_conf.eqx")
    model = eqx.tree_deserialise_leaves(model_path, model)
    print(f"Loaded model from {model_path}")

    # ------------------------------------------------------------------
    # Generate analysis batch (same seed for both target and learned)
    # ------------------------------------------------------------------
    ana_key = jax.random.PRNGKey(SEED_ANALYSIS)
    ana_queue_key, ana_tgt_key, ana_lrn_key = jax.random.split(ana_key, 3)

    z_init, wake_schedule = build_particle_queue(
        sources=SOURCES, n_particles=N_ANALYSIS,
        t_final=T_FINAL, dt=DT, key=ana_queue_key,
    )
    print(f"Analysis batch: N={N_ANALYSIS}, seed={SEED_ANALYSIS}")

    # --- Ground truth ---
    print("Simulating ground truth ...")
    tgt_all_px, tgt_all_py, tgt_all_S = simulate_ground_truth_full(
        z_init, wake_schedule, N_STEPS, DT, ana_tgt_key
    )
    tgt_all_z = jnp.stack([tgt_all_px, tgt_all_py], axis=-1)
    final_step = N_STEPS - 1

    # --- Learned model ---
    print("Simulating learned model ...")
    lrn_all_z, lrn_all_S = simulate_open_system_full(
        model, z_init, wake_schedule, SIGMA, N_STEPS, DT, ana_lrn_key
    )

    # ------------------------------------------------------------------
    # Compute all statistics
    # ------------------------------------------------------------------
    time_axis = np.arange(1, N_STEPS + 1) * DT

    # Mass curves
    tgt_mass_curve = compute_mass_curve(tgt_all_S)
    lrn_mass_curve = compute_mass_curve(lrn_all_S)

    # Final-time snapshots
    tgt_z_final = tgt_all_z[final_step]
    tgt_S_final = tgt_all_S[final_step]
    lrn_z_final = lrn_all_z[final_step]
    lrn_S_final = lrn_all_S[final_step]

    # Bifurcation
    tgt_frac_left, tgt_frac_right = compute_bifurcation_fraction(tgt_z_final, tgt_S_final)
    lrn_frac_left, lrn_frac_right = compute_bifurcation_fraction(lrn_z_final, lrn_S_final)

    # Marginals
    x_hist_tgt, x_bins = compute_weighted_marginal(
        np.array(tgt_z_final), np.array(tgt_S_final), dim=0, bins=60, range_lim=(-4, 4))
    y_hist_tgt, y_bins = compute_weighted_marginal(
        np.array(tgt_z_final), np.array(tgt_S_final), dim=1, bins=60, range_lim=(-2, 4))
    x_hist_lrn, _ = compute_weighted_marginal(
        np.array(lrn_z_final), np.array(lrn_S_final), dim=0, bins=60, range_lim=(-4, 4))
    y_hist_lrn, _ = compute_weighted_marginal(
        np.array(lrn_z_final), np.array(lrn_S_final), dim=1, bins=60, range_lim=(-2, 4))

    # Zones
    tgt_zones = compute_zone_fractions(tgt_z_final, tgt_S_final, ZONES)
    lrn_zones = compute_zone_fractions(lrn_z_final, lrn_S_final, ZONES)

    # Decay curves
    tgt_tau, tgt_decay_mean, tgt_decay_lo, tgt_decay_hi, tgt_decay_med = \
        compute_decay_curves(tgt_all_S, wake_schedule, DT)
    lrn_tau, lrn_decay_mean, lrn_decay_lo, lrn_decay_hi, lrn_decay_med = \
        compute_decay_curves(lrn_all_S, wake_schedule, DT)

    # Print summary
    print(f"\n{'='*50}")
    print(f"  Target: mass={tgt_mass_curve[-1]:.1f}, frac_left={tgt_frac_left:.3f}")
    print(f"  Learned: mass={lrn_mass_curve[-1]:.1f}, frac_left={lrn_frac_left:.3f}")
    print(f"\n  Zone fractions:")
    print(f"  {'Zone':<14s} {'Target':>8s} {'Learned':>8s} {'Error':>8s}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8}")
    for z in ZONES:
        err = abs(lrn_zones[z] - tgt_zones[z])
        print(f"  {z:<14s} {tgt_zones[z]:8.3f} {lrn_zones[z]:8.3f} {err:8.3f}")

    # ==================================================================
    # PLOT 1: Mass curves
    # ==================================================================
    print("\nPlotting ...")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_axis, tgt_mass_curve, 'b-', lw=2, label='Target')
    ax.plot(time_axis, lrn_mass_curve, 'r--', lw=2, label='Learned (data-driven conf)')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Total mass M(t)', fontsize=12)
    ax.set_title('Mass accumulation: target vs learned')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "analysis_mass_curves.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ==================================================================
    # PLOT 2: Weighted marginals
    # ==================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(x_bins, x_hist_tgt, 'b-', lw=1.5, label='Target')
    axes[0].plot(x_bins, x_hist_lrn, 'r--', lw=1.5, label='Learned')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('Weighted density', fontsize=12)
    axes[0].set_title(f'x-marginal at t={T_FINAL}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(y_bins, y_hist_tgt, 'b-', lw=1.5, label='Target')
    axes[1].plot(y_bins, y_hist_lrn, 'r--', lw=1.5, label='Learned')
    axes[1].set_xlabel('y', fontsize=12)
    axes[1].set_ylabel('Weighted density', fontsize=12)
    axes[1].set_title(f'y-marginal at t={T_FINAL}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "analysis_marginals.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ==================================================================
    # PLOT 3: Per-particle weight decay
    # ==================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(tgt_tau, tgt_decay_lo, tgt_decay_hi,
                    alpha=0.15, color='blue', label='Target 10-90%')
    ax.plot(tgt_tau, tgt_decay_mean, 'b-', lw=2, label='Target mean')
    ax.fill_between(lrn_tau, lrn_decay_lo, lrn_decay_hi,
                    alpha=0.15, color='red', label='Learned 10-90%')
    ax.plot(lrn_tau, lrn_decay_mean, 'r--', lw=2, label='Learned mean')
    ax.set_xlabel('Time since birth', fontsize=12)
    ax.set_ylabel('Weight $w = e^S$', fontsize=12)
    ax.set_title('Per-particle weight decay (aligned by birth time)')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "analysis_decay_curves.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ==================================================================
    # PLOT 4: Zone weight fractions
    # ==================================================================
    zone_names = list(ZONES.keys())
    zone_colors = ['tab:blue', 'tab:gray', 'tab:orange', 'tab:green', 'tab:red']

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(zone_names))
    width = 0.35
    tgt_vals = [tgt_zones[z] for z in zone_names]
    lrn_vals = [lrn_zones[z] for z in zone_names]

    bars_tgt = ax.bar(x_pos - width/2, tgt_vals, width, label='Target',
                      color=[c for c in zone_colors], alpha=0.6, edgecolor='black')
    bars_lrn = ax.bar(x_pos + width/2, lrn_vals, width, label='Learned',
                      color=[c for c in zone_colors], alpha=0.9, edgecolor='black',
                      hatch='//')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(zone_names, rotation=20, ha='right')
    ax.set_ylabel('Weight fraction', fontsize=12)
    ax.set_title('Zone weight fractions: target vs learned')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add error annotations
    for i, z in enumerate(zone_names):
        err = lrn_zones[z] - tgt_zones[z]
        y_top = max(tgt_vals[i], lrn_vals[i]) + 0.01
        ax.text(i, y_top, f'{err:+.3f}', ha='center', va='bottom', fontsize=9,
                color='red' if abs(err) > 0.03 else 'black')

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "analysis_zones.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ==================================================================
    # PLOT 5: 2D weighted density comparison
    # ==================================================================
    xb = np.linspace(-4, 4, 51)
    yb = np.linspace(-2, 4, 41)

    H_tgt = compute_weighted_hist2d(tgt_z_final, tgt_S_final, xb, yb)
    H_lrn = compute_weighted_hist2d(lrn_z_final, lrn_S_final, xb, yb)

    xc = 0.5 * (xb[:-1] + xb[1:])
    yc = 0.5 * (yb[:-1] + yb[1:])
    Xc, Yc = np.meshgrid(xc, yc)

    vmax = max(H_tgt.max(), H_lrn.max())
    levels = np.linspace(0, vmax, 25)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cf = axes[0].contourf(Xc, Yc, H_tgt, levels=levels, cmap='viridis')
    plt.colorbar(cf, ax=axes[0])
    axes[0].set_title('Target weighted density')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')

    cf = axes[1].contourf(Xc, Yc, H_lrn, levels=levels, cmap='viridis')
    plt.colorbar(cf, ax=axes[1])
    axes[1].set_title('Learned weighted density')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')

    H_diff = H_lrn - H_tgt
    d_absmax = max(abs(H_diff.min()), abs(H_diff.max()))
    dlev = np.linspace(-d_absmax, d_absmax, 25)
    cf = axes[2].contourf(Xc, Yc, H_diff, levels=dlev, cmap='RdBu_r')
    plt.colorbar(cf, ax=axes[2])
    axes[2].set_title('Density difference (learned - target)')
    axes[2].set_xlabel('x'); axes[2].set_ylabel('y')
    axes[2].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "analysis_density_2d.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ==================================================================
    # PLOT 6: Sample trajectories (target vs learned, side by side)
    # ==================================================================
    N_TRAJ = 30
    rng = np.random.RandomState(42)

    # Select early-born particles so trajectories are long enough to see
    wake_np = np.array(wake_schedule)
    early_mask = wake_np < N_STEPS // 3
    early_idx = np.where(early_mask)[0]
    traj_idx = rng.choice(early_idx, size=min(N_TRAJ, len(early_idx)),
                          replace=False)
    traj_idx.sort()

    tgt_z_np = np.array(tgt_all_z)    # (n_steps, N, 2)
    tgt_S_np = np.array(tgt_all_S)    # (n_steps, N)
    lrn_z_np = np.array(lrn_all_z)
    lrn_S_np = np.array(lrn_all_S)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, z_np, S_np, title in [
        (axes[0], tgt_z_np, tgt_S_np, 'Target (ground truth)'),
        (axes[1], lrn_z_np, lrn_S_np, 'Learned (data-driven conf)'),
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

        ax.set_xlim(-4, 4); ax.set_ylim(-2, 3.5)
        ax.set_xlabel('x', fontsize=12); ax.set_ylabel('y', fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label='Weight $w = e^S$')
    plt.suptitle(f'Sample trajectories ({N_TRAJ} particles, colored by weight)',
                 fontsize=14, y=1.01)
    plt.savefig(os.path.join(SCRIPT_DIR, "analysis_trajectories.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------
    # Save analysis data
    # ------------------------------------------------------------------
    np.savez(os.path.join(SCRIPT_DIR, "analysis_data.npz"),
             time_axis=time_axis,
             tgt_mass_curve=tgt_mass_curve,
             lrn_mass_curve=lrn_mass_curve,
             x_bins=x_bins, y_bins=y_bins,
             x_hist_tgt=x_hist_tgt, y_hist_tgt=y_hist_tgt,
             x_hist_lrn=x_hist_lrn, y_hist_lrn=y_hist_lrn,
             tgt_frac_left=tgt_frac_left, lrn_frac_left=lrn_frac_left,
             tgt_decay_tau=tgt_tau, tgt_decay_mean=tgt_decay_mean,
             tgt_decay_lo=tgt_decay_lo, tgt_decay_hi=tgt_decay_hi,
             lrn_decay_tau=lrn_tau, lrn_decay_mean=lrn_decay_mean,
             lrn_decay_lo=lrn_decay_lo, lrn_decay_hi=lrn_decay_hi,
             )

    print("\nDone. Saved:")
    print(f"  {SCRIPT_DIR}/analysis_mass_curves.png")
    print(f"  {SCRIPT_DIR}/analysis_marginals.png")
    print(f"  {SCRIPT_DIR}/analysis_decay_curves.png")
    print(f"  {SCRIPT_DIR}/analysis_zones.png")
    print(f"  {SCRIPT_DIR}/analysis_density_2d.png")
    print(f"  {SCRIPT_DIR}/analysis_trajectories.png")
    print(f"  {SCRIPT_DIR}/analysis_data.npz")
