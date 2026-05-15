#!/usr/bin/env python
"""
sweep_seeds.py — Seed robustness sweep (model 5.1c)

Tests whether learned potential is reproducible across random seeds.

Design (no confounds):
  - Target: FIXED seed=1, N=4000  (same ground-truth for ALL runs)
  - Learned: 10 different seeds    (NN init + training SDE noise)
  - N=4000 for both target and model
  - sigma=1.5 for both target and model
  - epochs=800, patience=400

Each seed_learned controls:
  1. Training particle queue (source positions + wake schedule)
  2. NN weight initialization
  3. SDE Brownian noise during training forward pass

Outputs:
  sweep_seeds/seed_{s}/          — per-seed figures
  sweep_seeds/summary_loss.png
  sweep_seeds/summary_critical_points.png
  sweep_seeds/summary_potential_overlay.png
  sweep_seeds/sweep_results.json  — metrics (replot without re-running)
  sweep_seeds/replot_data.npz     — arrays (replot without re-running)

Usage:
    python sweep_seeds/sweep_seeds.py > sweep_seeds/log_sweep_seeds.txt 2>&1
"""
import matplotlib
matplotlib.use('Agg')

import os
import sys
import json
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np

# Add parent dir so imports work when running from sweep_seeds/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import SourceDeathModel
from models.potential import PotentialNN
from training.data_loader import (
    build_particle_queue, analytical_death_rate_2d, analytical_potential,
)
from simulator.sde_solver import simulate_open_system_full


# ======================================================================
# Constants
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
SIGMA = 1.5              # same for ground-truth AND model
LAM_MASS = 5.0
N_EPOCHS = 800
PRINT_EVERY = 100
PATIENCE = 400

N_PARTICLES = 4000       # same for target AND model
SEED_TARGET = 1          # FIXED — ground-truth never changes
SEEDS_LEARNED = [2, 13, 42, 77, 123, 256, 444, 678, 999, 2025]

BARRIER_Y_SLICE = 0.5
Y_SLICES = [-0.5, 0.5, 1.5]  # for 1D profile overlay

SWEEP_DIR = os.path.dirname(__file__)  # sweep_seeds/


# ======================================================================
# Model components
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
# Ground-truth simulation (analytical)
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
# Loss functions
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
# Critical point extraction
# ======================================================================

def find_critical_points_1d(phi_1d, x_arr):
    result = {
        'x_min_left': None, 'x_min_right': None, 'x_saddle': None,
        'phi_min_left': None, 'phi_min_right': None, 'phi_saddle': None,
        'barrier': None, 'well_asymmetry': None,
    }
    grad = np.diff(phi_1d)
    local_mins = []
    local_maxs = []
    for i in range(len(grad) - 1):
        if grad[i] <= 0 and grad[i + 1] > 0:
            local_mins.append((i + 1, phi_1d[i + 1], x_arr[i + 1]))
        elif grad[i] >= 0 and grad[i + 1] < 0:
            local_maxs.append((i + 1, phi_1d[i + 1], x_arr[i + 1]))
    if len(local_mins) < 2 or len(local_maxs) < 1:
        return result
    local_mins.sort(key=lambda t: t[2])
    by_depth = sorted(local_mins, key=lambda t: t[1])
    two_deepest = sorted(by_depth[:2], key=lambda t: t[2])
    left_min = two_deepest[0]
    right_min = two_deepest[1]
    candidates = [(i, v, xv) for i, v, xv in local_maxs
                  if left_min[0] < i < right_min[0]]
    if not candidates:
        return result
    saddle = max(candidates, key=lambda t: t[1])
    deeper_val = min(left_min[1], right_min[1])
    barrier = saddle[1] - deeper_val
    asymmetry = left_min[1] - right_min[1]
    result.update({
        'x_min_left': float(left_min[2]),
        'x_min_right': float(right_min[2]),
        'x_saddle': float(saddle[2]),
        'phi_min_left': float(left_min[1]),
        'phi_min_right': float(right_min[1]),
        'phi_saddle': float(saddle[1]),
        'barrier': float(barrier),
        'well_asymmetry': float(asymmetry),
    })
    return result


# ======================================================================
# Single training run
# ======================================================================

def run_single(seed_learned, tgt_z, tgt_S, tgt_mass, tgt_all_px, tgt_all_py,
               tgt_all_S_full, out_dir):
    """Train one model with seed_learned. Target data passed in (fixed)."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n  seed_learned={seed_learned}")

    # --- Build TRAINING particle queue (seed_learned controls this) ---
    learn_master = jax.random.PRNGKey(seed_learned)
    queue_key, model_key, opt_key = jax.random.split(learn_master, 3)

    sources = [
        {'mu': jnp.array([0.0, -1.0]), 'sigma_x': 0.25, 'sigma_y': 0.10,
         'n_particles': N_PARTICLES},
    ]
    z_init, wake_schedule = build_particle_queue(
        sources=sources, n_particles=N_PARTICLES,
        t_final=T_FINAL, dt=DT, key=queue_key,
    )

    # --- Create model (seed_learned controls NN init) ---
    phi_nn = PotentialNN(model_key, d_latent=2, c_conf=0.0,
                         hidden_sizes=(16, 16))
    potential = PotentialWithConfinement(
        nn=phi_nn, b=TARGET_PARAMS['b'], d=TARGET_PARAMS['d'],
    )
    fixed_death = FixedDeathRate2D(**DEATH_PARAMS_2D)
    model = SourceDeathModel(potential=potential, death_rate=fixed_death)

    # --- Train (seed_learned controls SDE noise via opt_key) ---
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=3e-3, weight_decay=1e-4),
    )
    objective_fn = make_objective(N_STEPS, DT, SIGMA, LAM_MASS)
    step_fn = make_train_step(optimizer, objective_fn)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    history = {'loss': [], 'mmd': [], 'mass': []}
    best_loss = float('inf')
    best_epoch = 0
    best_model_leaves = None
    wait = 0

    for epoch in range(N_EPOCHS):
        opt_key, sim_key = jax.random.split(opt_key)
        model, opt_state, loss, (l_mmd, l_mass) = step_fn(
            model, opt_state, z_init, wake_schedule,
            tgt_z, tgt_S, tgt_mass, sim_key,
        )
        loss_val = float(loss)
        history['loss'].append(loss_val)
        history['mmd'].append(float(l_mmd))
        history['mass'].append(float(l_mass))

        if epoch % PRINT_EVERY == 0:
            print(f"    Epoch {epoch:4d}: Loss={loss_val:.5f} "
                  f"(MMD={float(l_mmd):.5f}, Mass={float(l_mass):.5f})")

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

    # --- Evaluate: 1D potential profiles ---
    x_line = np.linspace(-4, 4, 500)
    ref_true = float(analytical_potential(0.0, 0.0, TARGET_PARAMS))
    ref_learned = float(model.potential(jnp.array([0.0, 0.0])))

    profiles_learned = {}
    for y_val in Y_SLICES:
        phi_l = np.array(jax.vmap(
            lambda xi: model.potential(jnp.array([xi, y_val]))
        )(jnp.array(x_line))) - ref_learned
        profiles_learned[y_val] = phi_l

    # --- Evaluate: critical points ---
    y_sl = BARRIER_Y_SLICE
    phi_learned_barrier = profiles_learned[y_sl] if y_sl in profiles_learned else \
        np.array(jax.vmap(
            lambda xi: model.potential(jnp.array([xi, y_sl]))
        )(jnp.array(x_line))) - ref_learned

    learned_cp = find_critical_points_1d(phi_learned_barrier, x_line)

    if learned_cp['barrier'] is not None:
        print(f"    Learned CP: x_L={learned_cp['x_min_left']:.3f}, "
              f"x_R={learned_cp['x_min_right']:.3f}, "
              f"x_S={learned_cp['x_saddle']:.3f}, "
              f"barrier={learned_cp['barrier']:.3f}")
    else:
        print(f"    Learned CP: no double-well found")

    # --- Per-seed plots ---

    # 1. Training curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(history['loss'], label='Total', linewidth=1.5)
    ax.semilogy(history['mmd'], label='MMD', linewidth=1, alpha=0.8)
    ax.semilogy(history['mass'], label='Mass', linewidth=1, alpha=0.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title(f'seed_learned={seed_learned} (seed_target={SEED_TARGET})')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Landscape: contours + 1D slices
    x_grid = jnp.linspace(-4, 4, 200)
    y_grid = jnp.linspace(-2, 3.5, 200)
    Xg, Yg = jnp.meshgrid(x_grid, y_grid)

    Z_true = jax.vmap(jax.vmap(
        lambda x, y: analytical_potential(x, y, TARGET_PARAMS)
    ))(Xg, Yg)
    Z_learned = jax.vmap(
        lambda row_y: jax.vmap(
            lambda xi: model.potential(jnp.array([xi, row_y]))
        )(x_grid)
    )(y_grid)

    Z_true_a = Z_true - ref_true
    Z_learned_a = Z_learned - ref_learned

    pot_vmin = min(float(jnp.min(Z_true_a)), float(jnp.min(Z_learned_a)))
    pot_vmax = min(max(float(jnp.max(Z_true_a)), float(jnp.max(Z_learned_a))), 30.0)
    pot_levels = np.linspace(pot_vmin, pot_vmax, 31)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    cf = axes[0, 0].contourf(np.array(Xg), np.array(Yg), np.array(Z_learned_a),
                              levels=pot_levels, cmap='viridis', extend='both')
    plt.colorbar(cf, ax=axes[0, 0], label=r'$\hat\Phi$')
    axes[0, 0].set_title(f'Learned (seed={seed_learned})')
    axes[0, 0].set_xlabel('x'); axes[0, 0].set_ylabel('y')

    cf = axes[0, 1].contourf(np.array(Xg), np.array(Yg), np.array(Z_true_a),
                              levels=pot_levels, cmap='viridis', extend='both')
    plt.colorbar(cf, ax=axes[0, 1], label=r'$\Phi^*$')
    axes[0, 1].set_title('True Potential')
    axes[0, 1].set_xlabel('x'); axes[0, 1].set_ylabel('y')

    Z_diff = np.array(Z_learned_a) - np.array(Z_true_a)
    d_absmax = min(max(abs(Z_diff.min()), abs(Z_diff.max())), 15.0)
    diff_levels = np.linspace(-d_absmax, d_absmax, 31)
    cf = axes[0, 2].contourf(np.array(Xg), np.array(Yg), Z_diff,
                              levels=diff_levels, cmap='RdBu_r', extend='both')
    plt.colorbar(cf, ax=axes[0, 2], label=r'$\hat\Phi - \Phi^*$')
    axes[0, 2].set_title('Error')
    axes[0, 2].set_xlabel('x'); axes[0, 2].set_ylabel('y')

    for col, y_val in enumerate(Y_SLICES):
        ax = axes[1, col]
        phi_t = np.array([
            float(analytical_potential(float(xi), y_val, TARGET_PARAMS))
            for xi in x_line
        ]) - ref_true
        phi_l = profiles_learned[y_val]
        ax.plot(x_line, phi_t, 'b-', linewidth=2, label='True')
        ax.plot(x_line, phi_l, 'r--', linewidth=2, label='Learned')
        ax.set_xlabel('x'); ax.set_ylabel(r'$\Phi$')
        ax.set_title(f'y = {y_val}')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(-15, 15)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "landscape.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Snapshot: target vs learned at t=1,2,4,6
    VIS_TIMES = [1.0, 2.0, 4.0, 6.0]
    VIS_STEPS = [int(t / DT) - 1 for t in VIS_TIMES]

    # Simulate learned model for snapshot visualization
    snap_key = jax.random.PRNGKey(seed_learned + 50000)
    snap_queue_key, snap_sim_key = jax.random.split(snap_key)
    z_init_snap, wake_snap = build_particle_queue(
        sources=sources, n_particles=N_PARTICLES,
        t_final=T_FINAL, dt=DT, key=snap_queue_key,
    )
    sim_all_z, sim_all_S = simulate_open_system_full(
        model, z_init_snap, wake_snap, SIGMA, N_STEPS, DT, snap_sim_key
    )

    fig, axes_snap = plt.subplots(2, 4, figsize=(20, 10))
    for col, (t_vis, step_vis) in enumerate(zip(VIS_TIMES, VIS_STEPS)):
        # Top row: target
        ax_tgt = axes_snap[0, col]
        tgt_z_i = jnp.column_stack([tgt_all_px[step_vis], tgt_all_py[step_vis]])
        tgt_S_i = tgt_all_S_full[step_vis]
        tgt_w_i = np.array(jnp.clip(jnp.exp(tgt_S_i), 0.0, 1.0))
        alive = tgt_w_i > 1e-6
        ax_tgt.contourf(np.array(Xg), np.array(Yg), np.array(Z_true_a),
                        levels=pot_levels, cmap='viridis', alpha=0.3)
        ax_tgt.scatter(np.array(tgt_z_i[alive, 0]), np.array(tgt_z_i[alive, 1]),
                       c=tgt_w_i[alive], cmap='coolwarm', s=6, alpha=0.5,
                       vmin=0, vmax=1, edgecolors='none')
        ax_tgt.set_xlim(-4, 4); ax_tgt.set_ylim(-2, 3.5)
        ax_tgt.set_title(f'Target t={t_vis:.0f}')
        if col == 0:
            ax_tgt.set_ylabel('y (target)')

        # Bottom row: learned
        ax_sim = axes_snap[1, col]
        sim_z_i = sim_all_z[step_vis]
        sim_S_i = sim_all_S[step_vis]
        sim_w_i = np.array(jnp.clip(jnp.exp(sim_S_i), 0.0, 1.0))
        alive_sim = sim_w_i > 1e-6
        ax_sim.contourf(np.array(Xg), np.array(Yg), np.array(Z_learned_a),
                        levels=pot_levels, cmap='viridis', alpha=0.3)
        sc = ax_sim.scatter(
            np.array(sim_z_i[alive_sim, 0]), np.array(sim_z_i[alive_sim, 1]),
            c=sim_w_i[alive_sim], cmap='coolwarm', s=6, alpha=0.5,
            vmin=0, vmax=1, edgecolors='none')
        ax_sim.set_xlim(-4, 4); ax_sim.set_ylim(-2, 3.5)
        ax_sim.set_title(f'Learned t={t_vis:.0f}')
        if col == 0:
            ax_sim.set_ylabel('y (learned)')

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label=r'$w = e^S$')
    plt.savefig(os.path.join(out_dir, "snapshots.png"), dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'seed_learned': seed_learned,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'final_mmd': history['mmd'][-1],
        'final_mass': history['mass'][-1],
        'learned_cp': learned_cp,
        'history_loss': history['loss'],
        'history_mmd': history['mmd'],
        'history_mass': history['mass'],
        'profiles_learned': {str(k): v.tolist() for k, v in profiles_learned.items()},
        'ref_learned': ref_learned,
    }


# ======================================================================
# Main
# ======================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("Seed robustness sweep: model 5.1c")
    print("=" * 65)
    print(f"  seed_target  = {SEED_TARGET} (FIXED for all runs)")
    print(f"  seeds_learned = {SEEDS_LEARNED}")
    print(f"  N_particles  = {N_PARTICLES} (same for target AND model)")
    print(f"  sigma        = {SIGMA} (same for target AND model)")
    print(f"  epochs={N_EPOCHS}, patience={PATIENCE}, lam_mass={LAM_MASS}")
    print(f"  Barrier y-slice = {BARRIER_Y_SLICE}")
    print(f"  Total runs: {len(SEEDS_LEARNED)}")
    print("=" * 65)

    # ==================================================================
    # Generate ground-truth ONCE (seed_target=1, N=4000)
    # ==================================================================
    print("\nGenerating ground-truth (seed_target=1, N=4000) ...")

    tgt_master = jax.random.PRNGKey(SEED_TARGET)
    tgt_queue_key, tgt_sim_key = jax.random.split(tgt_master)

    sources_tgt = [
        {'mu': jnp.array([0.0, -1.0]), 'sigma_x': 0.25, 'sigma_y': 0.10,
         'n_particles': N_PARTICLES},
    ]
    tgt_z_init, tgt_wake = build_particle_queue(
        sources=sources_tgt, n_particles=N_PARTICLES,
        t_final=T_FINAL, dt=DT, key=tgt_queue_key,
    )
    tgt_all_px, tgt_all_py, tgt_all_S_full = simulate_ground_truth_full(
        tgt_z_init, tgt_wake, N_STEPS, DT, tgt_sim_key
    )

    final_step = N_STEPS - 1
    tgt_z = jnp.column_stack([tgt_all_px[final_step], tgt_all_py[final_step]])
    tgt_S = tgt_all_S_full[final_step]
    tgt_mass = float(jnp.sum(jnp.exp(tgt_S)))
    n_alive = int(jnp.sum(jnp.exp(tgt_S) > 0.01))
    print(f"  Target: mass={tgt_mass:.1f}, ~{n_alive} alive")

    # True critical points (same for all runs)
    x_line = np.linspace(-4, 4, 500)
    ref_true = float(analytical_potential(0.0, 0.0, TARGET_PARAMS))
    true_profiles = {}
    for y_val in Y_SLICES:
        phi_t = np.array([
            float(analytical_potential(float(xi), y_val, TARGET_PARAMS))
            for xi in x_line
        ]) - ref_true
        true_profiles[y_val] = phi_t

    true_cp = find_critical_points_1d(true_profiles[BARRIER_Y_SLICE], x_line)
    print(f"  True CP: x_L={true_cp['x_min_left']:.3f}, "
          f"x_R={true_cp['x_min_right']:.3f}, "
          f"x_S={true_cp['x_saddle']:.3f}, "
          f"barrier={true_cp['barrier']:.3f}, "
          f"asym={true_cp['well_asymmetry']:.3f}")

    # ==================================================================
    # Run all seeds
    # ==================================================================
    all_results = []

    for seed_learned in SEEDS_LEARNED:
        out_dir = os.path.join(SWEEP_DIR, f"seed_{seed_learned}")
        res = run_single(
            seed_learned, tgt_z, tgt_S, tgt_mass,
            tgt_all_px, tgt_all_py, tgt_all_S_full, out_dir
        )
        all_results.append(res)

        # Save intermediate results (without large arrays)
        json_safe = []
        for r in all_results:
            r_safe = {k: v for k, v in r.items()
                      if k not in ('history_loss', 'history_mmd',
                                   'history_mass', 'profiles_learned')}
            json_safe.append(r_safe)
        with open(os.path.join(SWEEP_DIR, "sweep_results.json"), 'w') as f:
            json.dump(json_safe, f, indent=2, default=str)

    # ==================================================================
    # Save full replot data (numpy arrays)
    # ==================================================================
    save_dict = {
        'x_line': x_line,
        'seeds_learned': np.array(SEEDS_LEARNED),
        'seed_target': np.array(SEED_TARGET),
    }
    # True profiles
    for y_val in Y_SLICES:
        save_dict[f'true_profile_y{y_val}'] = true_profiles[y_val]
    # Per-seed: training history + learned profiles
    for r in all_results:
        s = r['seed_learned']
        save_dict[f'loss_seed{s}'] = np.array(r['history_loss'])
        save_dict[f'mmd_seed{s}'] = np.array(r['history_mmd'])
        save_dict[f'mass_seed{s}'] = np.array(r['history_mass'])
        for y_val in Y_SLICES:
            save_dict[f'profile_seed{s}_y{y_val}'] = np.array(
                r['profiles_learned'][str(y_val)])

    np.savez(os.path.join(SWEEP_DIR, "replot_data.npz"), **save_dict)
    print(f"\nReplot data saved to {SWEEP_DIR}/replot_data.npz")

    # Save full JSON (with histories)
    json_full = []
    for r in all_results:
        json_full.append({
            'seed_learned': r['seed_learned'],
            'best_loss': r['best_loss'],
            'best_epoch': r['best_epoch'],
            'final_mmd': r['final_mmd'],
            'final_mass': r['final_mass'],
            'learned_cp': r['learned_cp'],
            'ref_learned': r['ref_learned'],
        })
    with open(os.path.join(SWEEP_DIR, "sweep_results.json"), 'w') as f:
        json.dump({
            'config': {
                'seed_target': SEED_TARGET,
                'n_particles': N_PARTICLES,
                'sigma': SIGMA,
                'n_epochs': N_EPOCHS,
                'patience': PATIENCE,
                'lam_mass': LAM_MASS,
                'barrier_y_slice': BARRIER_Y_SLICE,
            },
            'true_cp': true_cp,
            'runs': json_full,
        }, f, indent=2, default=str)

    # ==================================================================
    # Summary plot 1: Loss across seeds
    # ==================================================================
    seeds = [r['seed_learned'] for r in all_results]
    losses = [r['best_loss'] for r in all_results]
    mmds = [r['final_mmd'] for r in all_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: bar plot of best loss per seed
    x_pos = np.arange(len(seeds))
    axes[0].bar(x_pos, losses, color='tab:blue', alpha=0.8, edgecolor='black')
    axes[0].axhline(np.mean(losses), color='red', linestyle='--', linewidth=1.5,
                    label=f'mean={np.mean(losses):.4f}±{np.std(losses):.4f}')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(seeds, rotation=45, fontsize=9)
    axes[0].set_xlabel('seed_learned', fontsize=12)
    axes[0].set_ylabel('Best loss', fontsize=12)
    axes[0].set_title('Best loss per seed', fontsize=13)
    axes[0].legend(fontsize=10); axes[0].grid(True, alpha=0.3, axis='y')

    # Right: overlaid training curves
    for r in all_results:
        axes[1].semilogy(r['history_loss'], linewidth=1, alpha=0.7,
                         label=f"s={r['seed_learned']}")
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Total loss', fontsize=12)
    axes[1].set_title('Training curves (all seeds)', fontsize=13)
    axes[1].legend(fontsize=8, ncol=2); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SWEEP_DIR, "summary_loss.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ==================================================================
    # Summary plot 2: Critical points across seeds
    # ==================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    def plot_cp_bar(ax, key, true_val, ylabel, title):
        vals = []
        valid_seeds = []
        for r in all_results:
            v = r['learned_cp'].get(key)
            if v is not None:
                vals.append(v)
                valid_seeds.append(r['seed_learned'])
        if not vals:
            ax.set_title(f'{title} (no data)')
            return
        x_pos = np.arange(len(vals))
        ax.bar(x_pos, vals, color='tab:red', alpha=0.7, edgecolor='black')
        if true_val is not None:
            ax.axhline(true_val, color='blue', linestyle='--', linewidth=2,
                        label=f'True = {true_val:.3f}')
        ax.axhline(np.mean(vals), color='red', linestyle=':', linewidth=1.5,
                    label=f'mean={np.mean(vals):.3f}±{np.std(vals):.3f}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(valid_seeds, rotation=45, fontsize=9)
        ax.set_xlabel('seed_learned', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

    plot_cp_bar(axes[0, 0], 'x_min_left', true_cp['x_min_left'],
                r'$x_{min,left}$', 'Left minimum position')
    plot_cp_bar(axes[0, 1], 'x_min_right', true_cp['x_min_right'],
                r'$x_{min,right}$', 'Right minimum position')
    plot_cp_bar(axes[1, 0], 'x_saddle', true_cp['x_saddle'],
                r'$x_{saddle}$', 'Saddle position')
    plot_cp_bar(axes[1, 1], 'barrier', true_cp['barrier'],
                r'$\Delta\Phi$', f'Barrier height (y={BARRIER_Y_SLICE})')

    plt.tight_layout()
    plt.savefig(os.path.join(SWEEP_DIR, "summary_critical_points.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ==================================================================
    # Summary plot 3: Overlaid 1D potential profiles (KEY plot)
    # ==================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for col, y_val in enumerate(Y_SLICES):
        ax = axes[col]
        # True potential (thick blue)
        ax.plot(x_line, true_profiles[y_val], 'b-', linewidth=3,
                label='True', zorder=10)
        # All learned seeds (thin colored lines)
        for i, r in enumerate(all_results):
            prof = np.array(r['profiles_learned'][str(y_val)])
            ax.plot(x_line, prof, color=colors[i], linewidth=1, alpha=0.7,
                    label=f"s={r['seed_learned']}")
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel(r'$\Phi(x, y)$', fontsize=12)
        ax.set_title(f'y = {y_val}', fontsize=13)
        ax.set_ylim(-15, 15)
        ax.grid(True, alpha=0.3)
        if col == 2:
            ax.legend(fontsize=7, ncol=2, loc='upper right')

    plt.suptitle(
        f'Potential profiles across {len(SEEDS_LEARNED)} seeds '
        f'(seed_target={SEED_TARGET}, N={N_PARTICLES})',
        fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SWEEP_DIR, "summary_potential_overlay.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ==================================================================
    # Summary table
    # ==================================================================
    print(f"\n{'='*80}")
    print(f"Summary (y-slice={BARRIER_Y_SLICE})")
    print(f"  seed_target={SEED_TARGET}, N={N_PARTICLES}, sigma={SIGMA}")
    print(f"  True: x_L={true_cp['x_min_left']:.3f}, "
          f"x_R={true_cp['x_min_right']:.3f}, "
          f"x_S={true_cp['x_saddle']:.3f}, "
          f"barrier={true_cp['barrier']:.3f}, "
          f"asym={true_cp['well_asymmetry']:.3f}")
    print(f"{'='*80}")
    print(f"{'Seed':>6s}  {'Loss':>10s}  {'Epoch':>6s}  {'Barrier':>10s}  "
          f"{'x_left':>10s}  {'x_right':>10s}  {'x_saddle':>10s}  {'Asym':>10s}")
    print("-" * 80)
    for r in all_results:
        cp = r['learned_cp']
        def fmt(v):
            return f"{v:.4f}" if v is not None else "N/A"
        print(f"{r['seed_learned']:6d}  {r['best_loss']:10.6f}  "
              f"{r['best_epoch']:6d}  {fmt(cp['barrier']):>10s}  "
              f"{fmt(cp['x_min_left']):>10s}  {fmt(cp['x_min_right']):>10s}  "
              f"{fmt(cp['x_saddle']):>10s}  {fmt(cp['well_asymmetry']):>10s}")

    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    barriers = [r['learned_cp']['barrier'] for r in all_results
                if r['learned_cp']['barrier'] is not None]
    print(f"\nLoss: {mean_loss:.6f} ± {std_loss:.6f} "
          f"(CV = {std_loss/mean_loss*100:.1f}%)")
    if barriers:
        print(f"Barrier: {np.mean(barriers):.4f} ± {np.std(barriers):.4f} "
              f"(true={true_cp['barrier']:.4f})")

    print(f"\nResults saved to {SWEEP_DIR}/")
    print(f"  sweep_results.json  — metrics")
    print(f"  replot_data.npz     — arrays for replotting")
    print(f"  summary_loss.png")
    print(f"  summary_critical_points.png")
    print(f"  summary_potential_overlay.png")
