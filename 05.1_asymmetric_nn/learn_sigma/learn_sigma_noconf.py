#!/usr/bin/env python
"""
learn_sigma_noconf.py — Test learnability of sigma WITHOUT confinement

Compares:
  - Fixed sigma=1.5   (pure NN potential, no confinement)
  - Learnable sigma    (pure NN potential, no confinement, init=1.0)

Design:
  - Target: seed=1, N=1000, sigma_true=1.5 (with exact confinement)
  - Seeds: [2, 42, 256] (same for both conditions)
  - NO confinement: Phi = NN only (c_conf=0, no quartic wrapper)
  - Learnable sigma: sigma = exp(log_sigma), init log_sigma = log(1.0) = 0
  - epochs=1500, patience=600

Outputs:
  learn_sigma/noconf_fixed_seed_{s}/  — per-seed figures (fixed sigma, no conf)
  learn_sigma/noconf_learn_seed_{s}/  — per-seed figures (learnable sigma, no conf)
  learn_sigma/summary_*_noconf.png    — comparison plots
  learn_sigma/results_noconf.json     — metrics
  learn_sigma/replot_data_noconf.npz  — arrays for replotting

Usage:
    python learn_sigma/learn_sigma_noconf.py > learn_sigma/log_noconf.txt 2>&1
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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
SIGMA_TRUE = 1.5
SIGMA_INIT = 1.0          # learnable sigma starts here
LAM_MASS = 5.0
N_EPOCHS = 1500
PRINT_EVERY = 50
PATIENCE = 600

N_PARTICLES = 1000
SEED_TARGET = 1
SEEDS_LEARNED = [2, 42, 256]

BARRIER_Y_SLICE = 0.5
Y_SLICES = [-0.5, 0.5, 1.5]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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


class SourceDeathModelPureNN(eqx.Module):
    """SourceDeathModel with pure NN potential (no confinement), fixed sigma."""
    potential: PotentialNN
    death_rate: FixedDeathRate2D


class SourceDeathModelPureNNLearnableSigma(eqx.Module):
    """SourceDeathModel with pure NN potential + learnable sigma."""
    potential: PotentialNN
    death_rate: FixedDeathRate2D
    log_sigma: jax.Array     # learnable scalar

    @property
    def sigma(self):
        return jnp.exp(self.log_sigma)


# ======================================================================
# Ground-truth simulation (with exact confinement — target is unchanged)
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
        new_px = px + fx * dt + SIGMA_TRUE * nx
        new_py = py + fy * dt + SIGMA_TRUE * ny
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


def make_objective_fixed_sigma(n_steps, dt, sigma, lam_mass):
    """Objective with fixed sigma (not learnable)."""
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


def make_objective_learnable_sigma(n_steps, dt, lam_mass):
    """Objective where sigma comes from model.log_sigma (learnable)."""
    def objective(model, z_init, wake_schedule, tgt_z, tgt_S, tgt_mass, key):
        sigma = jnp.exp(model.log_sigma)
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
        return total, (l_mmd, l_mass, sigma)
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
# Shared: build queue + create pure NN (same keys for fair comparison)
# ======================================================================

def _build_queue_and_nn(seed_learned):
    """Deterministic queue + NN init from seed_learned."""
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
    # Pure NN — no confinement at all
    phi_nn = PotentialNN(model_key, d_latent=2, c_conf=0.0,
                         hidden_sizes=(16, 16))
    return z_init, wake_schedule, phi_nn, opt_key, sources


def _evaluate_model(potential_fn):
    """Compute profiles and critical points from a trained potential."""
    x_line = np.linspace(-4, 4, 500)
    ref_learned = float(potential_fn(jnp.array([0.0, 0.0])))

    profiles = {}
    for y_val in Y_SLICES:
        phi_l = np.array(jax.vmap(
            lambda xi: potential_fn(jnp.array([xi, y_val]))
        )(jnp.array(x_line))) - ref_learned
        profiles[y_val] = phi_l

    cp = find_critical_points_1d(profiles[BARRIER_Y_SLICE], x_line)
    return profiles, cp, ref_learned


# ======================================================================
# Single training run (fixed sigma, no confinement)
# ======================================================================

def run_fixed_sigma(seed_learned, tgt_z, tgt_S, tgt_mass,
                    tgt_all_px, tgt_all_py, tgt_all_S_full, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n  [Fixed sigma={SIGMA_TRUE}, NO confinement] "
          f"seed_learned={seed_learned}")

    z_init, wake_schedule, phi_nn, opt_key, sources = \
        _build_queue_and_nn(seed_learned)

    fixed_death = FixedDeathRate2D(**DEATH_PARAMS_2D)
    model = SourceDeathModelPureNN(potential=phi_nn, death_rate=fixed_death)

    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"    Total learnable params: {n_params}")

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=3e-3, weight_decay=1e-4),
    )
    objective_fn = make_objective_fixed_sigma(N_STEPS, DT, SIGMA_TRUE, LAM_MASS)
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

    profiles, cp, ref_learned = _evaluate_model(model.potential)
    if cp['barrier'] is not None:
        print(f"    CP: barrier={cp['barrier']:.3f}")
    else:
        print(f"    CP: no double-well found")

    # Training curve plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(history['loss'], label='Total', linewidth=1.5)
    ax.semilogy(history['mmd'], label='MMD', linewidth=1, alpha=0.8)
    ax.semilogy(history['mass'], label='Mass', linewidth=1, alpha=0.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title(f'Fixed sigma={SIGMA_TRUE}, no conf (seed={seed_learned})')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training.png"), dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'seed_learned': seed_learned,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'sigma': SIGMA_TRUE,
        'learned_cp': cp,
        'history_loss': history['loss'],
        'history_mmd': history['mmd'],
        'history_mass': history['mass'],
        'profiles': {str(k): v for k, v in profiles.items()},
        'ref_learned': ref_learned,
    }


# ======================================================================
# Single training run (learnable sigma, no confinement)
# ======================================================================

def run_learnable_sigma(seed_learned, tgt_z, tgt_S, tgt_mass,
                        tgt_all_px, tgt_all_py, tgt_all_S_full, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n  [Learnable sigma, NO confinement] "
          f"seed_learned={seed_learned}, sigma_init={SIGMA_INIT}")

    z_init, wake_schedule, phi_nn, opt_key, sources = \
        _build_queue_and_nn(seed_learned)

    fixed_death = FixedDeathRate2D(**DEATH_PARAMS_2D)
    model = SourceDeathModelPureNNLearnableSigma(
        potential=phi_nn,
        death_rate=fixed_death,
        log_sigma=jnp.array(jnp.log(SIGMA_INIT)),  # init at log(1.0) = 0
    )

    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"    Total learnable params: {n_params} "
          f"(337 Phi_nn + 1 log_sigma = 338)")

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=3e-3, weight_decay=1e-4),
    )
    objective_fn = make_objective_learnable_sigma(N_STEPS, DT, LAM_MASS)
    step_fn = make_train_step(optimizer, objective_fn)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    history = {'loss': [], 'mmd': [], 'mass': [], 'sigma': []}
    best_loss = float('inf')
    best_epoch = 0
    best_model_leaves = None
    best_sigma = SIGMA_INIT
    wait = 0

    for epoch in range(N_EPOCHS):
        opt_key, sim_key = jax.random.split(opt_key)
        model, opt_state, loss, (l_mmd, l_mass, sigma_val) = step_fn(
            model, opt_state, z_init, wake_schedule,
            tgt_z, tgt_S, tgt_mass, sim_key,
        )
        loss_val = float(loss)
        sigma_now = float(sigma_val)
        history['loss'].append(loss_val)
        history['mmd'].append(float(l_mmd))
        history['mass'].append(float(l_mass))
        history['sigma'].append(sigma_now)

        if epoch % PRINT_EVERY == 0:
            print(f"    Epoch {epoch:4d}: Loss={loss_val:.5f} "
                  f"(MMD={float(l_mmd):.5f}, Mass={float(l_mass):.5f}) "
                  f"sigma={sigma_now:.4f}")

        if loss_val < best_loss - 1e-6:
            best_loss = loss_val
            best_epoch = epoch
            best_sigma = sigma_now
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
    final_sigma = float(jnp.exp(model.log_sigma))
    print(f"    Best epoch {best_epoch}, loss={best_loss:.6f}, "
          f"sigma={final_sigma:.4f} (true={SIGMA_TRUE})")

    profiles, cp, ref_learned = _evaluate_model(model.potential)
    if cp['barrier'] is not None:
        print(f"    CP: x_L={cp['x_min_left']:.3f}, "
              f"x_R={cp['x_min_right']:.3f}, "
              f"barrier={cp['barrier']:.3f}")
    else:
        print(f"    CP: no double-well found")

    # --- Per-seed plots ---
    x_line = np.linspace(-4, 4, 500)
    x_grid = jnp.linspace(-4, 4, 200)
    y_grid = jnp.linspace(-2, 3.5, 200)
    Xg, Yg = jnp.meshgrid(x_grid, y_grid)

    ref_true = float(analytical_potential(0.0, 0.0, TARGET_PARAMS))
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

    # Training + sigma trajectory
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].semilogy(history['loss'], label='Total', linewidth=1.5)
    axes[0].semilogy(history['mmd'], label='MMD', linewidth=1, alpha=0.8)
    axes[0].semilogy(history['mass'], label='Mass', linewidth=1, alpha=0.8)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Training, no conf (seed={seed_learned})')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['sigma'], 'r-', linewidth=2)
    axes[1].axhline(SIGMA_TRUE, color='blue', linestyle='--', linewidth=1.5,
                    label=f'True sigma={SIGMA_TRUE}')
    axes[1].axhline(SIGMA_INIT, color='gray', linestyle=':', linewidth=1,
                    label=f'Init sigma={SIGMA_INIT}')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel(r'$\sigma$')
    axes[1].set_title(f'Sigma trajectory (seed={seed_learned})')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Landscape + slices
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    cf = axes[0, 0].contourf(np.array(Xg), np.array(Yg), np.array(Z_learned_a),
                              levels=pot_levels, cmap='viridis', extend='both')
    plt.colorbar(cf, ax=axes[0, 0], label=r'$\hat\Phi$')
    axes[0, 0].set_title(f'Learned no conf (seed={seed_learned}, '
                         f'$\\sigma$={final_sigma:.3f})')
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
        ax.plot(x_line, phi_t, 'b-', linewidth=2, label='True')
        ax.plot(x_line, profiles[y_val], 'r--', linewidth=2,
                label=f'Learned ($\\sigma$={final_sigma:.2f})')
        ax.set_xlabel('x'); ax.set_ylabel(r'$\Phi$')
        ax.set_title(f'y = {y_val}')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(-15, 15)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "landscape.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Snapshot: target vs learned
    VIS_TIMES = [1.0, 2.0, 4.0, 6.0]
    VIS_STEPS = [int(t / DT) - 1 for t in VIS_TIMES]
    snap_key = jax.random.PRNGKey(seed_learned + 50000)
    snap_queue_key, snap_sim_key = jax.random.split(snap_key)
    snap_sources = [
        {'mu': jnp.array([0.0, -1.0]), 'sigma_x': 0.25, 'sigma_y': 0.10,
         'n_particles': N_PARTICLES},
    ]
    z_init_snap, wake_snap = build_particle_queue(
        sources=snap_sources, n_particles=N_PARTICLES,
        t_final=T_FINAL, dt=DT, key=snap_queue_key,
    )
    sim_all_z, sim_all_S = simulate_open_system_full(
        model, z_init_snap, wake_snap, final_sigma, N_STEPS, DT, snap_sim_key
    )
    fig, axes_snap = plt.subplots(2, 4, figsize=(20, 10))
    for col, (t_vis, step_vis) in enumerate(zip(VIS_TIMES, VIS_STEPS)):
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
        ax_sim.set_title(f'Learned t={t_vis:.0f} ($\\sigma$={final_sigma:.2f})')
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
        'final_sigma': final_sigma,
        'best_sigma': best_sigma,
        'learned_cp': cp,
        'history_loss': history['loss'],
        'history_mmd': history['mmd'],
        'history_mass': history['mass'],
        'history_sigma': history['sigma'],
        'profiles': {str(k): v for k, v in profiles.items()},
        'ref_learned': ref_learned,
    }


# ======================================================================
# Main
# ======================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("Learnable sigma experiment (NO confinement — pure NN)")
    print("=" * 65)
    print(f"  seed_target   = {SEED_TARGET}")
    print(f"  seeds_learned = {SEEDS_LEARNED}")
    print(f"  N_particles   = {N_PARTICLES} (target AND model)")
    print(f"  sigma_true    = {SIGMA_TRUE}")
    print(f"  sigma_init    = {SIGMA_INIT} (learnable, start value)")
    print(f"  Confinement   = NONE (pure NN, c_conf=0)")
    print(f"  epochs={N_EPOCHS}, patience={PATIENCE}, lam_mass={LAM_MASS}")
    print(f"  Total runs: {len(SEEDS_LEARNED)} fixed + {len(SEEDS_LEARNED)} learnable"
          f" = {2 * len(SEEDS_LEARNED)}")
    print("=" * 65)

    # ==================================================================
    # Generate ground-truth ONCE (target always uses exact potential)
    # ==================================================================
    print(f"\nGenerating ground-truth (seed={SEED_TARGET}, N={N_PARTICLES},"
          f" sigma={SIGMA_TRUE}) ...")
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

    # True critical points
    x_line = np.linspace(-4, 4, 500)
    ref_true = float(analytical_potential(0.0, 0.0, TARGET_PARAMS))
    true_profiles = {}
    for y_val in Y_SLICES:
        true_profiles[y_val] = np.array([
            float(analytical_potential(float(xi), y_val, TARGET_PARAMS))
            for xi in x_line
        ]) - ref_true
    true_cp = find_critical_points_1d(true_profiles[BARRIER_Y_SLICE], x_line)
    print(f"  True CP: barrier={true_cp['barrier']:.3f}, "
          f"asym={true_cp['well_asymmetry']:.3f}")

    # ==================================================================
    # Train FIXED-sigma models (3 seeds, no confinement)
    # ==================================================================
    print(f"\n{'='*65}")
    print("Phase 1: Fixed sigma, NO confinement")
    print(f"{'='*65}")
    fixed_results = {}

    for seed_learned in SEEDS_LEARNED:
        out_dir = os.path.join(SCRIPT_DIR, f"noconf_fixed_seed_{seed_learned}")
        res = run_fixed_sigma(
            seed_learned, tgt_z, tgt_S, tgt_mass,
            tgt_all_px, tgt_all_py, tgt_all_S_full, out_dir
        )
        fixed_results[seed_learned] = res

    # ==================================================================
    # Train LEARNABLE-sigma models (3 seeds, no confinement)
    # ==================================================================
    print(f"\n{'='*65}")
    print("Phase 2: Learnable sigma, NO confinement")
    print(f"{'='*65}")
    learnable_results = []

    for seed_learned in SEEDS_LEARNED:
        out_dir = os.path.join(SCRIPT_DIR, f"noconf_learn_seed_{seed_learned}")
        res = run_learnable_sigma(
            seed_learned, tgt_z, tgt_S, tgt_mass,
            tgt_all_px, tgt_all_py, tgt_all_S_full, out_dir
        )
        learnable_results.append(res)

    # ==================================================================
    # Save replot data
    # ==================================================================
    save_dict = {
        'x_line': x_line,
        'seeds_learned': np.array(SEEDS_LEARNED),
        'sigma_true': np.array(SIGMA_TRUE),
        'sigma_init': np.array(SIGMA_INIT),
    }
    for y_val in Y_SLICES:
        save_dict[f'true_profile_y{y_val}'] = true_profiles[y_val]
    for r in learnable_results:
        s = r['seed_learned']
        save_dict[f'learn_loss_seed{s}'] = np.array(r['history_loss'])
        save_dict[f'learn_sigma_seed{s}'] = np.array(r['history_sigma'])
        for y_val in Y_SLICES:
            save_dict[f'learn_profile_seed{s}_y{y_val}'] = np.array(
                r['profiles'][str(y_val)])
    for s in SEEDS_LEARNED:
        fr = fixed_results[s]
        save_dict[f'fixed_loss_seed{s}'] = np.array(fr['history_loss'])
        for y_val in Y_SLICES:
            save_dict[f'fixed_profile_seed{s}_y{y_val}'] = fr['profiles'][str(y_val)]
    np.savez(os.path.join(SCRIPT_DIR, "replot_data_noconf.npz"), **save_dict)

    # Save JSON
    json_out = {
        'config': {
            'seed_target': SEED_TARGET,
            'n_particles': N_PARTICLES,
            'sigma_true': SIGMA_TRUE,
            'sigma_init': SIGMA_INIT,
            'n_epochs': N_EPOCHS,
            'patience': PATIENCE,
            'lam_mass': LAM_MASS,
            'confinement': 'none',
        },
        'true_cp': true_cp,
        'fixed_sigma_runs': {str(s): {
            'best_loss': fr['best_loss'],
            'best_epoch': fr['best_epoch'],
            'learned_cp': fr['learned_cp'],
            'sigma': SIGMA_TRUE,
        } for s, fr in fixed_results.items()},
        'learnable_sigma_runs': [{
            'seed_learned': r['seed_learned'],
            'best_loss': r['best_loss'],
            'best_epoch': r['best_epoch'],
            'final_sigma': r['final_sigma'],
            'best_sigma': r['best_sigma'],
            'learned_cp': r['learned_cp'],
        } for r in learnable_results],
    }
    with open(os.path.join(SCRIPT_DIR, "results_noconf.json"), 'w') as f:
        json.dump(json_out, f, indent=2, default=str)

    # ==================================================================
    # Summary plot 1: Sigma convergence (no confinement)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['tab:red', 'tab:green', 'tab:purple']
    for i, r in enumerate(learnable_results):
        ax.plot(r['history_sigma'], color=colors[i], linewidth=2,
                label=f"seed={r['seed_learned']} "
                      f"(final={r['final_sigma']:.3f})")
    ax.axhline(SIGMA_TRUE, color='blue', linestyle='--', linewidth=2,
               label=f'True $\\sigma$={SIGMA_TRUE}')
    ax.axhline(SIGMA_INIT, color='gray', linestyle=':', linewidth=1,
               label=f'Init $\\sigma$={SIGMA_INIT}')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(r'$\sigma = e^{\log\sigma}$', fontsize=12)
    ax.set_title('Sigma convergence (NO confinement)', fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "summary_sigma_noconf.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ==================================================================
    # Summary plot 2: Loss comparison (fixed vs learnable, no confinement)
    # ==================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    x_pos = np.arange(len(SEEDS_LEARNED))
    width = 0.35
    fixed_losses = [fixed_results[s]['best_loss'] for s in SEEDS_LEARNED]
    learn_losses = [r['best_loss'] for r in learnable_results]
    axes[0].bar(x_pos - width/2, fixed_losses, width, label='Fixed $\\sigma$=1.5',
                color='tab:blue', alpha=0.8, edgecolor='black')
    axes[0].bar(x_pos + width/2, learn_losses, width, label='Learnable $\\sigma$',
                color='tab:red', alpha=0.8, edgecolor='black')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(SEEDS_LEARNED)
    axes[0].set_xlabel('seed_learned', fontsize=12)
    axes[0].set_ylabel('Best loss', fontsize=12)
    axes[0].set_title('Loss: fixed vs learnable $\\sigma$ (no conf)', fontsize=13)
    axes[0].legend(fontsize=10); axes[0].grid(True, alpha=0.3, axis='y')

    # Training curves
    for i, s in enumerate(SEEDS_LEARNED):
        axes[1].semilogy(fixed_results[s]['history_loss'],
                         color=colors[i], linewidth=1, alpha=0.5,
                         linestyle='--', label=f'Fixed s={s}')
        axes[1].semilogy(learnable_results[i]['history_loss'],
                         color=colors[i], linewidth=1.5,
                         label=f'Learn s={s}')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Total loss', fontsize=12)
    axes[1].set_title('Training curves (no conf)', fontsize=13)
    axes[1].legend(fontsize=8, ncol=2); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "summary_loss_comparison_noconf.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ==================================================================
    # Summary plot 3: Potential profiles (fixed vs learnable, no conf)
    # ==================================================================
    fig, axes = plt.subplots(len(SEEDS_LEARNED), 3,
                              figsize=(18, 5 * len(SEEDS_LEARNED)))

    for row, (s, r) in enumerate(zip(SEEDS_LEARNED, learnable_results)):
        for col, y_val in enumerate(Y_SLICES):
            ax = axes[row, col]
            # True
            ax.plot(x_line, true_profiles[y_val], 'b-', linewidth=2.5,
                    label='True')
            # Fixed sigma, no conf
            ax.plot(x_line, fixed_results[s]['profiles'][str(y_val)].ravel(),
                    'g--', linewidth=1.5,
                    label=f'Fixed $\\sigma$=1.5')
            # Learnable sigma, no conf
            prof_l = np.array(r['profiles'][str(y_val)])
            ax.plot(x_line, prof_l, 'r--', linewidth=1.5,
                    label=f'Learn $\\sigma$={r["final_sigma"]:.2f}')

            ax.set_xlabel('x'); ax.set_ylabel(r'$\Phi$')
            ax.set_title(f'seed={s}, y={y_val} (no conf)')
            ax.set_ylim(-15, 15); ax.grid(True, alpha=0.3)
            if col == 0:
                ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "summary_potential_comparison_noconf.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ==================================================================
    # Summary plot 4: Barrier comparison (fixed vs learnable, no conf)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(SEEDS_LEARNED))
    width = 0.35
    fixed_barriers = [fixed_results[s]['learned_cp'].get('barrier') or 0
                      for s in SEEDS_LEARNED]
    learn_barriers = [r['learned_cp'].get('barrier') or 0
                      for r in learnable_results]
    ax.bar(x_pos - width/2, fixed_barriers, width, label='Fixed $\\sigma$=1.5',
           color='tab:blue', alpha=0.8, edgecolor='black')
    ax.bar(x_pos + width/2, learn_barriers, width, label='Learnable $\\sigma$',
           color='tab:red', alpha=0.8, edgecolor='black')
    ax.axhline(true_cp['barrier'], color='black', linestyle='--', linewidth=2,
               label=f'True barrier={true_cp["barrier"]:.2f}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(SEEDS_LEARNED)
    ax.set_xlabel('seed_learned', fontsize=12)
    ax.set_ylabel('Barrier height', fontsize=12)
    ax.set_title('Barrier: fixed vs learnable $\\sigma$ (no conf)', fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "summary_barrier_comparison_noconf.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ==================================================================
    # Print summary table
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"Summary (NO confinement)")
    print(f"  True: barrier={true_cp['barrier']:.3f}, "
          f"x_L={true_cp['x_min_left']:.3f}, "
          f"x_R={true_cp['x_min_right']:.3f}")
    print(f"{'='*70}")
    print(f"{'Condition':<25} {'Seed':>4}  {'Loss':>8}  {'Epoch':>5}  "
          f"{'Sigma':>6}  {'Barrier':>8}")
    print("-" * 70)
    for s in SEEDS_LEARNED:
        fr = fixed_results[s]
        b = fr['learned_cp'].get('barrier')
        b_str = f"{b:.3f}" if b else "N/A"
        print(f"{'Fixed sigma, no conf':<25} {s:>4}  {fr['best_loss']:>8.5f}  "
              f"{fr['best_epoch']:>5}  {SIGMA_TRUE:>6.3f}  {b_str:>8}")
    for r in learnable_results:
        b = r['learned_cp'].get('barrier')
        b_str = f"{b:.3f}" if b else "N/A"
        print(f"{'Learn sigma, no conf':<25} {r['seed_learned']:>4}  "
              f"{r['best_loss']:>8.5f}  {r['best_epoch']:>5}  "
              f"{r['final_sigma']:>6.3f}  {b_str:>8}")

    print(f"\nResults saved to {SCRIPT_DIR}/")
    print(f"  results_noconf.json, replot_data_noconf.npz")
