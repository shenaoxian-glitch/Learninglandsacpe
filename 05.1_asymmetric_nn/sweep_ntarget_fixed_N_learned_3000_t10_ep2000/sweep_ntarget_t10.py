#!/usr/bin/env python
"""
sweep_ntarget_t10.py — Target particle count sweep at T=10

Fixes N_learned=3000, varies N_target from 3000 to 15000 (step 1500)
to determine how much training data is needed for accurate learning.

  - T_FINAL=10.0 (near steady-state)
  - N_EPOCHS=2000, PATIENCE=800
  - 2D critical points (scipy-refined minima + vectorized saddle detection)
  - Saves trained models (.eqx) for each N_target

Experimental design:
  - Target: VARYING N_target with DIFFERENT seed per N_target
  - Learned: FIXED seed=99, FIXED N_learned=3000
  - sigma=1.5 for both, fixed

Sweep: N_target = [3000, 4500, 6000, 7500, 9000, 10500, 12000, 13500, 15000]
       seeds   = [1,    2,    3,    4,    5,    6,     7,     8,     9]

Includes distribution analysis:
  - Analysis batch: N_analysis=2000, seed=777
  - Compares: mass curves, bifurcation, marginals, decay curves,
    trajectories, zone fractions

Usage:
    python sweep_ntarget_fixed_N_learned_3000_t10_ep2000/sweep_ntarget_t10.py \
        > sweep_ntarget_fixed_N_learned_3000_t10_ep2000/log.txt 2>&1
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
from matplotlib.collections import LineCollection

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
T_FINAL = 10.0
N_STEPS = int(T_FINAL / DT)
SIGMA = 1.5
LAM_MASS = 5.0
N_EPOCHS = 2000
PRINT_EVERY = 100
PATIENCE = 800

N_LEARNED = 3000               # fixed learned particle count
SEED_LEARNED = 99
N_TARGET_LIST  = [3000, 4500, 6000, 7500, 9000, 10500, 12000, 13500, 15000]
TARGET_SEEDS   = [1,    2,    3,    4,    5,    6,     7,     8,     9]

# Analysis
N_ANALYSIS = 2000
SEED_ANALYSIS = 777
N_TRAJ_PLOT = 30
SELECT_N = N_TARGET_LIST  # all N_target values

Y_SLICES = [-0.5, 0.5, 1.5]

# Spatial zones
ZONES = {
    'left_well':   lambda x, y: x < -1.0,
    'saddle':      lambda x, y: (x >= -1.0) & (x <= 1.0),
    'right_well':  lambda x, y: x > 1.0,
    'source':      lambda x, y: y < 0.0,
    'death_zone':  lambda x, y: y > 2.2,
}

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
# Critical point extraction (2D)
# ======================================================================

def find_critical_points_2d(phi_fn, x_range=(-4, 4), y_range=(-2, 3.5),
                            grid_n=200):
    from scipy.optimize import minimize as sp_minimize

    result = {
        'x_min_left': None, 'y_min_left': None, 'phi_min_left': None,
        'x_min_right': None, 'y_min_right': None, 'phi_min_right': None,
        'x_saddle': None, 'y_saddle': None, 'phi_saddle': None,
        'barrier': None, 'well_asymmetry': None,
    }

    xg = jnp.linspace(*x_range, grid_n)
    yg = jnp.linspace(*y_range, grid_n)
    Zg = np.array(jax.vmap(
        lambda yi: jax.vmap(lambda xi: phi_fn(jnp.array([xi, yi])))(xg)
    )(yg))
    xg_np = np.array(xg)
    yg_np = np.array(yg)

    mid = grid_n // 2
    left_grid = Zg[:, :mid]
    right_grid = Zg[:, mid:]
    li, lj = np.unravel_index(left_grid.argmin(), left_grid.shape)
    ri, rj = np.unravel_index(right_grid.argmin(), right_grid.shape)
    rj += mid

    res_L = sp_minimize(lambda xy: float(phi_fn(jnp.array(xy))),
                        [xg_np[lj], yg_np[li]], method='Nelder-Mead',
                        options={'xatol': 1e-6, 'fatol': 1e-8})
    res_R = sp_minimize(lambda xy: float(phi_fn(jnp.array(xy))),
                        [xg_np[rj], yg_np[ri]], method='Nelder-Mead',
                        options={'xatol': 1e-6, 'fatol': 1e-8})

    xL, yL = res_L.x;  phiL = res_L.fun
    xR, yR = res_R.x;  phiR = res_R.fun

    max_x = (Zg[1:-1, 1:-1] > Zg[1:-1, :-2]) & (Zg[1:-1, 1:-1] > Zg[1:-1, 2:])
    min_y = (Zg[1:-1, 1:-1] < Zg[:-2, 1:-1]) & (Zg[1:-1, 1:-1] < Zg[2:, 1:-1])
    saddle_mask = max_x & min_y

    x_lo = min(xL, xR) + 0.2
    x_hi = max(xL, xR) - 0.2
    x_in_range = (xg_np[1:-1][None, :] > x_lo) & (xg_np[1:-1][None, :] < x_hi)
    saddle_mask = saddle_mask & x_in_range

    best_saddle = None
    if saddle_mask.any():
        saddle_vals = np.where(saddle_mask, Zg[1:-1, 1:-1], np.inf)
        si, sj = np.unravel_index(saddle_vals.argmin(), saddle_vals.shape)
        best_saddle = (float(xg_np[sj + 1]), float(yg_np[si + 1]))
        best_saddle_val = float(Zg[si + 1, sj + 1])

    result.update({
        'x_min_left': float(xL), 'y_min_left': float(yL),
        'phi_min_left': float(phiL),
        'x_min_right': float(xR), 'y_min_right': float(yR),
        'phi_min_right': float(phiR),
        'well_asymmetry': float(phiL - phiR),
    })
    if best_saddle is not None:
        deeper_val = min(phiL, phiR)
        result.update({
            'x_saddle': float(best_saddle[0]),
            'y_saddle': float(best_saddle[1]),
            'phi_saddle': float(best_saddle_val),
            'barrier': float(best_saddle_val - deeper_val),
        })
    return result


# ======================================================================
# Analysis helper functions
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


# ======================================================================
# Single training run
# ======================================================================

def run_single(n_target, tgt_z, tgt_S, tgt_mass, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n  N_target={n_target}")

    learn_master = jax.random.PRNGKey(SEED_LEARNED)
    queue_key, model_key, opt_key = jax.random.split(learn_master, 3)

    sources = [
        {'mu': jnp.array([0.0, -1.0]), 'sigma_x': 0.25, 'sigma_y': 0.10,
         'n_particles': N_LEARNED},
    ]
    z_init, wake_schedule = build_particle_queue(
        sources=sources, n_particles=N_LEARNED,
        t_final=T_FINAL, dt=DT, key=queue_key,
    )

    phi_nn = PotentialNN(model_key, d_latent=2, c_conf=0.0,
                         hidden_sizes=(16, 16))
    potential = PotentialWithConfinement(
        nn=phi_nn, b=TARGET_PARAMS['b'], d=TARGET_PARAMS['d'],
    )
    fixed_death = FixedDeathRate2D(**DEATH_PARAMS_2D)
    model = SourceDeathModel(potential=potential, death_rate=fixed_death)

    scaled_tgt_mass = tgt_mass * (N_LEARNED / n_target)
    print(f"    tgt_mass={tgt_mass:.1f} (N={n_target}) -> "
          f"scaled={scaled_tgt_mass:.1f} (N={N_LEARNED})")

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
            tgt_z, tgt_S, scaled_tgt_mass, sim_key,
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

    # Save model
    model_path = os.path.join(out_dir, "trained_model.eqx")
    eqx.tree_serialise_leaves(model_path, model)
    print(f"    Saved model to {model_path}")

    # Evaluate potential
    x_line = np.linspace(-4, 4, 500)
    ref_true = float(analytical_potential(0.0, 0.0, TARGET_PARAMS))
    ref_learned = float(model.potential(jnp.array([0.0, 0.0])))

    profiles = {}
    for y_val in Y_SLICES:
        phi_l = np.array(jax.vmap(
            lambda xi: model.potential(jnp.array([xi, y_val]))
        )(jnp.array(x_line))) - ref_learned
        profiles[y_val] = phi_l

    learned_cp = find_critical_points_2d(model.potential)

    if learned_cp['barrier'] is not None:
        print(f"    CP: ({learned_cp['x_min_left']:.3f},{learned_cp['y_min_left']:.3f}), "
              f"({learned_cp['x_min_right']:.3f},{learned_cp['y_min_right']:.3f}), "
              f"barrier={learned_cp['barrier']:.3f}")
    else:
        print(f"    CP: no saddle found")

    # Per-N_target plots
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

    # Training curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(history['loss'], label='Total', linewidth=1.5)
    ax.semilogy(history['mmd'], label='MMD', linewidth=1, alpha=0.8)
    ax.semilogy(history['mass'], label='Mass', linewidth=1, alpha=0.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title(f'N_target={n_target} (N_learned={N_LEARNED})')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training.png"), dpi=150,
                bbox_inches='tight')
    plt.close()

    # Landscape + slices
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    cf = axes[0, 0].contourf(np.array(Xg), np.array(Yg),
                              np.array(Z_learned_a),
                              levels=pot_levels, cmap='viridis', extend='both')
    plt.colorbar(cf, ax=axes[0, 0], label=r'$\hat\Phi$')
    axes[0, 0].set_title(f'Learned (N_tgt={n_target})')
    axes[0, 0].set_xlabel('x'); axes[0, 0].set_ylabel('y')

    cf = axes[0, 1].contourf(np.array(Xg), np.array(Yg),
                              np.array(Z_true_a),
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
        ax.plot(x_line, profiles[y_val], 'r--', linewidth=2, label='Learned')
        ax.set_xlabel('x'); ax.set_ylabel(r'$\Phi$')
        ax.set_title(f'y = {y_val}')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(-15, 15)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "landscape.png"), dpi=150,
                bbox_inches='tight')
    plt.close()

    return model, {
        'n_target': n_target,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'final_mmd': history['mmd'][-1],
        'final_mass': history['mass'][-1],
        'learned_cp': learned_cp,
        'history_loss': history['loss'],
        'history_mmd': history['mmd'],
        'history_mass': history['mass'],
        'profiles': {str(k): v for k, v in profiles.items()},
        'ref_learned': ref_learned,
    }


# ======================================================================
# Main
# ======================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("Target particle count sweep (T=10, ep=2000)")
    print("=" * 65)
    print(f"  seed_learned = {SEED_LEARNED}, N_learned = {N_LEARNED}")
    print(f"  N_target     = {N_TARGET_LIST}")
    print(f"  target_seeds = {TARGET_SEEDS}")
    print(f"  sigma        = {SIGMA} (fixed)")
    print(f"  T_final      = {T_FINAL}")
    print(f"  epochs={N_EPOCHS}, patience={PATIENCE}, lam_mass={LAM_MASS}")
    print(f"  Total runs: {len(N_TARGET_LIST)}")
    print("=" * 65)

    # ==================================================================
    # True critical points (2D)
    # ==================================================================
    x_line = np.linspace(-4, 4, 500)
    ref_true = float(analytical_potential(0.0, 0.0, TARGET_PARAMS))
    true_profiles = {}
    for y_val in Y_SLICES:
        true_profiles[y_val] = np.array([
            float(analytical_potential(float(xi), y_val, TARGET_PARAMS))
            for xi in x_line
        ]) - ref_true

    def true_phi_fn(z):
        return analytical_potential(z[0], z[1], TARGET_PARAMS)
    true_cp = find_critical_points_2d(true_phi_fn)
    print(f"\n  True CP: left=({true_cp['x_min_left']:.3f},{true_cp['y_min_left']:.3f}), "
          f"right=({true_cp['x_min_right']:.3f},{true_cp['y_min_right']:.3f}), "
          f"barrier={true_cp['barrier']:.3f}")

    # ==================================================================
    # Prepare analysis batch (seed=777, N=2000) — shared across all runs
    # ==================================================================
    print(f"\nPreparing analysis batch (seed={SEED_ANALYSIS}, "
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

    # Simulate analysis batch through true dynamics (once)
    print("  Simulating analysis batch through true dynamics ...")
    ana_tgt_px, ana_tgt_py, ana_tgt_S = simulate_ground_truth_full(
        ana_z_init, ana_wake, N_STEPS, DT, ana_sim_key
    )
    ana_tgt_z = jnp.stack([ana_tgt_px, ana_tgt_py], axis=-1)
    final_step = N_STEPS - 1
    time_axis = np.arange(1, N_STEPS + 1) * DT

    # Target analysis stats
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
    # Train all N_target values + collect analysis
    # ==================================================================
    all_results = []
    all_analysis = {}

    for n_target, seed_tgt in zip(N_TARGET_LIST, TARGET_SEEDS):
        print(f"\n{'='*50}")
        print(f"  N_target = {n_target}, seed_target = {seed_tgt}")
        print(f"{'='*50}")

        # Generate training target (seed per N_target)
        tgt_master = jax.random.PRNGKey(seed_tgt)
        tgt_queue_key, tgt_sim_key = jax.random.split(tgt_master)
        sources_tgt = [
            {'mu': jnp.array([0.0, -1.0]), 'sigma_x': 0.25, 'sigma_y': 0.10,
             'n_particles': n_target},
        ]
        tgt_z_init, tgt_wake = build_particle_queue(
            sources=sources_tgt, n_particles=n_target,
            t_final=T_FINAL, dt=DT, key=tgt_queue_key,
        )
        print(f"  Generating target (seed={seed_tgt}, N={n_target}) ...")
        tgt_all_px, tgt_all_py, tgt_all_S_full = simulate_ground_truth_full(
            tgt_z_init, tgt_wake, N_STEPS, DT, tgt_sim_key
        )
        tgt_z = jnp.column_stack([tgt_all_px[final_step],
                                   tgt_all_py[final_step]])
        tgt_S = tgt_all_S_full[final_step]
        tgt_mass = float(jnp.sum(jnp.exp(tgt_S)))
        print(f"  Target mass={tgt_mass:.1f}")

        # Train
        out_dir = os.path.join(SCRIPT_DIR, f"Ntgt{n_target}")
        model, res = run_single(n_target, tgt_z, tgt_S, tgt_mass, out_dir)
        all_results.append(res)

        # --- Analysis: simulate analysis batch through learned model ---
        print(f"    Analyzing (N_analysis={N_ANALYSIS}) ...")
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

        lrn_tau, lrn_mean, lrn_lo, lrn_hi, lrn_med = \
            compute_decay_curves(ana_lrn_S, ana_wake, DT)

        stats = {
            'mass_curve': lrn_mass_curve,
            'frac_left': lrn_frac_left,
            'frac_right': lrn_frac_right,
            'final_mass': float(lrn_mass_curve[-1]),
            'zone_fracs': lrn_zone_fracs,
            'x_hist': x_hist_lrn,
            'y_hist': y_hist_lrn,
            'decay_tau': lrn_tau,
            'decay_mean': lrn_mean,
            'decay_lo': lrn_lo,
            'decay_hi': lrn_hi,
            'decay_med': lrn_med,
            'all_z': np.array(ana_lrn_z),
            'all_S': np.array(ana_lrn_S),
        }

        print(f"    Bifurcation: {lrn_frac_left:.3f}L / {lrn_frac_right:.3f}R "
              f"(target: {tgt_frac_left:.3f}/{tgt_frac_right:.3f})")
        print(f"    Final mass: {lrn_mass_curve[-1]:.1f} "
              f"(target: {tgt_mass_curve[-1]:.1f})")
        print(f"    Zones: {lrn_zone_fracs}")

        all_analysis[n_target] = stats

    # ==================================================================
    # Save replot data
    # ==================================================================
    save_dict = {
        'x_line': x_line,
        'n_target_list': np.array(N_TARGET_LIST),
        'n_learned': np.array(N_LEARNED),
        'target_seeds': np.array(TARGET_SEEDS),
        'seed_learned': np.array(SEED_LEARNED),
        'time_axis': time_axis,
        'tgt_mass_curve': tgt_mass_curve,
        'tgt_frac_left': np.array(tgt_frac_left),
        'x_bins': x_bins, 'y_bins': y_bins,
        'x_hist_tgt': x_hist_tgt, 'y_hist_tgt': y_hist_tgt,
        'tgt_decay_tau': tgt_decay_tau,
        'tgt_decay_mean': tgt_decay_mean,
        'tgt_decay_lo': tgt_decay_lo,
        'tgt_decay_hi': tgt_decay_hi,
    }
    for y_val in Y_SLICES:
        save_dict[f'true_profile_y{y_val}'] = true_profiles[y_val]
    for r in all_results:
        n = r['n_target']
        save_dict[f'loss_Ntgt{n}'] = np.array(r['history_loss'])
        save_dict[f'mmd_Ntgt{n}'] = np.array(r['history_mmd'])
        save_dict[f'mass_Ntgt{n}'] = np.array(r['history_mass'])
        for y_val in Y_SLICES:
            save_dict[f'profile_Ntgt{n}_y{y_val}'] = r['profiles'][str(y_val)]
    for n, stats in all_analysis.items():
        save_dict[f'ana_mass_curve_Ntgt{n}'] = stats['mass_curve']
        save_dict[f'ana_frac_left_Ntgt{n}'] = np.array(stats['frac_left'])
        save_dict[f'ana_x_hist_Ntgt{n}'] = stats['x_hist']
        save_dict[f'ana_y_hist_Ntgt{n}'] = stats['y_hist']
        save_dict[f'ana_decay_tau_Ntgt{n}'] = stats['decay_tau']
        save_dict[f'ana_decay_mean_Ntgt{n}'] = stats['decay_mean']
        save_dict[f'ana_decay_lo_Ntgt{n}'] = stats['decay_lo']
        save_dict[f'ana_decay_hi_Ntgt{n}'] = stats['decay_hi']
    np.savez(os.path.join(SCRIPT_DIR, "replot_data.npz"), **save_dict)

    # Save JSON
    json_out = {
        'config': {
            'target_seeds': TARGET_SEEDS,
            'seed_learned': SEED_LEARNED,
            'n_learned': N_LEARNED,
            'n_target_list': N_TARGET_LIST,
            'sigma': SIGMA,
            't_final': T_FINAL,
            'n_epochs': N_EPOCHS,
            'patience': PATIENCE,
            'lam_mass': LAM_MASS,
        },
        'true_cp': true_cp,
        'analysis': {
            'tgt_frac_left': tgt_frac_left,
            'tgt_final_mass': float(tgt_mass_curve[-1]),
            'tgt_zone_fracs': tgt_zone_fracs,
        },
        'runs': [{
            'n_target': r['n_target'],
            'best_loss': r['best_loss'],
            'best_epoch': r['best_epoch'],
            'final_mmd': r['final_mmd'],
            'final_mass': r['final_mass'],
            'learned_cp': r['learned_cp'],
            'ref_learned': r['ref_learned'],
            'ana_frac_left': all_analysis[r['n_target']]['frac_left'],
            'ana_final_mass': all_analysis[r['n_target']]['final_mass'],
            'ana_zone_fracs': all_analysis[r['n_target']]['zone_fracs'],
        } for r in all_results],
    }
    with open(os.path.join(SCRIPT_DIR, "results.json"), 'w') as f:
        json.dump(json_out, f, indent=2, default=str)

    print(f"\nSaved replot_data.npz, results.json")

    # ==================================================================
    # SUMMARY PLOT 1: Loss vs N_target
    # ==================================================================
    ns = [r['n_target'] for r in all_results]
    losses = [r['best_loss'] for r in all_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(ns, losses, 'o-', color='tab:blue', linewidth=2, markersize=8)
    axes[0].set_xlabel('$N_{target}$ (training data size)', fontsize=12)
    axes[0].set_ylabel('Best loss', fontsize=12)
    axes[0].set_title(f'Loss vs $N_{{target}}$ ($N_{{learned}}$={N_LEARNED})',
                      fontsize=13)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    axes[0].set_xticks(ns)
    axes[0].set_xticklabels(ns, rotation=45, fontsize=9)

    for r in all_results:
        axes[1].semilogy(r['history_loss'], linewidth=1, alpha=0.8,
                         label=f"Ntgt={r['n_target']}")
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Total loss', fontsize=12)
    axes[1].set_title('Training curves', fontsize=13)
    axes[1].legend(fontsize=8, ncol=2); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "summary_loss.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved summary_loss.png")

    # ==================================================================
    # SUMMARY PLOT 2: Critical points vs N_target (2D)
    # ==================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    def plot_cp(ax, key, true_val, ylabel, title):
        vals = [(r['n_target'], r['learned_cp'].get(key))
                for r in all_results]
        valid = [(n, v) for n, v in vals if v is not None]
        if not valid:
            ax.set_title(f'{title} (no data)')
            return
        vn, vv = zip(*valid)
        ax.plot(vn, vv, 'o-', color='tab:red', linewidth=2, markersize=8)
        if true_val is not None:
            ax.axhline(true_val, color='blue', linestyle='--', linewidth=2,
                        label=f'True = {true_val:.3f}')
        ax.set_xlabel('$N_{target}$', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_xticks(N_TARGET_LIST)
        ax.set_xticklabels(N_TARGET_LIST, rotation=45, fontsize=8)

    plot_cp(axes[0, 0], 'x_min_left', true_cp['x_min_left'],
            r'$x_{min,left}$', 'Left well x-position')
    plot_cp(axes[0, 1], 'x_min_right', true_cp['x_min_right'],
            r'$x_{min,right}$', 'Right well x-position')
    plot_cp(axes[1, 0], 'x_saddle', true_cp['x_saddle'],
            r'$x_{saddle}$', 'Saddle x-position')
    plot_cp(axes[1, 1], 'barrier', true_cp['barrier'],
            r'$\Delta\Phi$', 'Barrier height (2D)')

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "summary_critical_points.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved summary_critical_points.png")

    # ==================================================================
    # SUMMARY PLOT 3: Potential overlay
    # ==================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_results)))

    for col, y_val in enumerate(Y_SLICES):
        ax = axes[col]
        ax.plot(x_line, true_profiles[y_val], 'k-', linewidth=3,
                label='True', zorder=10)
        for i, r in enumerate(all_results):
            ax.plot(x_line, r['profiles'][str(y_val)],
                    color=colors[i], linewidth=1.2, alpha=0.8,
                    label=f"Ntgt={r['n_target']}")
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel(r'$\Phi(x, y)$', fontsize=12)
        ax.set_title(f'y = {y_val}', fontsize=13)
        ax.set_ylim(-15, 15); ax.grid(True, alpha=0.3)
        if col == 2:
            ax.legend(fontsize=7, ncol=2, loc='upper right')

    plt.suptitle(f'Potential profiles vs $N_{{target}}$ '
                 f'($N_{{learned}}$={N_LEARNED})',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "summary_potential_overlay.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved summary_potential_overlay.png")

    # ==================================================================
    # SUMMARY PLOT 4: Force profiles
    # ==================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    dx = x_line[1] - x_line[0]

    for col, y_val in enumerate(Y_SLICES):
        ax = axes[col]
        F_true = -np.gradient(true_profiles[y_val], dx)
        ax.plot(x_line, F_true, 'k-', linewidth=3, label='True', zorder=10)
        for i, r in enumerate(all_results):
            F_l = -np.gradient(r['profiles'][str(y_val)], dx)
            ax.plot(x_line, F_l, color=colors[i], linewidth=1.2, alpha=0.8,
                    label=f"Ntgt={r['n_target']}")
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel(r'$F_x = -\partial\Phi/\partial x$', fontsize=12)
        ax.set_title(f'y = {y_val}', fontsize=13)
        ax.set_xlim(-3.5, 3.5); ax.grid(True, alpha=0.3)
        if col == 2:
            ax.legend(fontsize=7, ncol=2, loc='upper right')

    plt.suptitle('Force profiles vs $N_{target}$', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "summary_force_overlay.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved summary_force_overlay.png")

    # ==================================================================
    # ANALYSIS PLOT 1: Mass curves (3x3 grid)
    # ==================================================================
    n_sel = len(SELECT_N)
    n_cols = 3
    n_rows = (n_sel + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes_flat = axes.ravel()
    for i, n_val in enumerate(SELECT_N):
        ax = axes_flat[i]
        ax.plot(time_axis, tgt_mass_curve, 'b-', linewidth=2,
                label='Target', alpha=0.8)
        ax.plot(time_axis, all_analysis[n_val]['mass_curve'], 'r--',
                linewidth=2, label=f'Learned (Ntgt={n_val})', alpha=0.8)
        ax.set_xlabel('Time')
        ax.set_ylabel('$M(t)$')
        ax.set_title(f'$N_{{target}}$ = {n_val}')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    for i in range(n_sel, len(axes_flat)):
        axes_flat[i].set_visible(False)
    plt.suptitle(f'Total mass $M(t)$: target vs learned (T={T_FINAL})',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "analysis_mass_curves.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved analysis_mass_curves.png")

    # ==================================================================
    # ANALYSIS PLOT 2: Bifurcation fraction vs N_target
    # ==================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n_arr = N_TARGET_LIST
    lrn_fl = [all_analysis[n]['frac_left'] for n in n_arr]
    lrn_fr = [all_analysis[n]['frac_right'] for n in n_arr]

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
    axes[0].set_xlabel('$N_{target}$ (training data size)')
    axes[0].set_ylabel('Weight fraction')
    axes[0].set_title('Bifurcation: weight at x<0 vs x>0')
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim(0, 1)

    frac_errors = [abs(f - tgt_frac_left) for f in lrn_fl]
    axes[1].plot(n_arr, frac_errors, 'ro-', lw=2, ms=8)
    axes[1].set_xlabel('$N_{target}$')
    axes[1].set_ylabel('|frac_left error|')
    axes[1].set_title('Bifurcation fraction error vs $N_{target}$')
    axes[1].set_xscale('log'); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "analysis_bifurcation.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved analysis_bifurcation.png")

    # ==================================================================
    # ANALYSIS PLOT 3: x-marginals (3x3 grid)
    # ==================================================================
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes_flat = axes.ravel()
    for i, n_val in enumerate(SELECT_N):
        ax = axes_flat[i]
        ax.plot(x_bins, x_hist_tgt, 'b-', lw=2, label='Target')
        ax.plot(x_bins, all_analysis[n_val]['x_hist'], 'r--', lw=2,
                label=f'Learned (Ntgt={n_val})')
        ax.set_xlabel('x')
        ax.set_ylabel('Weighted density')
        ax.set_title(f'x-marginal ($N_{{tgt}}$={n_val})')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    for i in range(n_sel, len(axes_flat)):
        axes_flat[i].set_visible(False)
    plt.suptitle(f'Weighted x-marginals at t={T_FINAL}',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "analysis_marginals_x.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved analysis_marginals_x.png")

    # ==================================================================
    # ANALYSIS PLOT 4: y-marginals (3x3 grid)
    # ==================================================================
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes_flat = axes.ravel()
    for i, n_val in enumerate(SELECT_N):
        ax = axes_flat[i]
        ax.plot(y_bins, y_hist_tgt, 'b-', lw=2, label='Target')
        ax.plot(y_bins, all_analysis[n_val]['y_hist'], 'r--', lw=2,
                label=f'Learned (Ntgt={n_val})')
        ax.set_xlabel('y')
        ax.set_ylabel('Weighted density')
        ax.set_title(f'y-marginal ($N_{{tgt}}$={n_val})')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    for i in range(n_sel, len(axes_flat)):
        axes_flat[i].set_visible(False)
    plt.suptitle(f'Weighted y-marginals at t={T_FINAL}',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "analysis_marginals_y.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved analysis_marginals_y.png")

    # ==================================================================
    # ANALYSIS PLOT 5: Decay curves (3x3 grid)
    # ==================================================================
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes_flat = axes.ravel()
    for i, n_val in enumerate(SELECT_N):
        ax = axes_flat[i]
        s = all_analysis[n_val]
        ax.fill_between(tgt_decay_tau, tgt_decay_lo, tgt_decay_hi,
                        color='blue', alpha=0.15, label='Target 10-90%')
        ax.plot(tgt_decay_tau, tgt_decay_mean, 'b-', lw=2,
                label='Target mean')
        ax.fill_between(s['decay_tau'], s['decay_lo'], s['decay_hi'],
                        color='red', alpha=0.15, label='Learned 10-90%')
        ax.plot(s['decay_tau'], s['decay_mean'], 'r-', lw=2,
                label='Learned mean')
        ax.set_xlabel('Time since birth')
        ax.set_ylabel('Weight $e^S$')
        ax.set_title(f'Particle decay ($N_{{tgt}}$={n_val})')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    for i in range(n_sel, len(axes_flat)):
        axes_flat[i].set_visible(False)
    plt.suptitle('Per-particle weight decay (aligned by birth time)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "analysis_decay_curves.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved analysis_decay_curves.png")

    # ==================================================================
    # ANALYSIS PLOT 6: Sample trajectories (3x3 grid)
    # ==================================================================
    rng = np.random.RandomState(42)
    early_mask = np.array(ana_wake) < N_STEPS // 3
    early_idx = np.where(early_mask)[0]
    traj_idx = rng.choice(early_idx, size=min(N_TRAJ_PLOT, len(early_idx)),
                          replace=False)
    traj_idx.sort()
    wake_np = np.array(ana_wake)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    axes_flat = axes.ravel()

    for i, n_val in enumerate(SELECT_N):
        ax = axes_flat[i]
        lrn_z_np = all_analysis[n_val]['all_z']
        lrn_S_np = all_analysis[n_val]['all_S']
        for idx in traj_idx:
            birth = int(wake_np[idx])
            if birth >= N_STEPS:
                continue
            traj_x = lrn_z_np[birth:, idx, 0]
            traj_y = lrn_z_np[birth:, idx, 1]
            traj_w = np.exp(lrn_S_np[birth:, idx])
            points = np.column_stack([traj_x, traj_y]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap='coolwarm',
                                norm=plt.Normalize(0, 1))
            lc.set_array(traj_w[:-1])
            lc.set_linewidth(0.8)
            ax.add_collection(lc)
        ax.set_xlim(-4, 4); ax.set_ylim(-2, 3.5)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(f'Learned Ntgt={n_val}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    for i in range(n_sel, len(axes_flat)):
        axes_flat[i].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label='Weight $e^S$')
    plt.suptitle(f'Learned trajectories ({N_TRAJ_PLOT} particles)',
                 fontsize=14, y=1.01)
    plt.savefig(os.path.join(SCRIPT_DIR, "analysis_trajectories.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved analysis_trajectories.png")

    # ==================================================================
    # ANALYSIS PLOT 7: Zone weight fractions vs N_target
    # ==================================================================
    zone_names = list(ZONES.keys())
    zone_colors = ['tab:blue', 'tab:gray', 'tab:orange', 'tab:green',
                   'tab:red']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    x_pos = np.arange(len(N_TARGET_LIST))
    bar_width = 0.7
    bottom = np.zeros(len(N_TARGET_LIST))
    for iz, zname in enumerate(zone_names):
        vals = [all_analysis[n]['zone_fracs'][zname] for n in N_TARGET_LIST]
        ax.bar(x_pos, vals, bar_width, bottom=bottom, color=zone_colors[iz],
               alpha=0.7, label=zname)
        bottom += np.array(vals)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(N_TARGET_LIST, rotation=45)
    ax.set_xlabel('$N_{target}$ (training data size)')
    ax.set_ylabel('Weight fraction')
    ax.set_title('Learned: zone weight fractions')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2, axis='y')

    ax = axes[1]
    for iz, zname in enumerate(zone_names):
        errors = [abs(all_analysis[n]['zone_fracs'][zname]
                      - tgt_zone_fracs[zname]) for n in N_TARGET_LIST]
        ax.plot(N_TARGET_LIST, errors, 'o-', color=zone_colors[iz],
                lw=2, ms=6, label=zname)
    ax.set_xlabel('$N_{target}$')
    ax.set_ylabel('|zone fraction error|')
    ax.set_title('Zone fraction error vs $N_{target}$')
    ax.set_xscale('log')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    tgt_str = "Target: " + ", ".join(
        f"{k}={v:.3f}" for k, v in tgt_zone_fracs.items())
    fig.text(0.5, -0.02, tgt_str, ha='center', fontsize=10, style='italic')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "analysis_zones.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved analysis_zones.png")

    # ==================================================================
    # ANALYSIS PLOT 8: Summary metrics vs N_target
    # ==================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    mass_errors = [abs(all_analysis[n]['final_mass'] - tgt_mass_curve[-1])
                   / tgt_mass_curve[-1] for n in N_TARGET_LIST]
    axes[0].plot(N_TARGET_LIST, mass_errors, 'ko-', lw=2, ms=8)
    axes[0].set_xlabel('$N_{target}$')
    axes[0].set_ylabel('Relative mass error')
    axes[0].set_title('Final mass error vs $N_{target}$')
    axes[0].set_xscale('log'); axes[0].grid(True, alpha=0.3)

    axes[1].plot(N_TARGET_LIST, frac_errors, 'ro-', lw=2, ms=8)
    axes[1].set_xlabel('$N_{target}$')
    axes[1].set_ylabel('Bifurcation fraction error')
    axes[1].set_title('Bifurcation error vs $N_{target}$')
    axes[1].set_xscale('log'); axes[1].grid(True, alpha=0.3)

    x_marg_err = [np.sqrt(np.mean(
        (all_analysis[n]['x_hist'] - x_hist_tgt)**2))
        for n in N_TARGET_LIST]
    axes[2].plot(N_TARGET_LIST, x_marg_err, 'bs-', lw=2, ms=8)
    axes[2].set_xlabel('$N_{target}$')
    axes[2].set_ylabel('x-marginal RMSE')
    axes[2].set_title('x-marginal error vs $N_{target}$')
    axes[2].set_xscale('log'); axes[2].grid(True, alpha=0.3)

    plt.suptitle('Summary: distribution errors vs $N_{target}$',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "analysis_summary.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved analysis_summary.png")

    # ==================================================================
    # Summary tables
    # ==================================================================
    print(f"\n{'='*90}")
    print(f"Summary: N_learned={N_LEARNED}, seed_learned={SEED_LEARNED}")
    print(f"  True: left=({true_cp['x_min_left']:.3f},{true_cp['y_min_left']:.3f}), "
          f"right=({true_cp['x_min_right']:.3f},{true_cp['y_min_right']:.3f}), "
          f"barrier={true_cp['barrier']:.3f}")
    print(f"{'='*90}")
    print(f"{'Ntgt':>6s}  {'Loss':>10s}  {'Epoch':>6s}  {'Barrier':>10s}  "
          f"{'Left (x,y)':>16s}  {'Right (x,y)':>16s}  {'Saddle (x,y)':>16s}")
    print("-" * 90)
    for r in all_results:
        cp = r['learned_cp']
        def fmt_xy(xk, yk):
            xv = cp.get(xk)
            yv = cp.get(yk)
            if xv is not None and yv is not None:
                return f"({xv:.2f},{yv:.2f})"
            return "N/A"
        def fmt(v):
            return f"{v:.4f}" if v is not None else "N/A"
        print(f"{r['n_target']:6d}  {r['best_loss']:10.6f}  "
              f"{r['best_epoch']:6d}  {fmt(cp['barrier']):>10s}  "
              f"{fmt_xy('x_min_left','y_min_left'):>16s}  "
              f"{fmt_xy('x_min_right','y_min_right'):>16s}  "
              f"{fmt_xy('x_saddle','y_saddle'):>16s}")

    print(f"\n{'='*80}")
    print(f"Distribution analysis (N_analysis={N_ANALYSIS}, seed={SEED_ANALYSIS})")
    print(f"Target: frac_left={tgt_frac_left:.3f}, "
          f"final_mass={tgt_mass_curve[-1]:.1f}")
    zone_str = "  ".join(f"{k}={v:.3f}" for k, v in tgt_zone_fracs.items())
    print(f"Target zones: {zone_str}")
    print(f"{'='*80}")
    print(f"{'Ntgt':>6}  {'frac_L':>7}  {'frac_R':>7}  {'mass':>7}  "
          f"{'left_w':>7}  {'saddle':>7}  {'right_w':>7}  "
          f"{'source':>7}  {'death':>7}")
    print("-" * 80)
    for n in N_TARGET_LIST:
        s = all_analysis[n]
        zf = s['zone_fracs']
        print(f"{n:>6}  {s['frac_left']:>7.3f}  {s['frac_right']:>7.3f}  "
              f"{s['final_mass']:>7.1f}  "
              f"{zf['left_well']:>7.3f}  {zf['saddle']:>7.3f}  "
              f"{zf['right_well']:>7.3f}  {zf['source']:>7.3f}  "
              f"{zf['death_zone']:>7.3f}")

    print(f"\nAll outputs saved to {SCRIPT_DIR}/")
