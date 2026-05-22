#!/usr/bin/env python
"""
sweep_nparticles_step1b.py — N_learned sweep at N_target=12000

Step 1b base setup: sigma=1.5, NN(16,16), PotentialWithX4(c_x4=0.1), y_d=2.5.
Sweeps N_learned over 9 values from 400 to 8000, one unique seed per N.

Usage:
    python sweep_nparticles_step1b.py > log.txt 2>&1
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

from models import SourceDeathModel
from models.potential import PotentialNN
from models.death_rate import ExponentialDeathRate
from training.data_loader import (
    build_particle_queue, analytical_potential_no_y4, analytical_death_rate_exp,
)
from simulator.sde_solver import simulate_open_system_full


# ======================================================================
# Constants
# ======================================================================

TARGET_PARAMS = {
    'a': -1.0, 'b': -1.6, 'c': 3.5, 'e': 0.7, 'sigma': 1.5,
}
DEATH_PARAMS_EXP = {
    'A': 0.1, 'k': 5.0, 'y_d': 2.5, 'gamma_max': 100.0,
}

DT = 0.01
T_FINAL = 3.0
N_STEPS = int(T_FINAL / DT)
SIGMA = 1.5
LAM_MASS = 5.0
BANDWIDTHS = (0.005, 0.05, 0.5, 5.0)
ALIVE_THRESHOLD = 1e-3

N_EPOCHS = 2000
PRINT_EVERY = 100
PATIENCE = 800
LR = 3e-3

C_X4 = 0.1

TRUE_MU_X, TRUE_MU_Y = 0.0, -1.0
TRUE_SIGMA_X, TRUE_SIGMA_Y = 0.05, 0.02

Z_CLAMP = jnp.array([[-8.0, 8.0], [-3.0, 10.0]])

N_TARGET = 12000
SEED_TARGET = 4
N_LEARNED_LIST = [400, 600, 900, 1300, 1900, 2700, 3900, 5500, 8000]
SEED_LEARNED_LIST = [42, 43, 44, 45, 46, 47, 48, 49, 50]

Y_SLICES = [-0.5, 0.5, 1.5]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ======================================================================
# Model components
# ======================================================================

class PotentialWithX4(eqx.Module):
    nn: PotentialNN
    c_x4: float = eqx.field(static=True)

    def __call__(self, z):
        return self.nn(z) + self.c_x4 * z[0] ** 4


# ======================================================================
# Ground-truth simulation
# ======================================================================

_force_fn = jax.vmap(
    jax.grad(analytical_potential_no_y4, argnums=(0, 1)),
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
        new_px = jnp.clip(new_px, Z_CLAMP[0, 0], Z_CLAMP[0, 1])
        new_py = jnp.clip(new_py, Z_CLAMP[1, 0], Z_CLAMP[1, 1])
        gamma = analytical_death_rate_exp(px, py, DEATH_PARAMS_EXP)
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
    return (M_sim - target_mass) ** 2 / (target_mass ** 2 + 1e-8)


def weighted_mmd_both(x_sim, alpha_sim, x_obs, alpha_obs,
                      bandwidths=BANDWIDTHS):
    def k_matrix(a, b):
        diff = a[:, None, :] - b[None, :, :]
        dist_sq = jnp.sum(diff ** 2, axis=-1)
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


def make_objective(n_steps, dt, sigma, lam_mass, z_clamp=None):
    def objective(model, z_init, wake_schedule, tgt_z, tgt_S, tgt_mass, key):
        all_z, all_S = simulate_open_system_full(
            model, z_init, wake_schedule, sigma, n_steps, dt, key,
            z_clamp=z_clamp,
        )
        sim_z = all_z[-1]
        sim_S = all_S[-1]
        sim_alpha = jax.nn.softmax(sim_S)
        tgt_alpha = jax.nn.softmax(tgt_S)
        l_mmd = weighted_mmd_both(sim_z, sim_alpha, tgt_z, tgt_alpha)
        l_mass = mass_loss(sim_S, tgt_mass)
        return l_mmd + lam_mass * l_mass, (l_mmd, l_mass)
    return objective


def make_train_step(optimizer, objective_fn):
    @eqx.filter_jit
    def step(model, opt_state, z_init, wake_schedule,
             tgt_z, tgt_S, tgt_mass, key):
        def loss_fn(model):
            return objective_fn(
                model, z_init, wake_schedule,
                tgt_z, tgt_S, tgt_mass, key
            )
        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss, aux
    return step


# ======================================================================
# Valley diagnostics
# ======================================================================

def find_valleys_at_y(phi_fn, y_val, x_range=(-4.0, 4.0), n=500):
    """Argmin of phi(x, y_val) over x<0 and x>0 separately."""
    x_line = jnp.linspace(x_range[0], x_range[1], n)
    phi_vals = jax.vmap(lambda xi: phi_fn(jnp.array([xi, y_val])))(x_line)
    phi_np = np.array(phi_vals)
    x_np = np.array(x_line)

    left_mask = x_np < 0
    right_mask = x_np > 0
    left_idx = int(np.argmin(phi_np[left_mask]))
    right_idx = int(np.argmin(phi_np[right_mask]))

    x_left = float(x_np[left_mask][left_idx])
    phi_left = float(phi_np[left_mask][left_idx])
    x_right = float(x_np[right_mask][right_idx])
    phi_right = float(phi_np[right_mask][right_idx])

    return {
        'x_left': x_left, 'x_right': x_right,
        'phi_left': phi_left, 'phi_right': phi_right,
        'asymmetry': phi_right - phi_left,
    }


# ======================================================================
# Single training run
# ======================================================================

def run_single(n_learned, seed_learned, tgt_z, tgt_S, tgt_mass, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n  N_learned={n_learned}, seed={seed_learned}")

    learn_master = jax.random.PRNGKey(seed_learned)
    queue_key, model_key, opt_key = jax.random.split(learn_master, 3)

    sources = [
        {'mu': jnp.array([TRUE_MU_X, TRUE_MU_Y]),
         'sigma_x': TRUE_SIGMA_X, 'sigma_y': TRUE_SIGMA_Y,
         'n_particles': n_learned},
    ]
    z_init, wake_schedule = build_particle_queue(
        sources=sources, n_particles=n_learned,
        t_final=T_FINAL, dt=DT, key=queue_key,
    )

    phi_nn = PotentialNN(model_key, d_latent=2, c_conf=0.0,
                         hidden_sizes=(16, 16))
    potential = PotentialWithX4(nn=phi_nn, c_x4=C_X4)
    fixed_death = ExponentialDeathRate(**DEATH_PARAMS_EXP)
    model = SourceDeathModel(potential=potential, death_rate=fixed_death)

    scaled_tgt_mass = tgt_mass * (n_learned / N_TARGET)
    print(f"    tgt_mass={tgt_mass:.1f} -> scaled={scaled_tgt_mass:.1f}")

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=LR, weight_decay=1e-4),
    )
    objective_fn = make_objective(N_STEPS, DT, SIGMA, LAM_MASS, z_clamp=Z_CLAMP)
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

    # --- Save model ---
    model_path = os.path.join(out_dir, "trained_model.eqx")
    eqx.tree_serialise_leaves(model_path, model)

    # --- Profiles + valleys ---
    x_line = np.linspace(-4, 4, 500)
    ref_true = float(analytical_potential_no_y4(0.0, 0.0, TARGET_PARAMS))
    ref_learned = float(model.potential(jnp.array([0.0, 0.0])))

    profiles = {}
    for y_val in Y_SLICES:
        phi_l = np.array(jax.vmap(
            lambda xi: model.potential(jnp.array([xi, y_val]))
        )(jnp.array(x_line))) - ref_learned
        profiles[y_val] = phi_l

    learned_valleys = {}
    for y_val in [0.5, 1.5]:
        learned_valleys[y_val] = find_valleys_at_y(model.potential, y_val)

    # --- Per-N plots ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(history['loss'], label='Total', linewidth=1.5)
    ax.semilogy(history['mmd'], label='MMD', linewidth=1, alpha=0.8)
    ax.semilogy(history['mass'], label='Mass', linewidth=1, alpha=0.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title(f'N_learned={n_learned} seed={seed_learned} '
                 f'(best loss={best_loss:.5f} @ epoch {best_epoch})')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Landscape + slices
    x_grid = jnp.linspace(-4, 4, 200)
    y_grid = jnp.linspace(-2, 5, 200)
    Xg, Yg = jnp.meshgrid(x_grid, y_grid)
    Z_true = jax.vmap(jax.vmap(
        lambda x, y: analytical_potential_no_y4(x, y, TARGET_PARAMS)
    ))(Xg, Yg)
    Z_learned = jax.vmap(
        lambda row_y: jax.vmap(
            lambda xi: model.potential(jnp.array([xi, row_y]))
        )(x_grid)
    )(y_grid)
    Z_true_a = np.array(Z_true) - ref_true
    Z_learned_a = np.array(Z_learned) - ref_learned

    alive_mask = np.array(Yg) < 3.0
    pot_vmin = max(min(Z_true_a[alive_mask].min(), Z_learned_a[alive_mask].min()), -20.0)
    pot_vmax = min(max(Z_true_a[alive_mask].max(), Z_learned_a[alive_mask].max()), 20.0)
    pot_levels = np.linspace(pot_vmin, pot_vmax, 31)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    cf = axes[0, 0].contourf(np.array(Xg), np.array(Yg), Z_learned_a,
                              levels=pot_levels, cmap='viridis', extend='both')
    plt.colorbar(cf, ax=axes[0, 0], label=r'$\hat\Phi$')
    axes[0, 0].axhline(DEATH_PARAMS_EXP['y_d'], color='red', linestyle='--',
                        linewidth=1.5, alpha=0.5)
    axes[0, 0].set_title(f'Learned (N={n_learned})')
    axes[0, 0].set_xlabel('x'); axes[0, 0].set_ylabel('y')

    cf = axes[0, 1].contourf(np.array(Xg), np.array(Yg), Z_true_a,
                              levels=pot_levels, cmap='viridis', extend='both')
    plt.colorbar(cf, ax=axes[0, 1], label=r'$\Phi^*$')
    axes[0, 1].axhline(DEATH_PARAMS_EXP['y_d'], color='red', linestyle='--',
                        linewidth=1.5, alpha=0.5)
    axes[0, 1].set_title('True')
    axes[0, 1].set_xlabel('x'); axes[0, 1].set_ylabel('y')

    Z_diff = Z_learned_a - Z_true_a
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
            float(analytical_potential_no_y4(float(xi), y_val, TARGET_PARAMS))
            for xi in x_line
        ]) - ref_true
        ax.plot(x_line, phi_t, 'b-', linewidth=2, label='True')
        ax.plot(x_line, profiles[y_val], 'r--', linewidth=2, label='Learned')
        ax.set_xlabel('x'); ax.set_ylabel(r'$\Phi$')
        ax.set_title(f'y = {y_val}')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(-15, 15)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "landscape.png"), dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'n_learned': n_learned,
        'seed_learned': seed_learned,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'final_mmd': history['mmd'][-1],
        'final_mass': history['mass'][-1],
        'learned_valleys': learned_valleys,
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
    print("N_learned sweep — death-wall model (Step 1b base)")
    print("=" * 65)
    print(f"  N_target  = {N_TARGET} (seed={SEED_TARGET})")
    print(f"  N_learned = {N_LEARNED_LIST}")
    print(f"  seeds     = {SEED_LEARNED_LIST} (one per N)")
    print(f"  c_x4 = {C_X4}, y_d = {DEATH_PARAMS_EXP['y_d']}, sigma = {SIGMA}")
    print(f"  NN(16,16), epochs={N_EPOCHS}, patience={PATIENCE}, LR={LR}")
    print(f"  Total runs: {len(N_LEARNED_LIST)}")
    print("=" * 65)

    # --- Generate target once ---
    print(f"\nGenerating ground-truth (seed={SEED_TARGET}, N={N_TARGET}) ...")
    tgt_master = jax.random.PRNGKey(SEED_TARGET)
    tgt_queue_key, tgt_sim_key = jax.random.split(tgt_master)
    sources_tgt = [
        {'mu': jnp.array([TRUE_MU_X, TRUE_MU_Y]),
         'sigma_x': TRUE_SIGMA_X, 'sigma_y': TRUE_SIGMA_Y,
         'n_particles': N_TARGET},
    ]
    tgt_z_init, tgt_wake = build_particle_queue(
        sources=sources_tgt, n_particles=N_TARGET,
        t_final=T_FINAL, dt=DT, key=tgt_queue_key,
    )
    tgt_all_px, tgt_all_py, tgt_all_S_full = simulate_ground_truth_full(
        tgt_z_init, tgt_wake, N_STEPS, DT, tgt_sim_key
    )
    final_step = N_STEPS - 1
    tgt_z = jnp.column_stack([tgt_all_px[final_step], tgt_all_py[final_step]])
    tgt_S = tgt_all_S_full[final_step]
    tgt_mass = float(jnp.sum(jnp.exp(tgt_S)))
    n_alive_tgt = int(jnp.sum(jnp.exp(tgt_S) > ALIVE_THRESHOLD))
    print(f"  Target: mass={tgt_mass:.1f}, ~{n_alive_tgt} alive (N_target={N_TARGET})")

    # --- True profiles + true valleys ---
    x_line = np.linspace(-4, 4, 500)
    ref_true = float(analytical_potential_no_y4(0.0, 0.0, TARGET_PARAMS))
    true_profiles = {}
    for y_val in Y_SLICES:
        true_profiles[y_val] = np.array([
            float(analytical_potential_no_y4(float(xi), y_val, TARGET_PARAMS))
            for xi in x_line
        ]) - ref_true

    def true_phi_fn(z):
        return analytical_potential_no_y4(z[0], z[1], TARGET_PARAMS)
    true_valleys = {}
    for y_val in [0.5, 1.5]:
        true_valleys[y_val] = find_valleys_at_y(true_phi_fn, y_val)
    print(f"  True valleys y=0.5: left x={true_valleys[0.5]['x_left']:.3f}, "
          f"right x={true_valleys[0.5]['x_right']:.3f}, "
          f"asym={true_valleys[0.5]['asymmetry']:.3f}")
    print(f"  True valleys y=1.5: left x={true_valleys[1.5]['x_left']:.3f}, "
          f"right x={true_valleys[1.5]['x_right']:.3f}, "
          f"asym={true_valleys[1.5]['asymmetry']:.3f}")

    # --- Run all N values ---
    all_results = []
    for n_learned, seed_learned in zip(N_LEARNED_LIST, SEED_LEARNED_LIST):
        out_dir = os.path.join(SCRIPT_DIR, f"N{n_learned}")
        res = run_single(n_learned, seed_learned, tgt_z, tgt_S, tgt_mass, out_dir)
        all_results.append(res)

    # --- Save replot_data.npz ---
    save_dict = {
        'x_line': x_line,
        'n_learned_list': np.array(N_LEARNED_LIST),
        'seed_learned_list': np.array(SEED_LEARNED_LIST),
        'n_target': np.array(N_TARGET),
        'seed_target': np.array(SEED_TARGET),
    }
    for y_val in Y_SLICES:
        save_dict[f'true_profile_y{y_val}'] = true_profiles[y_val]
    for r in all_results:
        n = r['n_learned']
        save_dict[f'loss_N{n}'] = np.array(r['history_loss'])
        save_dict[f'mmd_N{n}'] = np.array(r['history_mmd'])
        save_dict[f'mass_N{n}'] = np.array(r['history_mass'])
        for y_val in Y_SLICES:
            save_dict[f'profile_N{n}_y{y_val}'] = r['profiles'][str(y_val)]
    np.savez(os.path.join(SCRIPT_DIR, "replot_data.npz"), **save_dict)

    # --- Save results.json ---
    json_out = {
        'config': {
            'seed_target': SEED_TARGET,
            'seed_learned_list': SEED_LEARNED_LIST,
            'n_target': N_TARGET,
            'n_learned_list': N_LEARNED_LIST,
            'sigma': SIGMA, 'c_x4': C_X4, 'y_d': DEATH_PARAMS_EXP['y_d'],
            'n_epochs': N_EPOCHS, 'patience': PATIENCE,
            'lam_mass': LAM_MASS, 'lr': LR,
        },
        'true_valleys': {str(k): v for k, v in true_valleys.items()},
        'tgt_mass': tgt_mass,
        'n_alive_tgt': n_alive_tgt,
        'runs': [{
            'n_learned': r['n_learned'],
            'seed_learned': r['seed_learned'],
            'best_loss': r['best_loss'],
            'best_epoch': r['best_epoch'],
            'final_mmd': r['final_mmd'],
            'final_mass': r['final_mass'],
            'learned_valleys': {str(k): v for k, v in r['learned_valleys'].items()},
            'ref_learned': r['ref_learned'],
        } for r in all_results],
    }
    with open(os.path.join(SCRIPT_DIR, "results.json"), 'w') as f:
        json.dump(json_out, f, indent=2, default=str)

    # ==================================================================
    # Summary plot 1: Loss vs N
    # ==================================================================
    ns = [r['n_learned'] for r in all_results]
    losses = [r['best_loss'] for r in all_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(ns, losses, 'o-', color='tab:blue', linewidth=2, markersize=8)
    axes[0].set_xlabel('N (learned particles)', fontsize=12)
    axes[0].set_ylabel('Best loss', fontsize=12)
    axes[0].set_title(f'Loss vs N (N_target={N_TARGET})', fontsize=13)
    axes[0].grid(True, alpha=0.3); axes[0].set_xscale('log')
    axes[0].set_xticks(ns)
    axes[0].set_xticklabels(ns, rotation=45, fontsize=9)

    for r in all_results:
        axes[1].semilogy(r['history_loss'], linewidth=1, alpha=0.8,
                         label=f"N={r['n_learned']}")
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Total loss', fontsize=12)
    axes[1].set_title('Training curves', fontsize=13)
    axes[1].legend(fontsize=8, ncol=2); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "summary_loss.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ==================================================================
    # Summary plot 2: Valley positions / asymmetry vs N
    # ==================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for col, y_val in enumerate([0.5, 1.5]):
        true_v = true_valleys[y_val]
        xl_vals = [r['learned_valleys'][y_val]['x_left'] for r in all_results]
        xr_vals = [r['learned_valleys'][y_val]['x_right'] for r in all_results]
        as_vals = [r['learned_valleys'][y_val]['asymmetry'] for r in all_results]

        ax = axes[0, col] if col < 2 else None
        # Use rows: top row positions, bottom row asymmetry
        ax_l = axes[0, col]
        ax_l.plot(ns, xl_vals, 'o-', color='tab:red', label='Learned left', markersize=7)
        ax_l.plot(ns, xr_vals, 's-', color='tab:green', label='Learned right', markersize=7)
        ax_l.axhline(true_v['x_left'], color='red', linestyle='--', alpha=0.7,
                     label=f"True left = {true_v['x_left']:.2f}")
        ax_l.axhline(true_v['x_right'], color='green', linestyle='--', alpha=0.7,
                     label=f"True right = {true_v['x_right']:.2f}")
        ax_l.set_xlabel('N'); ax_l.set_ylabel(r'$x_{valley}$')
        ax_l.set_title(f'Valley x-positions at y = {y_val}')
        ax_l.set_xscale('log'); ax_l.set_xticks(ns)
        ax_l.set_xticklabels(ns, rotation=45, fontsize=8)
        ax_l.legend(fontsize=8); ax_l.grid(True, alpha=0.3)

        ax_a = axes[1, col]
        ax_a.plot(ns, as_vals, 'o-', color='tab:purple', markersize=7,
                  label='Learned asymmetry')
        ax_a.axhline(true_v['asymmetry'], color='purple', linestyle='--', alpha=0.7,
                     label=f"True = {true_v['asymmetry']:.3f}")
        ax_a.set_xlabel('N'); ax_a.set_ylabel(r'$\Phi_{right} - \Phi_{left}$')
        ax_a.set_title(f'Valley asymmetry at y = {y_val}')
        ax_a.set_xscale('log'); ax_a.set_xticks(ns)
        ax_a.set_xticklabels(ns, rotation=45, fontsize=8)
        ax_a.legend(fontsize=8); ax_a.grid(True, alpha=0.3)

    # Use remaining column for a summary table-like text
    axes[0, 2].axis('off')
    axes[1, 2].axis('off')
    summary_txt = "True valleys:\n"
    for y_val in [0.5, 1.5]:
        v = true_valleys[y_val]
        summary_txt += (f"  y={y_val}: xL={v['x_left']:.3f}, xR={v['x_right']:.3f}, "
                        f"asym={v['asymmetry']:.3f}\n")
    summary_txt += f"\nN_target = {N_TARGET}\n"
    summary_txt += f"σ = {SIGMA}, c_x4 = {C_X4}, y_d = {DEATH_PARAMS_EXP['y_d']}\n"
    axes[0, 2].text(0.05, 0.95, summary_txt, fontsize=11, family='monospace',
                    verticalalignment='top', transform=axes[0, 2].transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "summary_valleys.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ==================================================================
    # Summary plot 3: Potential overlays at y slices
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
                    label=f"N={r['n_learned']}")
        ax.set_xlabel('x'); ax.set_ylabel(r'$\Phi$')
        ax.set_title(f'y = {y_val}')
        ax.set_ylim(-15, 15); ax.grid(True, alpha=0.3)
        if col == 2:
            ax.legend(fontsize=7, ncol=2, loc='upper right')
    plt.suptitle(f'Potential profiles vs N (N_target={N_TARGET})',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "summary_potential_overlay.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ==================================================================
    # Summary plot 4: Force overlays
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
                    label=f"N={r['n_learned']}")
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('x'); ax.set_ylabel(r'$F_x$')
        ax.set_title(f'y = {y_val}')
        ax.set_xlim(-3.5, 3.5); ax.grid(True, alpha=0.3)
        if col == 2:
            ax.legend(fontsize=7, ncol=2, loc='upper right')
    plt.suptitle('Force profiles vs N', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "summary_force_overlay.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ==================================================================
    # Summary table
    # ==================================================================
    print(f"\n{'='*100}")
    print(f"Summary: N_target={N_TARGET}")
    for y_val in [0.5, 1.5]:
        v = true_valleys[y_val]
        print(f"  True y={y_val}: xL={v['x_left']:.3f}, xR={v['x_right']:.3f}, "
              f"asym={v['asymmetry']:.3f}")
    print(f"{'='*100}")
    print(f"{'N':>6s} {'seed':>5s} {'Loss':>10s} {'Epoch':>6s} "
          f"{'y=0.5 xL/xR/asym':>26s} {'y=1.5 xL/xR/asym':>26s}")
    print("-" * 100)
    for r in all_results:
        v05 = r['learned_valleys'][0.5]
        v15 = r['learned_valleys'][1.5]
        line = (f"{r['n_learned']:6d} {r['seed_learned']:5d} "
                f"{r['best_loss']:10.6f} {r['best_epoch']:6d} "
                f"({v05['x_left']:+.2f},{v05['x_right']:+.2f},{v05['asymmetry']:+.2f})  "
                f"({v15['x_left']:+.2f},{v15['x_right']:+.2f},{v15['asymmetry']:+.2f})")
        print(line)
    print(f"{'='*100}")

    print("\nDone. Outputs:")
    print(f"  {SCRIPT_DIR}/N*/  (9 per-N folders)")
    print(f"  {SCRIPT_DIR}/summary_loss.png")
    print(f"  {SCRIPT_DIR}/summary_valleys.png")
    print(f"  {SCRIPT_DIR}/summary_potential_overlay.png")
    print(f"  {SCRIPT_DIR}/summary_force_overlay.png")
    print(f"  {SCRIPT_DIR}/results.json")
    print(f"  {SCRIPT_DIR}/replot_data.npz")
