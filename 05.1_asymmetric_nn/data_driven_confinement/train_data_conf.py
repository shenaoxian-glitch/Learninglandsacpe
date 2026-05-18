#!/usr/bin/env python
"""
train_data_conf.py — NN potential + data-driven anisotropic confinement

Replaces the true quartic confinement (exp(b)*x^4 + exp(d)*y^4) with
confinement coefficients C_x, C_y estimated from target data + known
death rate, following the Joint Energy-Survival Horizon method:

  1. Empirical horizon: R_99,i = 99th %ile of |z_i| in target data
  2. Survival horizon: R_gamma,y from gamma_term(y) = gamma_threshold
     (evaluated at well x-locations, no x survival horizon)
  3. Joint boundary: R_joint,i = max(R_99,i, R_gamma,i)
  4. Energy matching: C_i = 10*sigma^2 / (1.2 * R_joint,i)^4

Target data: N=2000 from sweep_nparticles_v3/analysis_data.npz
Result: C_x = 0.0777, C_y = 0.1061

Usage:
    cd 05.1_asymmetric_nn
    python data_driven_confinement/train_data_conf.py > data_driven_confinement/log.txt 2>&1
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

N_PARTICLES = 2000    # match target data size
DT = 0.01
T_FINAL = 6.0
N_STEPS = int(T_FINAL / DT)
SIGMA = 1.5
LAM_MASS = 5.0

SOURCES = [
    {'mu': jnp.array([0.0, -1.0]), 'sigma_x': 0.25, 'sigma_y': 0.10,
     'n_particles': N_PARTICLES},
]

N_EPOCHS = 800
PRINT_EVERY = 50
PATIENCE = 300

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ======================================================================
# Compute data-driven confinement from target data
# ======================================================================

def compute_confinement(tgt_z, sigma, death_params, well_x_locs,
                        gamma_threshold=5.0, safety=1.2, E_factor=10.0):
    """
    Joint Energy-Survival Horizon method for anisotropic confinement.

    Args:
        tgt_z: (N, 2) target particle positions
        sigma: diffusion coefficient
        death_params: dict with B, k2, y_max for gamma_term
        well_x_locs: list of x-coordinates of potential wells
        gamma_threshold: death rate at survival horizon
        safety: multiplicative safety margin on R_joint
        E_factor: energy target in units of sigma^2

    Returns:
        C_x, C_y, info_dict
    """
    B, k2, y_max = death_params['B'], death_params['k2'], death_params['y_max']

    # Step 1: Empirical horizon
    R99_x = float(np.percentile(np.abs(tgt_z[:, 0]), 99))
    R99_y = float(np.percentile(np.abs(tgt_z[:, 1]), 99))

    # Step 2: Survival horizon
    # x-direction: gamma_select decays with |x|, no survival horizon
    R_gamma_x = None

    # y-direction: evaluate gamma at well locations (where mass concentrates)
    # gamma at wells ≈ gamma_term(y) since gamma_select is negligible at |x|>1.5
    # Use min over wells (most permissive — if particles in either well survive, wall shouldn't be there)
    def gamma_at_well(x_well, y):
        A, w_x, k1, y_sel = (death_params['A'], death_params['w_x'],
                              death_params['k1'], death_params['y_select'])
        gs = A * np.exp(-x_well**2 / (2 * w_x**2)) * np.log1p(np.exp(k1 * (y - y_sel)))
        gt = B * np.log1p(np.exp(k2 * (y - y_max)))
        return gs + gt

    y_scan = np.linspace(0, 8, 100000)
    gamma_min = np.full_like(y_scan, np.inf)
    for x_w in well_x_locs:
        gamma_w = np.array([gamma_at_well(x_w, y) for y in y_scan])
        gamma_min = np.minimum(gamma_min, gamma_w)

    idx = np.argmax(gamma_min >= gamma_threshold)
    R_gamma_y = float(y_scan[idx]) if gamma_min[idx] >= gamma_threshold else None

    # Step 3: Joint boundary
    R_joint_x = R99_x
    R_joint_y = max(R99_y, R_gamma_y) if R_gamma_y is not None else R99_y

    # Step 4: Energy matching
    energy_target = E_factor * sigma**2
    R_bnd_x = safety * R_joint_x
    R_bnd_y = safety * R_joint_y
    C_x = energy_target / R_bnd_x**4
    C_y = energy_target / R_bnd_y**4

    info = {
        'R99_x': R99_x, 'R99_y': R99_y,
        'R_gamma_y': R_gamma_y,
        'R_joint_x': R_joint_x, 'R_joint_y': R_joint_y,
        'R_bnd_x': R_bnd_x, 'R_bnd_y': R_bnd_y,
        'C_x': C_x, 'C_y': C_y,
        'gamma_threshold': gamma_threshold,
        'safety': safety, 'E_factor': E_factor,
        'energy_target': energy_target,
    }
    return C_x, C_y, info


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


class PotentialWithAnisotropicConfinement(eqx.Module):
    """NN potential + data-driven anisotropic quartic confinement.

    Phi(z) = Phi_nn(z) + C_x * x^4 + C_y * y^4

    C_x, C_y are fixed (static) — only Phi_nn is learned.
    """
    nn: PotentialNN
    c_x: float = eqx.field(static=True)
    c_y: float = eqx.field(static=True)

    def __call__(self, z):
        x, y = z[0], z[1]
        conf = self.c_x * x**4 + self.c_y * y**4
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
    def objective(model, z_init, wake_schedule,
                  tgt_z, tgt_S, tgt_mass, key):
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
# Main
# ======================================================================

if __name__ == '__main__':

    print("=" * 65)
    print("Data-driven anisotropic confinement: NN Phi + C_x*x^4 + C_y*y^4")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Load saved target data (N=2000)
    # ------------------------------------------------------------------
    data_path = os.path.join(os.path.dirname(__file__), '..',
                             'sweep_nparticles_v3', 'analysis_data.npz')
    saved = np.load(data_path, allow_pickle=True)
    tgt_z_np = saved['tgt_z_final']   # (2000, 2)
    tgt_S_np = saved['tgt_S_final']   # (2000,)
    print(f"\nLoaded target data: {data_path}")
    print(f"  N = {tgt_z_np.shape[0]}, shape = {tgt_z_np.shape}")

    # ------------------------------------------------------------------
    # Compute confinement coefficients
    # ------------------------------------------------------------------
    # Well locations from true potential critical points
    well_x_locs = [-2.028, 1.804]

    C_x, C_y, conf_info = compute_confinement(
        tgt_z_np, SIGMA, DEATH_PARAMS_2D, well_x_locs,
        gamma_threshold=5.0, safety=1.2, E_factor=10.0,
    )

    print(f"\nConfinement computation:")
    print(f"  R_99:    x={conf_info['R99_x']:.3f}, y={conf_info['R99_y']:.3f}")
    print(f"  R_gamma: y={conf_info['R_gamma_y']:.3f} (gamma_thresh=5.0)")
    print(f"  R_joint: x={conf_info['R_joint_x']:.3f}, y={conf_info['R_joint_y']:.3f}")
    print(f"  R_bnd:   x={conf_info['R_bnd_x']:.3f}, y={conf_info['R_bnd_y']:.3f}")
    print(f"  C_x = {C_x:.6f}  (true exp(b) = {np.exp(-1.6):.6f}, ratio={C_x/np.exp(-1.6):.3f})")
    print(f"  C_y = {C_y:.6f}  (true exp(d) = {np.exp(-1.2):.6f}, ratio={C_y/np.exp(-1.2):.3f})")

    # Convert target data to JAX
    tgt_z = jnp.array(tgt_z_np)
    tgt_S = jnp.array(tgt_S_np)
    tgt_mass = float(jnp.sum(jnp.exp(tgt_S)))
    n_alive = int(jnp.sum(jnp.exp(tgt_S) > 0.01))
    print(f"  Target mass = {tgt_mass:.2f}, ~{n_alive} alive")

    # ------------------------------------------------------------------
    # Also generate ground-truth full trajectory (for snapshot plots)
    # ------------------------------------------------------------------
    # Use same seed as sweep_nparticles_v3 target (seed=42)
    print("\nGenerating ground-truth full trajectory for visualization ...")
    gt_key = jax.random.PRNGKey(42)
    gt_queue_key, gt_sim_key = jax.random.split(gt_key)
    gt_z_init, gt_wake = build_particle_queue(
        sources=SOURCES, n_particles=N_PARTICLES,
        t_final=T_FINAL, dt=DT, key=gt_queue_key,
    )
    tgt_all_px, tgt_all_py, tgt_all_S = simulate_ground_truth_full(
        gt_z_init, gt_wake, N_STEPS, DT, gt_sim_key
    )

    # ------------------------------------------------------------------
    # Build training particle queue (different seed)
    # ------------------------------------------------------------------
    key = jax.random.PRNGKey(4)
    key, queue_key, model_key = jax.random.split(key, 3)

    z_init, wake_schedule = build_particle_queue(
        sources=SOURCES, n_particles=N_PARTICLES,
        t_final=T_FINAL, dt=DT, key=queue_key,
    )
    print(f"  Training queue: {z_init.shape[0]} particles")

    # Scale target mass for particle count mismatch (here N_train == N_target)
    scaled_tgt_mass = tgt_mass  # same size, no scaling needed

    # ------------------------------------------------------------------
    # Create model
    # ------------------------------------------------------------------
    phi_nn = PotentialNN(model_key, d_latent=2, c_conf=0.0,
                         hidden_sizes=(16, 16))
    potential = PotentialWithAnisotropicConfinement(
        nn=phi_nn, c_x=C_x, c_y=C_y,
    )
    fixed_death = FixedDeathRate2D(**DEATH_PARAMS_2D)
    model = SourceDeathModel(potential=potential, death_rate=fixed_death)

    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"\nModel: Phi_nn(2->16->16->1) + {C_x:.4f}*x^4 + {C_y:.4f}*y^4")
    print(f"  Learnable params: {n_params}")
    print(f"  Death rate: fixed 2D analytical")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
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
    opt_key = jax.random.PRNGKey(42)

    print(f"\nTraining for {N_EPOCHS} epochs ...")
    print("-" * 65)

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
            print(f"Epoch {epoch:4d}: Loss={loss_val:.5f} "
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
            print(f"Early stopping at epoch {epoch}")
            break

    # Restore best model
    if best_model_leaves is not None:
        model = eqx.combine(best_model_leaves,
                             eqx.filter(model, lambda x: not eqx.is_array(x)))
        print(f"\nRestored best model from epoch {best_epoch} (loss={best_loss:.6f})")

    model_path = os.path.join(SCRIPT_DIR, "trained_model_data_conf.eqx")
    eqx.tree_serialise_leaves(model_path, model)
    print(f"Model saved to {model_path}")

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    results = {
        'confinement': conf_info,
        'training': {
            'n_particles': N_PARTICLES,
            'sigma': SIGMA,
            'n_epochs_run': len(history['loss']),
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'final_mmd': history['mmd'][-1],
            'final_mass': history['mass'][-1],
        },
        'comparison': {
            'true_exp_b': float(np.exp(-1.6)),
            'true_exp_d': float(np.exp(-1.2)),
            'C_x': C_x,
            'C_y': C_y,
            'ratio_Cx': C_x / np.exp(-1.6),
            'ratio_Cy': C_y / np.exp(-1.2),
        },
    }
    results_path = os.path.join(SCRIPT_DIR, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # ==================================================================
    # Visualization
    # ==================================================================

    # --- Grids ---
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

    # Align potentials at origin
    ref_true = float(analytical_potential(0.0, 0.0, TARGET_PARAMS))
    ref_learned = float(model.potential(jnp.array([0.0, 0.0])))
    Z_true = Z_true - ref_true
    Z_learned = Z_learned - ref_learned

    pot_vmin = min(float(jnp.min(Z_true)), float(jnp.min(Z_learned)))
    pot_vmax = min(max(float(jnp.max(Z_true)), float(jnp.max(Z_learned))), 30.0)
    pot_levels = np.linspace(pot_vmin, pot_vmax, 31)

    # --- Figure 1: Training curves ---
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs_arr = np.arange(len(history['loss']))
    ax.semilogy(epochs_arr, history['loss'], label='Total', linewidth=1.5)
    ax.semilogy(epochs_arr, history['mmd'], label='MMD', linewidth=1, alpha=0.8)
    ax.semilogy(epochs_arr, history['mass'], label='Mass', linewidth=1, alpha=0.8)
    ax.axvline(best_epoch, color='k', ls='--', alpha=0.4, label=f'Best (ep {best_epoch})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Training — data-driven confinement (C_x={C_x:.4f}, C_y={C_y:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "training.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Figure 2: Potential contours + 1D slices ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    cf = axes[0, 0].contourf(np.array(Xg), np.array(Yg), np.array(Z_learned),
                              levels=pot_levels, cmap='viridis', extend='both')
    plt.colorbar(cf, ax=axes[0, 0], label=r'$\Phi_{learned}$')
    axes[0, 0].set_title('Learned Potential')
    axes[0, 0].set_xlabel('x'); axes[0, 0].set_ylabel('y')

    cf = axes[0, 1].contourf(np.array(Xg), np.array(Yg), np.array(Z_true),
                              levels=pot_levels, cmap='viridis', extend='both')
    plt.colorbar(cf, ax=axes[0, 1], label=r'$\Phi^*$')
    axes[0, 1].set_title('True Potential')
    axes[0, 1].set_xlabel('x'); axes[0, 1].set_ylabel('y')

    Z_diff = np.array(Z_learned) - np.array(Z_true)
    d_absmax = min(max(abs(Z_diff.min()), abs(Z_diff.max())), 15.0)
    diff_levels = np.linspace(-d_absmax, d_absmax, 31)
    cf = axes[0, 2].contourf(np.array(Xg), np.array(Yg), Z_diff,
                              levels=diff_levels, cmap='RdBu_r', extend='both')
    plt.colorbar(cf, ax=axes[0, 2], label=r'$\Phi_{learned} - \Phi^*$')
    axes[0, 2].set_title('Potential Error')
    axes[0, 2].set_xlabel('x'); axes[0, 2].set_ylabel('y')

    # 1D slices
    y_slices = [-0.5, 0.5, 1.5]
    x_line = np.linspace(-4, 4, 300)
    for col, y_val in enumerate(y_slices):
        ax = axes[1, col]
        phi_true = np.array([
            float(analytical_potential(float(xi), y_val, TARGET_PARAMS))
            for xi in x_line
        ]) - ref_true
        phi_learned = np.array(jax.vmap(
            lambda xi: model.potential(jnp.array([xi, y_val]))
        )(jnp.array(x_line))) - ref_learned

        ax.plot(x_line, phi_true, 'b-', linewidth=2, label='True')
        ax.plot(x_line, phi_learned, 'r--', linewidth=2, label='Learned')
        ax.set_xlabel('x')
        ax.set_ylabel(r'$\Phi(x, y)$')
        ax.set_title(f'Slice at y = {y_val}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-15, 15)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "landscape.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Figure 3: Force field comparison ---
    n_arrow = 22
    xs_q = np.linspace(-3.5, 3.5, n_arrow)
    ys_q = np.linspace(-1.5, 3.0, n_arrow)
    Xq, Yq = np.meshgrid(xs_q, ys_q)

    grad_true = jax.grad(analytical_potential, argnums=(0, 1))
    Fx_true = np.zeros_like(Xq)
    Fy_true = np.zeros_like(Yq)
    for i in range(n_arrow):
        for j in range(n_arrow):
            gx, gy = grad_true(float(Xq[i, j]), float(Yq[i, j]), TARGET_PARAMS)
            Fx_true[i, j] = -float(gx)
            Fy_true[i, j] = -float(gy)

    grad_learned_fn = jax.grad(lambda z: model.potential(z))
    Fx_learned = np.zeros_like(Xq)
    Fy_learned = np.zeros_like(Yq)
    for i in range(n_arrow):
        for j in range(n_arrow):
            g = grad_learned_fn(jnp.array([Xq[i, j], Yq[i, j]]))
            Fx_learned[i, j] = -float(g[0])
            Fy_learned[i, j] = -float(g[1])

    mag_true = np.sqrt(Fx_true**2 + Fy_true**2)
    mag_learned = np.sqrt(Fx_learned**2 + Fy_learned**2)
    mag_max = max(mag_true.max(), mag_learned.max())

    fig, axes_f = plt.subplots(1, 2, figsize=(14, 6))
    for ax, Fx, Fy, mag, Z_bg, title in [
        (axes_f[0], Fx_true, Fy_true, mag_true, np.array(Z_true),
         r'True $\mathbf{F} = -\nabla\Phi^*$'),
        (axes_f[1], Fx_learned, Fy_learned, mag_learned, np.array(Z_learned),
         r'Learned $\mathbf{F} = -\nabla\Phi_{NN}$'),
    ]:
        ax.contourf(np.array(Xg), np.array(Yg), Z_bg,
                     levels=pot_levels, cmap='viridis', alpha=0.35)
        ax.contour(np.array(Xg), np.array(Yg), Z_bg,
                   levels=pot_levels, colors='0.5', linewidths=0.4, alpha=0.5)
        q = ax.quiver(Xq, Yq, Fx, Fy, mag,
                      cmap='hot_r', scale=mag_max * 5,
                      width=0.004, headwidth=3.5, headlength=4,
                      headaxislength=3.5, clim=(0, mag_max), zorder=3)
        ax.set_xlim(-3.8, 3.8); ax.set_ylim(-1.8, 3.3)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_aspect('equal')

    fig.subplots_adjust(right=0.88, wspace=0.25)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    fig.colorbar(q, cax=cbar_ax, label=r'$|\mathbf{F}|$')
    plt.savefig(os.path.join(SCRIPT_DIR, "force_field.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Figure 4: Snapshot comparison ---
    VIS_TIMES = [1.0, 2.0, 4.0, 6.0]
    VIS_STEPS = [int(t / DT) - 1 for t in VIS_TIMES]
    final_step = N_STEPS - 1

    key, eval_key, eval_queue_key = jax.random.split(key, 3)
    z_init_eval, wake_eval = build_particle_queue(
        sources=SOURCES, n_particles=N_PARTICLES,
        t_final=T_FINAL, dt=DT, key=eval_queue_key,
    )
    sim_all_z, sim_all_S = simulate_open_system_full(
        model, z_init_eval, wake_eval, SIGMA, N_STEPS, DT, eval_key
    )

    fig, axes_s = plt.subplots(2, 4, figsize=(20, 10))
    for col, (t_vis, step_vis) in enumerate(zip(VIS_TIMES, VIS_STEPS)):
        # Top row: target
        ax_tgt = axes_s[0, col]
        tgt_z_i = jnp.column_stack([tgt_all_px[step_vis], tgt_all_py[step_vis]])
        tgt_S_i = tgt_all_S[step_vis]
        tgt_w_i = np.array(jnp.clip(jnp.exp(tgt_S_i), 0.0, 1.0))
        alive = tgt_w_i > 1e-6

        ax_tgt.contourf(np.array(Xg), np.array(Yg), np.array(Z_true),
                        levels=pot_levels, cmap='viridis', alpha=0.3)
        ax_tgt.scatter(np.array(tgt_z_i[alive, 0]), np.array(tgt_z_i[alive, 1]),
                       c=tgt_w_i[alive], cmap='coolwarm', s=6, alpha=0.5,
                       vmin=0, vmax=1, edgecolors='none')
        ax_tgt.set_xlim(-4, 4); ax_tgt.set_ylim(-2, 3.5)
        ax_tgt.set_title(f'Target t={t_vis:.0f}')
        if col == 0:
            ax_tgt.set_ylabel('y (target)')

        # Bottom row: learned
        ax_sim = axes_s[1, col]
        sim_z_i = sim_all_z[step_vis]
        sim_S_i = sim_all_S[step_vis]
        sim_w_i = np.array(jnp.clip(jnp.exp(sim_S_i), 0.0, 1.0))
        alive_s = sim_w_i > 1e-6

        ax_sim.contourf(np.array(Xg), np.array(Yg), np.array(Z_learned),
                        levels=pot_levels, cmap='viridis', alpha=0.3)
        sc = ax_sim.scatter(np.array(sim_z_i[alive_s, 0]),
                            np.array(sim_z_i[alive_s, 1]),
                            c=sim_w_i[alive_s], cmap='coolwarm', s=6, alpha=0.5,
                            vmin=0, vmax=1, edgecolors='none')
        ax_sim.set_xlim(-4, 4); ax_sim.set_ylim(-2, 3.5)
        ax_sim.set_title(f'Learned t={t_vis:.0f}')
        if col == 0:
            ax_sim.set_ylabel('y (learned)')

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label=r'$w = e^S$')
    plt.savefig(os.path.join(SCRIPT_DIR, "snapshots.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Figure 5: Confinement comparison ---
    fig, axes_c = plt.subplots(1, 2, figsize=(14, 5))

    x_line_c = np.linspace(-5, 5, 500)
    conf_data_x = C_x * x_line_c**4
    conf_true_x = np.exp(-1.6) * x_line_c**4
    axes_c[0].plot(x_line_c, conf_data_x, 'r-', lw=2,
                   label=f'Data-driven $C_x={C_x:.4f}$')
    axes_c[0].plot(x_line_c, conf_true_x, 'b--', lw=2,
                   label=f'True $e^b={np.exp(-1.6):.4f}$')
    axes_c[0].axvline(conf_info['R_bnd_x'], color='r', ls=':', alpha=0.5, label='R_bnd,x')
    axes_c[0].axvline(-conf_info['R_bnd_x'], color='r', ls=':', alpha=0.5)
    axes_c[0].axhline(conf_info['energy_target'], color='gray', ls=':', alpha=0.5,
                      label=f'$10\\sigma^2 = {conf_info["energy_target"]:.1f}$')
    axes_c[0].set_xlabel('x'); axes_c[0].set_ylabel(r'$\Phi_{conf}$')
    axes_c[0].set_title('x-confinement')
    axes_c[0].legend(); axes_c[0].set_ylim(0, 50); axes_c[0].grid(True, alpha=0.3)

    y_line_c = np.linspace(-3, 5, 500)
    conf_data_y = C_y * y_line_c**4
    conf_true_y = np.exp(-1.2) * y_line_c**4
    axes_c[1].plot(y_line_c, conf_data_y, 'r-', lw=2,
                   label=f'Data-driven $C_y={C_y:.4f}$')
    axes_c[1].plot(y_line_c, conf_true_y, 'b--', lw=2,
                   label=f'True $e^d={np.exp(-1.2):.4f}$')
    axes_c[1].axvline(conf_info['R_bnd_y'], color='r', ls=':', alpha=0.5, label='R_bnd,y')
    axes_c[1].axvline(-conf_info['R_bnd_y'], color='r', ls=':', alpha=0.5)
    axes_c[1].axhline(conf_info['energy_target'], color='gray', ls=':', alpha=0.5,
                      label=f'$10\\sigma^2 = {conf_info["energy_target"]:.1f}$')
    axes_c[1].set_xlabel('y'); axes_c[1].set_ylabel(r'$\Phi_{conf}$')
    axes_c[1].set_title('y-confinement')
    axes_c[1].legend(); axes_c[1].set_ylim(0, 50); axes_c[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "confinement_comparison.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # --- Save plot data ---
    np.savez(os.path.join(SCRIPT_DIR, "plot_data.npz"),
             # Training history
             loss=np.array(history['loss']),
             mmd=np.array(history['mmd']),
             mass=np.array(history['mass']),
             best_epoch=best_epoch,
             # Target data used
             tgt_z=tgt_z_np, tgt_S=tgt_S_np,
             # Confinement
             C_x=C_x, C_y=C_y,
             # Potential grids
             x_grid=np.array(x_grid), y_grid=np.array(y_grid),
             Z_true=np.array(Z_true), Z_learned=np.array(Z_learned),
             ref_true=ref_true, ref_learned=ref_learned,
             # 1D slices
             x_line=x_line,
             )

    print("\nDone. Saved:")
    print(f"  {SCRIPT_DIR}/training.png")
    print(f"  {SCRIPT_DIR}/landscape.png")
    print(f"  {SCRIPT_DIR}/force_field.png")
    print(f"  {SCRIPT_DIR}/snapshots.png")
    print(f"  {SCRIPT_DIR}/confinement_comparison.png")
    print(f"  {SCRIPT_DIR}/results.json")
    print(f"  {SCRIPT_DIR}/plot_data.npz")
    print(f"  {SCRIPT_DIR}/trained_model_data_conf.eqx")
