#!/usr/bin/env python
"""
train_sd_nn_multi.py — multi-timepoint NN inference for potential + death rate

Uses neural networks (not parametric forms) to jointly learn:
  - Potential Phi(x,y): 2 -> 16 -> 16 -> 1  (reduced capacity, implicit regularization)
  - Death rate gamma(y): 1 -> 8 -> 8 -> 1   (y-only input, structural constraint)

Key architectural choices:
  1. Small potential network prevents spectral bias from fitting Brownian noise
     as high-frequency potential artifacts (the old 4-layer net had this problem).
  2. Death rate taking only y as input eliminates the x-direction degeneracy:
     the optimizer cannot use asymmetric death to compensate for potential errors,
     forcing the potential to learn correct lateral (x) gradients.
  3. Multi-timepoint transient matching breaks remaining parameter coupling.

Snapshot times: t = 1.0, 2.0, 3.0, 4.0, 6.0, 8.0

Usage:
    python train_sd_nn_multi.py
"""
import matplotlib
matplotlib.use('Agg')

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np

from models import create_sd_model
from training.data_loader import (
    build_particle_queue, analytical_death_rate, analytical_potential,
)
from simulator.sde_solver import simulate_open_system_full, simulate_open_system
from simulator.weighted_mmd import weighted_mmd_loss
from analysis.visualization import (
    plot_nn_landscape, plot_nn_death_rate, plot_training_history, _eval_grid,
)


# ==========================================================================
# 1. Ground-truth parameters
# ==========================================================================

TARGET_PARAMS = {'a': 0, 'b': -1.6, 'c': 3.5, 'd': -1.2, 'sigma': 1.5}
DEATH_PARAMS = {'y_threshold': 2.2}

N_PARTICLES = 2000
DT = 0.01
T_FINAL = 8.0
N_STEPS = int(T_FINAL / DT)  # 800
SIGMA = TARGET_PARAMS['sigma']

SOURCES = [
    {'mu': jnp.array([0.0, -1.0]), 'sigma_src': 0.15, 'n_particles': N_PARTICLES},
]

# Loss weights
LAM_MASS = 1.0

# Snapshot times: early transient + late steady state
SNAPSHOT_TIMES = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
SNAPSHOT_STEPS = [int(t / DT) - 1 for t in SNAPSHOT_TIMES]  # 0-indexed


# ==========================================================================
# 2. Ground-truth simulation (analytical, for target snapshots)
# ==========================================================================

_force_fn = jax.vmap(
    jax.grad(analytical_potential, argnums=(0, 1)),
    in_axes=(0, 0, None),
)


def simulate_ground_truth_full(z_init, wake_schedule, n_steps, dt, key):
    """Run analytical open-system simulation, return full trajectory."""
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

        gamma = analytical_death_rate(px, py, DEATH_PARAMS)
        new_S = S - gamma * dt

        return (new_px, new_py, new_S), (new_px, new_py, new_S)

    step_indices = jnp.arange(n_steps)
    _, (all_px, all_py, all_S) = jax.lax.scan(
        scan_body, (z_init[:, 0], z_init[:, 1], S0),
        (step_keys, step_indices)
    )
    return all_px, all_py, all_S


# ==========================================================================
# 3. Loss functions
# ==========================================================================

def mass_loss(S, target_mass):
    """(M_sim - M_target)^2 / M_target^2"""
    M_sim = jnp.sum(jnp.exp(S))
    return (M_sim - target_mass)**2 / (target_mass**2 + 1e-8)


def weighted_mmd_both(x_sim, alpha_sim, x_obs, alpha_obs,
                      bandwidths=(0.01, 1.0, 100.0)):
    """MMD^2 between two weighted point clouds."""
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


# ==========================================================================
# 4. Multi-snapshot objective (operates on NN model via equinox)
# ==========================================================================

def make_objective(snapshot_steps, n_steps, dt, sigma, lam_mass):
    """Build the multi-snapshot loss function for SourceDeathModel."""

    def objective(model, z_init, wake_schedule,
                  tgt_snap_z, tgt_snap_S, tgt_snap_mass, key):
        """
        Sum of (MMD + mass_loss) at each snapshot, averaged.
        """
        all_z, all_S = simulate_open_system_full(
            model, z_init, wake_schedule, sigma, n_steps, dt, key
        )

        total_mmd = 0.0
        total_mass_l = 0.0

        for i, step in enumerate(snapshot_steps):
            sim_z = all_z[step]
            sim_S = all_S[step]
            sim_alpha = jax.nn.softmax(sim_S)

            tgt_z = tgt_snap_z[i]
            tgt_alpha = jax.nn.softmax(tgt_snap_S[i])

            l_mmd = weighted_mmd_both(sim_z, sim_alpha, tgt_z, tgt_alpha)
            l_mass = mass_loss(sim_S, tgt_snap_mass[i])

            total_mmd += l_mmd
            total_mass_l += l_mass

        n_snaps = len(snapshot_steps)
        avg_mmd = total_mmd / n_snaps
        avg_mass = total_mass_l / n_snaps

        total = avg_mmd + lam_mass * avg_mass
        return total, (avg_mmd, avg_mass)

    return objective


# ==========================================================================
# 5. Training step (equinox-compatible)
# ==========================================================================

def make_train_step(optimizer, objective_fn):
    """Build JIT-compiled training step."""

    @eqx.filter_jit
    def step(model, opt_state, z_init, wake_schedule,
             tgt_snap_z, tgt_snap_S, tgt_snap_mass, key):
        def loss_fn(model):
            return objective_fn(
                model, z_init, wake_schedule,
                tgt_snap_z, tgt_snap_S, tgt_snap_mass, key
            )

        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss, aux

    return step


# ==========================================================================
# 6. Generate ground-truth snapshots
# ==========================================================================

print("=" * 65)
print("Multi-timepoint NN inference: Phi(x,y) + gamma(y)")
print("=" * 65)
print(f"  N_particles = {N_PARTICLES}")
print(f"  T_final = {T_FINAL}, dt = {DT}, n_steps = {N_STEPS}")
print(f"  sigma = {SIGMA} (fixed)")
print(f"  Snapshots at t = {SNAPSHOT_TIMES}")
print(f"  Potential arch: 2 -> 16 -> 16 -> 1")
print(f"  Death rate arch: 1(y) -> 8 -> 8 -> 1 -> softplus")
print(f"  Loss = avg(L_MMD + {LAM_MASS}*L_mass)")

key = jax.random.PRNGKey(4)
key, tgt_queue_key, tgt_sim_key = jax.random.split(key, 3)

print("\nGenerating ground-truth snapshots ...")

tgt_z_init, tgt_wake = build_particle_queue(
    sources=SOURCES, n_particles=N_PARTICLES,
    t_final=T_FINAL, dt=DT, key=tgt_queue_key,
)

tgt_all_px, tgt_all_py, tgt_all_S = simulate_ground_truth_full(
    tgt_z_init, tgt_wake, N_STEPS, DT, tgt_sim_key
)

# Extract and stack snapshots
tgt_snap_z_list = []
tgt_snap_S_list = []
tgt_snap_mass_list = []

for t, step in zip(SNAPSHOT_TIMES, SNAPSHOT_STEPS):
    z_i = jnp.column_stack([tgt_all_px[step], tgt_all_py[step]])
    S_i = tgt_all_S[step]
    mass_i = float(jnp.sum(jnp.exp(S_i)))
    n_alive = int(jnp.sum(jnp.exp(S_i) > 0.01))
    print(f"  t={t:.1f}: mass={mass_i:.2f}, ~{n_alive} alive")
    tgt_snap_z_list.append(z_i)
    tgt_snap_S_list.append(S_i)
    tgt_snap_mass_list.append(mass_i)

tgt_snap_z = jnp.stack(tgt_snap_z_list)      # (n_snaps, N, 2)
tgt_snap_S = jnp.stack(tgt_snap_S_list)      # (n_snaps, N)
tgt_snap_mass = jnp.array(tgt_snap_mass_list) # (n_snaps,)


# ==========================================================================
# 7. Build training particle queue
# ==========================================================================

key, queue_key = jax.random.split(key)
z_init, wake_schedule = build_particle_queue(
    sources=SOURCES, n_particles=N_PARTICLES,
    t_final=T_FINAL, dt=DT, key=queue_key,
)
print(f"  Training queue: {z_init.shape[0]} particles\n")


# ==========================================================================
# 8. Create NN model (reduced capacity)
# ==========================================================================

key, model_key = jax.random.split(key)
model = create_sd_model(
    model_key,
    d_latent=2,
    c_conf=0.01,
    parametric_death=False,
    potential_hidden=(16, 16),      # reduced from (16,32,32,16)
    death_y_only=True,             # gamma(y) not gamma(x,y)
    death_hidden=(8, 8),           # small 1D network
)

n_params_phi = sum(x.size for x in jax.tree.leaves(eqx.filter(model.potential, eqx.is_array)))
n_params_gamma = sum(x.size for x in jax.tree.leaves(eqx.filter(model.death_rate, eqx.is_array)))
print(f"SourceDeathModel created:")
print(f"  Phi: 2->16->16->1 ({n_params_phi} params)")
print(f"  gamma: 1(y)->8->8->1->softplus ({n_params_gamma} params)")
print(f"  Total: {n_params_phi + n_params_gamma} params")


# ==========================================================================
# 9. Train
# ==========================================================================

N_EPOCHS = 1000
PRINT_EVERY = 25
PATIENCE = 200

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=3e-3, weight_decay=1e-4),
)

objective_fn = make_objective(SNAPSHOT_STEPS, N_STEPS, DT, SIGMA, LAM_MASS)
step_fn = make_train_step(optimizer, objective_fn)

opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

history = {'loss': [], 'mmd': [], 'mass': []}
best_loss = float('inf')
best_epoch = 0
best_model_leaves = None
wait = 0
opt_key = jax.random.PRNGKey(42)

print(f"\nTraining for {N_EPOCHS} epochs ({len(SNAPSHOT_TIMES)} snapshots) ...")
print("-" * 65)

for epoch in range(N_EPOCHS):
    opt_key, sim_key = jax.random.split(opt_key)

    model, opt_state, loss, (l_mmd, l_mass) = step_fn(
        model, opt_state, z_init, wake_schedule,
        tgt_snap_z, tgt_snap_S, tgt_snap_mass, sim_key,
    )

    loss_val = float(loss)
    history['loss'].append(loss_val)
    history['mmd'].append(float(l_mmd))
    history['mass'].append(float(l_mass))

    if epoch % PRINT_EVERY == 0:
        print(
            f"Epoch {epoch:4d}: Loss={loss_val:.5f} "
            f"(MMD={float(l_mmd):.5f}, Mass={float(l_mass):.5f})"
        )

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
    print(f"Restored best model from epoch {best_epoch} (loss={best_loss:.6f})")

eqx.tree_serialise_leaves("trained_model_sd_nn_multi.eqx", model)
print("Model saved to trained_model_sd_nn_multi.eqx")


# ==========================================================================
# 10. Visualisation
# ==========================================================================

# --- Figure 1: Training curves ---
fig, ax = plt.subplots(figsize=(8, 5))
epochs_arr = np.arange(len(history['loss']))
ax.semilogy(epochs_arr, history['loss'], label='Total', linewidth=1.5)
ax.semilogy(epochs_arr, history['mmd'], label='MMD (avg)', linewidth=1, alpha=0.8)
ax.semilogy(epochs_arr, history['mass'], label='Mass (avg)', linewidth=1, alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss (NN multi-timepoint)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_nn_multi.png", dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 2: Learned landscape + death rate vs ground truth ---
fig, axes = plt.subplots(1, 4, figsize=(24, 5))

# Learned potential
plot_nn_landscape(model, ax=axes[0], title="Learned Potential (NN)")

# True potential
x_grid = jnp.linspace(-5, 5, 200)
y_grid = jnp.linspace(-2, 4, 200)
Xg, Yg = jnp.meshgrid(x_grid, y_grid)
Z_true = jax.vmap(jax.vmap(
    lambda x, y: analytical_potential(x, y, TARGET_PARAMS)
))(Xg, Yg)
cf = axes[1].contourf(np.array(Xg), np.array(Yg), np.array(Z_true),
                       levels=30, cmap='viridis')
plt.colorbar(cf, ax=axes[1], label=r'$\Phi^*(x,y)$')
axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
axes[1].set_title('True Potential')

# Learned death rate
gamma_true = analytical_death_rate(Xg, Yg, DEATH_PARAMS)
_, _, gamma_learned = _eval_grid(lambda z: model.death_rate(z), (-5, 5), (-2, 4), 200)

g_vmin = min(float(gamma_true.min()), float(gamma_learned.min()))
g_vmax = max(float(gamma_true.max()), float(gamma_learned.max()))

plot_nn_death_rate(model, ax=axes[2], title=r"Learned $\gamma(y)$",
                   vmin=g_vmin, vmax=g_vmax)

# True death rate
levels_g = np.linspace(g_vmin, g_vmax, 31)
cf = axes[3].contourf(np.array(Xg), np.array(Yg), np.array(gamma_true),
                       levels=levels_g, cmap='YlOrRd', extend='both')
plt.colorbar(cf, ax=axes[3], label=r'$\gamma^*(x,y)$')
axes[3].set_xlabel('x'); axes[3].set_ylabel('y')
axes[3].set_title(r"True $\gamma^*(x,y)$")

plt.tight_layout()
plt.savefig("landscape_nn_multi.png", dpi=150, bbox_inches='tight')
plt.close()


# --- Figure 3: Death rate 1D profile comparison ---
fig, ax = plt.subplots(figsize=(7, 5))
y_line = np.linspace(-2, 4, 200)
gamma_1d_true = np.array(jax.nn.softplus(jnp.array(y_line) - DEATH_PARAMS['y_threshold']))
# Evaluate learned gamma along y-axis (x=0)
gamma_1d_learned = np.array(jax.vmap(
    lambda y: model.death_rate(jnp.array([0.0, y]))
)(jnp.array(y_line)))
ax.plot(y_line, gamma_1d_true, 'b-', linewidth=2, label='True')
ax.plot(y_line, gamma_1d_learned, 'r--', linewidth=2, label='Learned')
ax.set_xlabel('y')
ax.set_ylabel(r'$\gamma(y)$')
ax.set_title('Death rate profile (y-only network)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("death_rate_nn_multi.png", dpi=150, bbox_inches='tight')
plt.close()


# --- Figure 4: Multi-snapshot particle comparison ---
VIS_TIMES = [1.0, 2.0, 4.0, 8.0]
VIS_STEPS = [int(t / DT) - 1 for t in VIS_TIMES]

key, eval_key, eval_queue_key = jax.random.split(key, 3)
z_init_eval, wake_eval = build_particle_queue(
    sources=SOURCES, n_particles=N_PARTICLES,
    t_final=T_FINAL, dt=DT, key=eval_queue_key,
)

sim_all_z, sim_all_S = simulate_open_system_full(
    model, z_init_eval, wake_eval, SIGMA, N_STEPS, DT, eval_key
)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for col, (t_vis, step_vis) in enumerate(zip(VIS_TIMES, VIS_STEPS)):
    # Top row: target
    ax_tgt = axes[0, col]
    tgt_z_i = jnp.column_stack([tgt_all_px[step_vis], tgt_all_py[step_vis]])
    tgt_S_i = tgt_all_S[step_vis]
    tgt_w_i = np.array(jnp.clip(jnp.exp(tgt_S_i), 0.0, 1.0))

    ax_tgt.contourf(np.array(Xg), np.array(Yg), np.array(Z_true),
                    levels=30, cmap='viridis', alpha=0.3)
    sc = ax_tgt.scatter(np.array(tgt_z_i[:, 0]), np.array(tgt_z_i[:, 1]),
                        c=tgt_w_i, cmap='coolwarm', s=6, alpha=0.5,
                        vmin=0, vmax=1, edgecolors='none')
    ax_tgt.set_xlim(-5, 5); ax_tgt.set_ylim(-2, 4)
    ax_tgt.set_title(f'Target t={t_vis:.0f}')
    if col == 0:
        ax_tgt.set_ylabel('y (target)')

    # Bottom row: simulated (learned NN)
    ax_sim = axes[1, col]
    sim_z_i = sim_all_z[step_vis]
    sim_S_i = sim_all_S[step_vis]
    sim_w_i = np.array(jnp.clip(jnp.exp(sim_S_i), 0.0, 1.0))

    # Get learned potential for background
    Z_learned = _eval_grid(model.potential, (-5, 5), (-2, 4), 200)[2]
    ax_sim.contourf(np.array(Xg), np.array(Yg), Z_learned,
                    levels=30, cmap='viridis', alpha=0.3)
    sc = ax_sim.scatter(np.array(sim_z_i[:, 0]), np.array(sim_z_i[:, 1]),
                        c=sim_w_i, cmap='coolwarm', s=6, alpha=0.5,
                        vmin=0, vmax=1, edgecolors='none')
    ax_sim.set_xlim(-5, 5); ax_sim.set_ylim(-2, 4)
    ax_sim.set_title(f'Learned t={t_vis:.0f}')
    if col == 0:
        ax_sim.set_ylabel('y (learned)')

fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
fig.colorbar(sc, cax=cbar_ax, label=r'$w = e^S$')

plt.savefig("snapshots_nn_multi.png", dpi=150, bbox_inches='tight')
plt.close()

print("\nDone. Figures saved:")
print("  training_nn_multi.png")
print("  landscape_nn_multi.png")
print("  death_rate_nn_multi.png")
print("  snapshots_nn_multi.png")
