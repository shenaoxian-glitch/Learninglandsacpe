#!/usr/bin/env python
"""
resume_freeze_nn.py — Freeze NN, train only source params (mu_x, mu_y).

Loads a trained model and optimizes only mu_x, mu_y with the NN potential
frozen. This removes the NN-source degeneracy and lets the source params
converge faster.

Usage:
    python resume_freeze_nn.py > log_freeze_nn.txt 2>&1
"""
import matplotlib
matplotlib.use('Agg')

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np
import pickle

from models.potential import PotentialNN
from training.data_loader import (
    build_particle_queue, analytical_death_rate_2d, analytical_potential,
)
from simulator.sde_solver import simulate_open_system_full

# ---- Constants (must match train_learnable_source.py) ----
TARGET_PARAMS = {
    'a': -1.0, 'b': -1.6, 'c': 3.5, 'd': -1.2,
    'e': 0.7, 'sigma': 1.5,
}
DEATH_PARAMS_2D = {
    'A': 2.0, 'w_x': 0.5, 'k1': 1.0, 'y_select': 1.0,
    'B': 5.0, 'k2': 3.0, 'y_max': 3.0,
}
N_PARTICLES = 6000
DT = 0.01
T_FINAL = 6.0
N_STEPS = int(T_FINAL / DT)
SIGMA = 2.5
TRUE_MU_X, TRUE_MU_Y = 0.0, -1.0
TRUE_SIGMA_X, TRUE_SIGMA_Y = 0.25, 0.10
INIT_MU_X, INIT_MU_Y = -0.5, -1.5
SOURCES = [{'mu': jnp.array([TRUE_MU_X, TRUE_MU_Y]),
            'sigma_x': TRUE_SIGMA_X, 'sigma_y': TRUE_SIGMA_Y,
            'n_particles': N_PARTICLES}]
LAM_MASS = 5.0

# ---- Phase 2 settings ----
CHECKPOINT_MODEL = "trained_model_learnable_source_N6000_ep7000_resumed.eqx"
N_EPOCHS = 1000
PRINT_EVERY = 20
LR_SOURCE = 5e-2  # higher lr since only training 2 params

# ---- Model classes ----
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
        gamma_select = (self.A * jnp.exp(-x**2 / (2 * self.w_x**2))
                        * jax.nn.softplus(self.k1 * (y - self.y_select)))
        gamma_term = self.B * jax.nn.softplus(self.k2 * (y - self.y_max))
        return gamma_select + gamma_term

class PotentialWithConfinement(eqx.Module):
    nn: PotentialNN
    b: float = eqx.field(static=True)
    d: float = eqx.field(static=True)
    def __call__(self, z):
        x, y = z[0], z[1]
        return self.nn(z) + jnp.exp(self.b) * x**4 + jnp.exp(self.d) * y**4

class SourceDeathModelWithSource(eqx.Module):
    potential: eqx.Module
    death_rate: eqx.Module
    mu_x: jax.Array
    mu_y: jax.Array
    sigma_x: float = eqx.field(static=True)
    sigma_y: float = eqx.field(static=True)

def make_z_init(model, eps):
    x = model.mu_x + eps[:, 0] * model.sigma_x
    y = model.mu_y + eps[:, 1] * model.sigma_y
    return jnp.column_stack([x, y])

# ---- Loss functions ----
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
    def objective(model, eps, wake_schedule, tgt_z, tgt_S, tgt_mass, key):
        z_init = make_z_init(model, eps)
        all_z, all_S = simulate_open_system_full(
            model, z_init, wake_schedule, sigma, n_steps, dt, key)
        sim_z, sim_S = all_z[-1], all_S[-1]
        sim_alpha = jax.nn.softmax(sim_S)
        tgt_alpha = jax.nn.softmax(tgt_S)
        l_mmd = weighted_mmd_both(sim_z, sim_alpha, tgt_z, tgt_alpha)
        l_mass = mass_loss(sim_S, tgt_mass)
        return l_mmd + lam_mass * l_mass, (l_mmd, l_mass)
    return objective


# ==========================================================================
# 1. Generate target data (same seeds as original)
# ==========================================================================

print("=" * 65)
print("Phase 2: Freeze NN, train only mu_x, mu_y")
print("=" * 65)
print(f"  Checkpoint: {CHECKPOINT_MODEL}")
print(f"  N_epochs = {N_EPOCHS}, lr_source = {LR_SOURCE}")

key = jax.random.PRNGKey(4)
key, tgt_queue_key, tgt_sim_key = jax.random.split(key, 3)

print("\nGenerating ground-truth snapshot ...")
tgt_z_init, tgt_wake = build_particle_queue(
    sources=SOURCES, n_particles=N_PARTICLES,
    t_final=T_FINAL, dt=DT, key=tgt_queue_key,
)

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
        (step_keys, step_indices))
    return all_px, all_py, all_S

tgt_all_px, tgt_all_py, tgt_all_S = simulate_ground_truth_full(
    tgt_z_init, tgt_wake, N_STEPS, DT, tgt_sim_key)
final_step = N_STEPS - 1
tgt_z = jnp.column_stack([tgt_all_px[final_step], tgt_all_py[final_step]])
tgt_S = tgt_all_S[final_step]
tgt_mass = float(jnp.sum(jnp.exp(tgt_S)))
print(f"  t={T_FINAL:.1f}: mass={tgt_mass:.2f}")

# Regenerate eps and wake_schedule (same seeds)
key, eps_key, wake_key = jax.random.split(key, 3)
eps = jax.random.normal(eps_key, (N_PARTICLES, 2))
birth_times = jnp.linspace(0, T_FINAL, N_PARTICLES, endpoint=False)
wake_schedule = (birth_times / DT).astype(int)
wake_schedule = jnp.clip(wake_schedule, 0, N_STEPS - 1)


# ==========================================================================
# 2. Load model and freeze NN
# ==========================================================================

print("\nLoading model ...")
key_dummy = jax.random.PRNGKey(0)
phi_nn = PotentialNN(key_dummy, d_latent=2, c_conf=0.0, hidden_sizes=(16, 16))
potential = PotentialWithConfinement(
    nn=phi_nn, b=TARGET_PARAMS['b'], d=TARGET_PARAMS['d'])
fixed_death = FixedDeathRate2D(**DEATH_PARAMS_2D)
model_skeleton = SourceDeathModelWithSource(
    potential=potential, death_rate=fixed_death,
    mu_x=jnp.array(0.0), mu_y=jnp.array(0.0),
    sigma_x=TRUE_SIGMA_X, sigma_y=TRUE_SIGMA_Y)

model = eqx.tree_deserialise_leaves(CHECKPOINT_MODEL, model_skeleton)
print(f"  Loaded from {CHECKPOINT_MODEL}")
print(f"  mu_x = {float(model.mu_x):.4f}, mu_y = {float(model.mu_y):.4f}")

# Freeze: only mu_x and mu_y are trainable
filter_spec = jax.tree.map(lambda _: False, model)
filter_spec = eqx.tree_at(lambda m: m.mu_x, filter_spec, True)
filter_spec = eqx.tree_at(lambda m: m.mu_y, filter_spec, True)

n_trainable = sum(x.size for x, f in zip(
    jax.tree.leaves(eqx.filter(model, eqx.is_array)),
    jax.tree.leaves(filter_spec)) if f)
print(f"  Trainable params: {n_trainable} (mu_x + mu_y only)")
print(f"  NN frozen: 337 params")


# ==========================================================================
# 3. Optimizer (source only)
# ==========================================================================

optimizer = optax.adam(learning_rate=LR_SOURCE)
trainable_params = eqx.filter(model, filter_spec)
opt_state = optimizer.init(trainable_params)

objective_fn = make_objective(N_STEPS, DT, SIGMA, LAM_MASS)

@eqx.filter_jit
def step(model, opt_state, eps, wake_schedule, tgt_z, tgt_S, tgt_mass, key):
    def loss_fn(trainable):
        full_model = eqx.combine(trainable, eqx.filter(model, lambda x: not filter_spec))
        # Need to use the tree_at approach for proper combination
        return objective_fn(full_model, eps, wake_schedule, tgt_z, tgt_S, tgt_mass, key)

    # Partition into trainable and frozen
    trainable = eqx.filter(model, filter_spec)
    frozen = eqx.filter(model, lambda x: not eqx.is_array(x))  # static parts

    def loss_fn_full(model):
        return objective_fn(model, eps, wake_schedule, tgt_z, tgt_S, tgt_mass, key)

    # Get grads for full model, then zero out frozen params
    (loss, aux), grads = eqx.filter_value_and_grad(loss_fn_full, has_aux=True)(model)

    # Zero out gradients for frozen (non-source) params
    grads = jax.tree.map(
        lambda g, f: g if f else jnp.zeros_like(g) if hasattr(g, 'shape') else g,
        grads, filter_spec)

    # Extract only trainable grads for optimizer
    trainable_grads = eqx.filter(grads, filter_spec)
    trainable_params = eqx.filter(model, filter_spec)

    updates, new_opt_state = optimizer.update(trainable_grads, opt_state, trainable_params)

    # Apply updates only to trainable params
    new_trainable = optax.apply_updates(trainable_params, updates)
    new_model = eqx.tree_at(lambda m: (m.mu_x, m.mu_y), model,
                             (new_trainable.mu_x, new_trainable.mu_y))

    return new_model, new_opt_state, loss, aux


# ==========================================================================
# 4. Train
# ==========================================================================

history = {'loss': [], 'mmd': [], 'mass': [], 'mu_x': [], 'mu_y': []}
best_loss = float('inf')
best_epoch = 0
best_mu_x = float(model.mu_x)
best_mu_y = float(model.mu_y)
opt_key = jax.random.PRNGKey(42)

print(f"\nTraining for {N_EPOCHS} epochs (NN frozen) ...")
print("-" * 65)

for epoch in range(N_EPOCHS):
    opt_key, sim_key = jax.random.split(opt_key)

    model, opt_state, loss, (l_mmd, l_mass) = step(
        model, opt_state, eps, wake_schedule,
        tgt_z, tgt_S, tgt_mass, sim_key)

    loss_val = float(loss)
    mu_x_val = float(model.mu_x)
    mu_y_val = float(model.mu_y)
    history['loss'].append(loss_val)
    history['mmd'].append(float(l_mmd))
    history['mass'].append(float(l_mass))
    history['mu_x'].append(mu_x_val)
    history['mu_y'].append(mu_y_val)

    if epoch % PRINT_EVERY == 0:
        print(
            f"Epoch {epoch:4d}: Loss={loss_val:.5f} "
            f"(MMD={float(l_mmd):.5f}, Mass={float(l_mass):.5f}) "
            f"mu_x={mu_x_val:.4f} mu_y={mu_y_val:.4f}")

    if loss_val < best_loss - 1e-6:
        best_loss = loss_val
        best_epoch = epoch
        best_mu_x = mu_x_val
        best_mu_y = mu_y_val

print(f"\nBest epoch {best_epoch}: loss={best_loss:.6f}")
print(f"  mu_x={best_mu_x:.4f} (true={TRUE_MU_X}, error={abs(best_mu_x - TRUE_MU_X):.4f})")
print(f"  mu_y={best_mu_y:.4f} (true={TRUE_MU_Y}, error={abs(best_mu_y - TRUE_MU_Y):.4f})")

final_mu_x = float(model.mu_x)
final_mu_y = float(model.mu_y)
print(f"\nFinal mu_x = {final_mu_x:.4f} (true = {TRUE_MU_X}, error = {abs(final_mu_x - TRUE_MU_X):.4f})")
print(f"Final mu_y = {final_mu_y:.4f} (true = {TRUE_MU_Y}, error = {abs(final_mu_y - TRUE_MU_Y):.4f})")

eqx.tree_serialise_leaves("trained_model_freeze_nn.eqx", model)

# Save checkpoint
with open("checkpoint_freeze_nn.pkl", "wb") as f:
    pickle.dump({
        'history': history,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
    }, f)
print("Model saved to trained_model_freeze_nn.eqx")
print("Checkpoint saved to checkpoint_freeze_nn.pkl")


# ==========================================================================
# 5. Visualisation
# ==========================================================================

x_grid = jnp.linspace(-4, 4, 200)
y_grid = jnp.linspace(-2, 3.5, 200)
Xg, Yg = jnp.meshgrid(x_grid, y_grid)
Z_true = jax.vmap(jax.vmap(
    lambda x, y: analytical_potential(x, y, TARGET_PARAMS)))(Xg, Yg)
Z_learned = jax.vmap(
    lambda row_y: jax.vmap(
        lambda xi: model.potential(jnp.array([xi, row_y]))
    )(x_grid))(y_grid)

ref_true = float(analytical_potential(0.0, 0.0, TARGET_PARAMS))
ref_learned = float(model.potential(jnp.array([0.0, 0.0])))
Z_true = Z_true - ref_true
Z_learned = Z_learned - ref_learned
pot_vmin = min(float(jnp.min(Z_true)), float(jnp.min(Z_learned)))
pot_vmax = min(max(float(jnp.max(Z_true)), float(jnp.max(Z_learned))), 30.0)
pot_levels = np.linspace(pot_vmin, pot_vmax, 31)

# --- Training curves ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
epochs_arr = np.arange(len(history['loss']))

axes[0].semilogy(epochs_arr, history['loss'], label='Total', linewidth=1.5)
axes[0].semilogy(epochs_arr, history['mmd'], label='MMD', linewidth=1, alpha=0.8)
axes[0].semilogy(epochs_arr, history['mass'], label='Mass', linewidth=1, alpha=0.8)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Loss (NN frozen)'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_arr, history['mu_x'], 'b-', linewidth=1.5)
axes[1].axhline(TRUE_MU_X, color='r', linestyle='--', linewidth=1.5, label=f'True = {TRUE_MU_X}')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel(r'$\mu_x$')
axes[1].set_title(r'$\mu_x$ (NN frozen)'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs_arr, history['mu_y'], 'b-', linewidth=1.5)
axes[2].axhline(TRUE_MU_Y, color='r', linestyle='--', linewidth=1.5, label=f'True = {TRUE_MU_Y}')
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel(r'$\mu_y$')
axes[2].set_title(r'$\mu_y$ (NN frozen)'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_freeze_nn.png", dpi=150, bbox_inches='tight')
plt.close()

# --- Snapshots ---
VIS_TIMES = [1.0, 2.0, 4.0, 6.0]
VIS_STEPS = [int(t / DT) - 1 for t in VIS_TIMES]
key, eval_key = jax.random.split(key)
z_init_eval = make_z_init(model, eps)
sim_all_z, sim_all_S = simulate_open_system_full(
    model, z_init_eval, wake_schedule, SIGMA, N_STEPS, DT, eval_key)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for col, (t_vis, step_vis) in enumerate(zip(VIS_TIMES, VIS_STEPS)):
    ax_tgt = axes[0, col]
    tgt_z_i = jnp.column_stack([tgt_all_px[step_vis], tgt_all_py[step_vis]])
    tgt_S_i = tgt_all_S[step_vis]
    tgt_w_i = np.array(jnp.clip(jnp.exp(tgt_S_i), 0.0, 1.0))
    alive = tgt_w_i > 1e-6
    ax_tgt.contourf(np.array(Xg), np.array(Yg), np.array(Z_true),
                    levels=pot_levels, cmap='viridis', alpha=0.3)
    sc = ax_tgt.scatter(np.array(tgt_z_i[alive, 0]), np.array(tgt_z_i[alive, 1]),
                        c=tgt_w_i[alive], cmap='coolwarm', s=6, alpha=0.5,
                        vmin=0, vmax=1, edgecolors='none')
    ax_tgt.set_xlim(-4, 4); ax_tgt.set_ylim(-2, 3.5)
    ax_tgt.set_title(f'Target t={t_vis:.0f}')
    if col == 0: ax_tgt.set_ylabel('y (target)')

    ax_sim = axes[1, col]
    sim_z_i = sim_all_z[step_vis]
    sim_S_i = sim_all_S[step_vis]
    sim_w_i = np.array(jnp.clip(jnp.exp(sim_S_i), 0.0, 1.0))
    alive = sim_w_i > 1e-6
    ax_sim.contourf(np.array(Xg), np.array(Yg), np.array(Z_learned),
                    levels=pot_levels, cmap='viridis', alpha=0.3)
    sc = ax_sim.scatter(np.array(sim_z_i[alive, 0]), np.array(sim_z_i[alive, 1]),
                        c=sim_w_i[alive], cmap='coolwarm', s=6, alpha=0.5,
                        vmin=0, vmax=1, edgecolors='none')
    ax_sim.set_xlim(-4, 4); ax_sim.set_ylim(-2, 3.5)
    ax_sim.set_title(f'Learned t={t_vis:.0f}')
    if col == 0: ax_sim.set_ylabel('y (learned)')

    ax_tgt.plot(TRUE_MU_X, TRUE_MU_Y, 'w*', markersize=12, markeredgecolor='k',
                markeredgewidth=0.8, zorder=10)
    ax_sim.plot(final_mu_x, final_mu_y, 'w*', markersize=12,
                markeredgecolor='k', markeredgewidth=0.8, zorder=10)

fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
fig.colorbar(sc, cax=cbar_ax, label=r'$w = e^S$')
fig.suptitle(f'NN frozen | mu_x: {final_mu_x:.3f} (true={TRUE_MU_X})  |  '
             f'mu_y: {final_mu_y:.3f} (true={TRUE_MU_Y})',
             fontsize=13, y=1.01)
plt.savefig("snapshots_freeze_nn.png", dpi=150, bbox_inches='tight')
plt.close()

print("\nFigures saved:")
print("  training_freeze_nn.png")
print("  snapshots_freeze_nn.png")
