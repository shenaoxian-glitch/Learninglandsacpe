#!/usr/bin/env python
"""
train_death_wall.py — Model 07: Potential-only training (plain x^4)

Ground truth: H = -(y-a)x^2 + exp(b)x^4 - cy + e*x  (no y^4)
Learned: PotentialNN only (no confinement term, no x^4 wrapper)
Source and death rate: FIXED at ground truth values.

Usage:
    python train_death_wall.py > log.txt 2>&1
"""
import matplotlib
matplotlib.use('Agg')

import os
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np
import pickle

from models import SourceDeathModel
from models.potential import PotentialNN
from models.death_rate import ExponentialDeathRate
from training.data_loader import (
    build_particle_queue, analytical_potential_no_y4, analytical_death_rate_exp,
)
from simulator.sde_solver import simulate_open_system_full


# ==========================================================================
# 1. Ground-truth parameters
# ==========================================================================

TARGET_PARAMS = {
    'a': -1.0, 'b': -1.6, 'c': 3.5,
    'e': 0.7, 'sigma': 1.0,
}

DEATH_PARAMS_EXP = {
    'A': 0.1, 'k': 5.0, 'y_d': 2.0, 'gamma_max': 100.0,
}

N_LEARNED = 3000
N_TARGET = 6000
DT = 0.01
T_FINAL = 3.0
N_STEPS = int(T_FINAL / DT)
SIGMA = 1.0

# True source parameters (FIXED, not learned)
TRUE_MU_X = 0.0
TRUE_MU_Y = -1.0
TRUE_SIGMA_X = 0.05
TRUE_SIGMA_Y = 0.02

# Training
N_EPOCHS = 2000
PRINT_EVERY = 40
PATIENCE = 800
LR = 3e-3

LAM_MASS = 5.0
BANDWIDTHS = (0.005, 0.05, 0.5, 5.0)
ALIVE_THRESHOLD = 1e-3

Z_CLAMP = jnp.array([[-8.0, 8.0], [-3.0, 10.0]])

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nn_32_32")
os.makedirs(SCRIPT_DIR, exist_ok=True)
SUFFIX = "07_nn32x32_N3000_Ntgt6000_t3_ep2000"


# ==========================================================================
# 2. Ground-truth simulation (analytical, no y^4, plain x^4)
# ==========================================================================

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


# ==========================================================================
# 3. Loss functions
# ==========================================================================

def mass_loss(S, target_mass):
    M_sim = jnp.sum(jnp.exp(S))
    return (M_sim - target_mass)**2 / (target_mass**2 + 1e-8)


def weighted_mmd_both(x_sim, alpha_sim, x_obs, alpha_obs,
                      bandwidths=BANDWIDTHS):
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


# ==========================================================================
# 4. Generate ground-truth target data
# ==========================================================================

print("=" * 65)
print("Model 07: Potential-only training (plain x^4, no y^4)")
print("=" * 65)
print(f"  Ground truth: H = -(y-a)x^2 + exp(b)x^4 - cy + e*x")
print(f"  Learned: PotentialNN (no confinement term)")
print(f"  N_learned = {N_LEARNED}, N_target = {N_TARGET}")
print(f"  T_final = {T_FINAL}, dt = {DT}, n_steps = {N_STEPS}")
print(f"  sigma = {SIGMA}, LR = {LR}")
print(f"  Death rate: FIXED exponential (A={DEATH_PARAMS_EXP['A']}, "
      f"k={DEATH_PARAMS_EXP['k']}, y_d={DEATH_PARAMS_EXP['y_d']})")
print(f"  Source: FIXED mu=({TRUE_MU_X}, {TRUE_MU_Y}), "
      f"sigma_x={TRUE_SIGMA_X}, sigma_y={TRUE_SIGMA_Y}")
print(f"  MMD bandwidths: {BANDWIDTHS}")
print(f"  Loss = L_MMD + {LAM_MASS}*L_mass")
print(f"  epochs={N_EPOCHS}, patience={PATIENCE}")

# Target data (N=6000)
SOURCES_TGT = [
    {'mu': jnp.array([TRUE_MU_X, TRUE_MU_Y]),
     'sigma_x': TRUE_SIGMA_X, 'sigma_y': TRUE_SIGMA_Y,
     'n_particles': N_TARGET},
]

key = jax.random.PRNGKey(4)
key, tgt_queue_key, tgt_sim_key = jax.random.split(key, 3)

print("\nGenerating ground-truth target snapshot ...")
tgt_z_init, tgt_wake = build_particle_queue(
    sources=SOURCES_TGT, n_particles=N_TARGET,
    t_final=T_FINAL, dt=DT, key=tgt_queue_key,
)
tgt_all_px, tgt_all_py, tgt_all_S = simulate_ground_truth_full(
    tgt_z_init, tgt_wake, N_STEPS, DT, tgt_sim_key
)

final_step = N_STEPS - 1
tgt_z = jnp.column_stack([tgt_all_px[final_step], tgt_all_py[final_step]])
tgt_S = tgt_all_S[final_step]
tgt_mass = float(jnp.sum(jnp.exp(tgt_S)))
n_alive = int(jnp.sum(jnp.exp(tgt_S) > ALIVE_THRESHOLD))
print(f"  t={T_FINAL:.1f}: mass={tgt_mass:.2f}, ~{n_alive} alive (N_target={N_TARGET})")

scaled_tgt_mass = tgt_mass * (N_LEARNED / N_TARGET)
print(f"  scaled_tgt_mass = {scaled_tgt_mass:.2f} (for N_learned={N_LEARNED})")


# ==========================================================================
# 5. Build training particle queue (FIXED source, N=3000)
# ==========================================================================

SOURCES_TRAIN = [
    {'mu': jnp.array([TRUE_MU_X, TRUE_MU_Y]),
     'sigma_x': TRUE_SIGMA_X, 'sigma_y': TRUE_SIGMA_Y,
     'n_particles': N_LEARNED},
]

key, train_queue_key = jax.random.split(key)
train_z_init, train_wake = build_particle_queue(
    sources=SOURCES_TRAIN, n_particles=N_LEARNED,
    t_final=T_FINAL, dt=DT, key=train_queue_key,
)
print(f"  Training queue: {N_LEARNED} particles, fixed source\n")


# ==========================================================================
# 6. Create model (potential-only, no confinement)
# ==========================================================================

key, model_key = jax.random.split(key)

potential = PotentialNN(model_key, d_latent=2, c_conf=0.0,
                        hidden_sizes=(32, 32))
fixed_death = ExponentialDeathRate(**DEATH_PARAMS_EXP)
model = SourceDeathModel(potential=potential, death_rate=fixed_death)

n_params = sum(x.size for x in jax.tree.leaves(
    eqx.filter(model, eqx.is_array)))
print(f"SourceDeathModel created:")
print(f"  Potential: PotentialNN(2->32->32->1, c_conf=0.0) — {n_params} params")
print(f"  Death rate: FIXED ExponentialDeathRate")
print(f"  Source: FIXED (not part of model)")


# ==========================================================================
# 7. Optimizer (single AdamW for NN params only)
# ==========================================================================

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=LR, weight_decay=1e-4),
)
print(f"  Optimizer: AdamW(lr={LR}, wd=1e-4)")

objective_fn = make_objective(N_STEPS, DT, SIGMA, LAM_MASS, z_clamp=Z_CLAMP)
step_fn = make_train_step(optimizer, objective_fn)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


# ==========================================================================
# 8. Train
# ==========================================================================

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
        model, opt_state, train_z_init, train_wake,
        tgt_z, tgt_S, scaled_tgt_mass, sim_key,
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

last_model = model

if best_model_leaves is not None:
    model = eqx.combine(best_model_leaves,
                         eqx.filter(model, lambda x: not eqx.is_array(x)))
    print(f"Restored best model from epoch {best_epoch} (loss={best_loss:.6f})")


# ==========================================================================
# 9. Save model + checkpoint
# ==========================================================================

model_path = os.path.join(SCRIPT_DIR, f"trained_model_{SUFFIX}.eqx")
eqx.tree_serialise_leaves(model_path, model)

checkpoint_model_path = os.path.join(SCRIPT_DIR, f"checkpoint_model_last_{SUFFIX}.eqx")
eqx.tree_serialise_leaves(checkpoint_model_path, last_model)

checkpoint_state_path = os.path.join(SCRIPT_DIR, f"checkpoint_state_{SUFFIX}.pkl")
with open(checkpoint_state_path, "wb") as f:
    pickle.dump({
        'opt_state': opt_state,
        'history': history,
        'epoch': epoch + 1,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'opt_key': opt_key,
    }, f)

print(f"Model saved to {model_path}")
print(f"Checkpoint saved to {checkpoint_model_path} + {checkpoint_state_path}")


# ==========================================================================
# 10. Visualisation
# ==========================================================================

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

ref_true = float(analytical_potential_no_y4(0.0, 0.0, TARGET_PARAMS))
ref_learned = float(model.potential(jnp.array([0.0, 0.0])))
Z_true = Z_true - ref_true
Z_learned = Z_learned - ref_learned

alive_mask = np.array(Yg) < 3.0
Z_true_np = np.array(Z_true)
Z_learned_np = np.array(Z_learned)
pot_vmin = max(min(Z_true_np[alive_mask].min(), Z_learned_np[alive_mask].min()), -20.0)
pot_vmax = min(max(Z_true_np[alive_mask].max(), Z_learned_np[alive_mask].max()), 20.0)
pot_levels = np.linspace(pot_vmin, pot_vmax, 31)


# --- Figure 1: Training curves ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

epochs_arr = np.arange(len(history['loss']))
axes[0].semilogy(epochs_arr, history['loss'], label='Total', linewidth=1.5)
axes[0].semilogy(epochs_arr, history['mmd'], label='MMD', linewidth=1, alpha=0.8)
axes[0].semilogy(epochs_arr, history['mass'], label='Mass', linewidth=1, alpha=0.8)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_arr, history['loss'], 'b-', linewidth=1.5)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Training Loss (linear)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, f"training_{SUFFIX}.png"),
            dpi=150, bbox_inches='tight')
plt.close()


# --- Figure 2: Force vector fields ---
n_arrow = 22
xs_q = np.linspace(-3.5, 3.5, n_arrow)
ys_q = np.linspace(-1.5, 4.5, n_arrow)
Xq, Yq = np.meshgrid(xs_q, ys_q)

grad_true = jax.grad(analytical_potential_no_y4, argnums=(0, 1))
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

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, Fx, Fy, mag, Z_bg, title in [
    (axes[0], Fx_true, Fy_true, mag_true, Z_true_np,
     r'True $\mathbf{F} = -\nabla\Phi^*$ (plain $x^4$)'),
    (axes[1], Fx_learned, Fy_learned, mag_learned, Z_learned_np,
     r'Learned $\mathbf{F} = -\nabla\Phi_{NN}$'),
]:
    cf = ax.contourf(np.array(Xg), np.array(Yg), Z_bg,
                     levels=pot_levels, cmap='viridis', alpha=0.35, extend='both')
    ax.contour(np.array(Xg), np.array(Yg), Z_bg,
               levels=pot_levels, colors='0.5', linewidths=0.4, alpha=0.5)
    q = ax.quiver(Xq, Yq, Fx, Fy, mag,
                  cmap='hot_r', scale=mag_max * 5,
                  width=0.004, headwidth=3.5, headlength=4,
                  headaxislength=3.5, clim=(0, mag_max), zorder=3)
    ax.axhline(DEATH_PARAMS_EXP['y_d'], color='red', linestyle='--',
               linewidth=1.5, alpha=0.5, label=f"death wall y_d={DEATH_PARAMS_EXP['y_d']}")
    ax.set_xlim(-3.8, 3.8); ax.set_ylim(-1.8, 4.8)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=8)

fig.subplots_adjust(right=0.88, wspace=0.25)
cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
fig.colorbar(q, cax=cbar_ax, label=r'$|\mathbf{F}|$')
plt.savefig(os.path.join(SCRIPT_DIR, f"force_field_{SUFFIX}.png"),
            dpi=150, bbox_inches='tight')
plt.close()


# --- Figure 3: Potential contours + 1D slices ---
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

cf = axes[0, 0].contourf(np.array(Xg), np.array(Yg), Z_learned_np,
                          levels=pot_levels, cmap='viridis', extend='both')
plt.colorbar(cf, ax=axes[0, 0], label=r'$\Phi_{learned}$')
axes[0, 0].axhline(DEATH_PARAMS_EXP['y_d'], color='red', linestyle='--',
                    linewidth=1.5, alpha=0.5)
axes[0, 0].set_title('Learned Potential')
axes[0, 0].set_xlabel('x'); axes[0, 0].set_ylabel('y')

cf = axes[0, 1].contourf(np.array(Xg), np.array(Yg), Z_true_np,
                          levels=pot_levels, cmap='viridis', extend='both')
plt.colorbar(cf, ax=axes[0, 1], label=r'$\Phi^*$')
axes[0, 1].axhline(DEATH_PARAMS_EXP['y_d'], color='red', linestyle='--',
                    linewidth=1.5, alpha=0.5)
axes[0, 1].set_title(r'True Potential ($\exp(b)x^4$, no $y^4$)')
axes[0, 1].set_xlabel('x'); axes[0, 1].set_ylabel('y')

Z_diff = Z_learned_np - Z_true_np
d_absmax = min(max(abs(Z_diff.min()), abs(Z_diff.max())), 15.0)
diff_levels = np.linspace(-d_absmax, d_absmax, 31)
cf = axes[0, 2].contourf(np.array(Xg), np.array(Yg), Z_diff,
                          levels=diff_levels, cmap='RdBu_r', extend='both')
plt.colorbar(cf, ax=axes[0, 2], label=r'$\Phi_{learned} - \Phi^*$')
axes[0, 2].set_title('Potential Error')
axes[0, 2].set_xlabel('x'); axes[0, 2].set_ylabel('y')

y_slices = [-0.5, 0.5, 1.5]
x_line = np.linspace(-4, 4, 300)
for col, y_val in enumerate(y_slices):
    ax = axes[1, col]
    phi_true = np.array([
        float(analytical_potential_no_y4(float(xi), y_val, TARGET_PARAMS))
        for xi in x_line
    ]) - ref_true
    phi_learned = np.array(jax.vmap(
        lambda xi: model.potential(jnp.array([xi, y_val]))
    )(jnp.array(x_line))) - ref_learned

    ax.plot(x_line, phi_true, 'b-', linewidth=2, label='True')
    ax.plot(x_line, phi_learned, 'r--', linewidth=2, label='Learned')
    ax.set_xlabel('x'); ax.set_ylabel(r'$\Phi(x, y)$')
    ax.set_title(f'Slice at y = {y_val}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-15, 15)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, f"landscape_{SUFFIX}.png"),
            dpi=150, bbox_inches='tight')
plt.close()


# --- Figure 4: Snapshot comparison ---
VIS_TIMES = [0.5, 1.0, 2.0, 3.0]
VIS_STEPS = [int(t / DT) - 1 for t in VIS_TIMES]

key, eval_key = jax.random.split(key)

sim_all_z, sim_all_S = simulate_open_system_full(
    model, train_z_init, train_wake, SIGMA, N_STEPS, DT, eval_key,
    z_clamp=Z_CLAMP,
)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for col, (t_vis, step_vis) in enumerate(zip(VIS_TIMES, VIS_STEPS)):
    # Target
    ax_tgt = axes[0, col]
    tgt_z_i = jnp.column_stack([tgt_all_px[step_vis], tgt_all_py[step_vis]])
    tgt_S_i = tgt_all_S[step_vis]
    tgt_w_i = np.array(jnp.clip(jnp.exp(tgt_S_i), 0.0, 1.0))
    alive = tgt_w_i > ALIVE_THRESHOLD

    ax_tgt.contourf(np.array(Xg), np.array(Yg), Z_true_np,
                    levels=pot_levels, cmap='viridis', alpha=0.3, extend='both')
    if alive.sum() > 0:
        sc = ax_tgt.scatter(np.array(tgt_z_i[alive, 0]), np.array(tgt_z_i[alive, 1]),
                            c=tgt_w_i[alive], cmap='coolwarm', s=6, alpha=0.5,
                            vmin=0, vmax=1, edgecolors='none')
    ax_tgt.axhline(DEATH_PARAMS_EXP['y_d'], color='red', linestyle='--',
                   linewidth=1, alpha=0.5)
    ax_tgt.plot(TRUE_MU_X, TRUE_MU_Y, 'w*', markersize=12, markeredgecolor='k',
                markeredgewidth=0.8, zorder=10)
    ax_tgt.set_xlim(-4, 4); ax_tgt.set_ylim(-2, 5)
    ax_tgt.set_title(f'Target t={t_vis:.1f} ({alive.sum()} alive)')
    if col == 0:
        ax_tgt.set_ylabel('y (target)')

    # Learned
    ax_sim = axes[1, col]
    sim_z_i = sim_all_z[step_vis]
    sim_S_i = sim_all_S[step_vis]
    sim_w_i = np.array(jnp.clip(jnp.exp(sim_S_i), 0.0, 1.0))
    alive_sim = sim_w_i > ALIVE_THRESHOLD

    ax_sim.contourf(np.array(Xg), np.array(Yg), Z_learned_np,
                    levels=pot_levels, cmap='viridis', alpha=0.3, extend='both')
    if alive_sim.sum() > 0:
        sc = ax_sim.scatter(np.array(sim_z_i[alive_sim, 0]),
                            np.array(sim_z_i[alive_sim, 1]),
                            c=sim_w_i[alive_sim], cmap='coolwarm', s=6, alpha=0.5,
                            vmin=0, vmax=1, edgecolors='none')
    ax_sim.axhline(DEATH_PARAMS_EXP['y_d'], color='red', linestyle='--',
                   linewidth=1, alpha=0.5)
    ax_sim.plot(TRUE_MU_X, TRUE_MU_Y, 'w*', markersize=12, markeredgecolor='k',
                markeredgewidth=0.8, zorder=10)
    ax_sim.set_xlim(-4, 4); ax_sim.set_ylim(-2, 5)
    ax_sim.set_title(f'Learned t={t_vis:.1f} ({alive_sim.sum()} alive)')
    if col == 0:
        ax_sim.set_ylabel('y (learned)')

fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
if alive_sim.sum() > 0:
    fig.colorbar(sc, cax=cbar_ax, label=r'$w = e^S$')
fig.suptitle(f'Model 07: plain $\\exp(b)x^4$ | death wall y_d={DEATH_PARAMS_EXP["y_d"]}',
             fontsize=13, y=1.01)
plt.savefig(os.path.join(SCRIPT_DIR, f"snapshots_{SUFFIX}.png"),
            dpi=150, bbox_inches='tight')
plt.close()


print("\nDone. Figures saved:")
print(f"  training_{SUFFIX}.png")
print(f"  force_field_{SUFFIX}.png")
print(f"  landscape_{SUFFIX}.png")
print(f"  snapshots_{SUFFIX}.png")
