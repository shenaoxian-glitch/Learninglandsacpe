#!/usr/bin/env python
"""
train_learnable_source_mux.py — Model 6.1: NN potential + learnable mu_x

Phase 1: Learn Phi_nn + mu_x only. mu_y is fixed at true value (-1.0).

Key differences from model 6:
  - N_learned=3000 (model), N_target=8000 (data) — separate counts
  - T_FINAL=10.0 (near steady-state)
  - N_EPOCHS=2000, PATIENCE=800
  - SIGMA=1.5 (matching 5.1 sweeps)
  - mu_y FIXED at true value (-1.0)
  - INIT_MU_X=0.5 (true=0.0)

Source distribution (reparameterization trick):
    x_i(0) = mu_x + eps_i^x * sigma_x     (mu_x learnable)
    y_i(0) = mu_y + eps_i^y * sigma_y     (mu_y FIXED)

Usage:
    python train_learnable_source_mux.py > log.txt 2>&1
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
from training.data_loader import (
    build_particle_queue, analytical_death_rate_2d, analytical_potential,
)
from simulator.sde_solver import simulate_open_system_full


# ==========================================================================
# 1. Ground-truth parameters
# ==========================================================================

TARGET_PARAMS = {
    'a': -1.0, 'b': -1.6, 'c': 3.5, 'd': -1.2,
    'e': 0.7, 'sigma': 1.5,
}
DEATH_PARAMS_2D = {
    'A': 2.0, 'w_x': 0.5, 'k1': 1.0, 'y_select': 1.0,
    'B': 5.0, 'k2': 3.0, 'y_max': 3.0,
}

N_LEARNED = 3000       # model (training) particles
N_TARGET = 8000        # target data particles
DT = 0.01
T_FINAL = 10.0
N_STEPS = int(T_FINAL / DT)  # 1000
SIGMA = 1.5

# True source parameters
TRUE_MU_X = 0.0
TRUE_MU_Y = -1.0
TRUE_SIGMA_X = 0.25
TRUE_SIGMA_Y = 0.10

# Learnable source: initial guess for mu_x (mu_y fixed)
INIT_MU_X = 0.0

# Loss weights
LAM_MASS = 5.0

# Training
N_EPOCHS = 2000
PRINT_EVERY = 40
PATIENCE = 800

LR_NN = 3e-3
LR_SOURCE = 2e-2

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "init_mux_0.0")
os.makedirs(SCRIPT_DIR, exist_ok=True)
SUFFIX = "mux_N3000_Ntgt8000_t10_ep2000"


# ==========================================================================
# 2. Model components
# ==========================================================================

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


# ==========================================================================
# 3. Model with learnable mu_x (mu_y fixed)
# ==========================================================================

class SourceDeathModelLearnMuX(eqx.Module):
    """SourceDeathModel + learnable mu_x. mu_y is FIXED (static)."""
    potential: eqx.Module
    death_rate: eqx.Module
    mu_x: jax.Array              # learnable
    mu_y: float = eqx.field(static=True)   # FIXED
    sigma_x: float = eqx.field(static=True)
    sigma_y: float = eqx.field(static=True)


def make_z_init(model, eps):
    """Reparameterization: z_init = mu + eps * sigma."""
    x = model.mu_x + eps[:, 0] * model.sigma_x
    y = model.mu_y + eps[:, 1] * model.sigma_y
    return jnp.column_stack([x, y])


# ==========================================================================
# 4. Ground-truth simulation (analytical)
# ==========================================================================

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


# ==========================================================================
# 5. Loss functions
# ==========================================================================

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
    def objective(model, eps, wake_schedule,
                  tgt_z, tgt_S, tgt_mass, key):
        z_init = make_z_init(model, eps)
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
    def step(model, opt_state, eps, wake_schedule,
             tgt_z, tgt_S, tgt_mass, key):
        def loss_fn(model):
            return objective_fn(
                model, eps, wake_schedule,
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
# 8. Generate ground-truth target snapshot (N=8000)
# ==========================================================================

print("=" * 65)
print("Model 6.1: NN potential + learnable mu_x (mu_y fixed)")
print("=" * 65)
print(f"  N_learned = {N_LEARNED}, N_target = {N_TARGET}")
print(f"  T_final = {T_FINAL}, dt = {DT}, n_steps = {N_STEPS}")
print(f"  sigma = {SIGMA} (fixed)")
print(f"  Potential: Phi_nn(2->16->16->1) + exp(b)x^4 + exp(d)y^4 (b,d fixed)")
print(f"  Death rate: FIXED 2D analytical")
print(f"  Loss = L_MMD + {LAM_MASS}*L_mass")
print(f"\n  True source: mu=({TRUE_MU_X}, {TRUE_MU_Y}), "
      f"sigma_x={TRUE_SIGMA_X}, sigma_y={TRUE_SIGMA_Y}")
print(f"  Init mu_x = {INIT_MU_X} (true = {TRUE_MU_X})")
print(f"  mu_y = {TRUE_MU_Y} (FIXED, not learned)")
print(f"  Learnable: Phi_nn weights + mu_x")
print(f"  LR: NN=AdamW({LR_NN}), mu_x=Adam({LR_SOURCE})")
print(f"  epochs={N_EPOCHS}, patience={PATIENCE}")

# Target data
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
n_alive = int(jnp.sum(jnp.exp(tgt_S) > 0.01))
print(f"  t={T_FINAL:.1f}: mass={tgt_mass:.2f}, ~{n_alive} alive (N_target={N_TARGET})")

# Scale target mass for particle count mismatch
scaled_tgt_mass = tgt_mass * (N_LEARNED / N_TARGET)
print(f"  scaled_tgt_mass = {scaled_tgt_mass:.2f} (for N_learned={N_LEARNED})")


# ==========================================================================
# 9. Build training particle queue (reparameterized, N=3000)
# ==========================================================================

key, eps_key, wake_key = jax.random.split(key, 3)
eps = jax.random.normal(eps_key, (N_LEARNED, 2))

birth_times = jnp.linspace(0, T_FINAL, N_LEARNED, endpoint=False)
wake_schedule = (birth_times / DT).astype(int)
wake_schedule = jnp.clip(wake_schedule, 0, N_STEPS - 1)

print(f"  Training queue: {N_LEARNED} particles, eps pre-sampled\n")


# ==========================================================================
# 10. Create model
# ==========================================================================

key, model_key = jax.random.split(key)

phi_nn = PotentialNN(model_key, d_latent=2, c_conf=0.0,
                     hidden_sizes=(16, 16))
potential = PotentialWithConfinement(
    nn=phi_nn, b=TARGET_PARAMS['b'], d=TARGET_PARAMS['d'],
)
fixed_death = FixedDeathRate2D(**DEATH_PARAMS_2D)

model = SourceDeathModelLearnMuX(
    potential=potential,
    death_rate=fixed_death,
    mu_x=jnp.array(INIT_MU_X),
    mu_y=TRUE_MU_Y,          # FIXED (static)
    sigma_x=TRUE_SIGMA_X,
    sigma_y=TRUE_SIGMA_Y,
)

n_params_phi = sum(x.size for x in jax.tree.leaves(
    eqx.filter(model.potential, eqx.is_array)))
n_params_total = sum(x.size for x in jax.tree.leaves(
    eqx.filter(model, eqx.is_array)))
print(f"SourceDeathModelLearnMuX created:")
print(f"  Phi_nn: {n_params_phi} params")
print(f"  mu_x: 1 param (init={INIT_MU_X}, true={TRUE_MU_X})")
print(f"  mu_y: {TRUE_MU_Y} (FIXED, not in optimizer)")
print(f"  Total learnable: {n_params_total} params")


# ==========================================================================
# 11. Optimizer: NN params get AdamW(3e-3), mu_x gets Adam(2e-2)
# ==========================================================================

params = eqx.filter(model, eqx.is_array)
param_labels = jax.tree.map(lambda _: 'nn', params)
param_labels = eqx.tree_at(lambda p: p.mu_x, param_labels, 'source')

optimizer = optax.multi_transform(
    {
        'nn': optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=LR_NN, weight_decay=1e-4),
        ),
        'source': optax.adam(learning_rate=LR_SOURCE),
    },
    param_labels,
)
print(f"  Optimizer: NN=AdamW(lr={LR_NN}), mu_x=Adam(lr={LR_SOURCE})")

objective_fn = make_objective(N_STEPS, DT, SIGMA, LAM_MASS)
step_fn = make_train_step(optimizer, objective_fn)
opt_state = optimizer.init(params)


# ==========================================================================
# 12. Train
# ==========================================================================

history = {'loss': [], 'mmd': [], 'mass': [], 'mu_x': []}
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
        model, opt_state, eps, wake_schedule,
        tgt_z, tgt_S, scaled_tgt_mass, sim_key,
    )

    loss_val = float(loss)
    mu_x_val = float(model.mu_x)
    history['loss'].append(loss_val)
    history['mmd'].append(float(l_mmd))
    history['mass'].append(float(l_mass))
    history['mu_x'].append(mu_x_val)

    if epoch % PRINT_EVERY == 0:
        print(
            f"Epoch {epoch:4d}: Loss={loss_val:.5f} "
            f"(MMD={float(l_mmd):.5f}, Mass={float(l_mass):.5f}) "
            f"mu_x={mu_x_val:.4f}"
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

final_mu_x = float(model.mu_x)
print(f"\nFinal mu_x = {final_mu_x:.4f} (true = {TRUE_MU_X}, "
      f"init = {INIT_MU_X}, error = {abs(final_mu_x - TRUE_MU_X):.4f})")
print(f"mu_y = {TRUE_MU_Y} (FIXED)")


# ==========================================================================
# 13. Save model + checkpoint
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
# 14. Visualisation
# ==========================================================================

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

ref_true = float(analytical_potential(0.0, 0.0, TARGET_PARAMS))
ref_learned = float(model.potential(jnp.array([0.0, 0.0])))
Z_true = Z_true - ref_true
Z_learned = Z_learned - ref_learned

pot_vmin = min(float(jnp.min(Z_true)), float(jnp.min(Z_learned)))
pot_vmax = min(max(float(jnp.max(Z_true)), float(jnp.max(Z_learned))), 30.0)
pot_levels = np.linspace(pot_vmin, pot_vmax, 31)


# --- Figure 1: Training curves + mu_x convergence ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

epochs_arr = np.arange(len(history['loss']))
axes[0].semilogy(epochs_arr, history['loss'], label='Total', linewidth=1.5)
axes[0].semilogy(epochs_arr, history['mmd'], label='MMD', linewidth=1, alpha=0.8)
axes[0].semilogy(epochs_arr, history['mass'], label='Mass', linewidth=1, alpha=0.8)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_arr, history['mu_x'], 'b-', linewidth=1.5,
             label=r'$\mu_x$ (learned)')
axes[1].axhline(TRUE_MU_X, color='r', linestyle='--', linewidth=1.5,
                label=f'True $\\mu_x$ = {TRUE_MU_X}')
axes[1].axhline(INIT_MU_X, color='gray', linestyle=':', linewidth=1,
                label=f'Init $\\mu_x$ = {INIT_MU_X}')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel(r'$\mu_x$')
axes[1].set_title(r'Source Position $\mu_x$ Convergence')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# mu_x error over time
mu_x_error = np.abs(np.array(history['mu_x']) - TRUE_MU_X)
axes[2].semilogy(epochs_arr, mu_x_error, 'b-', linewidth=1.5)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel(r'$|\mu_x - \mu_x^*|$')
axes[2].set_title(r'$\mu_x$ Error')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, f"training_{SUFFIX}.png"),
            dpi=150, bbox_inches='tight')
plt.close()


# --- Figure 2: Force vector fields ---
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

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, Fx, Fy, mag, Z_bg, title in [
    (axes[0], Fx_true, Fy_true, mag_true, np.array(Z_true),
     r'True $\mathbf{F} = -\nabla\Phi^*$'),
    (axes[1], Fx_learned, Fy_learned, mag_learned, np.array(Z_learned),
     r'Learned $\mathbf{F} = -\nabla\Phi_{NN}$'),
]:
    cf = ax.contourf(np.array(Xg), np.array(Yg), Z_bg,
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
plt.savefig(os.path.join(SCRIPT_DIR, f"force_field_{SUFFIX}.png"),
            dpi=150, bbox_inches='tight')
plt.close()


# --- Figure 3: Potential contours + 1D slices ---
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
VIS_TIMES = [1.0, 3.0, 6.0, 10.0]
VIS_STEPS = [int(t / DT) - 1 for t in VIS_TIMES]

key, eval_key = jax.random.split(key)

z_init_eval = make_z_init(model, eps)
sim_all_z, sim_all_S = simulate_open_system_full(
    model, z_init_eval, wake_schedule, SIGMA, N_STEPS, DT, eval_key
)

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
    if col == 0:
        ax_tgt.set_ylabel('y (target)')

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
    if col == 0:
        ax_sim.set_ylabel('y (learned)')

    # Mark source positions
    ax_tgt.plot(TRUE_MU_X, TRUE_MU_Y, 'w*', markersize=12, markeredgecolor='k',
                markeredgewidth=0.8, zorder=10)
    ax_sim.plot(float(model.mu_x), TRUE_MU_Y, 'w*', markersize=12,
                markeredgecolor='k', markeredgewidth=0.8, zorder=10)

fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
fig.colorbar(sc, cax=cbar_ax, label=r'$w = e^S$')
fig.suptitle(f'mu_x: {INIT_MU_X}->{final_mu_x:.3f} (true={TRUE_MU_X})  |  '
             f'mu_y={TRUE_MU_Y} (fixed)',
             fontsize=13, y=1.01)
plt.savefig(os.path.join(SCRIPT_DIR, f"snapshots_{SUFFIX}.png"),
            dpi=150, bbox_inches='tight')
plt.close()


print("\nDone. Figures saved:")
print(f"  training_{SUFFIX}.png")
print(f"  force_field_{SUFFIX}.png")
print(f"  landscape_{SUFFIX}.png")
print(f"  snapshots_{SUFFIX}.png")
