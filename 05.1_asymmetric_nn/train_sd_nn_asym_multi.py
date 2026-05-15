#!/usr/bin/env python
"""
train_sd_nn_asym_multi.py — NN potential + fixed 2D death rate

Model 5.1b: Asymmetric potential learned via NN, death rate fixed to
analytical 2D function from model 3.1 (selection corridor + terminal).

The 2D death rate kills undifferentiated (x≈0) cells that reach the
selection stage, pushing mass toward bifurcated branches. With death
rate fixed, gradient signal goes entirely to the potential NN — no
Phi/gamma degeneracy.

Usage:
    python train_sd_nn_asym_multi.py > log4.txt 2>&1
"""
import matplotlib
matplotlib.use('Agg')

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np

from models import SourceDeathModel
from models.potential import PotentialNN
from training.data_loader import (
    build_particle_queue, analytical_death_rate_2d, analytical_potential,
)
from simulator.sde_solver import simulate_open_system_full
from analysis.visualization import (
    plot_nn_landscape, _eval_grid,
)


# ==========================================================================
# 1. Ground-truth parameters
# ==========================================================================

# Asymmetric potential (from model 3.1)
TARGET_PARAMS = {
    'a': -1.0, 'b': -1.6, 'c': 3.5, 'd': -1.2,
    'e': 0.7,   # asymmetry: linear tilt in x
    'sigma': 1.5,
}

# 2D death rate (from model 3.1)
DEATH_PARAMS_2D = {
    'A': 2.0, 'w_x': 0.5, 'k1': 1.0, 'y_select': 1.0,
    'B': 5.0, 'k2': 3.0, 'y_max': 3.0,
}

N_PARTICLES = 1000
DT = 0.01
T_FINAL = 6.0
N_STEPS = int(T_FINAL / DT)  # 600
SIGMA = 2.5

# Oval Gaussian source (from model 3.1)
SOURCES = [
    {'mu': jnp.array([0.0, -1.0]), 'sigma_x': 0.25, 'sigma_y': 0.10,
     'n_particles': N_PARTICLES},
]

# Loss weights
LAM_MASS = 5.0


# ==========================================================================
# 2. Fixed death rate as Equinox module (non-learnable)
# ==========================================================================

class FixedDeathRate2D(eqx.Module):
    """Non-learnable 2D death rate from model 3.1.

    All parameters are static — excluded from gradient computation.
    gamma(x,y) = A*exp(-x^2/(2*w_x^2))*softplus(k1*(y-y_select))
                 + B*softplus(k2*(y-y_max))
    """
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
    """NN potential + fixed quartic confinement from true parameters.

    Phi(z) = Phi_nn(z) + exp(b)*x^4 + exp(d)*y^4

    The quartic terms are fixed (static) — only Phi_nn is learned.
    This gives the NN the confinement for free, so it only needs to
    learn the coupling (-(y-a)x^2) and tilt (-cy + e*x) terms.
    """
    nn: PotentialNN
    b: float = eqx.field(static=True)
    d: float = eqx.field(static=True)

    def __call__(self, z):
        x, y = z[0], z[1]
        conf = jnp.exp(self.b) * x**4 + jnp.exp(self.d) * y**4
        return self.nn(z) + conf


# ==========================================================================
# 3. Ground-truth simulation (analytical, for target snapshot)
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

        # 2D death rate
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
# 4. Loss functions
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
# 5. Single-snapshot objective
# ==========================================================================

def make_objective(n_steps, dt, sigma, lam_mass):
    """Build the single-snapshot loss function for SourceDeathModel."""

    def objective(model, z_init, wake_schedule,
                  tgt_z, tgt_S, tgt_mass, key):
        all_z, all_S = simulate_open_system_full(
            model, z_init, wake_schedule, sigma, n_steps, dt, key
        )

        # Final time step
        sim_z = all_z[-1]
        sim_S = all_S[-1]
        sim_alpha = jax.nn.softmax(sim_S)
        tgt_alpha = jax.nn.softmax(tgt_S)

        l_mmd = weighted_mmd_both(sim_z, sim_alpha, tgt_z, tgt_alpha)
        l_mass = mass_loss(sim_S, tgt_mass)

        total = l_mmd + lam_mass * l_mass
        return total, (l_mmd, l_mass)

    return objective


# ==========================================================================
# 6. Training step (equinox-compatible)
# ==========================================================================

def make_train_step(optimizer, objective_fn):
    """Build JIT-compiled training step."""

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
# 7. Generate ground-truth snapshot
# ==========================================================================

print("=" * 65)
print("Model 5.1c: NN potential + fixed quartic + FIXED 2D death rate")
print("=" * 65)
print(f"  N_particles = {N_PARTICLES}")
print(f"  T_final = {T_FINAL}, dt = {DT}, n_steps = {N_STEPS}")
print(f"  sigma = {SIGMA} (fixed, exploration noise)")
print(f"  Training snapshot: t = {T_FINAL} (single steady-state)")
print(f"  Potential: Phi_nn(2->16->16->1) + exp(b)x^4 + exp(d)y^4 (b,d fixed)")
print(f"  Death rate: FIXED 2D analytical (not learned)")
print(f"  Loss = L_MMD + {LAM_MASS}*L_mass")
print(f"\n  Ground-truth potential (asymmetric):")
print(f"    H = -(y-a)x^2 + exp(b)x^4 - cy + exp(d)y^4 + e*x")
print(f"    a={TARGET_PARAMS['a']}, b={TARGET_PARAMS['b']}, "
      f"c={TARGET_PARAMS['c']}, d={TARGET_PARAMS['d']}, e={TARGET_PARAMS['e']}")
print(f"  Ground-truth death rate (2D, from 3.1):")
print(f"    gamma_select = A*exp(-x^2/(2*w_x^2))*softplus(k1*(y-y_select))")
print(f"    gamma_term   = B*softplus(k2*(y-y_max))")
print(f"    A={DEATH_PARAMS_2D['A']}, w_x={DEATH_PARAMS_2D['w_x']}, "
      f"k1={DEATH_PARAMS_2D['k1']}, y_select={DEATH_PARAMS_2D['y_select']}")
print(f"    B={DEATH_PARAMS_2D['B']}, k2={DEATH_PARAMS_2D['k2']}, "
      f"y_max={DEATH_PARAMS_2D['y_max']}")
print(f"  Source: oval Gaussian, mu=(0,-1), sigma_x=0.25, sigma_y=0.10")

key = jax.random.PRNGKey(4)
key, tgt_queue_key, tgt_sim_key = jax.random.split(key, 3)

print("\nGenerating ground-truth snapshot ...")

tgt_z_init, tgt_wake = build_particle_queue(
    sources=SOURCES, n_particles=N_PARTICLES,
    t_final=T_FINAL, dt=DT, key=tgt_queue_key,
)

tgt_all_px, tgt_all_py, tgt_all_S = simulate_ground_truth_full(
    tgt_z_init, tgt_wake, N_STEPS, DT, tgt_sim_key
)

# Extract single snapshot at t=T_FINAL
final_step = N_STEPS - 1
tgt_z = jnp.column_stack([tgt_all_px[final_step], tgt_all_py[final_step]])
tgt_S = tgt_all_S[final_step]
tgt_mass = float(jnp.sum(jnp.exp(tgt_S)))
n_alive = int(jnp.sum(jnp.exp(tgt_S) > 0.01))
print(f"  t={T_FINAL:.1f}: mass={tgt_mass:.2f}, ~{n_alive} alive particles")


# ==========================================================================
# 8. Build training particle queue
# ==========================================================================

key, queue_key = jax.random.split(key)
z_init, wake_schedule = build_particle_queue(
    sources=SOURCES, n_particles=N_PARTICLES,
    t_final=T_FINAL, dt=DT, key=queue_key,
)
print(f"  Training queue: {z_init.shape[0]} particles\n")


# ==========================================================================
# 9. Create model: NN potential + fixed 2D death rate
# ==========================================================================

key, model_key = jax.random.split(key)

# Learnable NN (c_conf=0 — quartic confinement provided externally)
phi_nn = PotentialNN(model_key, d_latent=2, c_conf=0.0,
                     hidden_sizes=(16, 16))

# Wrap with fixed quartic confinement: Phi = Phi_nn + exp(b)*x^4 + exp(d)*y^4
potential = PotentialWithConfinement(
    nn=phi_nn, b=TARGET_PARAMS['b'], d=TARGET_PARAMS['d'],
)

# Fixed (non-learnable) death rate
fixed_death = FixedDeathRate2D(**DEATH_PARAMS_2D)

model = SourceDeathModel(potential=potential, death_rate=fixed_death)

n_params_phi = sum(x.size for x in jax.tree.leaves(eqx.filter(model.potential, eqx.is_array)))
n_params_total = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
print(f"SourceDeathModel created:")
print(f"  Phi = Phi_nn(2->16->16->1) + exp({TARGET_PARAMS['b']})*x^4 + exp({TARGET_PARAMS['d']})*y^4")
print(f"  Phi_nn: {n_params_phi} learnable params (c_conf=0)")
print(f"  Quartic conf: exp(b)={jnp.exp(TARGET_PARAMS['b']):.4f}, exp(d)={jnp.exp(TARGET_PARAMS['d']):.4f} (fixed)")
print(f"  gamma: FixedDeathRate2D (0 learnable params)")
print(f"  Total learnable: {n_params_total} params")


# ==========================================================================
# 10. Train
# ==========================================================================

N_EPOCHS = 400
PRINT_EVERY = 10
PATIENCE = 200

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

print(f"\nTraining for {N_EPOCHS} epochs (single snapshot at t={T_FINAL}) ...")
print("-" * 65)

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

eqx.tree_serialise_leaves("trained_model_sd_nn_asym_conf.eqx", model)
print("Model saved to trained_model_sd_nn_asym_conf.eqx")


# ==========================================================================
# 11. Visualisation
# ==========================================================================

# --- Shared grids ---
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

# Align potentials: subtract value at origin (constant offset is arbitrary)
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
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title(f'Training (NN + fixed quartic conf + fixed 2D death, t={T_FINAL:.0f})')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_nn_asym_conf.png", dpi=150, bbox_inches='tight')
plt.close()


# --- Figure 2: Force vector fields F = -grad(Phi) ---
n_arrow = 22
xs_q = np.linspace(-3.5, 3.5, n_arrow)
ys_q = np.linspace(-1.5, 3.0, n_arrow)
Xq, Yq = np.meshgrid(xs_q, ys_q)

# True force
grad_true = jax.grad(analytical_potential, argnums=(0, 1))
Fx_true = np.zeros_like(Xq)
Fy_true = np.zeros_like(Yq)
for i in range(n_arrow):
    for j in range(n_arrow):
        gx, gy = grad_true(float(Xq[i, j]), float(Yq[i, j]), TARGET_PARAMS)
        Fx_true[i, j] = -float(gx)
        Fy_true[i, j] = -float(gy)

# Learned force
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
    # Potential contours as background
    cf = ax.contourf(np.array(Xg), np.array(Yg), Z_bg,
                     levels=pot_levels, cmap='viridis', alpha=0.35)
    ax.contour(np.array(Xg), np.array(Yg), Z_bg,
               levels=pot_levels, colors='0.5', linewidths=0.4, alpha=0.5)

    # Force arrows — proportional length, shared scale
    q = ax.quiver(Xq, Yq, Fx, Fy, mag,
                  cmap='hot_r', scale=mag_max * 5,
                  width=0.004, headwidth=3.5, headlength=4,
                  headaxislength=3.5, clim=(0, mag_max),
                  zorder=3)

    ax.set_xlim(-3.8, 3.8)
    ax.set_ylim(-1.8, 3.3)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_aspect('equal')

fig.subplots_adjust(right=0.88, wspace=0.25)
cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
fig.colorbar(q, cax=cbar_ax, label=r'$|\mathbf{F}|$')
plt.savefig("force_field_conf.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"\n  Force magnitude: true max={mag_true.max():.1f} mean={mag_true.mean():.1f},"
      f" learned max={mag_learned.max():.1f} mean={mag_learned.mean():.1f}")


# --- Figure 3: Potential contours + 1D slices ---
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# Top row: contour comparison + error
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

# Bottom row: 1D slices
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
plt.savefig("landscape_nn_asym_conf.png", dpi=150, bbox_inches='tight')
plt.close()


# --- Figure 4: Snapshot particle comparison ---
VIS_TIMES = [1.0, 2.0, 4.0, 6.0]
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

    # Bottom row: simulated (learned potential + fixed death)
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

fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
fig.colorbar(sc, cax=cbar_ax, label=r'$w = e^S$')

plt.savefig("snapshots_nn_asym_conf.png", dpi=150, bbox_inches='tight')
plt.close()

# --- Figure 5: Ground-truth 2D death rate ---
fig, ax = plt.subplots(figsize=(8, 6))
G_death = jax.vmap(jax.vmap(
    lambda x, y: analytical_death_rate_2d(x, y, DEATH_PARAMS_2D)
))(Xg, Yg)
d_levels = np.linspace(0, float(jnp.max(G_death).clip(max=15.0)), 31)
cf = ax.contourf(np.array(Xg), np.array(Yg), np.array(G_death),
                 levels=d_levels, cmap='magma_r', extend='max')
ax.contour(np.array(Xg), np.array(Yg), np.array(G_death),
           levels=d_levels, colors='0.3', linewidths=0.4, alpha=0.5)
plt.colorbar(cf, ax=ax, label=r'$\gamma(x, y)$')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Ground-truth 2D death rate (fixed)', fontsize=13)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig("death_rate_2d_conf.png", dpi=150, bbox_inches='tight')
plt.close()

print("\nDone. Figures saved:")
print("  training_nn_asym_conf.png")
print("  force_field_conf.png")
print("  landscape_nn_asym_conf.png")
print("  snapshots_nn_asym_conf.png")
print("  death_rate_2d_conf.png")
