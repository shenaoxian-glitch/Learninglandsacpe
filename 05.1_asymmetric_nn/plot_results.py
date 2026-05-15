#!/usr/bin/env python
"""Plot results for model 5.1c using saved model weights."""
import matplotlib
matplotlib.use('Agg')

import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from models import SourceDeathModel
from models.potential import PotentialNN
from training.data_loader import (
    analytical_potential, analytical_death_rate_2d, build_particle_queue,
)
from simulator.sde_solver import simulate_open_system_full

# --- Rebuild model and load weights ---

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
        gs = self.A * jnp.exp(-x**2/(2*self.w_x**2)) * jax.nn.softplus(self.k1*(y - self.y_select))
        gt = self.B * jax.nn.softplus(self.k2*(y - self.y_max))
        return gs + gt

class PotentialWithConfinement(eqx.Module):
    nn: PotentialNN
    b: float = eqx.field(static=True)
    d: float = eqx.field(static=True)
    def __call__(self, z):
        x, y = z[0], z[1]
        return self.nn(z) + jnp.exp(self.b) * x**4 + jnp.exp(self.d) * y**4

TARGET_PARAMS = {'a': -1.0, 'b': -1.6, 'c': 3.5, 'd': -1.2, 'e': 0.7, 'sigma': 1.5}
DEATH_PARAMS_2D = {'A': 2.0, 'w_x': 0.5, 'k1': 1.0, 'y_select': 1.0,
                   'B': 5.0, 'k2': 3.0, 'y_max': 3.0}

key = jax.random.PRNGKey(0)
phi_nn = PotentialNN(key, d_latent=2, c_conf=0.0, hidden_sizes=(16, 16))
potential = PotentialWithConfinement(nn=phi_nn, b=TARGET_PARAMS['b'], d=TARGET_PARAMS['d'])
fixed_death = FixedDeathRate2D(**DEATH_PARAMS_2D)
model = SourceDeathModel(potential=potential, death_rate=fixed_death)
model = eqx.tree_deserialise_leaves("trained_model_sd_nn_asym_conf.eqx", model)

# --- Compute grids ---
# Focus on the physically relevant region
x_range = (-3.5, 3.5)
y_range = (-1.5, 3.0)
n_fine = 200

x_f = np.linspace(*x_range, n_fine)
y_f = np.linspace(*y_range, n_fine)
Xf, Yf = np.meshgrid(x_f, y_f)

Z_true = np.array(jax.vmap(jax.vmap(
    lambda x, y: analytical_potential(x, y, TARGET_PARAMS)
))(jnp.array(Xf), jnp.array(Yf)))

Z_learned = np.array(jax.vmap(
    lambda row_y: jax.vmap(
        lambda xi: model.potential(jnp.array([xi, row_y]))
    )(jnp.array(x_f))
)(jnp.array(y_f)))

# Potentials are defined up to a constant — normalize by subtracting min
# so both share the same baseline for visual comparison
Z_true_n = Z_true - Z_true.min()
Z_learned_n = Z_learned - Z_learned.min()

vmax = min(max(np.percentile(Z_true_n, 97), np.percentile(Z_learned_n, 97)), 30.0)
levels = np.linspace(0, vmax, 40)

print(f"True potential range: [{Z_true.min():.1f}, {Z_true.max():.1f}]")
print(f"Learned potential range: [{Z_learned.min():.1f}, {Z_learned.max():.1f}]")
print(f"Offset (learned min - true min): {Z_learned.min() - Z_true.min():.1f}")
print(f"Normalized color scale: [0, {vmax:.1f}]")


# ==============================================================
# Figure 1: Potential contours — 1 row, 3 columns
# ==============================================================
# Data aspect: x spans 7, y spans 4.5 → each panel is taller than wide
# No aspect='equal' — let panels fill the allocated space naturally
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

cf0 = axes[0].contourf(Xf, Yf, Z_true_n, levels=levels, cmap='viridis', extend='both')
axes[0].contour(Xf, Yf, Z_true_n, levels=levels, colors='k', linewidths=0.3, alpha=0.4)
plt.colorbar(cf0, ax=axes[0], label=r'$\Phi - \Phi_{min}$')
axes[0].set_title('True Potential', fontsize=12)
axes[0].set_xlabel('x'); axes[0].set_ylabel('y')

cf1 = axes[1].contourf(Xf, Yf, Z_learned_n, levels=levels, cmap='viridis', extend='both')
axes[1].contour(Xf, Yf, Z_learned_n, levels=levels, colors='k', linewidths=0.3, alpha=0.4)
plt.colorbar(cf1, ax=axes[1], label=r'$\Phi - \Phi_{min}$')
axes[1].set_title('Learned Potential', fontsize=12)
axes[1].set_xlabel('x'); axes[1].set_ylabel('y')

Z_diff = Z_learned_n - Z_true_n
d_abs = min(max(abs(np.percentile(Z_diff, 2)), abs(np.percentile(Z_diff, 98))), 15.0)
diff_levels = np.linspace(-d_abs, d_abs, 40)
cf2 = axes[2].contourf(Xf, Yf, Z_diff, levels=diff_levels, cmap='RdBu_r', extend='both')
axes[2].contour(Xf, Yf, Z_diff, levels=diff_levels[::4], colors='k', linewidths=0.3, alpha=0.3)
plt.colorbar(cf2, ax=axes[2], label=r'$\Phi_{learned} - \Phi^*$')
axes[2].set_title('Error', fontsize=12)
axes[2].set_xlabel('x'); axes[2].set_ylabel('y')

plt.tight_layout()
plt.savefig("potential_contours.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: potential_contours.png")


# ==============================================================
# Figure 2: Force streamlines (cleaner than quiver)
# ==============================================================

# Compute forces on a fine grid for streamplot
n_stream = 100
x_s = np.linspace(*x_range, n_stream)
y_s = np.linspace(*y_range, n_stream)
Xs, Ys = np.meshgrid(x_s, y_s)

grad_true_fn = jax.grad(analytical_potential, argnums=(0, 1))
grad_learned_fn = jax.grad(lambda z: model.potential(z))

Fx_t = np.zeros((n_stream, n_stream))
Fy_t = np.zeros((n_stream, n_stream))
Fx_l = np.zeros((n_stream, n_stream))
Fy_l = np.zeros((n_stream, n_stream))

for i in range(n_stream):
    for j in range(n_stream):
        gx, gy = grad_true_fn(float(Xs[i, j]), float(Ys[i, j]), TARGET_PARAMS)
        Fx_t[i, j] = -float(gx)
        Fy_t[i, j] = -float(gy)
        g = grad_learned_fn(jnp.array([Xs[i, j], Ys[i, j]]))
        Fx_l[i, j] = -float(g[0])
        Fy_l[i, j] = -float(g[1])

speed_t = np.sqrt(Fx_t**2 + Fy_t**2)
speed_l = np.sqrt(Fx_l**2 + Fy_l**2)
speed_max = max(speed_t.max(), speed_l.max())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, Fx, Fy, speed, Z_bg, title in [
    (axes[0], Fx_t, Fy_t, speed_t, Z_true_n, r'True $\mathbf{F} = -\nabla\Phi^*$'),
    (axes[1], Fx_l, Fy_l, speed_l, Z_learned_n, r'Learned $\mathbf{F} = -\nabla\Phi_{NN}$'),
]:
    # Potential as background
    ax.contourf(Xf, Yf, Z_bg, levels=levels, cmap='viridis', alpha=0.5)
    ax.contour(Xf, Yf, Z_bg, levels=levels, colors='k', linewidths=0.2, alpha=0.3)

    # Streamlines colored by speed, shared norm
    lw = 1.5 * speed / (speed_max + 1e-8)
    strm = ax.streamplot(x_s, y_s, Fx, Fy,
                         color=speed, cmap='magma_r',
                         norm=mcolors.Normalize(vmin=0, vmax=speed_max),
                         linewidth=lw, density=1.8, arrowsize=1.2,
                         arrowstyle='->')
    plt.colorbar(strm.lines, ax=ax, label=r'$|\mathbf{F}|$')

    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=12)

plt.tight_layout()
plt.savefig("force_streamlines.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: force_streamlines.png")

print(f"\nForce stats:")
print(f"  True:    max={speed_t.max():.1f}, mean={speed_t.mean():.1f}")
print(f"  Learned: max={speed_l.max():.1f}, mean={speed_l.mean():.1f}")


# ==============================================================
# Figure 3: 1D slices
# ==============================================================
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
y_slices = [-0.5, 0.5, 1.0, 1.5]
x_line = np.linspace(-3.5, 3.5, 300)

for ax, y_val in zip(axes.flat, y_slices):
    phi_true = np.array([
        float(analytical_potential(float(xi), y_val, TARGET_PARAMS))
        for xi in x_line
    ])
    phi_learned = np.array(jax.vmap(
        lambda xi: model.potential(jnp.array([xi, y_val]))
    )(jnp.array(x_line)))

    # Align by subtracting each curve's minimum so shapes are comparable
    phi_true_n = phi_true - phi_true.min()
    phi_learned_n = phi_learned - phi_learned.min()

    ax.plot(x_line, phi_true_n, 'b-', linewidth=2.5, label='True')
    ax.plot(x_line, phi_learned_n, 'r--', linewidth=2.5, label='Learned')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel(r'$\Phi - \Phi_{min}$', fontsize=11)
    ax.set_title(f'y = {y_val}', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("potential_slices.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: potential_slices.png")


# ==============================================================
# Figure 4: Snapshot particle comparison (target vs learned)
# ==============================================================

SIGMA = TARGET_PARAMS['sigma']
N_PARTICLES = 4000
T_FINAL = 6.0
DT = 0.01
N_STEPS = int(T_FINAL / DT)

SOURCES = [
    {'mu': jnp.array([0.0, -1.0]), 'sigma_x': 0.25, 'sigma_y': 0.10,
     'n_particles': N_PARTICLES},
]

# --- Ground-truth simulation (analytical) ---
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


print("\nSimulating for snapshot comparison ...")
sim_key = jax.random.PRNGKey(42)

# Target (ground truth)
k1, k2, k3, k4 = jax.random.split(sim_key, 4)
tgt_z_init, tgt_wake = build_particle_queue(
    sources=SOURCES, n_particles=N_PARTICLES,
    t_final=T_FINAL, dt=DT, key=k1,
)
tgt_all_px, tgt_all_py, tgt_all_S = simulate_ground_truth_full(
    tgt_z_init, tgt_wake, N_STEPS, DT, k2
)

# Learned model
eval_z_init, eval_wake = build_particle_queue(
    sources=SOURCES, n_particles=N_PARTICLES,
    t_final=T_FINAL, dt=DT, key=k3,
)
sim_all_z, sim_all_S = simulate_open_system_full(
    model, eval_z_init, eval_wake, SIGMA, N_STEPS, DT, k4
)

# Plot
VIS_TIMES = [1.0, 2.0, 4.0, 6.0]
VIS_STEPS = [int(t / DT) - 1 for t in VIS_TIMES]

fig, axes = plt.subplots(2, 4, figsize=(16, 9))

for col, (t_vis, step_vis) in enumerate(zip(VIS_TIMES, VIS_STEPS)):
    # Top row: target
    ax_tgt = axes[0, col]
    tgt_z_i = jnp.column_stack([tgt_all_px[step_vis], tgt_all_py[step_vis]])
    tgt_S_i = tgt_all_S[step_vis]
    tgt_w_i = np.array(jnp.clip(jnp.exp(tgt_S_i), 0.0, 1.0))
    alive_t = tgt_w_i > 1e-6

    ax_tgt.contourf(Xf, Yf, Z_true_n, levels=levels, cmap='viridis', alpha=0.3)
    ax_tgt.contour(Xf, Yf, Z_true_n, levels=levels, colors='k', linewidths=0.2, alpha=0.3)
    ax_tgt.scatter(np.array(tgt_z_i[alive_t, 0]), np.array(tgt_z_i[alive_t, 1]),
                   c=tgt_w_i[alive_t], cmap='coolwarm', s=4, alpha=0.5,
                   vmin=0, vmax=1, edgecolors='none')
    ax_tgt.set_xlim(*x_range); ax_tgt.set_ylim(*y_range)
    ax_tgt.set_title(f'Target t={t_vis:.0f}', fontsize=11)
    if col == 0:
        ax_tgt.set_ylabel('y (target)')

    # Bottom row: learned
    ax_sim = axes[1, col]
    sim_z_i = sim_all_z[step_vis]
    sim_S_i = sim_all_S[step_vis]
    sim_w_i = np.array(jnp.clip(jnp.exp(sim_S_i), 0.0, 1.0))
    alive_s = sim_w_i > 1e-6

    ax_sim.contourf(Xf, Yf, Z_learned_n, levels=levels, cmap='viridis', alpha=0.3)
    ax_sim.contour(Xf, Yf, Z_learned_n, levels=levels, colors='k', linewidths=0.2, alpha=0.3)
    sc = ax_sim.scatter(np.array(sim_z_i[alive_s, 0]), np.array(sim_z_i[alive_s, 1]),
                        c=sim_w_i[alive_s], cmap='coolwarm', s=4, alpha=0.5,
                        vmin=0, vmax=1, edgecolors='none')
    ax_sim.set_xlim(*x_range); ax_sim.set_ylim(*y_range)
    ax_sim.set_title(f'Learned t={t_vis:.0f}', fontsize=11)
    if col == 0:
        ax_sim.set_ylabel('y (learned)')

fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
fig.colorbar(sc, cax=cbar_ax, label=r'$w = e^S$')

plt.savefig("snapshots_nn_asym_conf.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: snapshots_nn_asym_conf.png")
