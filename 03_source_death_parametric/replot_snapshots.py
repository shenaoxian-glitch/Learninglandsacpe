#!/usr/bin/env python
"""Replot snapshots_multi.png from saved learned params (no training needed)."""
import matplotlib
matplotlib.use('Agg')

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from training.data_loader import build_particle_queue

# ---------- constants (must match train_sd_parametric_multi.py) ----------
TARGET_PARAMS = {'a': 0, 'b': -1.6, 'c': 3.5, 'd': -1.2, 'sigma': 1.5}
TRUE_DEATH = {'y_c': 2.2, 'k': 1.0}
N_PARTICLES = 2000
DT = 0.01
T_FINAL = 8.0
N_STEPS = int(T_FINAL / DT)
SIGMA = TARGET_PARAMS['sigma']
SOURCES = [
    {'mu': jnp.array([0.0, -1.0]), 'sigma_src': 0.15, 'n_particles': N_PARTICLES},
]

# ---------- learned params from log_multi_regional4.txt ----------
LEARNED_PARAMS = {
    'a': jnp.array(-0.2495),
    'b': jnp.array(-1.4099),
    'c': jnp.array(3.1661),
    'd': jnp.array(-1.2663),
    'y_c': jnp.array(1.9472),
    'log_k': jnp.array(float(jnp.log(1.7038))),
}

# ---------- functions (copied from train_sd_parametric_multi.py) ----------
def potential(x, y, p):
    return -(y - p['a']) * x**2 + jnp.exp(p['b']) * x**4 - p['c'] * y + jnp.exp(p['d']) * y**4

_force_fn = jax.vmap(jax.grad(potential, argnums=(0, 1)), in_axes=(0, 0, None))

def simulate_open_full(learnable, z_init, wake_schedule, n_steps, dt, key):
    pot_params = {k: learnable[k] for k in ['a', 'b', 'c', 'd']}
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
        gx, gy = _force_fn(px, py, pot_params)
        fx, fy = -gx, -gy
        kx, ky = jax.random.split(step_key)
        nx = jax.random.normal(kx, px.shape) * jnp.sqrt(dt)
        ny = jax.random.normal(ky, py.shape) * jnp.sqrt(dt)
        new_px = px + fx * dt + SIGMA * nx
        new_py = py + fy * dt + SIGMA * ny
        k = jnp.exp(learnable['log_k'])
        gamma = jax.nn.softplus(k * (py - learnable['y_c']))
        new_S = S - gamma * dt
        return (new_px, new_py, new_S), (new_px, new_py, new_S)

    step_indices = jnp.arange(n_steps)
    _, (all_px, all_py, all_S) = jax.lax.scan(
        scan_body, (z_init[:, 0], z_init[:, 1], S0), (step_keys, step_indices)
    )
    return all_px, all_py, all_S

# ---------- run simulations ----------
key = jax.random.PRNGKey(4)
key, tgt_queue_key, tgt_sim_key = jax.random.split(key, 3)

tgt_z_init, tgt_wake = build_particle_queue(
    sources=SOURCES, n_particles=N_PARTICLES,
    t_final=T_FINAL, dt=DT, key=tgt_queue_key,
)

true_learnable = {
    'a': jnp.array(float(TARGET_PARAMS['a'])),
    'b': jnp.array(float(TARGET_PARAMS['b'])),
    'c': jnp.array(float(TARGET_PARAMS['c'])),
    'd': jnp.array(float(TARGET_PARAMS['d'])),
    'y_c': jnp.array(float(TRUE_DEATH['y_c'])),
    'log_k': jnp.array(float(jnp.log(TRUE_DEATH['k']))),
}

print("Running ground-truth simulation ...")
tgt_all_px, tgt_all_py, tgt_all_S = simulate_open_full(
    true_learnable, tgt_z_init, tgt_wake, N_STEPS, DT, tgt_sim_key
)

key, eval_key, eval_queue_key = jax.random.split(key, 3)
z_init_eval, wake_eval = build_particle_queue(
    sources=SOURCES, n_particles=N_PARTICLES,
    t_final=T_FINAL, dt=DT, key=eval_queue_key,
)

print("Running learned simulation ...")
sim_all_px, sim_all_py, sim_all_S = simulate_open_full(
    LEARNED_PARAMS, z_init_eval, wake_eval, N_STEPS, DT, eval_key
)

# ---------- potential grids ----------
x_grid = jnp.linspace(-5, 5, 200)
y_grid = jnp.linspace(-2, 4, 200)
Xg, Yg = jnp.meshgrid(x_grid, y_grid)
Z_true = jax.vmap(jax.vmap(
    lambda x, y: potential(x, y, TARGET_PARAMS)
))(Xg, Yg)
learned_pot_params = {k: LEARNED_PARAMS[k] for k in ['a', 'b', 'c', 'd']}
Z_learned = jax.vmap(jax.vmap(
    lambda x, y: potential(x, y, learned_pot_params)
))(Xg, Yg)

# ---------- Figure 4: snapshot comparison ----------
VIS_TIMES = [1.0, 2.0, 4.0, 8.0]
VIS_STEPS = [int(t / DT) - 1 for t in VIS_TIMES]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for col, (t_vis, step_vis) in enumerate(zip(VIS_TIMES, VIS_STEPS)):
    # Top row: target
    ax_tgt = axes[0, col]
    tgt_px_i = tgt_all_px[step_vis]
    tgt_py_i = tgt_all_py[step_vis]
    tgt_S_i = tgt_all_S[step_vis]
    tgt_w_i = np.array(jnp.clip(jnp.exp(tgt_S_i), 0.0, 1.0))
    alive = tgt_w_i > 1e-6

    ax_tgt.contourf(np.array(Xg), np.array(Yg), np.array(Z_true),
                    levels=30, cmap='viridis', alpha=0.3)
    sc = ax_tgt.scatter(np.array(tgt_px_i[alive]), np.array(tgt_py_i[alive]),
                        c=tgt_w_i[alive], cmap='coolwarm', s=6, alpha=0.5,
                        vmin=0, vmax=1, edgecolors='none')
    ax_tgt.set_xlim(-5, 5); ax_tgt.set_ylim(-2, 4)
    ax_tgt.set_title(f'Target t={t_vis:.0f}')
    if col == 0:
        ax_tgt.set_ylabel('y (target)')

    # Bottom row: learned
    ax_sim = axes[1, col]
    sim_px_i = sim_all_px[step_vis]
    sim_py_i = sim_all_py[step_vis]
    sim_S_i = sim_all_S[step_vis]
    sim_w_i = np.array(jnp.clip(jnp.exp(sim_S_i), 0.0, 1.0))
    alive = sim_w_i > 1e-6

    ax_sim.contourf(np.array(Xg), np.array(Yg), np.array(Z_learned),
                    levels=30, cmap='viridis', alpha=0.3)
    sc = ax_sim.scatter(np.array(sim_px_i[alive]), np.array(sim_py_i[alive]),
                        c=sim_w_i[alive], cmap='coolwarm', s=6, alpha=0.5,
                        vmin=0, vmax=1, edgecolors='none')
    ax_sim.set_xlim(-5, 5); ax_sim.set_ylim(-2, 4)
    ax_sim.set_title(f'Learned t={t_vis:.0f}')
    if col == 0:
        ax_sim.set_ylabel('y (learned)')

fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
fig.colorbar(sc, cax=cbar_ax, label=r'$w = e^S$')

plt.savefig("snapshots_multi.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved snapshots_multi.png")
