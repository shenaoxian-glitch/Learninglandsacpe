"""
train_sd.py — open-system source + death model (parametric death rate)

Physics: empty system at t=0. Sources continuously inject particles.
Each particle has a unique birth time t0 and position x0 ~ N(mu, sigma_src^2 I).
Particles flow down potential Phi, bifurcate, and die via
gamma(z) = softplus(k * (y - y_c)) with learnable y_c and k.

Only Phi, y_c, and k are learned. Sigma is fixed.

Usage:
    python train_sd.py
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
    generate_open_system_data, build_particle_queue,
    analytical_potential, analytical_death_rate,
)
from training.trainer import train_sd
from simulator.sde_solver import simulate_open_system
from analysis.visualization import (
    plot_nn_landscape,
    plot_nn_death_rate,
    plot_training_history,
    plot_comparison,
    _eval_grid,
)


# ==========================================================================
# 1. Ground-truth parameters
# ==========================================================================

TARGET_PARAMS = {
    'a': 0, 'b': -1.6, 'c': 3.5, 'd': -1.2, 'sigma': 1.5,
}

DEATH_PARAMS = {
    'y_threshold': 2.2,  # target: death activates above y=2.2
}

# Source: 1000 particles, each born at a unique time in [0, T_FINAL)
SOURCES = [
    {'mu': jnp.array([0.0, -1.0]), 'sigma_src': 0.15, 'n_particles': 1000},
]

DT = 0.01
T_FINAL = 8.0
N_STEPS = int(T_FINAL / DT)  # 800 steps
N_PARTICLES = 1000

SIGMA = TARGET_PARAMS['sigma']  # fixed, not learned

LOSS_CONFIG = {
    'lam_mass': 1.0,
    'lam_sparse': 0.0,   # no L1 needed for parametric death (only 2 params)
}


# ==========================================================================
# 2. Generate ground-truth steady-state target
# ==========================================================================

print("Generating open-system ground-truth data ...")
print(f"  t_final={T_FINAL}, dt={DT}, n_steps={N_STEPS}")
print(f"  {N_PARTICLES} particles, birth rate={N_PARTICLES/T_FINAL:.0f}/time unit")
print(f"  Death: softplus(y - {DEATH_PARAMS['y_threshold']})")

key = jax.random.PRNGKey(4)
key, target_key = jax.random.split(key)

target_pos, target_S, target_mass, z_init_gt, wake_schedule_gt = generate_open_system_data(
    params=TARGET_PARAMS,
    death_params=DEATH_PARAMS,
    sources=SOURCES,
    n_particles=N_PARTICLES,
    t_final=T_FINAL,
    dt=DT,
    key=target_key,
)

target_w = jnp.clip(jnp.exp(target_S), 0.0, 1.0)
print(f"  Target: {target_pos.shape[0]} particles, total_mass={target_mass:.2f}")
print(f"  Target w: mean={float(target_w.mean()):.3f}, "
      f"min={float(target_w.min()):.6f}, max={float(target_w.max()):.3f}")


# ==========================================================================
# 3. Build particle queue for training (same structure, fresh noise)
# ==========================================================================

key, queue_key = jax.random.split(key)
z_init, wake_schedule = build_particle_queue(
    sources=SOURCES,
    n_particles=N_PARTICLES,
    t_final=T_FINAL,
    dt=DT,
    key=queue_key,
)
print(f"  Training queue: {z_init.shape[0]} pre-allocated particles")


# ==========================================================================
# 4. Create model (Phi + parametric death rate)
# ==========================================================================

key, model_key = jax.random.split(key)
model = create_sd_model(
    model_key,
    d_latent=2,
    c_conf=0.01,
    parametric_death=True,
    y_c_init=0.0,   # start far from true value (2.2) to test learning
    k_init=1.0,
)
print(f"SourceDeathModel created (Phi + parametric gamma).")
print(f"  Initial: y_c={float(model.death_rate.y_c):.2f}, "
      f"k={float(jnp.exp(model.death_rate.log_k)):.2f}\n")


# ==========================================================================
# 5. Train
# ==========================================================================

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=3e-3, weight_decay=1e-4),
)

model, history = train_sd(
    model=model,
    z_init=z_init,
    wake_schedule=wake_schedule,
    target_pos=target_pos,
    target_mass=target_mass,
    sigma=SIGMA,
    optimizer=optimizer,
    dt=DT,
    n_steps=N_STEPS,
    n_epochs=1000,
    lam_mass=LOSS_CONFIG['lam_mass'],
    lam_sparse=LOSS_CONFIG['lam_sparse'],
    key=jax.random.PRNGKey(42),
    use_fresh_keys=True,
    print_every=25,
    patience=200,
)

# Print learned death rate parameters
y_c_learned = float(model.death_rate.y_c)
k_learned = float(jnp.exp(model.death_rate.log_k))
print(f"\nLearned death rate: gamma(z) = softplus({k_learned:.3f} * (y - {y_c_learned:.3f}))")
print(f"  True:   y_c = {DEATH_PARAMS['y_threshold']}, k = 1.0")

eqx.tree_serialise_leaves("trained_model_sd.eqx", model)
print("Model saved to trained_model_sd.eqx")


# ==========================================================================
# 6. Visualise
# ==========================================================================

# --- Training curves ---
plot_training_history(history, title="Open-System Training Loss")
plt.savefig("training_history_sd.png", dpi=150, bbox_inches='tight')
plt.close()

# --- Learned landscape & death rate vs ground truth ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

plot_nn_landscape(model, ax=axes[0], title="Learned Potential")

# Shared death rate colorscale
x_grid = jnp.linspace(-5, 5, 200)
y_grid = jnp.linspace(-2, 4, 200)
Xg, Yg = jnp.meshgrid(x_grid, y_grid)
gamma_true = analytical_death_rate(Xg, Yg, DEATH_PARAMS)
_, _, gamma_learned = _eval_grid(lambda z: model.death_rate(z), (-5, 5), (-2, 4), 200)

g_vmin = min(float(gamma_true.min()), float(gamma_learned.min()))
g_vmax = max(float(gamma_true.max()), float(gamma_learned.max()))

plot_nn_death_rate(model, ax=axes[1], title=r"Learned $\gamma(z)$", vmin=g_vmin, vmax=g_vmax)

levels = np.linspace(g_vmin, g_vmax, 31)
cf = axes[2].contourf(Xg, Yg, gamma_true, levels=levels, cmap='YlOrRd', extend='both')
plt.colorbar(cf, ax=axes[2], label=r'$\gamma^*(x,y)$')
axes[2].set_xlabel('x'); axes[2].set_ylabel('y')
axes[2].set_title(r"Ground Truth $\gamma^*(x,y)$")

plt.tight_layout()
plt.savefig("learned_landscape_sd.png", dpi=150, bbox_inches='tight')
plt.close()

# --- Compare: both panels show raw w = exp(S) ---
key, sim_key = jax.random.split(key)

key, sim_queue_key = jax.random.split(key)
z_init_sim, wake_schedule_sim = build_particle_queue(
    sources=SOURCES, n_particles=N_PARTICLES, t_final=T_FINAL,
    dt=DT, key=sim_queue_key,
)

sim_z, sim_S, sim_alpha = simulate_open_system(
    model, z_init_sim, wake_schedule_sim, SIGMA, N_STEPS, DT, sim_key
)

sim_w = jnp.clip(jnp.exp(sim_S), 0.0, 1.0)

fig = plot_comparison(
    target_pos, target_w,
    sim_z, sim_w,
    model=model,
    target_potential_fn=lambda z: analytical_potential(z[0], z[1], TARGET_PARAMS),
    weight_label=r'$w = e^S$',
)
plt.savefig("comparison_sd.png", dpi=150, bbox_inches='tight')
plt.close()

print("\nDone. Figures saved to training_history_sd.png, learned_landscape_sd.png, comparison_sd.png")
