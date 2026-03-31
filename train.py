"""
train.py — multi-timepoint short-time transition matching
on the neural-network Waddington landscape with proliferation.

Howe & Mani format: D = {(t_i, X_i, t_{i+1}, X_{i+1})}

Usage:
    python train.py
"""
import matplotlib
matplotlib.use('Agg')

import jax
import jax.numpy as jnp
import optax

from models import create_model
from training.data_loader import generate_multi_timepoint_data
from training.trainer import train
from simulator.sde_solver import simulate
from analysis.visualization import (
    plot_nn_landscape,
    plot_nn_proliferation,
    plot_weighted_particles,
    plot_training_history,
    plot_comparison,
)

import matplotlib.pyplot as plt


# ==========================================================================
# 1. Ground-truth parameters
# ==========================================================================

TARGET_PARAMS = {
    'a': 0, 'b': -1.6, 'c': 3.5, 'd': -1.2, 'sigma': 1.0,
}

PROLIF_PARAMS = {
    'x0': 0.0, 'y0': -1.0,
    'beta': 2.0, 'gamma': 0.8, 'delta': 0.3,
}

# Multiple snapshot times for short-time transition matching
# Longer intervals (T=1.0 each) give R more time to accumulate weight signal
SNAPSHOT_TIMES = [0.0, 1.0, 2.0, 3.0]

DT = 0.005
N_PARTICLES = 2000

LOSS_CONFIG = {
    'lam_cons': 1.0,
    'lam_reg': 0.0001,
}


# ==========================================================================
# 2. Generate multi-timepoint target data
# ==========================================================================

print("Generating multi-timepoint ground-truth data ...")
key = jax.random.PRNGKey(4)
key, target_key = jax.random.split(key)

transitions, snapshots = generate_multi_timepoint_data(
    params=TARGET_PARAMS,
    prolif_params=PROLIF_PARAMS,
    n_particles=N_PARTICLES,
    snapshot_times=SNAPSHOT_TIMES[1:],  # [0.5, 1.0, 1.5, 2.0]; t=0 is prepended
    dt=DT,
    key=target_key,
    x0_mean=jnp.array([0.0, -1.0]),
    x0_std=0.1,
)

print(f"  {len(snapshots)} snapshots at t = {[f'{s[0]:.1f}' for s in snapshots]}")
print(f"  {len(transitions)} transitions:")
for tr in transitions:
    print(f"    [{tr['t0']:.1f} -> {tr['t1']:.1f}] ({tr['n_steps']} steps)")


# ==========================================================================
# 3. Create neural-network model
# ==========================================================================

key, model_key = jax.random.split(key)
model = create_model(
    model_key,
    d_latent=2,
    d_signal=0,
    c_conf=0.01,
    sigma_init=0.5,
    use_tilt=False,
    state_dependent_noise=False,
)
print("Model created.\n")


# ==========================================================================
# 4. Train with multi-timepoint short-time transition matching
# ==========================================================================

optimizer = optax.adam(learning_rate=3e-3)

model, history = train(
    model=model,
    transitions=transitions,
    optimizer=optimizer,
    dt=DT,
    loss_config=LOSS_CONFIG,
    n_epochs=2000,
    key=jax.random.PRNGKey(42),
    use_fresh_keys=True,
    print_every=100,
    patience=500,
)

sigma_learned = float(jnp.exp(model.noise.log_sigma))
print(f"\nLearned sigma: {sigma_learned:.3f}  (true: {TARGET_PARAMS['sigma']})")


# ==========================================================================
# 5. Visualise
# ==========================================================================

# --- Training curves ---
plot_training_history(history, title="Multi-Timepoint Training Loss")
plt.savefig("training_history.png", dpi=150, bbox_inches='tight')
plt.close()

# --- Learned landscape & proliferation vs ground truth ---
from training.data_loader import analytical_proliferation
from analysis.visualization import _eval_grid

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

plot_nn_landscape(model, ax=axes[0], title="Learned Potential")

# Compute shared R colorscale between learned and ground truth
x_grid = jnp.linspace(-5, 5, 200)
y_grid = jnp.linspace(-2, 4, 200)
Xg, Yg = jnp.meshgrid(x_grid, y_grid)
R_true = analytical_proliferation(Xg, Yg, PROLIF_PARAMS)
_, _, R_learned = _eval_grid(lambda z: model.proliferation(z), (-5, 5), (-2, 4), 200)

r_vmin = min(float(R_true.min()), float(R_learned.min()))
r_vmax = max(float(R_true.max()), float(R_learned.max()))

plot_nn_proliferation(model, ax=axes[1], title="Learned R(z)", vmin=r_vmin, vmax=r_vmax)

# Ground-truth R with same scale
import numpy as np
levels = np.linspace(r_vmin, r_vmax, 31)
cf = axes[2].contourf(Xg, Yg, R_true, levels=levels, cmap='RdBu_r', extend='both')
plt.colorbar(cf, ax=axes[2], label='R*(x,y)')
axes[2].contour(Xg, Yg, R_true, levels=[0], colors='black', linewidths=1.5, linestyles='--')
axes[2].set_xlabel('x'); axes[2].set_ylabel('y')
axes[2].set_title("Ground Truth R*(x,y)")

plt.tight_layout()
plt.savefig("learned_landscape.png", dpi=150, bbox_inches='tight')
plt.close()

# --- Compare at final timepoint ---
# Simulate full trajectory from t=0 to t_final using learned model
key, sim_key, init_key = jax.random.split(key, 3)
z0 = jnp.array([0.0, -1.0]) + 0.1 * jax.random.normal(
    init_key, (N_PARTICLES, 2)
)
total_steps = int(SNAPSHOT_TIMES[-1] / DT)
sim_z, sim_S, sim_alpha = simulate(model, z0, total_steps, DT, sim_key)

# Final target snapshot
_, target_final_pos, target_final_alpha = snapshots[-1]

fig = plot_comparison(
    target_final_pos, target_final_alpha,
    sim_z, sim_alpha,
    model=model,
)
plt.savefig("comparison.png", dpi=150, bbox_inches='tight')
plt.close()

print("\nDone. Figures saved to training_history.png, learned_landscape.png, comparison.png")
