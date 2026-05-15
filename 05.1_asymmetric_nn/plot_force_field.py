#!/usr/bin/env python
"""Plot force vector fields F = -grad(Phi) for true vs learned potential."""
import matplotlib
matplotlib.use('Agg')

import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np

from models import SourceDeathModel
from models.potential import PotentialNN
from training.data_loader import analytical_potential

# Re-create the FixedDeathRate2D so we can deserialize the model
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


# Ground-truth params
TARGET_PARAMS = {
    'a': -1.0, 'b': -1.6, 'c': 3.5, 'd': -1.2, 'e': 0.7, 'sigma': 1.5,
}
DEATH_PARAMS_2D = {
    'A': 2.0, 'w_x': 0.5, 'k1': 1.0, 'y_select': 1.0,
    'B': 5.0, 'k2': 3.0, 'y_max': 3.0,
}

# Rebuild model skeleton and load weights
key = jax.random.PRNGKey(0)
potential = PotentialNN(key, d_latent=2, c_conf=0.01, hidden_sizes=(16, 16))
fixed_death = FixedDeathRate2D(**DEATH_PARAMS_2D)
model = SourceDeathModel(potential=potential, death_rate=fixed_death)
model = eqx.tree_deserialise_leaves("trained_model_sd_nn_asym_fixed_death.eqx", model)

# --- Compute force fields on a grid ---
x_range = (-4, 4)
y_range = (-2, 3.5)
n_grid = 25  # arrow density

xs = np.linspace(*x_range, n_grid)
ys = np.linspace(*y_range, n_grid)
Xq, Yq = np.meshgrid(xs, ys)

# True force: F = -grad(Phi*)
grad_true = jax.grad(analytical_potential, argnums=(0, 1))
Fx_true = np.zeros_like(Xq)
Fy_true = np.zeros_like(Yq)
for i in range(n_grid):
    for j in range(n_grid):
        gx, gy = grad_true(float(Xq[i, j]), float(Yq[i, j]), TARGET_PARAMS)
        Fx_true[i, j] = -float(gx)
        Fy_true[i, j] = -float(gy)

# Learned force: F = -grad(Phi_NN)
grad_learned = jax.grad(lambda z: model.potential(z))
Fx_learned = np.zeros_like(Xq)
Fy_learned = np.zeros_like(Yq)
for i in range(n_grid):
    for j in range(n_grid):
        g = grad_learned(jnp.array([Xq[i, j], Yq[i, j]]))
        Fx_learned[i, j] = -float(g[0])
        Fy_learned[i, j] = -float(g[1])

# Force magnitude for color
mag_true = np.sqrt(Fx_true**2 + Fy_true**2)
mag_learned = np.sqrt(Fx_learned**2 + Fy_learned**2)

# Also compute potential on a finer grid for background contours
n_bg = 150
x_bg = np.linspace(*x_range, n_bg)
y_bg = np.linspace(*y_range, n_bg)
Xbg, Ybg = np.meshgrid(x_bg, y_bg)
Z_true = np.array(jax.vmap(jax.vmap(
    lambda x, y: analytical_potential(x, y, TARGET_PARAMS)
))(jnp.array(Xbg), jnp.array(Ybg)))
Z_learned = np.array(jax.vmap(
    lambda row: jax.vmap(
        lambda xi: model.potential(jnp.array([xi, row]))
    )(jnp.array(x_bg))
)(jnp.array(y_bg)))

# Shared color scale for potential background
pot_vmin = min(Z_true.min(), Z_learned.min())
pot_vmax = min(max(Z_true.max(), Z_learned.max()), 30.0)
pot_levels = np.linspace(pot_vmin, pot_vmax, 25)

# Shared arrow scale
mag_max = max(mag_true.max(), mag_learned.max())

# --- Plot 1: Normalized arrows (direction) + magnitude as color, per-panel scale ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, Fx, Fy, mag, Z_bg, title, vmax_mag in [
    (axes[0], Fx_true, Fy_true, mag_true, Z_true,
     r'True Force $\mathbf{F} = -\nabla\Phi^*$', mag_true.max()),
    (axes[1], Fx_learned, Fy_learned, mag_learned, Z_learned,
     r'Learned Force $\mathbf{F} = -\nabla\Phi_{NN}$', mag_learned.max()),
]:
    # Potential contours as background
    ax.contourf(Xbg, Ybg, Z_bg, levels=pot_levels, cmap='viridis', alpha=0.3)
    ax.contour(Xbg, Ybg, Z_bg, levels=pot_levels, colors='gray',
               linewidths=0.3, alpha=0.4)

    # Normalize arrows to unit length (show direction only)
    norm = np.where(mag > 1e-8, mag, 1.0)
    Fx_n = Fx / norm
    Fy_n = Fy / norm

    q = ax.quiver(Xq, Yq, Fx_n, Fy_n, mag,
                  cmap='inferno', scale=30, width=0.005,
                  headwidth=4, headlength=5, clim=(0, vmax_mag))
    plt.colorbar(q, ax=ax, label=r'$|\mathbf{F}|$', shrink=0.8)

    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig("force_field_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: force_field_comparison.png")

# --- Plot 2: Same scale for both (shared colorbar) to show magnitude gap ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, Fx, Fy, mag, Z_bg, title in [
    (axes[0], Fx_true, Fy_true, mag_true, Z_true,
     r'True Force $\mathbf{F} = -\nabla\Phi^*$'),
    (axes[1], Fx_learned, Fy_learned, mag_learned, Z_learned,
     r'Learned Force $\mathbf{F} = -\nabla\Phi_{NN}$'),
]:
    ax.contourf(Xbg, Ybg, Z_bg, levels=pot_levels, cmap='viridis', alpha=0.3)
    ax.contour(Xbg, Ybg, Z_bg, levels=pot_levels, colors='gray',
               linewidths=0.3, alpha=0.4)

    # Arrows scaled proportionally (same scale for both panels)
    q = ax.quiver(Xq, Yq, Fx, Fy, mag,
                  cmap='inferno', scale=mag_max * 6, width=0.005,
                  headwidth=4, headlength=5, clim=(0, mag_max))

    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
fig.colorbar(q, cax=cbar_ax, label=r'$|\mathbf{F}|$')
plt.savefig("force_field_shared_scale.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: force_field_shared_scale.png")

print(f"\nForce magnitude stats:")
print(f"  True:    max={mag_true.max():.2f}, mean={mag_true.mean():.2f}")
print(f"  Learned: max={mag_learned.max():.2f}, mean={mag_learned.mean():.2f}")
print(f"  Ratio (true/learned): {mag_true.mean() / (mag_learned.mean() + 1e-8):.1f}x")
