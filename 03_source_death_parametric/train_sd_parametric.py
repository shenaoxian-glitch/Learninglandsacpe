#!/usr/bin/env python
"""
train_sd_parametric.py — parametric inference for open-system source+death model

Diagnostic script: uses the EXACT analytical potential form and parametric death
rate (no neural networks) to test whether the open-system framework can recover
ground-truth parameters.  This isolates framework/identifiability issues from
NN capacity issues.

Learnable parameters (6 total):
  Potential:   a, b, c, d   in  H = -(y-a)x^2 + exp(b)x^4 - cy + exp(d)y^4
  Death rate:  y_c, log_k   in  gamma(z) = softplus(exp(log_k) * (y - y_c))

Fixed:  sigma = 1.5 (diffusion coefficient, same as train_sd.py)

Key differences from train_sd.py (NN version):
  - Potential is the correct analytical form (symmetric in x by construction)
  - Only 6 scalar parameters instead of thousands of NN weights
  - If this fails to recover params, the issue is fundamental (identifiability)
  - If it succeeds, the issue is NN-specific (capacity, optimization landscape)

Usage:
    python train_sd_parametric.py
"""
import matplotlib
matplotlib.use('Agg')

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np

from training.data_loader import (
    build_particle_queue,
    generate_open_system_data,
    analytical_death_rate,
)


# ==========================================================================
# 1. Ground-truth parameters
# ==========================================================================

TARGET_PARAMS = {'a': 0, 'b': -1.6, 'c': 3.5, 'd': -1.2, 'sigma': 1.5}
TRUE_DEATH = {'y_c': 2.2, 'k': 1.0}

N_PARTICLES = 2000
DT = 0.01
T_FINAL = 8.0
N_STEPS = int(T_FINAL / DT)  # 800
SIGMA = TARGET_PARAMS['sigma']  # fixed, not learned

SOURCES = [
    {'mu': jnp.array([0.0, -1.0]), 'sigma_src': 0.15, 'n_particles': N_PARTICLES},
]

LAM_MASS = 1.0  # mass matching weight


# ==========================================================================
# 2. Analytical potential, forces, and death rate (JAX autodiff)
# ==========================================================================

def potential(x, y, p):
    """H = -(y - a)x^2 + exp(b)x^4 - cy + exp(d)y^4"""
    return -(y - p['a']) * x**2 + jnp.exp(p['b']) * x**4 - p['c'] * y + jnp.exp(p['d']) * y**4


# Force = -grad(H) w.r.t. (x, y), vectorized over particles
_force_fn = jax.vmap(jax.grad(potential, argnums=(0, 1)), in_axes=(0, 0, None))


# ==========================================================================
# 3. Open-system simulation (differentiable w.r.t. learnable params)
# ==========================================================================

def simulate_open(learnable, z_init, wake_schedule, n_steps, dt, key):
    """
    Euler-Maruyama for open system with particle wake-up schedule.
    Differentiable w.r.t. all entries in `learnable`.

    Args:
        learnable: dict with 'a','b','c','d' (potential) and 'y_c','log_k' (death)
        z_init: (N, 2) birth positions
        wake_schedule: (N,) int, birth step for each particle
        n_steps, dt: simulation config
        key: PRNG key

    Returns:
        final_z: (N, 2) final positions
        final_S: (N,) raw log-weights
    """
    pot_params = {k: learnable[k] for k in ['a', 'b', 'c', 'd']}

    N = z_init.shape[0]
    S0 = jnp.full(N, -100.0)  # all dormant
    step_keys = jax.random.split(key, n_steps)

    def scan_body(carry, inputs):
        px, py, S = carry
        step_key, step_idx = inputs

        # 1. Wake up particles at their scheduled birth step
        is_waking = (wake_schedule == step_idx)
        S = jnp.where(is_waking, 0.0, S)
        px = jnp.where(is_waking, z_init[:, 0], px)
        py = jnp.where(is_waking, z_init[:, 1], py)

        # 2. Drift: F = -grad(H)
        gx, gy = _force_fn(px, py, pot_params)
        fx, fy = -gx, -gy

        # 3. Diffusion (sigma fixed)
        kx, ky = jax.random.split(step_key)
        nx = jax.random.normal(kx, px.shape) * jnp.sqrt(dt)
        ny = jax.random.normal(ky, py.shape) * jnp.sqrt(dt)

        new_px = px + fx * dt + SIGMA * nx
        new_py = py + fy * dt + SIGMA * ny

        # 4. Death: dS = -gamma(y) dt,  gamma = softplus(k*(y - y_c))
        k = jnp.exp(learnable['log_k'])
        gamma = jax.nn.softplus(k * (py - learnable['y_c']))
        new_S = S - gamma * dt

        return (new_px, new_py, new_S), None

    step_indices = jnp.arange(n_steps)
    (final_px, final_py, final_S), _ = jax.lax.scan(
        scan_body, (z_init[:, 0], z_init[:, 1], S0), (step_keys, step_indices)
    )

    return jnp.column_stack([final_px, final_py]), final_S


# ==========================================================================
# 4. Loss functions
# ==========================================================================

def weighted_mmd(x_sim, alpha, x_obs, bandwidths=(0.01, 1.0, 100.0)):
    """Weighted MMD^2 between weighted sim and uniform obs."""
    def k_matrix(a, b):
        diff = a[:, None, :] - b[None, :, :]
        dist_sq = jnp.sum(diff**2, axis=-1)
        K = jnp.zeros_like(dist_sq)
        for bw in bandwidths:
            K = K + jnp.exp(-dist_sq / (2 * bw))
        return K

    M = x_obs.shape[0]
    K_xx = k_matrix(x_sim, x_sim)
    K_yy = k_matrix(x_obs, x_obs)
    K_xy = k_matrix(x_sim, x_obs)

    term_xx = jnp.sum(alpha[:, None] * alpha[None, :] * K_xx)
    term_yy = jnp.mean(K_yy)
    term_xy = jnp.sum(alpha[:, None] * K_xy) / M

    return term_xx + term_yy - 2 * term_xy


def mass_loss(final_S, target_mass):
    """(M_sim - M_target)^2 / M_target^2"""
    M_sim = jnp.sum(jnp.exp(final_S))
    return (M_sim - target_mass)**2 / (target_mass**2 + 1e-8)


# ==========================================================================
# 5. Objective function (differentiable w.r.t. learnable)
# ==========================================================================

def objective(learnable, z_init, wake_schedule, target_pos, target_mass, key):
    """L = L_MMD(weighted) + lam_mass * L_mass"""
    final_z, final_S = simulate_open(
        learnable, z_init, wake_schedule, N_STEPS, DT, key
    )

    # Normalize weights via logsumexp
    log_alpha = final_S - jax.nn.logsumexp(final_S)
    alpha = jnp.exp(log_alpha)

    l_mmd = weighted_mmd(final_z, alpha, target_pos)
    l_mass = mass_loss(final_S, target_mass)

    return l_mmd + LAM_MASS * l_mass, (l_mmd, l_mass)


loss_and_grad_fn = jax.jit(jax.value_and_grad(objective, argnums=0, has_aux=True))


# ==========================================================================
# 6. Generate ground-truth target
# ==========================================================================

print("=" * 65)
print("Parametric inference: open-system source + death model")
print("=" * 65)
print(f"  N_particles = {N_PARTICLES}")
print(f"  T_final = {T_FINAL}, dt = {DT}, n_steps = {N_STEPS}")
print(f"  sigma = {SIGMA} (fixed)")
print(f"  Loss = L_MMD + {LAM_MASS} * L_mass")

key = jax.random.PRNGKey(4)
key, target_key = jax.random.split(key)

print("\nGenerating ground-truth data ...")
target_pos, target_S, target_mass, _, _ = generate_open_system_data(
    params=TARGET_PARAMS,
    death_params={'y_threshold': TRUE_DEATH['y_c']},
    sources=SOURCES,
    n_particles=N_PARTICLES,
    t_final=T_FINAL,
    dt=DT,
    key=target_key,
)

target_w = jnp.clip(jnp.exp(target_S), 0.0, 1.0)
print(f"  Target: {target_pos.shape[0]} particles, total_mass = {target_mass:.2f}")
print(f"  w: mean={float(target_w.mean()):.4f}, "
      f"min={float(target_w.min()):.6f}, max={float(target_w.max()):.4f}")


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
# 8. Initialize learnable parameters and optimizer
# ==========================================================================

learnable = {
    'a': jnp.array(0.5),       # true: 0
    'b': jnp.array(-1.0),      # true: -1.6
    'c': jnp.array(2.0),       # true: 3.5
    'd': jnp.array(-0.8),      # true: -1.2
    'y_c': jnp.array(0.0),     # true: 2.2
    'log_k': jnp.array(0.5),   # true: 0.0 (k=1.0)
}

print("Initial vs True parameters:")
print(f"  a:     {float(learnable['a']):+.3f}   (true: {TARGET_PARAMS['a']})")
print(f"  b:     {float(learnable['b']):+.3f}   (true: {TARGET_PARAMS['b']})")
print(f"  c:     {float(learnable['c']):+.3f}   (true: {TARGET_PARAMS['c']})")
print(f"  d:     {float(learnable['d']):+.3f}   (true: {TARGET_PARAMS['d']})")
print(f"  y_c:   {float(learnable['y_c']):+.3f}   (true: {TRUE_DEATH['y_c']})")
print(f"  log_k: {float(learnable['log_k']):+.3f}   (true: 0.0, k=1.0)")

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=0.01),
)
opt_state = optimizer.init(learnable)


# ==========================================================================
# 9. Training loop
# ==========================================================================

N_EPOCHS = 500
PRINT_EVERY = 10
USE_FRESH_KEYS = True

history = {'loss': [], 'mmd': [], 'mass': [], 'params': {
    'a': [], 'b': [], 'c': [], 'd': [], 'y_c': [], 'log_k': [],
}}
best_loss = float('inf')
best_params = None
best_epoch = 0
opt_key = jax.random.PRNGKey(42)

print(f"\nTraining for {N_EPOCHS} epochs ...")
print("-" * 65)

for epoch in range(N_EPOCHS):
    if USE_FRESH_KEYS:
        opt_key, sim_key = jax.random.split(opt_key)
    else:
        sim_key = opt_key

    (loss, (l_mmd, l_mass)), grads = loss_and_grad_fn(
        learnable, z_init, wake_schedule, target_pos, target_mass, sim_key,
    )

    updates, opt_state = optimizer.update(grads, opt_state)
    learnable = optax.apply_updates(learnable, updates)

    # Physical constraint: c > 0 (drive strength must be positive)
    learnable['c'] = jnp.maximum(learnable['c'], 0.001)

    loss_val = float(loss)
    history['loss'].append(loss_val)
    history['mmd'].append(float(l_mmd))
    history['mass'].append(float(l_mass))
    for p in history['params']:
        history['params'][p].append(float(learnable[p]))

    # Checkpoint best
    if loss_val < best_loss - 1e-6:
        best_loss = loss_val
        best_params = {k: v for k, v in learnable.items()}
        best_epoch = epoch

    if epoch % PRINT_EVERY == 0:
        k_val = float(jnp.exp(learnable['log_k']))
        print(
            f"Epoch {epoch:4d}: Loss={loss_val:.5f} "
            f"(MMD={float(l_mmd):.5f}, Mass={float(l_mass):.5f})  "
            f"a={float(learnable['a']):+.3f} b={float(learnable['b']):+.3f} "
            f"c={float(learnable['c']):+.3f} d={float(learnable['d']):+.3f} "
            f"y_c={float(learnable['y_c']):+.3f} k={k_val:.3f}"
        )


# Restore best
if best_params is not None:
    learnable = best_params
    print(f"\nRestored best params from epoch {best_epoch} (loss={best_loss:.6f})")


# ==========================================================================
# 10. Results
# ==========================================================================

k_learned = float(jnp.exp(learnable['log_k']))

print("\n" + "=" * 65)
print("Final Results")
print("=" * 65)
print(f"  {'Param':<8} {'Learned':>10} {'True':>10} {'Error':>10}")
print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
for name, true_val in [('a', 0), ('b', -1.6), ('c', 3.5), ('d', -1.2)]:
    learned = float(learnable[name])
    print(f"  {name:<8} {learned:>+10.4f} {true_val:>+10.4f} {learned - true_val:>+10.4f}")
print(f"  {'y_c':<8} {float(learnable['y_c']):>+10.4f} {TRUE_DEATH['y_c']:>+10.4f} "
      f"{float(learnable['y_c']) - TRUE_DEATH['y_c']:>+10.4f}")
print(f"  {'k':<8} {k_learned:>10.4f} {TRUE_DEATH['k']:>10.4f} "
      f"{k_learned - TRUE_DEATH['k']:>+10.4f}")


# ==========================================================================
# 11. Visualisation
# ==========================================================================

# --- Figure 1: Training curves + parameter convergence ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
epochs = np.arange(len(history['loss']))
ax.semilogy(epochs, history['loss'], label='Total', linewidth=1.5)
ax.semilogy(epochs, history['mmd'], label='MMD', linewidth=1, alpha=0.8)
ax.semilogy(epochs, history['mass'], label='Mass', linewidth=1, alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
true_vals = {'a': 0, 'b': -1.6, 'c': 3.5, 'd': -1.2, 'y_c': 2.2, 'log_k': 0.0}
for name in ['a', 'b', 'c', 'd', 'y_c', 'log_k']:
    ax.plot(epochs, history['params'][name], label=name, linewidth=1.2)
    ax.axhline(true_vals[name], color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Value')
ax.set_title('Parameter Convergence')
ax.legend(ncol=2, fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_parametric.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved training_parametric.png")


# --- Figure 2: Potential comparison (true vs learned) ---
x_grid = jnp.linspace(-5, 5, 200)
y_grid = jnp.linspace(-2, 4, 200)
Xg, Yg = jnp.meshgrid(x_grid, y_grid)

Z_true = potential(Xg, Yg, TARGET_PARAMS)
learned_pot_params = {k: learnable[k] for k in ['a', 'b', 'c', 'd']}
Z_learned = potential(Xg, Yg, learned_pot_params)

# Shared colorscale
vmin = min(float(Z_true.min()), float(Z_learned.min()))
vmax = min(float(Z_true.max()), float(Z_learned.max()))
vmax = min(vmax, 50.0)  # cap for visibility
levels = np.linspace(vmin, vmax, 31)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, Z, title in [(axes[0], Z_true, 'True Potential'),
                      (axes[1], Z_learned, 'Learned Potential')]:
    cf = ax.contourf(np.array(Xg), np.array(Yg), np.array(Z),
                     levels=levels, cmap='viridis', extend='both')
    plt.colorbar(cf, ax=ax, label=r'$\Phi(x,y)$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)

plt.tight_layout()
plt.savefig("potential_parametric.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved potential_parametric.png")


# --- Figure 3: Death rate comparison + particle comparison ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Death rate: 2D contour (true)
gamma_true = analytical_death_rate(Xg, Yg, {'y_threshold': TRUE_DEATH['y_c']})
k_l = float(jnp.exp(learnable['log_k']))
gamma_learned = jax.nn.softplus(k_l * (Yg - float(learnable['y_c'])))

g_vmin = 0.0
g_vmax = max(float(gamma_true.max()), float(gamma_learned.max()))
g_vmax = min(g_vmax, 5.0)
g_levels = np.linspace(g_vmin, g_vmax, 31)

cf0 = axes[0].contourf(np.array(Xg), np.array(Yg), np.array(gamma_true),
                        levels=g_levels, cmap='YlOrRd', extend='both')
plt.colorbar(cf0, ax=axes[0], label=r'$\gamma^*(x,y)$')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title(r'True $\gamma^*$')

cf1 = axes[1].contourf(np.array(Xg), np.array(Yg), np.array(gamma_learned),
                        levels=g_levels, cmap='YlOrRd', extend='both')
plt.colorbar(cf1, ax=axes[1], label=r'$\gamma_\theta(x,y)$')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title(r'Learned $\gamma_\theta$')

# Death rate: 1D y-profile at x=0
y_line = np.linspace(-2, 4, 200)
gamma_1d_true = np.array(jax.nn.softplus(jnp.array(y_line) - TRUE_DEATH['y_c']))
gamma_1d_learned = np.array(jax.nn.softplus(
    k_l * (jnp.array(y_line) - float(learnable['y_c']))
))
axes[2].plot(y_line, gamma_1d_true, 'b-', linewidth=2, label='True')
axes[2].plot(y_line, gamma_1d_learned, 'r--', linewidth=2, label='Learned')
axes[2].set_xlabel('y')
axes[2].set_ylabel(r'$\gamma(y)$')
axes[2].set_title(r'Death rate profile (x=0)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("death_rate_parametric.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved death_rate_parametric.png")


# --- Figure 4: Particle distribution comparison ---
# Run forward simulation with learned params
key, eval_key, eval_queue_key = jax.random.split(key, 3)
z_init_eval, wake_eval = build_particle_queue(
    sources=SOURCES, n_particles=N_PARTICLES,
    t_final=T_FINAL, dt=DT, key=eval_queue_key,
)

sim_z, sim_S = simulate_open(
    learnable, z_init_eval, wake_eval, N_STEPS, DT, eval_key
)
sim_w = jnp.clip(jnp.exp(sim_S), 0.0, 1.0)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, pos, w, title, Z_bg in [
    (axes[0], target_pos, target_w, 'Target (ground truth)', Z_true),
    (axes[1], np.array(sim_z), np.array(sim_w), 'Simulated (learned params)', Z_learned),
]:
    # Background: potential contour (true for target, learned for sim)
    ax.contourf(np.array(Xg), np.array(Yg), np.array(Z_bg),
                levels=30, cmap='viridis', alpha=0.3)
    sc = ax.scatter(np.array(pos[:, 0]), np.array(pos[:, 1]),
                    c=np.array(w), cmap='coolwarm', s=8, alpha=0.6,
                    vmin=0, vmax=1, edgecolors='none')
    plt.colorbar(sc, ax=ax, label=r'$w = e^S$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-2, 4)

plt.tight_layout()
plt.savefig("particles_parametric.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved particles_parametric.png")

print("\nDone. All figures saved.")
