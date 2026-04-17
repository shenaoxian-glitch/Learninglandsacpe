#!/usr/bin/env python
"""
train_asym_potential_multi.py — learn asymmetric potential with fixed 2D death rate

Model 3.1: Based on model 03 (parametric).
  - Potential: H = -(y-a)x² + exp(b)x⁴ - cy + exp(d)y⁴ + e·x
    → 5 learnable parameters (a, b, c, d, e). The e·x term breaks x-symmetry.
  - Death rate: FIXED 2D function (not learned)
    γ(x,y) = γ_select(x,y) + γ_term(y)
    γ_select = A·exp(-x²/(2·w_x²))·softplus(k1·(y - y_select))  [death corridor]
    γ_term   = B·softplus(k2·(y - y_max))                         [terminal wall]
  - Source: FIXED oval Gaussian with separate σ_x, σ_y

Training uses a single steady-state snapshot at t=8 (no multi-timepoint).
Multi-timepoint objective is kept commented out for future use.

Usage:
    python train_asym_potential_multi.py > log.txt 2>&1
"""
import matplotlib
matplotlib.use('Agg')

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np

from training.data_loader import build_particle_queue, death_rate_2d


# ==========================================================================
# 1. Ground-truth parameters
# ==========================================================================

TARGET_PARAMS = {
    'a': -1.0, 'b': -1.6, 'c': 3.5, 'd': -1.2,
    'e': 0.7,   # asymmetry: linear tilt in x
    'sigma': 1.5,
}

# Fixed 2D death rate parameters
# NOTE on terminal wall stability: B=5, k2=3, y_max=3.5 gives steep onset.
# With dt=0.01 and σ=1.5, per-step noise in y is ~0.15, so particles rarely
# penetrate deep into the wall zone. A soft clamp (GAMMA_MAX) is applied as
# a safety net. If boundary artifacts appear, consider lowering k2 or
# switching from Euler-Maruyama to Heun integration.
DEATH_PARAMS = {
    'A': 2.0,          # selection death amplitude
    'w_x': 0.5,        # x-corridor width (narrow → only x≈0 cells die)
    'k1': 1.0,         # y-onset steepness for selection
    'y_select': 1.0,   # selection begins at this y
    'B': 5.0,          # terminal death amplitude
    'k2': 3.0,         # terminal wall steepness
    'y_max': 3.0,      # hard upper boundary
}
GAMMA_MAX = 50.0  # numerical safety clamp: caps dS per step to -0.5

N_PARTICLES = 2000
DT = 0.01
T_FINAL = 10.0
N_STEPS = int(T_FINAL / DT)  # 1000
SIGMA = TARGET_PARAMS['sigma']

# Oval Gaussian source (wider in x than y)
SOURCES = [
    {'mu': jnp.array([0.0, -1.0]), 'sigma_x': 0.25, 'sigma_y': 0.10,
     'n_particles': N_PARTICLES},
]

LAM_MASS = 0.0

# --- Multi-snapshot config (kept for future use, not active) ---
# SNAPSHOT_TIMES = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
# SNAPSHOT_STEPS = [int(t / DT) - 1 for t in SNAPSHOT_TIMES]


# ==========================================================================
# 2. Asymmetric potential and forces
# ==========================================================================

def potential(x, y, p):
    """H = -(y - a)x² + exp(b)x⁴ - cy + exp(d)y⁴ + e·x"""
    return (-(y - p['a']) * x**2
            + jnp.exp(p['b']) * x**4
            - p['c'] * y
            + jnp.exp(p['d']) * y**4
            + p['e'] * x)


_force_fn = jax.vmap(jax.grad(potential, argnums=(0, 1)), in_axes=(0, 0, None))


# ==========================================================================
# 3. Simulation with full trajectory output
# ==========================================================================

def simulate_open_full(learnable, death_params, z_init, wake_schedule,
                       n_steps, dt, key):
    """
    Euler-Maruyama for open system with fixed 2D death rate.

    Only potential params (a,b,c,d,e) are in `learnable`.
    Death rate params are fixed constants passed via `death_params`.

    Returns:
        all_px: (n_steps, N) x-positions at each step
        all_py: (n_steps, N) y-positions at each step
        all_S:  (n_steps, N) log-weights at each step
    """
    pot_params = {k: learnable[k] for k in ['a', 'b', 'c', 'd', 'e']}
    N = z_init.shape[0]
    S0 = jnp.full(N, -100.0)
    step_keys = jax.random.split(key, n_steps)

    def scan_body(carry, inputs):
        px, py, S = carry
        step_key, step_idx = inputs

        # Wake up particles at their scheduled birth step
        is_waking = (wake_schedule == step_idx)
        S = jnp.where(is_waking, 0.0, S)
        px = jnp.where(is_waking, z_init[:, 0], px)
        py = jnp.where(is_waking, z_init[:, 1], py)

        # Drift: F = -∇H
        gx, gy = _force_fn(px, py, pot_params)
        fx, fy = -gx, -gy

        # Diffusion
        kx, ky = jax.random.split(step_key)
        nx = jax.random.normal(kx, px.shape) * jnp.sqrt(dt)
        ny = jax.random.normal(ky, py.shape) * jnp.sqrt(dt)
        new_px = px + fx * dt + SIGMA * nx
        new_py = py + fy * dt + SIGMA * ny

        # Death: fixed 2D rate (not learned), clamped for numerical safety
        gamma = death_rate_2d(px, py, death_params)
        gamma = jnp.minimum(gamma, GAMMA_MAX)
        new_S = S - gamma * dt

        return (new_px, new_py, new_S), (new_px, new_py, new_S)

    step_indices = jnp.arange(n_steps)
    _, (all_px, all_py, all_S) = jax.lax.scan(
        scan_body, (z_init[:, 0], z_init[:, 1], S0), (step_keys, step_indices)
    )
    return all_px, all_py, all_S


# ==========================================================================
# 4. Loss functions
# ==========================================================================

def weighted_mmd_both(x_sim, alpha_sim, x_obs, alpha_obs,
                      bandwidths=(0.01, 1.0, 100.0)):
    """MMD² between two weighted point clouds."""
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


def mass_loss(S, target_mass):
    """(M_sim - M_target)² / M_target²"""
    M_sim = jnp.sum(jnp.exp(S))
    return (M_sim - target_mass)**2 / (target_mass**2 + 1e-8)


# ==========================================================================
# 5. Single-snapshot objective (t=T_FINAL)
# ==========================================================================

def objective(learnable, death_params, z_init, wake_schedule,
              tgt_px, tgt_py, tgt_S, tgt_mass, key):
    """
    MMD + mass loss at the single steady-state snapshot (t=8).
    Only potential params (a,b,c,d,e) are optimized; death rate is fixed.
    """
    all_px, all_py, all_S = simulate_open_full(
        learnable, death_params, z_init, wake_schedule, N_STEPS, DT, key
    )

    # Final time step
    step = N_STEPS - 1
    sim_z = jnp.column_stack([all_px[step], all_py[step]])
    sim_S = all_S[step]
    sim_alpha = jax.nn.softmax(sim_S)

    tgt_z = jnp.column_stack([tgt_px, tgt_py])
    tgt_alpha = jax.nn.softmax(tgt_S)

    l_mmd = weighted_mmd_both(sim_z, sim_alpha, tgt_z, tgt_alpha)
    l_mass = mass_loss(sim_S, tgt_mass)

    total = l_mmd + LAM_MASS * l_mass
    return total, (l_mmd, l_mass)


# --- Multi-snapshot objective (kept for future use) ---
# def objective_multi(learnable, death_params, z_init, wake_schedule,
#                     tgt_snap_px, tgt_snap_py, tgt_snap_S, tgt_snap_mass, key):
#     """
#     Sum of (MMD + mass) losses at each snapshot time, averaged.
#     Activate by uncommenting SNAPSHOT_TIMES/SNAPSHOT_STEPS above.
#     """
#     all_px, all_py, all_S = simulate_open_full(
#         learnable, death_params, z_init, wake_schedule, N_STEPS, DT, key
#     )
#     total_mmd = 0.0
#     total_mass_l = 0.0
#     for i, step in enumerate(SNAPSHOT_STEPS):
#         sim_z = jnp.column_stack([all_px[step], all_py[step]])
#         sim_S = all_S[step]
#         sim_alpha = jax.nn.softmax(sim_S)
#         tgt_z = jnp.column_stack([tgt_snap_px[i], tgt_snap_py[i]])
#         tgt_alpha = jax.nn.softmax(tgt_snap_S[i])
#         total_mmd += weighted_mmd_both(sim_z, sim_alpha, tgt_z, tgt_alpha)
#         total_mass_l += mass_loss(sim_S, tgt_snap_mass[i])
#     n_snaps = len(SNAPSHOT_STEPS)
#     avg_mmd = total_mmd / n_snaps
#     avg_mass = total_mass_l / n_snaps
#     total = avg_mmd + LAM_MASS * avg_mass
#     return total, (avg_mmd, avg_mass)


loss_and_grad_fn = jax.jit(jax.value_and_grad(objective, argnums=0, has_aux=True))


# ==========================================================================
# 6. Generate ground-truth snapshot at t=8
# ==========================================================================

print("=" * 65)
print("Model 3.1: Asymmetric potential inference (fixed 2D death rate)")
print("=" * 65)
print(f"  N_particles = {N_PARTICLES}")
print(f"  T_final = {T_FINAL}, dt = {DT}, n_steps = {N_STEPS}")
print(f"  sigma = {SIGMA} (fixed)")
print(f"  Training snapshot: t = {T_FINAL} (single steady-state)")
print(f"  Loss = L_MMD + {LAM_MASS}*L_mass")
print(f"\n  Potential: H = -(y-a)x² + exp(b)x⁴ - cy + exp(d)y⁴ + e·x")
print(f"  True params: a={TARGET_PARAMS['a']}, b={TARGET_PARAMS['b']}, "
      f"c={TARGET_PARAMS['c']}, d={TARGET_PARAMS['d']}, e={TARGET_PARAMS['e']}")
print(f"\n  Death rate (FIXED): γ = γ_select(x,y) + γ_term(y)")
print(f"    γ_select: A={DEATH_PARAMS['A']}, w_x={DEATH_PARAMS['w_x']}, "
      f"k1={DEATH_PARAMS['k1']}, y_select={DEATH_PARAMS['y_select']}")
print(f"    γ_term:   B={DEATH_PARAMS['B']}, k2={DEATH_PARAMS['k2']}, "
      f"y_max={DEATH_PARAMS['y_max']}")
print(f"    GAMMA_MAX clamp = {GAMMA_MAX}")
print(f"\n  Source (FIXED oval): mu=(0,-1), sigma_x=0.25, sigma_y=0.10")

key = jax.random.PRNGKey(4)
key, tgt_queue_key, tgt_sim_key = jax.random.split(key, 3)

print("\nGenerating ground-truth snapshot ...")

# Target particle queue (oval Gaussian)
tgt_z_init, tgt_wake = build_particle_queue(
    sources=SOURCES, n_particles=N_PARTICLES,
    t_final=T_FINAL, dt=DT, key=tgt_queue_key,
)

# Run true simulation with fixed death rate (full trajectory for viz)
true_learnable = {
    'a': jnp.array(float(TARGET_PARAMS['a'])),
    'b': jnp.array(float(TARGET_PARAMS['b'])),
    'c': jnp.array(float(TARGET_PARAMS['c'])),
    'd': jnp.array(float(TARGET_PARAMS['d'])),
    'e': jnp.array(float(TARGET_PARAMS['e'])),
}

tgt_all_px, tgt_all_py, tgt_all_S = simulate_open_full(
    true_learnable, DEATH_PARAMS, tgt_z_init, tgt_wake, N_STEPS, DT, tgt_sim_key
)

# Extract single snapshot at t=T_FINAL
final_step = N_STEPS - 1
tgt_px = tgt_all_px[final_step]
tgt_py = tgt_all_py[final_step]
tgt_S = tgt_all_S[final_step]
tgt_mass = float(jnp.sum(jnp.exp(tgt_S)))
n_alive = int(jnp.sum(jnp.exp(tgt_S) > 0.01))
print(f"  t={T_FINAL:.1f}: mass={tgt_mass:.2f}, ~{n_alive} alive particles")


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
# 8. Initialize learnable parameters (potential only) and optimizer
# ==========================================================================

learnable = {
    'a': jnp.array(0.0),       # true: -1.0
    'b': jnp.array(-1.0),      # true: -1.6
    'c': jnp.array(2.0),       # true: 3.5
    'd': jnp.array(-0.8),      # true: -1.2
    'e': jnp.array(0.0),       # true: 0.7 (start symmetric)
}

print("Initial vs True parameters (potential only):")
print(f"  a:  {float(learnable['a']):+.3f}   (true: {TARGET_PARAMS['a']})")
print(f"  b:  {float(learnable['b']):+.3f}   (true: {TARGET_PARAMS['b']})")
print(f"  c:  {float(learnable['c']):+.3f}   (true: {TARGET_PARAMS['c']})")
print(f"  d:  {float(learnable['d']):+.3f}   (true: {TARGET_PARAMS['d']})")
print(f"  e:  {float(learnable['e']):+.3f}   (true: {TARGET_PARAMS['e']})")

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
    'a': [], 'b': [], 'c': [], 'd': [], 'e': [],
}}
best_loss = float('inf')
best_params = None
best_epoch = 0
opt_key = jax.random.PRNGKey(42)

print(f"\nTraining for {N_EPOCHS} epochs (single snapshot at t={T_FINAL}) ...")
print("-" * 65)

for epoch in range(N_EPOCHS):
    if USE_FRESH_KEYS:
        opt_key, sim_key = jax.random.split(opt_key)
    else:
        sim_key = opt_key

    (loss, (l_mmd, l_mass)), grads = loss_and_grad_fn(
        learnable, DEATH_PARAMS, z_init, wake_schedule,
        tgt_px, tgt_py, tgt_S, tgt_mass,
        sim_key,
    )

    updates, opt_state = optimizer.update(grads, opt_state)
    learnable = optax.apply_updates(learnable, updates)

    # Physical constraint: c > 0 (ensures downward y-slope)
    learnable['c'] = jnp.maximum(learnable['c'], 0.001)

    loss_val = float(loss)
    history['loss'].append(loss_val)
    history['mmd'].append(float(l_mmd))
    history['mass'].append(float(l_mass))
    for p in history['params']:
        history['params'][p].append(float(learnable[p]))

    if loss_val < best_loss - 1e-6:
        best_loss = loss_val
        best_params = {k: v for k, v in learnable.items()}
        best_epoch = epoch

    if epoch % PRINT_EVERY == 0:
        print(
            f"Epoch {epoch:4d}: Loss={loss_val:.5f} "
            f"(MMD={float(l_mmd):.5f}, Mass={float(l_mass):.5f})  "
            f"a={float(learnable['a']):+.3f} b={float(learnable['b']):+.3f} "
            f"c={float(learnable['c']):+.3f} d={float(learnable['d']):+.3f} "
            f"e={float(learnable['e']):+.3f}"
        )


# Restore best
if best_params is not None:
    learnable = best_params
    print(f"\nRestored best params from epoch {best_epoch} (loss={best_loss:.6f})")


# ==========================================================================
# 10. Results
# ==========================================================================

print("\n" + "=" * 65)
print("Final Results — Asymmetric Potential (fixed 2D death rate)")
print("=" * 65)
print(f"  {'Param':<8} {'Learned':>10} {'True':>10} {'Error':>10}")
print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
for name, true_val in [('a', -1.0), ('b', -1.6), ('c', 3.5), ('d', -1.2), ('e', 0.7)]:
    learned = float(learnable[name])
    print(f"  {name:<8} {learned:>+10.4f} {true_val:>+10.4f} {learned - true_val:>+10.4f}")


# ==========================================================================
# 11. Visualisation
# ==========================================================================

# --- Figure 1: Training curves + parameter convergence ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
epochs_arr = np.arange(len(history['loss']))
ax.semilogy(epochs_arr, history['loss'], label='Total', linewidth=1.5)
ax.semilogy(epochs_arr, history['mmd'], label='MMD', linewidth=1, alpha=0.8)
ax.semilogy(epochs_arr, history['mass'], label='Mass', linewidth=1, alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title(f'Training Loss (single snapshot t={T_FINAL:.0f})')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
true_vals = {'a': -1.0, 'b': -1.6, 'c': 3.5, 'd': -1.2, 'e': 0.7}
for name in ['a', 'b', 'c', 'd', 'e']:
    line, = ax.plot(epochs_arr, history['params'][name], label=name, linewidth=1.2)
    ax.axhline(true_vals[name], color=line.get_color(), linestyle='--', alpha=0.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Value')
ax.set_title('Parameter Convergence')
ax.legend(ncol=2, fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_asym.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved training_asym.png")


# --- Figure 2: Potential comparison (true vs learned) ---
x_grid = jnp.linspace(-5, 5, 200)
y_grid = jnp.linspace(-2, 4, 200)
Xg, Yg = jnp.meshgrid(x_grid, y_grid)

Z_true = potential(Xg, Yg, TARGET_PARAMS)
learned_pot_params = {k: learnable[k] for k in ['a', 'b', 'c', 'd', 'e']}
Z_learned = potential(Xg, Yg, learned_pot_params)

vmin = min(float(Z_true.min()), float(Z_learned.min()))
vmax = min(float(Z_true.max()), float(Z_learned.max()))
vmax = min(vmax, 50.0)
levels = np.linspace(vmin, vmax, 31)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, Z, title in [(axes[0], Z_true, 'True Potential'),
                      (axes[1], Z_learned, 'Learned Potential')]:
    cf = ax.contourf(np.array(Xg), np.array(Yg), np.array(Z),
                     levels=levels, cmap='viridis', extend='both')
    plt.colorbar(cf, ax=ax, label=r'$\Phi(x,y)$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)

# Difference plot
Z_diff = Z_learned - Z_true
diff_max = max(abs(float(Z_diff.min())), abs(float(Z_diff.max())))
diff_max = min(diff_max, 20.0)
cf = axes[2].contourf(np.array(Xg), np.array(Yg), np.array(Z_diff),
                       levels=np.linspace(-diff_max, diff_max, 31),
                       cmap='RdBu_r', extend='both')
plt.colorbar(cf, ax=axes[2], label=r'$\Phi_{learned} - \Phi_{true}$')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
axes[2].set_title('Potential Error')

plt.tight_layout()
plt.savefig("potential_asym.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved potential_asym.png")


# --- Figure 3: Fixed 2D death rate landscape ---
gamma_2d = death_rate_2d(Xg, Yg, DEATH_PARAMS)
g_vmax = min(float(gamma_2d.max()), 10.0)
g_levels = np.linspace(0.0, g_vmax, 31)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cf0 = axes[0].contourf(np.array(Xg), np.array(Yg), np.array(gamma_2d),
                        levels=g_levels, cmap='YlOrRd', extend='both')
plt.colorbar(cf0, ax=axes[0], label=r'$\gamma(x,y)$')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title(r'Fixed 2D Death Rate $\gamma(x,y) = \gamma_{select} + \gamma_{term}$')

# 1D slices at x=0 and x=±1.5
y_line = np.linspace(-2, 4, 200)
for x_val, style, label in [(0.0, '-', 'x=0 (corridor center)'),
                              (1.5, '--', 'x=1.5 (escaped)'),
                              (-1.5, ':', 'x=-1.5 (escaped)')]:
    x_arr = jnp.full_like(jnp.array(y_line), x_val)
    gamma_slice = death_rate_2d(x_arr, jnp.array(y_line), DEATH_PARAMS)
    axes[1].plot(y_line, np.array(gamma_slice), style, linewidth=2, label=label)

axes[1].set_xlabel('y')
axes[1].set_ylabel(r'$\gamma$')
axes[1].set_title('Death Rate Profiles (1D slices)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("death_rate_asym.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved death_rate_asym.png")


# --- Figure 4: Snapshot particle comparison (from full trajectory) ---
VIS_TIMES = [1.0, 3.0, 6.0, 10.0]
VIS_STEPS = [int(t / DT) - 1 for t in VIS_TIMES]

# Run learned simulation for visualization
key, eval_key, eval_queue_key = jax.random.split(key, 3)
z_init_eval, wake_eval = build_particle_queue(
    sources=SOURCES, n_particles=N_PARTICLES,
    t_final=T_FINAL, dt=DT, key=eval_queue_key,
)

sim_all_px, sim_all_py, sim_all_S = simulate_open_full(
    learnable, DEATH_PARAMS, z_init_eval, wake_eval, N_STEPS, DT, eval_key
)

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

    # Bottom row: simulated (learned)
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

# Colorbar
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
fig.colorbar(sc, cax=cbar_ax, label=r'$w = e^S$')

plt.savefig("snapshots_asym.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved snapshots_asym.png")


# --- Figure 5: Alive particle mass vs time ---
time_axis = np.arange(1, N_STEPS + 1) * DT  # t after each step

tgt_mass_t = np.array([float(jnp.sum(jnp.exp(tgt_all_S[i]))) for i in range(N_STEPS)])
sim_mass_t = np.array([float(jnp.sum(jnp.exp(sim_all_S[i]))) for i in range(N_STEPS)])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(time_axis, tgt_mass_t, 'b-', linewidth=1.5, label='Target (true)', alpha=0.8)
ax.plot(time_axis, sim_mass_t, 'r--', linewidth=1.5, label='Learned', alpha=0.8)
ax.set_xlabel('Time')
ax.set_ylabel('Total mass  $\\sum e^{S_i}$')
ax.set_title('Alive Particle Mass vs Time')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("mass_vs_time.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved mass_vs_time.png")

print("\nDone. All figures saved.")
