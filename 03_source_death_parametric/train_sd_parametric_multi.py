#!/usr/bin/env python
"""
train_sd_parametric_multi.py — multi-timepoint parametric inference

Improvement over train_sd_parametric.py (single steady-state matching):
matches particle distributions at multiple intermediate snapshot times
during the transient phase to break parameter coupling degeneracy.

Key insight: parameters that produce identical steady states have different
transient dynamics. "Fast slope + fast death" vs "slow slope + slow death"
push particles at different speeds, so their wavefront positions at t=1,2,3
differ even if their t=8 distributions converge. Matching these intermediate
snapshots provides enough constraints to uniquely identify the parameters.

Snapshot times: t = 1.0, 2.0, 3.0, 4.0, 6.0, 8.0
  - Early (t=1-3): captures wavefront propagation, most informative
  - Late (t=6-8): ensures steady-state also matches

Regional mass matching: additionally constrains death rate spatial location
by matching mass in y-axis sub-regions (y<1.5, 1.5≤y<2.5, y≥2.5).
This breaks the y_c/k trade-off by penalizing death rates that kill
particles at the wrong y-coordinate.

Usage:
    python train_sd_parametric_multi.py
"""
import matplotlib
matplotlib.use('Agg')

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np

from training.data_loader import build_particle_queue, analytical_death_rate


# ==========================================================================
# 1. Ground-truth parameters (identical to train_sd_parametric.py)
# ==========================================================================

TARGET_PARAMS = {'a': 0, 'b': -1.6, 'c': 3.5, 'd': -1.2, 'sigma': 1.5}
TRUE_DEATH = {'y_c': 2.2, 'k': 1.0}

N_PARTICLES = 2000
DT = 0.01
T_FINAL = 8.0
N_STEPS = int(T_FINAL / DT)  # 800
SIGMA = TARGET_PARAMS['sigma']

SOURCES = [
    {'mu': jnp.array([0.0, -1.0]), 'sigma_src': 0.15, 'n_particles': N_PARTICLES},
]

LAM_MASS = 1.0
LAM_REGIONAL = 0.01

# Regional mass matching: 5 narrow regions around the death transition zone
REGION_BOUNDS = [1.0, 1.5, 2.0, 2.5]  # 5 regions: y<1|1-1.5|1.5-2|2-2.5|y≥2.5
REGIONAL_EPS = 1.0

# Snapshot configuration: emphasize early transient
SNAPSHOT_TIMES = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
SNAPSHOT_STEPS = [int(t / DT) - 1 for t in SNAPSHOT_TIMES]  # 0-indexed


# ==========================================================================
# 2. Analytical potential, forces, and death rate
# ==========================================================================

def potential(x, y, p):
    """H = -(y - a)x^2 + exp(b)x^4 - cy + exp(d)y^4"""
    return -(y - p['a']) * x**2 + jnp.exp(p['b']) * x**4 - p['c'] * y + jnp.exp(p['d']) * y**4


_force_fn = jax.vmap(jax.grad(potential, argnums=(0, 1)), in_axes=(0, 0, None))


# ==========================================================================
# 3. Simulation with full trajectory output (for snapshot extraction)
# ==========================================================================

def simulate_open_full(learnable, z_init, wake_schedule, n_steps, dt, key):
    """
    Euler-Maruyama for open system, returning state at EVERY step.

    Returns:
        all_px: (n_steps, N) x-positions at each step
        all_py: (n_steps, N) y-positions at each step
        all_S:  (n_steps, N) log-weights at each step
    """
    pot_params = {k: learnable[k] for k in ['a', 'b', 'c', 'd']}
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

        # Drift: F = -grad(H)
        gx, gy = _force_fn(px, py, pot_params)
        fx, fy = -gx, -gy

        # Diffusion
        kx, ky = jax.random.split(step_key)
        nx = jax.random.normal(kx, px.shape) * jnp.sqrt(dt)
        ny = jax.random.normal(ky, py.shape) * jnp.sqrt(dt)
        new_px = px + fx * dt + SIGMA * nx
        new_py = py + fy * dt + SIGMA * ny

        # Death: dS = -gamma(y) dt
        k = jnp.exp(learnable['log_k'])
        gamma = jax.nn.softplus(k * (py - learnable['y_c']))
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
    """
    MMD^2 between two weighted point clouds.

    Both alpha_sim and alpha_obs should be normalized (sum to 1).
    Dormant particles (S=-100) get alpha ~0 via softmax, so they
    naturally don't contribute.
    """
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
    """(M_sim - M_target)^2 / M_target^2"""
    M_sim = jnp.sum(jnp.exp(S))
    return (M_sim - target_mass)**2 / (target_mass**2 + 1e-8)


def regional_mass_loss(sim_py, sim_S, tgt_py, tgt_S, bounds=REGION_BOUNDS,
                       eps=REGIONAL_EPS):
    """
    Log-mass MSE across y-axis regions.

    Divides space into len(bounds)+1 regions along y-axis.
    For each region, computes (log(M_sim + eps) - log(M_tgt + eps))^2.
    Returns average over all regions.
    """
    loss = 0.0
    n_regions = len(bounds) + 1

    for j in range(n_regions):
        if j == 0:
            sim_mask = sim_py < bounds[0]
            tgt_mask = tgt_py < bounds[0]
        elif j == n_regions - 1:
            sim_mask = sim_py >= bounds[-1]
            tgt_mask = tgt_py >= bounds[-1]
        else:
            sim_mask = (sim_py >= bounds[j-1]) & (sim_py < bounds[j])
            tgt_mask = (tgt_py >= bounds[j-1]) & (tgt_py < bounds[j])

        M_sim = jnp.sum(jnp.where(sim_mask, jnp.exp(sim_S), 0.0))
        M_tgt = jnp.sum(jnp.where(tgt_mask, jnp.exp(tgt_S), 0.0))

        loss += (jnp.log(M_sim + eps) - jnp.log(M_tgt + eps))**2

    return loss / n_regions


# ==========================================================================
# 5. Multi-snapshot objective
# ==========================================================================

def objective(learnable, z_init, wake_schedule,
              tgt_snap_px, tgt_snap_py, tgt_snap_S, tgt_snap_mass, key):
    """
    Sum of (MMD + mass_loss + regional_mass_loss) at each snapshot time, averaged.

    Target snapshots have pre-computed positions and log-weights.
    Both sim and target use softmax(S) for weights, so dormant
    particles (S=-100) automatically get near-zero weight.
    """
    all_px, all_py, all_S = simulate_open_full(
        learnable, z_init, wake_schedule, N_STEPS, DT, key
    )

    total_mmd = 0.0
    total_mass_l = 0.0
    total_regional = 0.0

    for i, step in enumerate(SNAPSHOT_STEPS):
        # Sim snapshot at this time
        sim_z = jnp.column_stack([all_px[step], all_py[step]])
        sim_S = all_S[step]
        sim_alpha = jax.nn.softmax(sim_S)

        # Target snapshot (pre-computed)
        tgt_z = jnp.column_stack([tgt_snap_px[i], tgt_snap_py[i]])
        tgt_alpha = jax.nn.softmax(tgt_snap_S[i])

        l_mmd = weighted_mmd_both(sim_z, sim_alpha, tgt_z, tgt_alpha)
        l_mass = mass_loss(sim_S, tgt_snap_mass[i])
        l_regional = regional_mass_loss(
            all_py[step], sim_S, tgt_snap_py[i], tgt_snap_S[i],
        )

        total_mmd += l_mmd
        total_mass_l += l_mass
        total_regional += l_regional

    n_snaps = len(SNAPSHOT_STEPS)
    avg_mmd = total_mmd / n_snaps
    avg_mass = total_mass_l / n_snaps
    avg_regional = total_regional / n_snaps

    total = avg_mmd + LAM_MASS * avg_mass + LAM_REGIONAL * avg_regional
    return total, (avg_mmd, avg_mass, avg_regional)


loss_and_grad_fn = jax.jit(jax.value_and_grad(objective, argnums=0, has_aux=True))


# ==========================================================================
# 6. Generate ground-truth snapshots at all time points
# ==========================================================================

print("=" * 65)
print("Multi-timepoint parametric inference: transient matching")
print("=" * 65)
print(f"  N_particles = {N_PARTICLES}")
print(f"  T_final = {T_FINAL}, dt = {DT}, n_steps = {N_STEPS}")
print(f"  sigma = {SIGMA} (fixed)")
print(f"  Snapshots at t = {SNAPSHOT_TIMES}")
print(f"  Region bounds (y): {REGION_BOUNDS}")
print(f"  Loss = avg(L_MMD + {LAM_MASS}*L_mass + {LAM_REGIONAL}*L_regional)")

key = jax.random.PRNGKey(4)
key, tgt_queue_key, tgt_sim_key = jax.random.split(key, 3)

print("\nGenerating ground-truth snapshots ...")

# Target particle queue
tgt_z_init, tgt_wake = build_particle_queue(
    sources=SOURCES, n_particles=N_PARTICLES,
    t_final=T_FINAL, dt=DT, key=tgt_queue_key,
)

# Run true simulation → full trajectory
true_learnable = {
    'a': jnp.array(float(TARGET_PARAMS['a'])),
    'b': jnp.array(float(TARGET_PARAMS['b'])),
    'c': jnp.array(float(TARGET_PARAMS['c'])),
    'd': jnp.array(float(TARGET_PARAMS['d'])),
    'y_c': jnp.array(float(TRUE_DEATH['y_c'])),
    'log_k': jnp.array(float(jnp.log(TRUE_DEATH['k']))),
}

tgt_all_px, tgt_all_py, tgt_all_S = simulate_open_full(
    true_learnable, tgt_z_init, tgt_wake, N_STEPS, DT, tgt_sim_key
)

# Extract snapshots and stack into arrays for JIT
tgt_snap_px_list = []
tgt_snap_py_list = []
tgt_snap_S_list = []
tgt_snap_mass_list = []

for i, (t, step) in enumerate(zip(SNAPSHOT_TIMES, SNAPSHOT_STEPS)):
    px_i = tgt_all_px[step]
    py_i = tgt_all_py[step]
    S_i = tgt_all_S[step]
    mass_i = float(jnp.sum(jnp.exp(S_i)))
    n_alive = int(jnp.sum(jnp.exp(S_i) > 0.01))
    print(f"  t={t:.1f}: mass={mass_i:.2f}, ~{n_alive} alive particles")
    tgt_snap_px_list.append(px_i)
    tgt_snap_py_list.append(py_i)
    tgt_snap_S_list.append(S_i)
    tgt_snap_mass_list.append(mass_i)

# Stack into JAX arrays: (n_snaps, N)
tgt_snap_px = jnp.stack(tgt_snap_px_list)
tgt_snap_py = jnp.stack(tgt_snap_py_list)
tgt_snap_S = jnp.stack(tgt_snap_S_list)
tgt_snap_mass = jnp.array(tgt_snap_mass_list)


# ==========================================================================
# 7. Build training particle queue (separate from target)
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

history = {'loss': [], 'mmd': [], 'mass': [], 'regional': [], 'params': {
    'a': [], 'b': [], 'c': [], 'd': [], 'y_c': [], 'log_k': [],
}}
best_loss = float('inf')
best_params = None
best_epoch = 0
opt_key = jax.random.PRNGKey(42)

print(f"\nTraining for {N_EPOCHS} epochs (6 snapshot times) ...")
print("-" * 65)

for epoch in range(N_EPOCHS):
    if USE_FRESH_KEYS:
        opt_key, sim_key = jax.random.split(opt_key)
    else:
        sim_key = opt_key

    (loss, (l_mmd, l_mass, l_regional)), grads = loss_and_grad_fn(
        learnable, z_init, wake_schedule,
        tgt_snap_px, tgt_snap_py, tgt_snap_S, tgt_snap_mass,
        sim_key,
    )

    updates, opt_state = optimizer.update(grads, opt_state)
    learnable = optax.apply_updates(learnable, updates)

    # Physical constraint: c > 0
    learnable['c'] = jnp.maximum(learnable['c'], 0.001)

    loss_val = float(loss)
    history['loss'].append(loss_val)
    history['mmd'].append(float(l_mmd))
    history['mass'].append(float(l_mass))
    history['regional'].append(float(l_regional))
    for p in history['params']:
        history['params'][p].append(float(learnable[p]))

    if loss_val < best_loss - 1e-6:
        best_loss = loss_val
        best_params = {k: v for k, v in learnable.items()}
        best_epoch = epoch

    if epoch % PRINT_EVERY == 0:
        k_val = float(jnp.exp(learnable['log_k']))
        print(
            f"Epoch {epoch:4d}: Loss={loss_val:.5f} "
            f"(MMD={float(l_mmd):.5f}, Mass={float(l_mass):.5f}, "
            f"Reg={float(l_regional):.5f})  "
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
print("Final Results — Multi-Timepoint Transient Matching")
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
ax.semilogy(epochs, history['mmd'], label='MMD (avg)', linewidth=1, alpha=0.8)
ax.semilogy(epochs, history['mass'], label='Mass (avg)', linewidth=1, alpha=0.8)
ax.semilogy(epochs, history['regional'], label='Regional (avg)', linewidth=1, alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss (multi-timepoint)')
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
plt.savefig("training_multi.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved training_multi.png")


# --- Figure 2: Potential comparison (true vs learned) ---
x_grid = jnp.linspace(-5, 5, 200)
y_grid = jnp.linspace(-2, 4, 200)
Xg, Yg = jnp.meshgrid(x_grid, y_grid)

Z_true = potential(Xg, Yg, TARGET_PARAMS)
learned_pot_params = {k: learnable[k] for k in ['a', 'b', 'c', 'd']}
Z_learned = potential(Xg, Yg, learned_pot_params)

vmin = min(float(Z_true.min()), float(Z_learned.min()))
vmax = min(float(Z_true.max()), float(Z_learned.max()))
vmax = min(vmax, 50.0)
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
plt.savefig("potential_multi.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved potential_multi.png")


# --- Figure 3: Death rate comparison ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

gamma_true = analytical_death_rate(Xg, Yg, {'y_threshold': TRUE_DEATH['y_c']})
k_l = float(jnp.exp(learnable['log_k']))
gamma_learned = jax.nn.softplus(k_l * (Yg - float(learnable['y_c'])))

g_vmax = min(max(float(gamma_true.max()), float(gamma_learned.max())), 5.0)
g_levels = np.linspace(0.0, g_vmax, 31)

cf0 = axes[0].contourf(np.array(Xg), np.array(Yg), np.array(gamma_true),
                        levels=g_levels, cmap='YlOrRd', extend='both')
plt.colorbar(cf0, ax=axes[0], label=r'$\gamma^*(x,y)$')
axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
axes[0].set_title(r'True $\gamma^*$')

cf1 = axes[1].contourf(np.array(Xg), np.array(Yg), np.array(gamma_learned),
                        levels=g_levels, cmap='YlOrRd', extend='both')
plt.colorbar(cf1, ax=axes[1], label=r'$\gamma_\theta(x,y)$')
axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
axes[1].set_title(r'Learned $\gamma_\theta$')

y_line = np.linspace(-2, 4, 200)
gamma_1d_true = np.array(jax.nn.softplus(jnp.array(y_line) - TRUE_DEATH['y_c']))
gamma_1d_learned = np.array(jax.nn.softplus(
    k_l * (jnp.array(y_line) - float(learnable['y_c']))
))
axes[2].plot(y_line, gamma_1d_true, 'b-', linewidth=2, label='True')
axes[2].plot(y_line, gamma_1d_learned, 'r--', linewidth=2, label='Learned')
axes[2].set_xlabel('y'); axes[2].set_ylabel(r'$\gamma(y)$')
axes[2].set_title(r'Death rate profile (x=0)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("death_rate_multi.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved death_rate_multi.png")


# --- Figure 4: Multi-snapshot particle comparison ---
# Show 4 selected snapshots: t=1, 2, 4, 8
VIS_TIMES = [1.0, 2.0, 4.0, 8.0]
VIS_STEPS = [int(t / DT) - 1 for t in VIS_TIMES]

# Run learned simulation for visualization
key, eval_key, eval_queue_key = jax.random.split(key, 3)
z_init_eval, wake_eval = build_particle_queue(
    sources=SOURCES, n_particles=N_PARTICLES,
    t_final=T_FINAL, dt=DT, key=eval_queue_key,
)

sim_all_px, sim_all_py, sim_all_S = simulate_open_full(
    learnable, z_init_eval, wake_eval, N_STEPS, DT, eval_key
)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for col, (t_vis, step_vis) in enumerate(zip(VIS_TIMES, VIS_STEPS)):
    # Top row: target
    ax_tgt = axes[0, col]
    tgt_px_i = tgt_all_px[step_vis]
    tgt_py_i = tgt_all_py[step_vis]
    tgt_S_i = tgt_all_S[step_vis]
    tgt_w_i = np.array(jnp.clip(jnp.exp(tgt_S_i), 0.0, 1.0))
    alive = tgt_w_i > 1e-6  # exclude dormant particles (S=-100)

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
    alive = sim_w_i > 1e-6  # exclude dormant particles (S=-100)

    ax_sim.contourf(np.array(Xg), np.array(Yg), np.array(Z_learned),
                    levels=30, cmap='viridis', alpha=0.3)
    sc = ax_sim.scatter(np.array(sim_px_i[alive]), np.array(sim_py_i[alive]),
                        c=sim_w_i[alive], cmap='coolwarm', s=6, alpha=0.5,
                        vmin=0, vmax=1, edgecolors='none')
    ax_sim.set_xlim(-5, 5); ax_sim.set_ylim(-2, 4)
    ax_sim.set_title(f'Learned t={t_vis:.0f}')
    if col == 0:
        ax_sim.set_ylabel('y (learned)')

# Add single colorbar
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
fig.colorbar(sc, cax=cbar_ax, label=r'$w = e^S$')

plt.savefig("snapshots_multi.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved snapshots_multi.png")

print("\nDone. All figures saved.")
