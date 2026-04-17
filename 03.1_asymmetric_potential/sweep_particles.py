#!/usr/bin/env python
"""
sweep_particles.py — sweep N_PARTICLES from 2000 to 4000 (step 200)

For each N: train 500 epochs, save figures to sweep/N{n}/.
After all runs: plot learned params vs N (summary figure).
"""
import matplotlib
matplotlib.use('Agg')
import os
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np

from training.data_loader import build_particle_queue, death_rate_2d

# ==========================================================================
# Constants (shared across all runs)
# ==========================================================================

TARGET_PARAMS = {
    'a': -1.0, 'b': -1.6, 'c': 3.5, 'd': -1.2,
    'e': 0.7, 'sigma': 1.5,
}
DEATH_PARAMS = {
    'A': 2.0, 'w_x': 0.5, 'k1': 1.0, 'y_select': 1.0,
    'B': 5.0, 'k2': 3.0, 'y_max': 3.0,
}
GAMMA_MAX = 50.0
DT = 0.01
T_FINAL = 10.0
N_STEPS = int(T_FINAL / DT)
SIGMA = TARGET_PARAMS['sigma']
LAM_MASS = 0.0
N_EPOCHS = 500
PRINT_EVERY = 50
TRUE_VALS = {'a': -1.0, 'b': -1.6, 'c': 3.5, 'd': -1.2, 'e': 0.7}


# ==========================================================================
# Core functions
# ==========================================================================

def potential(x, y, p):
    return (-(y - p['a']) * x**2 + jnp.exp(p['b']) * x**4
            - p['c'] * y + jnp.exp(p['d']) * y**4 + p['e'] * x)

_force_fn = jax.vmap(jax.grad(potential, argnums=(0, 1)), in_axes=(0, 0, None))


def simulate_open_full(learnable, death_params, z_init, wake_schedule,
                       n_steps, dt, key):
    pot_params = {k: learnable[k] for k in ['a', 'b', 'c', 'd', 'e']}
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
        gamma = jnp.minimum(death_rate_2d(px, py, death_params), GAMMA_MAX)
        new_S = S - gamma * dt
        return (new_px, new_py, new_S), (new_px, new_py, new_S)

    step_indices = jnp.arange(n_steps)
    _, (all_px, all_py, all_S) = jax.lax.scan(
        scan_body, (z_init[:, 0], z_init[:, 1], S0), (step_keys, step_indices)
    )
    return all_px, all_py, all_S


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
    return (jnp.sum(alpha_sim[:, None] * alpha_sim[None, :] * K_xx)
            + jnp.sum(alpha_obs[:, None] * alpha_obs[None, :] * K_yy)
            - 2 * jnp.sum(alpha_sim[:, None] * alpha_obs[None, :] * K_xy))


def mass_loss(S, target_mass):
    M_sim = jnp.sum(jnp.exp(S))
    return (M_sim - target_mass)**2 / (target_mass**2 + 1e-8)


def objective(learnable, death_params, z_init, wake_schedule,
              tgt_px, tgt_py, tgt_S, tgt_mass, key):
    all_px, all_py, all_S = simulate_open_full(
        learnable, death_params, z_init, wake_schedule, N_STEPS, DT, key
    )
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


loss_and_grad_fn = jax.jit(jax.value_and_grad(objective, argnums=0, has_aux=True))


# ==========================================================================
# Single run
# ==========================================================================

def run_one(n_particles, master_key, output_dir):
    """Full training + figures for one N value. Returns final learned params."""
    os.makedirs(output_dir, exist_ok=True)

    sources = [{'mu': jnp.array([0.0, -1.0]), 'sigma_x': 0.25, 'sigma_y': 0.10,
                'n_particles': n_particles}]

    key = master_key
    key, tgt_queue_key, tgt_sim_key = jax.random.split(key, 3)

    # --- Target ---
    tgt_z_init, tgt_wake = build_particle_queue(
        sources=sources, n_particles=n_particles,
        t_final=T_FINAL, dt=DT, key=tgt_queue_key)

    true_learnable = {k: jnp.array(float(TARGET_PARAMS[k])) for k in 'abcde'}

    tgt_all_px, tgt_all_py, tgt_all_S = simulate_open_full(
        true_learnable, DEATH_PARAMS, tgt_z_init, tgt_wake, N_STEPS, DT, tgt_sim_key)

    final_step = N_STEPS - 1
    tgt_px = tgt_all_px[final_step]
    tgt_py = tgt_all_py[final_step]
    tgt_S = tgt_all_S[final_step]
    tgt_mass = float(jnp.sum(jnp.exp(tgt_S)))
    n_alive = int(jnp.sum(jnp.exp(tgt_S) > 0.01))
    print(f"  Target: mass={tgt_mass:.1f}, ~{n_alive} alive")

    # --- Training queue ---
    key, queue_key = jax.random.split(key)
    z_init, wake_schedule = build_particle_queue(
        sources=sources, n_particles=n_particles,
        t_final=T_FINAL, dt=DT, key=queue_key)

    # --- Train ---
    learnable = {
        'a': jnp.array(0.0), 'b': jnp.array(-1.0),
        'c': jnp.array(2.0), 'd': jnp.array(-0.8), 'e': jnp.array(0.0),
    }
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(0.01))
    opt_state = optimizer.init(learnable)

    history = {'loss': [], 'mmd': [], 'mass': [], 'params': {p: [] for p in 'abcde'}}
    best_loss, best_params, best_epoch = float('inf'), None, 0
    opt_key = jax.random.PRNGKey(42)

    for epoch in range(N_EPOCHS):
        opt_key, sim_key = jax.random.split(opt_key)
        (loss, (l_mmd, l_mass)), grads = loss_and_grad_fn(
            learnable, DEATH_PARAMS, z_init, wake_schedule,
            tgt_px, tgt_py, tgt_S, tgt_mass, sim_key)
        updates, opt_state = optimizer.update(grads, opt_state)
        learnable = optax.apply_updates(learnable, updates)
        learnable['c'] = jnp.maximum(learnable['c'], 0.001)

        loss_val = float(loss)
        history['loss'].append(loss_val)
        history['mmd'].append(float(l_mmd))
        history['mass'].append(float(l_mass))
        for p in 'abcde':
            history['params'][p].append(float(learnable[p]))

        if loss_val < best_loss - 1e-6:
            best_loss = loss_val
            best_params = {k: v for k, v in learnable.items()}
            best_epoch = epoch

        if epoch % PRINT_EVERY == 0:
            print(f"  Epoch {epoch:4d}: Loss={loss_val:.5f}  "
                  f"a={float(learnable['a']):+.3f} b={float(learnable['b']):+.3f} "
                  f"c={float(learnable['c']):+.3f} d={float(learnable['d']):+.3f} "
                  f"e={float(learnable['e']):+.3f}")

    if best_params is not None:
        learnable = best_params
    print(f"  Best epoch {best_epoch}, loss={best_loss:.6f}")

    # --- Figures ---
    epochs_arr = np.arange(N_EPOCHS)

    # Fig 1: training + convergence
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].semilogy(epochs_arr, history['loss'], label='Total', linewidth=1.5)
    axes[0].semilogy(epochs_arr, history['mmd'], label='MMD', linewidth=1, alpha=0.8)
    axes[0].semilogy(epochs_arr, history['mass'], label='Mass', linewidth=1, alpha=0.8)
    axes[0].set(xlabel='Epoch', ylabel='Loss',
                title=f'Training Loss (N={n_particles})')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    for name in 'abcde':
        ln, = axes[1].plot(epochs_arr, history['params'][name], label=name, linewidth=1.2)
        axes[1].axhline(TRUE_VALS[name], color=ln.get_color(), linestyle='--', alpha=0.5)
    axes[1].set(xlabel='Epoch', ylabel='Value', title='Parameter Convergence')
    axes[1].legend(ncol=2, fontsize=9); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Fig 2: potential comparison
    x_grid = jnp.linspace(-5, 5, 200)
    y_grid = jnp.linspace(-2, 4, 200)
    Xg, Yg = jnp.meshgrid(x_grid, y_grid)
    Z_true = potential(Xg, Yg, TARGET_PARAMS)
    Z_learned = potential(Xg, Yg, {k: learnable[k] for k in 'abcde'})

    vmin = min(float(Z_true.min()), float(Z_learned.min()))
    vmax = min(float(Z_true.max()), float(Z_learned.max()), 50.0)
    levels = np.linspace(vmin, vmax, 31)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, Z, title in [(axes[0], Z_true, 'True'), (axes[1], Z_learned, 'Learned')]:
        cf = ax.contourf(np.array(Xg), np.array(Yg), np.array(Z),
                         levels=levels, cmap='viridis', extend='both')
        plt.colorbar(cf, ax=ax); ax.set(xlabel='x', ylabel='y', title=title)
    Z_diff = Z_learned - Z_true
    dm = min(max(abs(float(Z_diff.min())), abs(float(Z_diff.max()))), 20.0)
    cf = axes[2].contourf(np.array(Xg), np.array(Yg), np.array(Z_diff),
                           levels=np.linspace(-dm, dm, 31), cmap='RdBu_r', extend='both')
    plt.colorbar(cf, ax=axes[2]); axes[2].set(xlabel='x', ylabel='y', title='Error')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/potential.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Fig 3: mass vs time
    key, eval_key, eval_queue_key = jax.random.split(key, 3)
    z_eval, wake_eval = build_particle_queue(
        sources=sources, n_particles=n_particles,
        t_final=T_FINAL, dt=DT, key=eval_queue_key)
    sim_all_px, sim_all_py, sim_all_S = simulate_open_full(
        learnable, DEATH_PARAMS, z_eval, wake_eval, N_STEPS, DT, eval_key)

    time_axis = np.arange(1, N_STEPS + 1) * DT
    tgt_mass_t = np.array([float(jnp.sum(jnp.exp(tgt_all_S[i]))) for i in range(N_STEPS)])
    sim_mass_t = np.array([float(jnp.sum(jnp.exp(sim_all_S[i]))) for i in range(N_STEPS)])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_axis, tgt_mass_t, 'b-', linewidth=1.5, label='Target', alpha=0.8)
    ax.plot(time_axis, sim_mass_t, 'r--', linewidth=1.5, label='Learned', alpha=0.8)
    ax.set(xlabel='Time', ylabel=r'Total mass $\sum e^{S_i}$',
           title=f'Mass vs Time (N={n_particles})')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mass_vs_time.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Fig 4: snapshots
    VIS_TIMES = [1.0, 3.0, 6.0, 10.0]
    VIS_STEPS = [int(t / DT) - 1 for t in VIS_TIMES]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for col, (t_vis, step_vis) in enumerate(zip(VIS_TIMES, VIS_STEPS)):
        for row, (all_px, all_py, all_S_arr, Z, label) in enumerate([
            (tgt_all_px, tgt_all_py, tgt_all_S, Z_true, 'Target'),
            (sim_all_px, sim_all_py, sim_all_S, Z_learned, 'Learned'),
        ]):
            ax = axes[row, col]
            px_i, py_i, S_i = all_px[step_vis], all_py[step_vis], all_S_arr[step_vis]
            w_i = np.array(jnp.clip(jnp.exp(S_i), 0.0, 1.0))
            alive = w_i > 1e-6
            ax.contourf(np.array(Xg), np.array(Yg), np.array(Z),
                        levels=30, cmap='viridis', alpha=0.3)
            sc = ax.scatter(np.array(px_i[alive]), np.array(py_i[alive]),
                            c=w_i[alive], cmap='coolwarm', s=6, alpha=0.5,
                            vmin=0, vmax=1, edgecolors='none')
            ax.set_xlim(-5, 5); ax.set_ylim(-2, 4)
            ax.set_title(f'{label} t={t_vis:.0f}')
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label=r'$w = e^S$')
    plt.savefig(f"{output_dir}/snapshots.png", dpi=150, bbox_inches='tight')
    plt.close()

    result = {name: float(learnable[name]) for name in 'abcde'}
    result['best_loss'] = best_loss
    print(f"  Final: " + "  ".join(f"{k}={v:+.4f}" for k, v in result.items()
                                    if k != 'best_loss'))
    return result


# ==========================================================================
# Main sweep
# ==========================================================================

if __name__ == '__main__':
    N_VALUES = list(range(2000, 4001, 200))
    all_results = {}
    master_key = jax.random.PRNGKey(4)

    for n in N_VALUES:
        master_key, run_key = jax.random.split(master_key)
        print(f"\n{'='*60}")
        print(f"  N_PARTICLES = {n}  ({N_VALUES.index(n)+1}/{len(N_VALUES)})")
        print(f"{'='*60}")
        all_results[n] = run_one(n, run_key, f"sweep/N{n}")

    # ==========================================================================
    # Summary plot: learned params vs N
    # ==========================================================================

    ns = np.array(N_VALUES)
    param_names = list('abcde')

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for i, name in enumerate(param_names):
        ax = axes_flat[i]
        learned = np.array([all_results[n][name] for n in N_VALUES])
        true_val = TRUE_VALS[name]

        ax.plot(ns, learned, 'o-', linewidth=1.5, markersize=5, label='Learned')
        ax.axhline(true_val, color='red', linestyle='--', linewidth=1.5,
                    alpha=0.7, label=f'True ({true_val})')
        ax.set_xlabel('N particles')
        ax.set_ylabel(f'{name} value')
        ax.set_title(f'Parameter {name}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # 6th subplot: loss vs N
    ax = axes_flat[5]
    losses = np.array([all_results[n]['best_loss'] for n in N_VALUES])
    ax.plot(ns, losses, 'ko-', linewidth=1.5, markersize=5)
    ax.set_xlabel('N particles')
    ax.set_ylabel('Best loss')
    ax.set_title('Best Training Loss')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sweep/summary_params_vs_N.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved sweep/summary_params_vs_N.png")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  Summary: Learned Parameters vs N_PARTICLES")
    print(f"{'='*70}")
    print(f"  {'N':>6}  {'a':>8}  {'b':>8}  {'c':>8}  {'d':>8}  {'e':>8}  {'loss':>10}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
    print(f"  {'TRUE':>6}  {-1.0:>+8.4f}  {-1.6:>+8.4f}  {+3.5:>+8.4f}  "
          f"{-1.2:>+8.4f}  {+0.7:>+8.4f}")
    for n in N_VALUES:
        r = all_results[n]
        print(f"  {n:>6}  {r['a']:>+8.4f}  {r['b']:>+8.4f}  {r['c']:>+8.4f}  "
              f"{r['d']:>+8.4f}  {r['e']:>+8.4f}  {r['best_loss']:>10.6f}")
    print()
