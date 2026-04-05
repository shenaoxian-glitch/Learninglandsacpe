import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Analytical potential and proliferation (ground-truth for synthetic data)
# ---------------------------------------------------------------------------

def analytical_potential(x, y, params):
    """H = -(y - a)x^2 + exp(b)x^4 - cy + exp(d)y^4"""
    return (-(y - params['a']) * (x ** 2)
            + jnp.exp(params['b']) * (x ** 4)
            - params['c'] * y
            + jnp.exp(params['d']) * (y ** 4))


_analytical_force = jax.vmap(
    jax.grad(analytical_potential, argnums=(0, 1)),
    in_axes=(0, 0, None),
)


def analytical_proliferation(x, y, prolif_params):
    """R(x,y) = beta * exp(-dist^2 / 2gamma^2) - delta"""
    dist_sq = ((x - prolif_params['x0']) ** 2
               + (y - prolif_params['y0']) ** 2)
    return (prolif_params['beta']
            * jnp.exp(-dist_sq / (2 * prolif_params['gamma'] ** 2))
            - prolif_params['delta'])


def analytical_death_rate(x, y, death_params):
    """gamma*(x,y) = softplus(y - y_threshold). Non-negative, smooth."""
    return jax.nn.softplus(y - death_params['y_threshold'])


# ---------------------------------------------------------------------------
# Multi-timepoint target data generator
# Howe & Mani format: D = {(t0, X0, t1, X1, s(t))}
# ---------------------------------------------------------------------------

def generate_multi_timepoint_data(
    params,
    prolif_params,
    n_particles,
    snapshot_times,
    dt,
    key,
    x0_mean=None,
    x0_std=0.1,
):
    """
    Generate ground-truth snapshots at multiple time points by running
    the analytical SDE + proliferation.

    Args:
        params: potential parameters dict
        prolif_params: proliferation parameters dict
        n_particles: number of particles
        snapshot_times: list/array of times to record snapshots, e.g. [0.0, 0.5, 1.0, 2.0]
        dt: integration step size
        key: PRNG key
        x0_mean: (2,) initial mean position
        x0_std: initial noise scale

    Returns:
        transitions: list of dicts, each containing:
            't0': start time
            't1': end time
            'X0': (N, 2) positions at t0
            'X1': (N, 2) positions at t1
            'n_steps': number of integration steps between t0 and t1
        snapshots: list of (time, positions, alpha) for visualization
    """
    if x0_mean is None:
        x0_mean = jnp.array([0.0, -1.0])
    snapshot_times = list(snapshot_times)

    # Initialize particles
    k1, k2, scan_key = jax.random.split(key, 3)
    px = x0_mean[0] + x0_std * jax.random.normal(k1, (n_particles,))
    py = x0_mean[1] + x0_std * jax.random.normal(k2, (n_particles,))
    S = jnp.zeros(n_particles)

    total_steps = int(snapshot_times[-1] / dt)
    step_keys = jax.random.split(scan_key, total_steps)

    # Convert snapshot times to step indices
    snapshot_step_indices = [int(t / dt) for t in snapshot_times]

    # Run full simulation, recording at snapshot steps
    def scan_body(carry, step_key):
        px, py, S = carry
        gx, gy = _analytical_force(px, py, params)
        fx, fy = -gx, -gy

        kx, ky = jax.random.split(step_key)
        nx = jax.random.normal(kx, px.shape) * jnp.sqrt(dt)
        ny = jax.random.normal(ky, py.shape) * jnp.sqrt(dt)

        new_px = px + fx * dt + params['sigma'] * nx
        new_py = py + fy * dt + params['sigma'] * ny

        R = analytical_proliferation(px, py, prolif_params)
        new_S = S + R * dt

        return (new_px, new_py, new_S), (new_px, new_py, new_S)

    (final_px, final_py, final_S), (all_px, all_py, all_S) = jax.lax.scan(
        scan_body, (px, py, S), step_keys
    )
    # all_px: (total_steps, N) — state *after* each step

    # Collect snapshots (prepend initial state at t=0)
    # At each snapshot, RESAMPLE particles proportional to their cumulative
    # weights. This simulates real scRNA-seq: you observe more cells where
    # proliferation is high, because high-R regions produce more daughter cells.
    # Without resampling, R only affects weights (not position density),
    # making it invisible to the MMD loss which compares position distributions.

    snapshots = []
    resample_key = jax.random.PRNGKey(999)  # fixed for reproducibility

    # t=0 snapshot: uniform weights, no resampling needed
    init_pos = jnp.column_stack([px, py])
    init_alpha = jnp.ones(n_particles) / n_particles
    snapshots.append((0.0, init_pos, init_alpha))

    for i, (t, step_idx) in enumerate(zip(snapshot_times, snapshot_step_indices)):
        if step_idx == 0:
            continue
        idx = step_idx - 1  # scan output is 0-indexed, after each step
        pos = jnp.column_stack([all_px[idx], all_py[idx]])
        log_w = all_S[idx]
        alpha = jnp.exp(log_w - jax.nn.logsumexp(log_w))

        # Multinomial resampling: draw n_particles samples with replacement,
        # probability proportional to alpha. Positions with high alpha are
        # duplicated, encoding R's effect into the position density.
        resample_key, rk = jax.random.split(resample_key)
        alpha_np = np.asarray(alpha)
        alpha_np = alpha_np / alpha_np.sum()  # ensure exact sum=1 for numpy
        indices = np.random.default_rng(int(rk[0])).choice(
            n_particles, size=n_particles, replace=True, p=alpha_np
        )
        resampled_pos = pos[indices]
        # After resampling, weights are uniform again
        resampled_alpha = jnp.ones(n_particles) / n_particles

        snapshots.append((t, resampled_pos, resampled_alpha))

    # Build transition pairs: (t_i, X_i) -> (t_{i+1}, X_{i+1})
    # For each transition, compute the ground-truth log mass ratio log_m_obs.
    # This is log(mean(exp(S_interval))) where S_interval is the log-weight
    # accumulated within that interval only (not cumulative from t=0).
    # Since simulate_transition resets S=0 at each interval start, log_m_obs
    # must also reflect per-interval accumulation.
    transitions = []
    for i in range(len(snapshots) - 1):
        t0, X0, _ = snapshots[i]
        t1, X1, _ = snapshots[i + 1]
        n_steps_segment = int(round((t1 - t0) / dt))

        # Compute per-interval log mass ratio from ground-truth R
        # Approximate: run a mini-simulation of R along the interval using
        # the snapshot positions (held fixed) to accumulate S over the interval.
        # S_i = integral_t0^t1 R(z_i(t)) dt ≈ R(z_i_start) * (t1-t0) for short intervals.
        # For accuracy, use the step-level data if available.
        if i == 0:
            # First interval: S accumulated from t=0 to snapshot_times[0]
            step_idx = snapshot_step_indices[0] - 1
            interval_S = all_S[step_idx]  # cumulative S from t=0
        else:
            # Later intervals: S accumulated within interval only
            # S_interval = S(t_{i+1}) - S(t_i)
            prev_idx = snapshot_step_indices[i - 1] - 1
            curr_idx = snapshot_step_indices[i] - 1
            interval_S = all_S[curr_idx] - all_S[prev_idx]

        # log_m_obs = log(mean(exp(S_interval)))
        # Use logsumexp for numerical stability
        N = interval_S.shape[0]
        log_m_obs = float(jax.nn.logsumexp(interval_S) - jnp.log(N))

        transitions.append({
            't0': t0,
            't1': t1,
            'X0': X0,
            'X1': X1,
            'n_steps': n_steps_segment,
            'log_m_obs': log_m_obs,
        })

    return transitions, snapshots


# ---------------------------------------------------------------------------
# Single-timepoint (backward compatible)
# ---------------------------------------------------------------------------

def generate_target_data(
    params,
    prolif_params,
    n_particles,
    n_steps,
    dt,
    key,
    x0_mean=None,
    x0_std=0.1,
):
    """
    Generate ground-truth endpoint data (single timepoint).

    Returns:
        positions: (N, 2) final (x, y)
        alpha: (N,) normalized weights
    """
    t_final = n_steps * dt
    transitions, snapshots = generate_multi_timepoint_data(
        params, prolif_params, n_particles,
        snapshot_times=[t_final],
        dt=dt, key=key, x0_mean=x0_mean, x0_std=x0_std,
    )
    _, pos, alpha = snapshots[-1]
    return pos, alpha


# ---------------------------------------------------------------------------
# Open system data generation (particle queue with wake-up schedule)
# ---------------------------------------------------------------------------

def build_particle_queue(sources, n_particles, t_final, dt, key, d=2):
    """
    Pre-allocate particles with individual (t0, x0) birth events.

    Each particle has a unique birth time uniformly spread over [0, t_final]
    and a birth position sampled from N(mu, sigma_src^2 I).

    Args:
        sources: list of dicts, each with 'mu' (d,), 'sigma_src', 'n_particles'
            n_particles: how many particles from this source
        n_particles: total particle count (ignored if sources specify per-source counts)
        t_final: simulation duration
        dt: step size
        key: PRNG key
        d: spatial dimension

    Returns:
        z_init: (N_total, d) birth positions
        wake_schedule: (N_total,) int array, birth step for each particle
    """
    all_positions = []
    all_wake_steps = []
    n_steps = int(round(t_final / dt))

    for src in sources:
        mu = jnp.array(src['mu'])
        sigma_src = src['sigma_src']
        n = src.get('n_particles', n_particles)

        # Uniform birth times over [0, t_final)
        birth_times = jnp.linspace(0, t_final, n, endpoint=False)
        wake_steps = (birth_times / dt).astype(int)
        wake_steps = jnp.clip(wake_steps, 0, n_steps - 1)

        key, k = jax.random.split(key)
        noise = sigma_src * jax.random.normal(k, (n, d))
        positions = mu[None, :] + noise

        all_positions.append(positions)
        all_wake_steps.append(wake_steps)

    z_init = jnp.concatenate(all_positions, axis=0)
    wake_schedule = jnp.concatenate(all_wake_steps, axis=0)
    return z_init, wake_schedule


def generate_open_system_data(
    params,
    death_params,
    sources,
    n_particles,
    t_final,
    dt,
    key,
):
    """
    Generate ground-truth steady-state snapshot for an open system.

    Starts with 0 active particles. Each particle has a unique birth
    time and position. Particles flow under analytical potential, die
    via analytical death rate. Returns the final-time snapshot.

    Args:
        params: potential parameters dict (includes 'sigma')
        death_params: {'y_threshold': float}
        sources: list of source dicts with 'mu', 'sigma_src', 'n_particles'
        n_particles: total particles (used if source doesn't specify)
        t_final: total simulation time
        dt: integration step size
        key: PRNG key

    Returns:
        target_pos: (N_total, 2) raw final positions of all particles
        target_S: (N_total,) raw log-weights of all particles
        target_mass: float, total active mass sum(exp(S))
        z_init: (N_total, 2) all birth positions
        wake_schedule: (N_total,) birth step for each particle
    """
    n_steps = int(round(t_final / dt))
    key, queue_key, sim_key = jax.random.split(key, 3)

    z_init, wake_schedule = build_particle_queue(
        sources, n_particles, t_final, dt, queue_key
    )
    N_total = z_init.shape[0]

    # All start dormant
    S0 = jnp.full(N_total, -100.0)
    step_keys = jax.random.split(sim_key, n_steps)
    sigma = params['sigma']

    def scan_body(carry, inputs):
        px, py, S = carry
        step_key, step_idx = inputs

        # Wake-up
        is_waking = (wake_schedule == step_idx)
        S = jnp.where(is_waking, 0.0, S)
        px = jnp.where(is_waking, z_init[:, 0], px)
        py = jnp.where(is_waking, z_init[:, 1], py)

        # Drift
        gx, gy = _analytical_force(px, py, params)
        fx, fy = -gx, -gy

        kx, ky = jax.random.split(step_key)
        nx = jax.random.normal(kx, px.shape) * jnp.sqrt(dt)
        ny = jax.random.normal(ky, py.shape) * jnp.sqrt(dt)

        new_px = px + fx * dt + sigma * nx
        new_py = py + fy * dt + sigma * ny

        # Death
        gamma = analytical_death_rate(px, py, death_params)
        new_S = S - gamma * dt

        return (new_px, new_py, new_S), None

    step_indices = jnp.arange(n_steps)
    (final_px, final_py, final_S), _ = jax.lax.scan(
        scan_body, (z_init[:, 0], z_init[:, 1], S0), (step_keys, step_indices)
    )

    final_pos = jnp.column_stack([final_px, final_py])
    active_mass = float(jnp.sum(jnp.exp(final_S)))

    return final_pos, final_S, active_mass, z_init, wake_schedule
