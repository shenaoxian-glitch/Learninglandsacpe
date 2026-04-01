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
