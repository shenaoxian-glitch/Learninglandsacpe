import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# 2D death rate: selection corridor + terminal boundary
# ---------------------------------------------------------------------------

def death_rate_2d(x, y, params):
    """
    γ(x,y) = γ_select(x,y) + γ_term(y)

    Selection death: penalizes undifferentiated cells (x≈0) that reach
    the selection stage (y > y_select). Cells that commit to a lineage
    (|x| >> 0) escape via the Gaussian corridor.

      γ_select = A · exp(-x²/(2·w_x²)) · softplus(k1·(y - y_select))

    Terminal death: hard upper boundary on developmental progression.

      γ_term = B · softplus(k2·(y - y_max))
    """
    gamma_select = (
        params['A']
        * jnp.exp(-x**2 / (2 * params['w_x']**2))
        * jax.nn.softplus(params['k1'] * (y - params['y_select']))
    )
    gamma_term = (
        params['B']
        * jax.nn.softplus(params['k2'] * (y - params['y_max']))
    )
    return gamma_select + gamma_term


# ---------------------------------------------------------------------------
# Oval-Gaussian particle queue (anisotropic source)
# ---------------------------------------------------------------------------

def build_particle_queue(sources, n_particles, t_final, dt, key):
    """
    Pre-allocate particles with individual (t0, x0) birth events.

    Each source specifies separate sigma_x / sigma_y for an oval Gaussian.
    Birth times are uniformly spread over [0, t_final].

    Args:
        sources: list of dicts with keys:
            'mu': (2,) center position
            'sigma_x': source width in x
            'sigma_y': source width in y
            'n_particles': how many from this source
        n_particles: fallback if source doesn't specify
        t_final: simulation duration
        dt: step size
        key: PRNG key

    Returns:
        z_init: (N_total, 2) birth positions
        wake_schedule: (N_total,) int array of birth steps
    """
    all_positions = []
    all_wake_steps = []
    n_steps = int(round(t_final / dt))

    for src in sources:
        mu = jnp.array(src['mu'])
        sigma_x = src['sigma_x']
        sigma_y = src['sigma_y']
        n = src.get('n_particles', n_particles)

        # Uniform birth times over [0, t_final)
        birth_times = jnp.linspace(0, t_final, n, endpoint=False)
        wake_steps = (birth_times / dt).astype(int)
        wake_steps = jnp.clip(wake_steps, 0, n_steps - 1)

        key, kx, ky = jax.random.split(key, 3)
        noise_x = sigma_x * jax.random.normal(kx, (n,))
        noise_y = sigma_y * jax.random.normal(ky, (n,))
        positions = jnp.column_stack([mu[0] + noise_x, mu[1] + noise_y])

        all_positions.append(positions)
        all_wake_steps.append(wake_steps)

    z_init = jnp.concatenate(all_positions, axis=0)
    wake_schedule = jnp.concatenate(all_wake_steps, axis=0)
    return z_init, wake_schedule
