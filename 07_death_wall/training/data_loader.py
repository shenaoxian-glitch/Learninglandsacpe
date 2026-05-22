import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Analytical potential and death rate (ground-truth for synthetic data)
# ---------------------------------------------------------------------------

def analytical_potential(x, y, params):
    """H = -(y - a)x^2 + exp(b)x^4 - cy + exp(d)y^4 + e*x

    Asymmetric potential: the e*x term breaks x-symmetry.
    """
    return (-(y - params['a']) * (x ** 2)
            + jnp.exp(params['b']) * (x ** 4)
            - params['c'] * y
            + jnp.exp(params['d']) * (y ** 4)
            + params['e'] * x)


_analytical_force = jax.vmap(
    jax.grad(analytical_potential, argnums=(0, 1)),
    in_axes=(0, 0, None),
)


def analytical_death_rate(x, y, death_params):
    """gamma*(x,y) = softplus(y - y_threshold). Non-negative, smooth, y-only."""
    return jax.nn.softplus(y - death_params['y_threshold'])


def analytical_death_rate_2d(x, y, params):
    """
    2D death rate from model 3.1: selection corridor + terminal boundary.

    gamma(x,y) = gamma_select(x,y) + gamma_term(y)
      gamma_select = A * exp(-x^2/(2*w_x^2)) * softplus(k1*(y - y_select))
      gamma_term   = B * softplus(k2*(y - y_max))
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


def analytical_potential_no_y4(x, y, params):
    """H = -(y - a)x^2 + exp(b)x^4 - cy + e*x

    No y^4 confinement — particles confined in y by death wall instead.
    x-confinement is decoupled from y (global x^4).
    """
    return (-(y - params['a']) * (x ** 2)
            + jnp.exp(params['b']) * (x ** 4)
            - params['c'] * y
            + params['e'] * x)


def analytical_potential_no_y4_coupled(x, y, params):
    """H = -(y - a)x^2 + exp(b)x^4*(y - a) - cy + e*x

    No y^4 confinement — particles confined in y by death wall instead.
    x-confinement coupled to y: both x^2 and x^4 terms scale with (y-a).
    H = (y-a)*[exp(b)*x^4 - x^2] - c*y + e*x
    """
    return (-(y - params['a']) * (x ** 2)
            + jnp.exp(params['b']) * (x ** 4) * (y - params['a'])
            - params['c'] * y
            + params['e'] * x)


def analytical_death_rate_exp(x, y, params):
    """Exponential death wall: gamma(y) = A * exp(k * (y - y_d)), capped."""
    raw = params['A'] * jnp.exp(params['k'] * (y - params['y_d']))
    return jnp.minimum(raw, params['gamma_max'])


# ---------------------------------------------------------------------------
# Open system data generation (particle queue with wake-up schedule)
# ---------------------------------------------------------------------------

def build_particle_queue(sources, n_particles, t_final, dt, key):
    """
    Pre-allocate particles with individual (t0, x0) birth events.

    Supports anisotropic (oval) Gaussian sources via separate sigma_x / sigma_y.
    Falls back to isotropic sigma_src if sigma_x/sigma_y not specified.

    Args:
        sources: list of dicts with keys:
            'mu': (2,) center position
            'sigma_x', 'sigma_y': anisotropic source widths (preferred)
            OR 'sigma_src': isotropic source width (fallback)
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
        n = src.get('n_particles', n_particles)

        # Uniform birth times over [0, t_final)
        birth_times = jnp.linspace(0, t_final, n, endpoint=False)
        wake_steps = (birth_times / dt).astype(int)
        wake_steps = jnp.clip(wake_steps, 0, n_steps - 1)

        # Support oval (anisotropic) or isotropic source
        if 'sigma_x' in src and 'sigma_y' in src:
            key, kx, ky = jax.random.split(key, 3)
            noise_x = src['sigma_x'] * jax.random.normal(kx, (n,))
            noise_y = src['sigma_y'] * jax.random.normal(ky, (n,))
            positions = jnp.column_stack([mu[0] + noise_x, mu[1] + noise_y])
        else:
            sigma_src = src['sigma_src']
            key, k = jax.random.split(key)
            noise = sigma_src * jax.random.normal(k, (n, 2))
            positions = mu[None, :] + noise

        all_positions.append(positions)
        all_wake_steps.append(wake_steps)

    z_init = jnp.concatenate(all_positions, axis=0)
    wake_schedule = jnp.concatenate(all_wake_steps, axis=0)
    return z_init, wake_schedule
