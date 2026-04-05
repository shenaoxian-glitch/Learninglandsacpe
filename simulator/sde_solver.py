import jax
import jax.numpy as jnp


def _normalize_log_weights(S):
    """Numerically stable weight normalization via explicit logsumexp."""
    log_alpha = S - jax.nn.logsumexp(S)
    return jnp.exp(log_alpha)


def simulate(model, z0, n_steps, dt, key, signals=None):
    """
    Euler-Maruyama integration of the augmented SDE (final state only).

    dz = [-nabla_z Phi(z) - Psi(s)] dt + sigma(z) dW
    dS = R(z, s) dt

    Args:
        model: WaddingtonModel
        z0: (N, d) initial positions
        n_steps: number of time steps
        dt: step size
        key: PRNG key
        signals: (n_steps, d_signal) array or None

    Returns:
        final_z: (N, d)
        final_S: (N,) log-weights
        alpha:   (N,) normalized weights via logsumexp
    """
    N = z0.shape[0]
    S0 = jnp.zeros(N)
    step_keys = jax.random.split(key, n_steps)

    use_signal = signals is not None and model.tilt is not None

    def scan_body(carry, inputs):
        z, S = carry

        if use_signal:
            step_key, signal = inputs
        else:
            step_key = inputs

        # Drift: -nabla_z Phi(z)
        grad_phi = jax.vmap(jax.grad(lambda zi: model.potential(zi)))(z)
        drift = -grad_phi

        if use_signal:
            psi = model.tilt(signal)
            drift = drift - psi[None, :]

        # Diffusion
        sigma = jax.vmap(model.noise)(z)
        dW = jax.random.normal(step_key, z.shape) * jnp.sqrt(dt)

        new_z = z + drift * dt + sigma * dW

        # Proliferation: dS = R(z) dt
        if use_signal:
            R_vals = jax.vmap(lambda zi: model.proliferation(zi, signal))(z)
        else:
            R_vals = jax.vmap(lambda zi: model.proliferation(zi))(z)
        new_S = S + R_vals * dt

        return (new_z, new_S), None

    xs = (step_keys, signals) if use_signal else step_keys
    (final_z, final_S), _ = jax.lax.scan(scan_body, (z0, S0), xs)

    alpha = _normalize_log_weights(final_S)
    return final_z, final_S, alpha


def simulate_transition(model, z0, n_steps, dt, key, signal=None):
    """
    Short-time transition simulation for multi-timepoint fitting.

    Simulates from z0 for n_steps steps with fresh log-weights (S0=0).
    Used for Howe & Mani style (t_i, X_i) -> (t_{i+1}, X_{i+1}) matching.

    Args:
        model: WaddingtonModel
        z0: (N, d) initial positions (observed snapshot at t_i)
        n_steps: steps in this transition interval
        dt: step size
        key: PRNG key
        signal: (d_signal,) constant signal for this interval, or None

    Returns:
        final_z: (N, d)
        final_S: (N,) raw log-weights (for mass matching loss)
        alpha:   (N,) normalized weights via logsumexp
    """
    N = z0.shape[0]
    S0 = jnp.zeros(N)
    step_keys = jax.random.split(key, n_steps)

    use_signal = signal is not None and model.tilt is not None

    def scan_body(carry, step_key):
        z, S = carry

        grad_phi = jax.vmap(jax.grad(lambda zi: model.potential(zi)))(z)
        drift = -grad_phi

        if use_signal:
            psi = model.tilt(signal)
            drift = drift - psi[None, :]

        sigma = jax.vmap(model.noise)(z)
        dW = jax.random.normal(step_key, z.shape) * jnp.sqrt(dt)
        new_z = z + drift * dt + sigma * dW

        if use_signal:
            R_vals = jax.vmap(lambda zi: model.proliferation(zi, signal))(z)
        else:
            R_vals = jax.vmap(lambda zi: model.proliferation(zi))(z)
        new_S = S + R_vals * dt

        return (new_z, new_S), None

    (final_z, final_S), _ = jax.lax.scan(scan_body, (z0, S0), step_keys)

    alpha = _normalize_log_weights(final_S)
    return final_z, final_S, alpha


def simulate_with_history(model, z0, n_steps, dt, key, signals=None):
    """
    Same as simulate but returns the full trajectory history.

    Additional returns:
        hist_z: (n_steps, N, d) position history
        hist_S: (n_steps, N) log-weight history
    """
    N = z0.shape[0]
    S0 = jnp.zeros(N)
    step_keys = jax.random.split(key, n_steps)

    use_signal = signals is not None and model.tilt is not None

    def scan_body(carry, inputs):
        z, S = carry

        if use_signal:
            step_key, signal = inputs
        else:
            step_key = inputs

        grad_phi = jax.vmap(jax.grad(lambda zi: model.potential(zi)))(z)
        drift = -grad_phi

        if use_signal:
            psi = model.tilt(signal)
            drift = drift - psi[None, :]

        sigma = jax.vmap(model.noise)(z)
        dW = jax.random.normal(step_key, z.shape) * jnp.sqrt(dt)
        new_z = z + drift * dt + sigma * dW

        if use_signal:
            R_vals = jax.vmap(lambda zi: model.proliferation(zi, signal))(z)
        else:
            R_vals = jax.vmap(lambda zi: model.proliferation(zi))(z)
        new_S = S + R_vals * dt

        return (new_z, new_S), (z, S)

    xs = (step_keys, signals) if use_signal else step_keys
    (final_z, final_S), (hist_z, hist_S) = jax.lax.scan(
        scan_body, (z0, S0), xs
    )

    alpha = _normalize_log_weights(final_S)
    return final_z, final_S, alpha, hist_z, hist_S


# ======================================================================
# Source-Death model: open system with particle queue
# ======================================================================

def simulate_open_system(model, z_init, wake_schedule, sigma, n_steps, dt, key):
    """
    Simulate an open system with pre-allocated particle queue.

    Particles start dormant (S=-100). At scheduled steps, batches wake up:
    their S is set to 0 and position is reset to z_init (source location).
    All particles drift under -grad(Phi), diffuse with fixed sigma,
    and lose weight via dS = -gamma(z) dt.

    Args:
        model: SourceDeathModel (potential + death_rate; noise not used,
               sigma is passed explicitly as a fixed constant)
        z_init: (N_total, d) initial positions for all particles
                (source positions with Gaussian noise, pre-sampled)
        wake_schedule: (N_total,) int array — step index at which each
                       particle wakes up. Particles with wake_step > n_steps
                       never activate.
        sigma: float, fixed diffusion coefficient
        n_steps: total integration steps
        dt: step size
        key: PRNG key

    Returns:
        final_z: (N_total, d)
        final_S: (N_total,) raw log-weights
        alpha:   (N_total,) normalized weights (over active particles)
    """
    N_total = z_init.shape[0]
    S0 = jnp.full(N_total, -100.0)  # all dormant
    step_keys = jax.random.split(key, n_steps)

    def scan_body(carry, inputs):
        z, S = carry
        step_key, step_idx = inputs

        # 1. Wake-up: particles whose schedule matches this step
        is_waking = (wake_schedule == step_idx)
        S = jnp.where(is_waking, 0.0, S)
        z = jnp.where(is_waking[:, None], z_init, z)

        # 2. Drift: -grad Phi
        grad_phi = jax.vmap(jax.grad(lambda zi: model.potential(zi)))(z)
        drift = -grad_phi

        # 3. Diffusion with fixed sigma
        dW = jax.random.normal(step_key, z.shape) * jnp.sqrt(dt)
        new_z = z + drift * dt + sigma * dW

        # 4. Death: dS = -gamma(z) dt
        gamma_vals = jax.vmap(model.death_rate)(z)
        new_S = S - gamma_vals * dt

        return (new_z, new_S), None

    step_indices = jnp.arange(n_steps)
    (final_z, final_S), _ = jax.lax.scan(
        scan_body, (z_init, S0), (step_keys, step_indices)
    )

    alpha = _normalize_log_weights(final_S)
    return final_z, final_S, alpha
