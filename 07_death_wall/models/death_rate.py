import equinox as eqx
import jax
import jax.numpy as jnp


class DeathRateNN(eqx.Module):
    """
    Neural network death rate: gamma_phi(z) >= 0

    Architecture: d_input -> hidden_sizes -> 1 -> softplus
    Hidden activation: softplus
    Output: softplus(raw) -- guarantees non-negativity

    When y_only=True, d_input=1 and only z[1] (the y coordinate) is used.
    This structurally eliminates x-direction degeneracy: the optimizer
    cannot use asymmetric death to compensate for potential errors,
    forcing the potential to learn correct lateral gradients.

    Used in the Feynman-Kac weight equation: dS = -gamma(z) dt
    Since gamma >= 0, log-weights S are strictly non-increasing,
    and effective mass w = exp(S) in (0, 1].
    """
    layers: list
    y_only: bool = eqx.field(static=True)

    def __init__(self, key, d_latent=2, y_only=False, hidden_sizes=(16, 16)):
        self.y_only = y_only
        d_input = 1 if y_only else d_latent
        sizes = [d_input] + list(hidden_sizes) + [1]
        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = [
            eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i])
            for i in range(len(sizes) - 1)
        ]

    def __call__(self, z):
        """
        z: (d_latent,)
        Returns: scalar death rate gamma >= 0
        """
        x = z[1:2] if self.y_only else z
        for layer in self.layers[:-1]:
            x = jax.nn.softplus(layer(x))
        raw = jnp.squeeze(self.layers[-1](x))
        return jax.nn.softplus(raw)


class DeathRateParametric(eqx.Module):
    """
    Parametric death rate: gamma(z) = softplus(k * (y - y_c))

    Only 2 learnable parameters:
        y_c: threshold coordinate (death activates above y_c)
        log_k: log of steepness (k = exp(log_k) > 0)

    Ground truth: softplus(y - 2.2) => y_c=2.2, k=1.0
    """
    y_c: jnp.ndarray    # scalar, learnable threshold
    log_k: jnp.ndarray  # scalar, learnable log-steepness

    def __init__(self, y_c_init=0.0, k_init=1.0):
        self.y_c = jnp.array(y_c_init)
        self.log_k = jnp.array(jnp.log(k_init))

    def __call__(self, z):
        """
        z: (d_latent,) where z[1] = y coordinate
        Returns: scalar death rate gamma >= 0
        """
        y = z[1]
        k = jnp.exp(self.log_k)
        return jax.nn.softplus(k * (y - self.y_c))


class ExponentialDeathRate(eqx.Module):
    """
    Sharp exponential death wall: gamma(y) = A * exp(k * (y - y_d))

    Capped at gamma_max to prevent float32 overflow at large y.
    Once gamma > gamma_max the particle is already dead; the cap
    prevents exp -> Inf -> NaN gradients without changing physics.

    All parameters are static (not learnable).
    """
    A: float = eqx.field(static=True)
    k: float = eqx.field(static=True)
    y_d: float = eqx.field(static=True)
    gamma_max: float = eqx.field(static=True)

    def __call__(self, z):
        y = z[1]
        raw = self.A * jnp.exp(self.k * (y - self.y_d))
        return jnp.minimum(raw, self.gamma_max)
