import equinox as eqx
import jax
import jax.numpy as jnp


class DeathRateNN(eqx.Module):
    """
    Neural network death rate: gamma_phi(z) >= 0

    Architecture: d_latent -> 16 -> 16 -> 1
    Hidden activation: softplus
    Output: softplus(raw) -- guarantees non-negativity

    Used in the Feynman-Kac weight equation: dS = -gamma(z) dt
    Since gamma >= 0, log-weights S are strictly non-increasing,
    and effective mass w = exp(S) in (0, 1].
    """
    layers: list

    def __init__(self, key, d_latent=2):
        keys = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(d_latent, 16, key=keys[0]),
            eqx.nn.Linear(16, 16, key=keys[1]),
            eqx.nn.Linear(16, 1, key=keys[2]),
        ]

    def __call__(self, z):
        """
        z: (d_latent,)
        Returns: scalar death rate gamma >= 0
        """
        x = z
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
