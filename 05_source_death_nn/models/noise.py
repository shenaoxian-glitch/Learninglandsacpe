import equinox as eqx
import jax
import jax.numpy as jnp


class NoiseScalar(eqx.Module):
    """
    Isotropic noise parameterized in log-space for guaranteed positivity.
    sigma = exp(log_sigma)
    """
    log_sigma: jnp.ndarray

    def __init__(self, sigma_init=1.0):
        self.log_sigma = jnp.log(jnp.array(sigma_init, dtype=jnp.float32))

    @property
    def sigma(self):
        return jnp.exp(self.log_sigma)

    def __call__(self, z):
        """Returns noise vector of same shape as z, all entries = sigma."""
        return jnp.full_like(z, jnp.exp(self.log_sigma))


class NoiseNN(eqx.Module):
    """
    State-dependent diagonal noise: sigma(z) via MLP.
    Output guaranteed positive via softplus.
    """
    layers: list

    def __init__(self, key, d_latent=2):
        keys = jax.random.split(key, 2)
        self.layers = [
            eqx.nn.Linear(d_latent, 16, key=keys[0]),
            eqx.nn.Linear(16, d_latent, key=keys[1]),
        ]

    def __call__(self, z):
        """z: (d_latent,) -> (d_latent,) positive noise values."""
        x = jax.nn.softplus(self.layers[0](z))
        return jax.nn.softplus(self.layers[1](x))
