import equinox as eqx
import jax
import jax.numpy as jnp


class ProliferationNN(eqx.Module):
    """
    Neural network proliferation rate: R_theta(z; s)

    Architecture: (d_latent + d_signal) -> 16 -> 16 -> 1
    Hidden activation: softplus
    Output: unconstrained (positive = proliferation, negative = apoptosis)
    """
    layers: list
    d_signal: int = eqx.field(static=True)

    def __init__(self, key, d_latent=2, d_signal=0):
        keys = jax.random.split(key, 3)
        d_in = d_latent + d_signal
        self.layers = [
            eqx.nn.Linear(d_in, 16, key=keys[0]),
            eqx.nn.Linear(16, 16, key=keys[1]),
            eqx.nn.Linear(16, 1, key=keys[2]),
        ]
        self.d_signal = d_signal

    def __call__(self, z, s=None):
        """
        z: (d_latent,), s: (d_signal,) or None
        Returns: scalar proliferation rate R
        """
        if self.d_signal > 0 and s is not None:
            x = jnp.concatenate([z, s])
        else:
            x = z
        for layer in self.layers[:-1]:
            x = jax.nn.softplus(layer(x))
        return jnp.squeeze(self.layers[-1](x))
