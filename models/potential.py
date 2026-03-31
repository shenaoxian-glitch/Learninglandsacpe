import equinox as eqx
import jax
import jax.numpy as jnp


def xavier_linear(in_features, out_features, key):
    """Linear layer with Xavier (Glorot) uniform initialization."""
    layer = eqx.nn.Linear(in_features, out_features, key=key)
    w_key = jax.random.split(key)[0]
    limit = jnp.sqrt(6.0 / (in_features + out_features))
    weight = jax.random.uniform(w_key, (out_features, in_features), minval=-limit, maxval=limit)
    return eqx.tree_at(lambda l: l.weight, layer, weight)


class PotentialNN(eqx.Module):
    """
    Neural network potential with confinement.

    Phi(z) = Phi_nn(z) + c_conf * ||z||^4

    Architecture: d_latent -> 16 -> 32 -> 32 -> 16 -> 1
    Activation: softplus (smooth, consistent with Howe & Mani)
    Initialization: Xavier uniform
    """
    layers: list
    c_conf: float = eqx.field(static=True)

    def __init__(self, key, d_latent=2, c_conf=0.01):
        keys = jax.random.split(key, 5)
        self.layers = [
            xavier_linear(d_latent, 16, keys[0]),
            xavier_linear(16, 32, keys[1]),
            xavier_linear(32, 32, keys[2]),
            xavier_linear(32, 16, keys[3]),
            xavier_linear(16, 1, keys[4]),
        ]
        self.c_conf = c_conf

    def __call__(self, z):
        """z: (d_latent,) -> scalar potential value."""
        x = z
        for layer in self.layers[:-1]:
            x = jax.nn.softplus(layer(x))
        x = self.layers[-1](x)
        confinement = self.c_conf * jnp.sum(z ** 2) ** 2
        return jnp.squeeze(x) + confinement
