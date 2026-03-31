import jax
import equinox as eqx

from .potential import PotentialNN
from .tilt import TiltLinear
from .proliferation import ProliferationNN
from .noise import NoiseScalar, NoiseNN


class WaddingtonModel(eqx.Module):
    """Top-level model: potential + tilt + proliferation + noise."""
    potential: PotentialNN
    tilt: TiltLinear       # None when no external signal
    proliferation: ProliferationNN
    noise: eqx.Module      # NoiseScalar or NoiseNN


def create_model(
    key,
    d_latent=2,
    d_signal=0,
    c_conf=0.01,
    sigma_init=1.0,
    use_tilt=False,
    state_dependent_noise=False,
):
    """Factory function to create a WaddingtonModel with all sub-modules."""
    k1, k2, k3, k4 = jax.random.split(key, 4)

    potential = PotentialNN(k1, d_latent=d_latent, c_conf=c_conf)
    tilt = TiltLinear(k2, d_signal=max(d_signal, 1), d_latent=d_latent) if use_tilt else None
    proliferation = ProliferationNN(k3, d_latent=d_latent, d_signal=d_signal)

    if state_dependent_noise:
        noise = NoiseNN(k4, d_latent=d_latent)
    else:
        noise = NoiseScalar(sigma_init=sigma_init)

    return WaddingtonModel(
        potential=potential, tilt=tilt,
        proliferation=proliferation, noise=noise,
    )
