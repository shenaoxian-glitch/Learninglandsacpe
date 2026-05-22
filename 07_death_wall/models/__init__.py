import jax
import equinox as eqx

from .potential import PotentialNN
from .tilt import TiltLinear
from .proliferation import ProliferationNN
from .death_rate import DeathRateNN, DeathRateParametric
from .noise import NoiseScalar, NoiseNN


class WaddingtonModel(eqx.Module):
    """Top-level model: potential + tilt + proliferation + noise."""
    potential: PotentialNN
    tilt: TiltLinear       # None when no external signal
    proliferation: ProliferationNN
    noise: eqx.Module      # NoiseScalar or NoiseNN


class SourceDeathModel(eqx.Module):
    """Model with potential + death rate. Sigma is fixed externally, not learned."""
    potential: PotentialNN
    death_rate: eqx.Module  # DeathRateNN or DeathRateParametric


def create_model(
    key,
    d_latent=2,
    d_signal=0,
    c_conf=0.01,
    sigma_init=1.0,
    use_tilt=False,
    state_dependent_noise=False,
    beta_max=3.0,
):
    """Factory function to create a WaddingtonModel with all sub-modules."""
    k1, k2, k3, k4 = jax.random.split(key, 4)

    potential = PotentialNN(k1, d_latent=d_latent, c_conf=c_conf)
    tilt = TiltLinear(k2, d_signal=max(d_signal, 1), d_latent=d_latent) if use_tilt else None
    proliferation = ProliferationNN(k3, d_latent=d_latent, d_signal=d_signal, beta_max=beta_max)

    if state_dependent_noise:
        noise = NoiseNN(k4, d_latent=d_latent)
    else:
        noise = NoiseScalar(sigma_init=sigma_init)

    return WaddingtonModel(
        potential=potential, tilt=tilt,
        proliferation=proliferation, noise=noise,
    )


def create_sd_model(key, d_latent=2, c_conf=0.01, parametric_death=False,
                    y_c_init=0.0, k_init=1.0,
                    potential_hidden=(16, 32, 32, 16),
                    death_y_only=False, death_hidden=(16, 16)):
    """Factory function to create a SourceDeathModel (Phi + gamma only).

    Args:
        parametric_death: if True, use DeathRateParametric(y_c, k) instead of NN
        y_c_init: initial threshold for parametric death
        k_init: initial steepness for parametric death
        potential_hidden: hidden layer sizes for PotentialNN
        death_y_only: if True, death rate network only takes y as input
        death_hidden: hidden layer sizes for DeathRateNN
    """
    k1, k2 = jax.random.split(key)
    potential = PotentialNN(k1, d_latent=d_latent, c_conf=c_conf,
                            hidden_sizes=potential_hidden)
    if parametric_death:
        death_rate = DeathRateParametric(y_c_init=y_c_init, k_init=k_init)
    else:
        death_rate = DeathRateNN(k2, d_latent=d_latent, y_only=death_y_only,
                                 hidden_sizes=death_hidden)
    return SourceDeathModel(potential=potential, death_rate=death_rate)
