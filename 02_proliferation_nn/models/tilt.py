import equinox as eqx


class TiltLinear(eqx.Module):
    """
    Linear tilt: Psi(s) in R^d_latent

    Tilted potential: phi(z; s) = Phi(z) + z^T Psi(s)
    Drift contribution: -Psi(s) (gradient of the linear term)
    """
    linear: eqx.nn.Linear

    def __init__(self, key, d_signal=1, d_latent=2):
        self.linear = eqx.nn.Linear(d_signal, d_latent, key=key)

    def __call__(self, s):
        """s: (d_signal,) -> (d_latent,) tilt vector."""
        return self.linear(s)
