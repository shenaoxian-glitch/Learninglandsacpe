import jax
import jax.numpy as jnp


def find_equilibrium(potential_fn, z_init, lr=0.01, n_steps=2000, tol=1e-8):
    """
    Find an equilibrium of `potential_fn` by minimising ||nabla Phi||^2
    via gradient descent.

    Args:
        potential_fn: callable(z) -> scalar
        z_init: (d,) starting point
        lr: learning rate
        n_steps: max iterations
        tol: convergence threshold on ||nabla Phi||^2

    Returns:
        z_eq: (d,) equilibrium position
    """
    grad_norm_sq = lambda z: jnp.sum(jax.grad(potential_fn)(z) ** 2)

    def body(carry, _):
        z, converged = carry
        g = jax.grad(grad_norm_sq)(z)
        new_z = jnp.where(converged, z, z - lr * g)
        new_converged = converged | (grad_norm_sq(new_z) < tol)
        return (new_z, new_converged), None

    (z_eq, _), _ = jax.lax.scan(body, (z_init, False), jnp.arange(n_steps))
    return z_eq


def classify_equilibrium(potential_fn, z_eq):
    """
    Classify an equilibrium by the eigenvalues of the Hessian.

    Returns: 'stable', 'unstable', or 'saddle'
    """
    H = jax.hessian(potential_fn)(z_eq)
    eigs = jnp.linalg.eigvalsh(H)

    if jnp.all(eigs > 0):
        return 'stable'
    elif jnp.all(eigs < 0):
        return 'unstable'
    return 'saddle'


def trace_equilibria(potential_fn_factory, z_init, param_values):
    """
    Pseudo-arclength continuation: track equilibria as a scalar parameter varies.

    Args:
        potential_fn_factory: callable(param_val) -> potential_fn(z)
        z_init: (d,) starting equilibrium at param_values[0]
        param_values: 1-D array of parameter values to sweep

    Returns:
        equilibria: (len(param_values), d) array of equilibrium locations
        labels: list of classification strings
    """
    equilibria = []
    labels = []
    z = z_init

    for p in param_values:
        pot_fn = potential_fn_factory(float(p))
        z = find_equilibrium(pot_fn, z)
        equilibria.append(z)
        labels.append(classify_equilibrium(pot_fn, z))

    return jnp.stack(equilibria), labels
