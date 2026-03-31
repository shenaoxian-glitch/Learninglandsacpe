import jax
import jax.numpy as jnp


def gaussian_kernel_multi(a, b, bandwidths=(0.01, 1.0, 100.0)):
    """Multi-bandwidth Gaussian kernel matrix."""
    diff = a[:, None, :] - b[None, :, :]
    dist_sq = jnp.sum(diff ** 2, axis=-1)
    K = jnp.zeros_like(dist_sq)
    for bw in bandwidths:
        K = K + jnp.exp(-dist_sq / (2 * bw))
    return K


def weighted_mmd_loss(x_sim, alpha, x_obs):
    """
    Weighted MMD^2 between weighted simulated and uniformly-weighted observed data.

    L = sum_ij ai aj K(xi,xj) + (1/M^2) sum_ab K(ya,yb) - (2/M) sum_ia ai K(xi,ya)

    Args:
        x_sim: (N, d) simulated positions
        alpha: (N,) normalized weights (sum = 1)
        x_obs: (M, d) observed positions (uniform 1/M)
    """
    M = x_obs.shape[0]

    K_xx = gaussian_kernel_multi(x_sim, x_sim)
    K_yy = gaussian_kernel_multi(x_obs, x_obs)
    K_xy = gaussian_kernel_multi(x_sim, x_obs)

    term_xx = jnp.sum(alpha[:, None] * alpha[None, :] * K_xx)
    term_yy = jnp.mean(K_yy)
    term_xy = jnp.sum(alpha[:, None] * K_xy) / M

    return term_xx + term_yy - 2 * term_xy


def conservation_loss(model, z, alpha, signal=None):
    """
    Population conservation penalty: L_cons = |int R(x) rho(x) dx|^2

    Approximated as |sum_i alpha_i R(z_i)|^2.
    Penalizes net population growth/decay, encouraging steady state.
    """
    if signal is not None and model.proliferation.d_signal > 0:
        R_vals = jax.vmap(lambda zi: model.proliferation(zi, signal))(z)
    else:
        R_vals = jax.vmap(lambda zi: model.proliferation(zi))(z)
    return (jnp.sum(alpha * R_vals)) ** 2


def gradient_reg_loss(model, z, signal=None):
    """
    L1 (Lasso) gradient regularization on R: (1/N) sum_i ||nabla_z R(z_i)||_1

    Encourages sparsity in the proliferation gradient field.
    """
    if signal is not None and model.proliferation.d_signal > 0:
        grad_fn = jax.vmap(jax.grad(lambda zi: model.proliferation(zi, signal)))
    else:
        grad_fn = jax.vmap(jax.grad(lambda zi: model.proliferation(zi)))
    grad_R = grad_fn(z)
    return jnp.mean(jnp.sum(jnp.abs(grad_R), axis=-1))


def total_loss(model, x_sim, alpha, x_obs, z_for_reg,
               lam_cons=0.1, lam_reg=0.01, signal=None):
    """
    Combined loss:
        L = L_MMD(weighted) + lam_cons * L_conservation + lam_reg * ||nabla R||_1

    Returns:
        total: scalar loss
        aux: (l_mmd, l_cons, l_reg) for logging
    """
    l_mmd = weighted_mmd_loss(x_sim, alpha, x_obs)
    l_cons = conservation_loss(model, z_for_reg, alpha, signal)
    l_reg = gradient_reg_loss(model, z_for_reg, signal)
    return l_mmd + lam_cons * l_cons + lam_reg * l_reg, (l_mmd, l_cons, l_reg)
