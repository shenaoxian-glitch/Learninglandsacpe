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


def mass_matching_loss(final_S, log_m_obs):
    """
    Mass matching loss: penalizes mismatch between simulated and observed
    absolute mass amplification over a transition interval.

    L_mass = (log_m_sim - log_m_obs)^2

    where log_m_sim = logsumexp(S) - log(N) is the log of the mean weight
    (i.e., average mass amplification factor), and log_m_obs is the ground-truth
    log mass ratio computed from the analytical proliferation.

    This restores the absolute mass information that normalized MMD erases (Q1),
    making R's overall level observable.

    Args:
        final_S: (N,) raw log-weights from SDE integration (NOT normalized)
        log_m_obs: scalar, ground-truth log(mean(exp(S))) for this interval
    """
    N = final_S.shape[0]
    log_m_sim = jax.nn.logsumexp(final_S) - jnp.log(N)
    return (log_m_sim - log_m_obs) ** 2


def sparsity_reg_loss(model, z, signal=None):
    """
    L1 sparsity regularization on R values: (1/N) sum_i |R(z_i)|

    Encourages R to be exactly zero across most of the state space (Q3).
    This is the correct biological prior: proliferation/apoptosis is highly
    localized. Unlike gradient_reg (|nabla R|, i.e. Total Variation), this
    penalizes R being nonzero anywhere, not just R having spatial variation.
    """
    if signal is not None and model.proliferation.d_signal > 0:
        R_vals = jax.vmap(lambda zi: model.proliferation(zi, signal))(z)
    else:
        R_vals = jax.vmap(lambda zi: model.proliferation(zi))(z)
    return jnp.mean(jnp.abs(R_vals))


def total_loss(model, x_sim, alpha, x_obs, z_for_reg, final_S, log_m_obs,
               lam_mass=1.0, lam_sparse=0.01, signal=None):
    """
    Combined loss:
        L = L_MMD(weighted) + lam_mass * L_mass + lam_sparse * L_sparsity

    - L_MMD: weighted MMD between simulated and observed positions
    - L_mass: mass matching loss restoring absolute growth info (Q1 fix)
    - L_sparsity: L1 on R values encouraging spatial sparsity (Q3 fix)

    Conservation loss (Q2) is deliberately removed: it forces E[R]=0,
    which is wrong for systems with net growth/death.

    Returns:
        total: scalar loss
        aux: (l_mmd, l_mass, l_sparse) for logging
    """
    l_mmd = weighted_mmd_loss(x_sim, alpha, x_obs)
    l_mass = mass_matching_loss(final_S, log_m_obs)
    l_sparse = sparsity_reg_loss(model, z_for_reg, signal)
    return l_mmd + lam_mass * l_mass + lam_sparse * l_sparse, (l_mmd, l_mass, l_sparse)
