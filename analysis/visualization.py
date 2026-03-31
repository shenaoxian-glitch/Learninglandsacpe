import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def _eval_grid(fn, xlim, ylim, n_grid):
    """Evaluate a scalar function on a 2-D meshgrid."""
    xs = jnp.linspace(*xlim, n_grid)
    ys = jnp.linspace(*ylim, n_grid)
    X, Y = jnp.meshgrid(xs, ys)
    flat = jnp.column_stack([X.ravel(), Y.ravel()])
    Z = jax.vmap(fn)(flat).reshape(X.shape)
    return np.asarray(X), np.asarray(Y), np.asarray(Z)


# ------------------------------------------------------------------
# Landscape
# ------------------------------------------------------------------

def plot_nn_landscape(model, xlim=(-5, 5), ylim=(-2, 4), n_grid=200,
                      ax=None, title="Learned Potential"):
    """Contour plot of the NN potential."""
    X, Y, Z = _eval_grid(model.potential, xlim, ylim, n_grid)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(X, Y, Z, levels=30, cmap=cm.viridis)
    plt.colorbar(cf, ax=ax, label=r'$\Phi(z)$')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(title)
    return ax


def plot_nn_landscape_3d(model, xlim=(-5, 5), ylim=(-2, 4), n_grid=200,
                         title="Learned Potential (3D)"):
    """3-D surface plot of the NN potential."""
    X, Y, Z = _eval_grid(model.potential, xlim, ylim, n_grid)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.9, edgecolor='none')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel(r'$\Phi$')
    ax.set_title(title)
    ax.view_init(elev=30, azim=160)
    return ax


# ------------------------------------------------------------------
# Proliferation field
# ------------------------------------------------------------------

def plot_nn_proliferation(model, xlim=(-5, 5), ylim=(-2, 4), n_grid=200,
                          ax=None, title="Learned R(z)", vmin=None, vmax=None):
    """Contour plot of the learned proliferation rate."""
    X, Y, Z = _eval_grid(lambda z: model.proliferation(z), xlim, ylim, n_grid)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    if vmin is None:
        vmin = Z.min()
    if vmax is None:
        vmax = Z.max()
    levels = np.linspace(vmin, vmax, 31)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap='RdBu_r', extend='both')
    plt.colorbar(cf, ax=ax, label='R(z)')
    ax.contour(X, Y, Z, levels=[0], colors='black', linewidths=1.5, linestyles='--')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(title)
    return ax


# ------------------------------------------------------------------
# Particles
# ------------------------------------------------------------------

def plot_weighted_particles(positions, alpha, model=None,
                            xlim=(-5, 5), ylim=(-2, 4),
                            title="Weighted Particles"):
    """Scatter plot of weighted particles, optionally on top of the potential."""
    fig, ax = plt.subplots(figsize=(8, 6))

    if model is not None:
        X, Y, Z = _eval_grid(model.potential, xlim, ylim, 200)
        ax.contourf(X, Y, Z, levels=25, cmap='viridis', alpha=0.5)

    sc = ax.scatter(
        np.asarray(positions[:, 0]),
        np.asarray(positions[:, 1]),
        c=np.asarray(alpha), cmap='hot_r', s=15, alpha=0.8, edgecolors='none',
    )
    plt.colorbar(sc, ax=ax, label=r'weight $\alpha$')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    return ax


# ------------------------------------------------------------------
# Training curves
# ------------------------------------------------------------------

def plot_training_history(history, title="Training Loss"):
    """Plot total loss and components over epochs."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(history['loss'], label='Total')
    axes[0].set_yscale('log'); axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch'); axes[0].legend()

    axes[1].plot(history['mmd'], label='MMD', color='tab:blue')
    axes[1].set_yscale('log'); axes[1].set_title('MMD Loss')
    axes[1].set_xlabel('Epoch'); axes[1].legend()

    axes[2].plot(history['cons'], label='Conservation', color='tab:orange')
    axes[2].plot(history['reg'], label='Grad Reg', color='tab:green')
    axes[2].set_yscale('log'); axes[2].set_title('Regularisers')
    axes[2].set_xlabel('Epoch'); axes[2].legend()

    fig.suptitle(title)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# Side-by-side comparison (target vs learned)
# ------------------------------------------------------------------

def plot_comparison(target_pos, target_alpha, sim_pos, sim_alpha,
                    model=None, xlim=(-5, 5), ylim=(-2, 4)):
    """Side-by-side scatter: target data vs learned simulation with shared colorscale."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Compute shared color limits across both datasets
    all_alpha = np.concatenate([np.asarray(target_alpha), np.asarray(sim_alpha)])
    vmin, vmax = float(all_alpha.min()), float(all_alpha.max())

    for ax, pos, alpha, label in [
        (axes[0], target_pos, target_alpha, "Target (Ground Truth)"),
        (axes[1], sim_pos, sim_alpha, "Learned Model"),
    ]:
        if model is not None:
            X, Y, Z = _eval_grid(model.potential, xlim, ylim, 200)
            ax.contourf(X, Y, Z, levels=25, cmap='viridis', alpha=0.5)

        sc = ax.scatter(
            np.asarray(pos[:, 0]), np.asarray(pos[:, 1]),
            c=np.asarray(alpha), cmap='hot_r', s=15, alpha=0.8, edgecolors='none',
            vmin=vmin, vmax=vmax,
        )
        plt.colorbar(sc, ax=ax, label=r'$\alpha$')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(label)
        ax.set_xlim(xlim); ax.set_ylim(ylim)

    plt.tight_layout()
    return fig
