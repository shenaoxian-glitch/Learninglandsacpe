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
                      ax=None, title="Learned Potential", vmin=None, vmax=None):
    """Contour plot of the NN potential."""
    X, Y, Z = _eval_grid(model.potential, xlim, ylim, n_grid)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    if vmin is not None and vmax is not None:
        levels = np.linspace(vmin, vmax, 31)
        cf = ax.contourf(X, Y, Z, levels=levels, cmap=cm.viridis, extend='both')
    else:
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
# Death rate field
# ------------------------------------------------------------------

def plot_nn_death_rate(model, xlim=(-5, 5), ylim=(-2, 4), n_grid=200,
                       ax=None, title=r"Learned $\gamma(z)$", vmin=None, vmax=None):
    """Contour plot of the learned death rate (non-negative)."""
    X, Y, Z = _eval_grid(lambda z: model.death_rate(z), xlim, ylim, n_grid)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    if vmin is None:
        vmin = Z.min()
    if vmax is None:
        vmax = Z.max()
    levels = np.linspace(vmin, vmax, 31)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap='YlOrRd', extend='both')
    plt.colorbar(cf, ax=ax, label=r'$\gamma(z)$')
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
    has_regs = ('mass' in history or 'sparse' in history
                or 'cons' in history or 'reg' in history)
    n_panels = 3 if has_regs else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))

    axes[0].plot(history['loss'], label='Total')
    axes[0].set_yscale('log'); axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch'); axes[0].legend()

    axes[1].plot(history['mmd'], label='MMD', color='tab:blue')
    axes[1].set_yscale('log'); axes[1].set_title('MMD Loss')
    axes[1].set_xlabel('Epoch'); axes[1].legend()

    if has_regs:
        ax2 = axes[2]
        has_mass = 'mass' in history
        has_sparse = 'sparse' in history
        has_cons = 'cons' in history
        has_reg = 'reg' in history

        if has_mass:
            ax2.plot(history['mass'], label='Mass Matching', color='tab:orange')
        elif has_cons:
            ax2.plot(history['cons'], label='Conservation', color='tab:orange')

        if has_sparse:
            ax2.plot(history['sparse'], label='L1 Sparsity', color='tab:green')
        elif has_reg:
            ax2.plot(history['reg'], label='Grad Reg', color='tab:green')

        ax2.set_yscale('log'); ax2.set_title('Regularisers')
        ax2.set_xlabel('Epoch'); ax2.legend()

    fig.suptitle(title)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# Side-by-side comparison (target vs learned)
# ------------------------------------------------------------------

def plot_comparison(target_pos, target_alpha, sim_pos, sim_alpha,
                    model=None, target_potential_fn=None,
                    xlim=(-5, 5), ylim=(-2, 4),
                    weight_label=r'$\alpha$'):
    """
    Side-by-side scatter: target data vs learned simulation with shared colorscale.

    Args:
        target_pos: (N, 2) target positions
        target_alpha: (N,) target weights (or raw w)
        sim_pos: (N, 2) simulated positions
        sim_alpha: (N,) simulated weights (or raw w)
        model: model with .potential (for learned potential background)
        target_potential_fn: callable z->(scalar) for ground-truth potential
        xlim, ylim: plot limits
        weight_label: colorbar label (e.g. r'$w$' or r'$\alpha$')
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Compute shared color limits across both datasets
    all_w = np.concatenate([np.asarray(target_alpha), np.asarray(sim_alpha)])
    vmin, vmax = float(all_w.min()), float(all_w.max())

    # Compute potential grids for both panels first, to establish shared colorscale
    Z_target, Z_learned = None, None
    if target_potential_fn is not None:
        X_t, Y_t, Z_target = _eval_grid(target_potential_fn, xlim, ylim, 200)
    if model is not None:
        X_l, Y_l, Z_learned = _eval_grid(model.potential, xlim, ylim, 200)

    # Shared potential colorscale across both panels
    pot_vmin, pot_vmax = None, None
    if Z_target is not None and Z_learned is not None:
        pot_vmin = min(float(Z_target.min()), float(Z_learned.min()))
        pot_vmax = max(float(Z_target.max()), float(Z_learned.max()))
    elif Z_target is not None:
        pot_vmin, pot_vmax = float(Z_target.min()), float(Z_target.max())
    elif Z_learned is not None:
        pot_vmin, pot_vmax = float(Z_learned.min()), float(Z_learned.max())

    pot_levels = np.linspace(pot_vmin, pot_vmax, 26) if pot_vmin is not None else 25

    # Left panel: target data with ground-truth potential background
    ax = axes[0]
    if Z_target is not None:
        ax.contourf(X_t, Y_t, Z_target, levels=pot_levels, cmap='viridis', alpha=0.5)
    elif Z_learned is not None:
        ax.contourf(X_l, Y_l, Z_learned, levels=pot_levels, cmap='viridis', alpha=0.5)

    sc = ax.scatter(
        np.asarray(target_pos[:, 0]), np.asarray(target_pos[:, 1]),
        c=np.asarray(target_alpha), cmap='hot_r', s=15, alpha=0.8, edgecolors='none',
        vmin=vmin, vmax=vmax,
    )
    plt.colorbar(sc, ax=ax, label=weight_label)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title("Target (Ground Truth)")
    ax.set_xlim(xlim); ax.set_ylim(ylim)

    # Right panel: learned simulation with learned potential background
    ax = axes[1]
    if Z_learned is not None:
        ax.contourf(X_l, Y_l, Z_learned, levels=pot_levels, cmap='viridis', alpha=0.5)

    sc = ax.scatter(
        np.asarray(sim_pos[:, 0]), np.asarray(sim_pos[:, 1]),
        c=np.asarray(sim_alpha), cmap='hot_r', s=15, alpha=0.8, edgecolors='none',
        vmin=vmin, vmax=vmax,
    )
    plt.colorbar(sc, ax=ax, label=weight_label)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title("Learned Model")
    ax.set_xlim(xlim); ax.set_ylim(ylim)

    plt.tight_layout()
    return fig
