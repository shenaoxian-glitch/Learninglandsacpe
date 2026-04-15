import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import optax

# =============================================================================
# PARAMETERS
# =============================================================================

PARAMS = {
    'a': 0,      # Bifurcation point (y=0)
    'b': -1.6,   # x-scale stability, distance between branches^-1
    'c': 3.5,    # Drive strength (tilt), distance between bifurcation and new stable points
    'd': -1.2,   # Wall steepness
    'sigma': 1   # Noise strength
}

# Proliferation rate parameters R(x, y) = beta * exp(-((x-x0)^2 + (y-y0)^2) / (2*gamma^2)) - delta
PROLIF_PARAMS = {
    'x0': 0.0,     # Center of proliferation zone (at the saddle/stem cell state)
    'y0': -1.0,    # Near initial y position (progenitor zone)
    'beta': 0.5,   # Max proliferation rate
    'gamma': 0.8,  # Width of proliferation zone
    'delta': 0.1   # Basal apoptosis rate
}

# Simulation controls
N_PARTICLES = 1000
T_TOTAL = 2
dt = 0.005
n_steps = int(T_TOTAL / dt)
x0 = 0.0
dx = 0.1
y0 = -1.0
dy = 0.1

# =============================================================================
# POTENTIAL LANDSCAPE
# =============================================================================

def potential(x, y, p):
    """H = -(y - a)x^2 + exp(b)*x^4 - c*y + exp(d)*y^4"""
    return -(y - p['a']) * (x**2) + jnp.exp(p['b']) * (x**4) - p['c'] * y + jnp.exp(p['d']) * (y**4)


force_fn = jax.vmap(jax.grad(potential, argnums=(0, 1)), in_axes=(0, 0, None))


def force(x, y, p):
    """Fx = -dH/dx, Fy = -dH/dy"""
    grad_x, grad_y = force_fn(x, y, p)
    return -grad_x, -grad_y


# =============================================================================
# PROLIFERATION RATE R(x, y)
# =============================================================================

def proliferation_rate(x, y, rp=PROLIF_PARAMS):
    """
    Net proliferation rate: R(x, y) = beta * exp(-((x-x0)^2 + (y-y0)^2) / (2*gamma^2)) - delta

    Positive in the progenitor/stem zone, approaches -delta elsewhere (basal apoptosis).
    Differentiable w.r.t. x, y for end-to-end JAX autodiff.
    """
    dist_sq = (x - rp['x0'])**2 + (y - rp['y0'])**2
    return rp['beta'] * jnp.exp(-dist_sq / (2 * rp['gamma']**2)) - rp['delta']


# =============================================================================
# AUGMENTED SDE: (x, y, S) where S = log-weight
# =============================================================================

@jax.jit
def augmented_scan_step(carry, step_key):
    """
    Augmented state: carry = (px, py, log_weights S)

    Coordinate update (SDE):
        dx = -dH/dx * dt + sigma * dW_x
        dy = -dH/dy * dt + sigma * dW_y

    Log-weight update (ODE, no noise):
        dS = R(x, y) * dt
    """
    px, py, S = carry

    # Deterministic force (drift)
    fx, fy = force(px, py, PARAMS)

    # Stochastic noise (diffusion)
    key_x, key_y = jax.random.split(step_key)
    noise_x = jax.random.normal(key_x, px.shape) * jnp.sqrt(dt)
    noise_y = jax.random.normal(key_y, py.shape) * jnp.sqrt(dt)

    # Update coordinates
    new_px = px + fx * dt + PARAMS['sigma'] * noise_x
    new_py = py + fy * dt + PARAMS['sigma'] * noise_y

    # Update log-weights: dS = R(x, y) * dt
    new_S = S + proliferation_rate(px, py) * dt

    return (new_px, new_py, new_S), (px, py, S)


# =============================================================================
# SIMULATION WITH PROLIFERATION (for parameter inference)
# =============================================================================

def get_sim_data_weighted(params, key, steps):
    """
    Runs the augmented SDE simulation and returns:
        - positions: (N, 2) array of final (x, y)
        - alpha: (N,) normalized weights via softmax(S_final)
    """
    k1, k2, scan_key = jax.random.split(key, 3)
    px = x0 + dx * jax.random.normal(k1, (N_PARTICLES,))
    py = y0 + dy * jax.random.normal(k2, (N_PARTICLES,))
    S = jnp.zeros(N_PARTICLES)  # Initial log-weights: w0 = 1, S0 = 0

    def step_fn(carry, step_key):
        curr_px, curr_py, curr_S = carry
        fx, fy = force(curr_px, curr_py, params)

        kx, ky = jax.random.split(step_key)
        n_x = jax.random.normal(kx, curr_px.shape) * jnp.sqrt(dt)
        n_y = jax.random.normal(ky, curr_py.shape) * jnp.sqrt(dt)

        new_px = curr_px + fx * dt + params['sigma'] * n_x
        new_py = curr_py + fy * dt + params['sigma'] * n_y
        new_S = curr_S + proliferation_rate(curr_px, curr_py) * dt

        return (new_px, new_py, new_S), None

    step_keys = jax.random.split(scan_key, steps)
    (final_px, final_py, final_S), _ = jax.lax.scan(step_fn, (px, py, S), step_keys)

    positions = jnp.column_stack([final_px, final_py])
    alpha = jax.nn.softmax(final_S)  # Numerically stable normalization

    return positions, alpha


get_sim_data_weighted = jax.jit(get_sim_data_weighted, static_argnums=(2,))


# =============================================================================
# WEIGHTED MMD LOSS
# =============================================================================

@jax.jit
def weighted_mmd_loss(x1, alpha, x2):
    """
    Weighted MMD^2 between:
        - simulated particles x1 (N, 2) with weights alpha (N,)
        - observed data x2 (M, 2) with uniform weights 1/M

    L = sum_ij alpha_i * alpha_j * K(x1_i, x1_j)
      + (1/M^2) * sum_ab K(x2_a, x2_b)
      - (2/M) * sum_ia alpha_i * K(x1_i, x2_a)
    """
    M = x2.shape[0]

    def k(a, b):
        """Multi-bandwidth Gaussian kernel sum."""
        diff = a[:, None, :] - b[None, :, :]
        dist_sq = jnp.sum(diff**2, axis=-1)
        return (
            jnp.exp(-dist_sq / (2 * 0.01))
            + jnp.exp(-dist_sq / (2 * 1.0))
            + jnp.exp(-dist_sq / (2 * 100.0))
        )

    K_xx = k(x1, x1)   # (N, N)
    K_yy = k(x2, x2)   # (M, M)
    K_xy = k(x1, x2)   # (N, M)

    # Weighted terms
    term1 = jnp.sum(alpha[:, None] * alpha[None, :] * K_xx)
    term2 = jnp.mean(K_yy)
    term3 = jnp.sum(alpha[:, None] * K_xy) / M

    return term1 + term2 - 2 * term3


# =============================================================================
# OBJECTIVE FUNCTION (end-to-end differentiable)
# =============================================================================

def objective(params, fixed_params, target_data, key):
    """
    Gradient flows through:
        params -> trajectories -> R(x,y) traversed -> log-weights S
        -> alpha = softmax(S) -> Weighted MMD loss
    """
    full_params = {**fixed_params, **params}
    sim_data, alpha = get_sim_data_weighted(full_params, key, n_steps)
    return weighted_mmd_loss(sim_data, alpha, target_data)


loss_and_grad_fn = jax.jit(jax.value_and_grad(objective, argnums=0))


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def plot_landscape(params=PARAMS):
    x_grid = jnp.linspace(-5, 5, 200)
    y_grid = jnp.linspace(-2, 4, 200)
    X, Y = jnp.meshgrid(x_grid, y_grid)
    Z = potential(X, Y, params)

    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.9, edgecolor='none')
    ax1.set_title("2D Landscape")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel(r"$\phi$")
    ax1.view_init(elev=30, azim=160)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.contourf(X, Y, Z, levels=30, cmap=cm.viridis)
    ax2.set_title("Top View")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")

    plt.tight_layout()
    plt.show()


def plot_proliferation_rate(rp=PROLIF_PARAMS):
    x_grid = jnp.linspace(-5, 5, 200)
    y_grid = jnp.linspace(-2, 4, 200)
    X, Y = jnp.meshgrid(x_grid, y_grid)
    R = proliferation_rate(X, Y, rp)

    plt.figure(figsize=(6, 5))
    cf = plt.contourf(X, Y, R, levels=30, cmap='RdBu_r')
    plt.colorbar(cf, label='R(x, y)')
    plt.contour(X, Y, R, levels=[0], colors='black', linewidths=1.5, linestyles='--')
    plt.title("Proliferation Rate R(x, y)\n(dashed = zero crossing)")
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout()
    plt.show()


def plot_weighted_particles(positions, alpha, params=PARAMS, title="Weighted Particles"):
    x_grid = jnp.linspace(-5, 5, 200)
    y_grid = jnp.linspace(-2, 4, 200)
    X, Y = jnp.meshgrid(x_grid, y_grid)
    Z = potential(X, Y, params)

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=25, cmap='viridis', alpha=0.6)
    plt.colorbar(label=r'Potential $\phi$')
    plt.scatter(
        positions[:, 0], positions[:, 1],
        c=alpha, cmap='hot_r', s=20, alpha=0.8,
        edgecolors='none', label='Particles'
    )
    plt.colorbar(label='Weight α')
    plt.axhline(PARAMS['a'], color='white', linestyle='--', label='Bifurcation (y=0)')
    plt.title(title)
    plt.xlabel("x"); plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN: PARAMETER INFERENCE WITH WEIGHTED MMD
# =============================================================================

if __name__ == "__main__":
    # --- Setup ---
    key = jax.random.PRNGKey(4)

    # Plot landscape and proliferation rate
    plot_landscape()
    plot_proliferation_rate()

    # --- Generate Target Data ---
    target_params = PARAMS.copy()
    key, target_key = jax.random.split(key)

    print("Generating target data...")
    target_data, target_alpha = get_sim_data_weighted(target_params, target_key, n_steps)
    print(f"Target data shape: {target_data.shape}, alpha sum: {target_alpha.sum():.4f}")

    plot_weighted_particles(target_data, target_alpha, title=f"Target Distribution (T={T_TOTAL})")

    # --- Optimization ---
    current_params = {'a': 0.9, 'b': -1.2, 'c': 2.1, 'd': -1.4, 'sigma': 0.5}
    fixed_params = {}

    print(f"\nTarget:  a={target_params['a']}, b={target_params['b']:.1f}, "
          f"c={target_params['c']}, d={target_params['d']:.1f}, sigma={target_params['sigma']}")
    print(f"Initial: a={current_params['a']}, b={current_params['b']:.1f}, "
          f"c={current_params['c']}, d={current_params['d']:.1f}, sigma={current_params['sigma']}\n")

    optimizer = optax.adam(learning_rate=0.1)
    opt_state = optimizer.init(current_params)
    opt_key = jax.random.PRNGKey(42)

    for i in range(1000):
        loss, grads = loss_and_grad_fn(current_params, fixed_params, target_data, opt_key)
        updates, opt_state = optimizer.update(grads, opt_state)
        current_params = optax.apply_updates(current_params, updates)

        # Physical constraints
        current_params['c'] = jnp.maximum(current_params['c'], 0.001)
        current_params['sigma'] = jnp.maximum(current_params['sigma'], 0.01)

        if loss < 0.0001:
            print(f"Converged at iteration {i}.")
            break

        if i % 50 == 0:
            print(f"Iter {i:4d}: Loss={float(loss):.5f} | "
                  f"a={float(current_params['a']):.3f}, "
                  f"b={float(current_params['b']):.3f}, "
                  f"c={float(current_params['c']):.3f}, "
                  f"d={float(current_params['d']):.3f}, "
                  f"sigma={float(current_params['sigma']):.3f}")

    print(f"\nFinal Estimated a: {float(current_params['a']):.3f}  (True: {target_params['a']})")
    print(f"Final Estimated b: {float(current_params['b']):.3f}  (True: {target_params['b']})")
    print(f"Final Estimated c: {float(current_params['c']):.3f}  (True: {target_params['c']})")
    print(f"Final Estimated d: {float(current_params['d']):.3f}  (True: {target_params['d']})")
    print(f"Final Estimated sigma: {float(current_params['sigma']):.3f}  (True: {target_params['sigma']})")
