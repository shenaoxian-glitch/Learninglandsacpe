import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from simulator.sde_solver import simulate_transition
from simulator.weighted_mmd import weighted_mmd_loss, mass_matching_loss, sparsity_reg_loss


# ======================================================================
# Multi-timepoint short-time transition matching
# Howe & Mani style: for each (t_i, X_i) -> (t_{i+1}, X_{i+1}),
# simulate from X_i, compare against X_{i+1} via weighted MMD.
# ======================================================================

def make_transition_loss(n_steps, dt, lam_mass, lam_sparse):
    """
    Build a loss function for a single transition (t_i, X_i) -> (t_{i+1}, X_{i+1}).

    n_steps is static (fixed per transition segment) so JAX can trace.
    """

    @eqx.filter_jit
    def transition_loss(model, X0, X1, log_m_obs, key):
        """
        Simulate from X0 for n_steps, compare result against X1.

        Returns:
            total: scalar loss
            aux: (l_mmd, l_mass, l_sparse)
        """
        final_z, final_S, alpha = simulate_transition(model, X0, n_steps, dt, key)

        l_mmd = weighted_mmd_loss(final_z, alpha, X1)
        l_mass = mass_matching_loss(final_S, log_m_obs)
        l_sparse = sparsity_reg_loss(model, final_z)

        total = l_mmd + lam_mass * l_mass + lam_sparse * l_sparse
        return total, (l_mmd, l_mass, l_sparse)

    return transition_loss


def make_multi_step(optimizer, transitions, dt, loss_config):
    """
    Build a JIT-compiled training step that sums loss over all transitions.

    Args:
        optimizer: optax optimizer
        transitions: list of dicts with 'X0', 'X1', 'n_steps', 'log_m_obs'
        dt: integration step size
        loss_config: dict with 'lam_mass', 'lam_sparse'
    """
    lam_mass = loss_config['lam_mass']
    lam_sparse = loss_config['lam_sparse']

    # Pre-build a loss fn for each transition segment (different n_steps => different traces)
    unique_steps = {}
    for tr in transitions:
        ns = tr['n_steps']
        if ns not in unique_steps:
            unique_steps[ns] = make_transition_loss(ns, dt, lam_mass, lam_sparse)

    # Map each transition to its loss fn
    loss_fns = [unique_steps[tr['n_steps']] for tr in transitions]

    # Extract log_m_obs as JAX arrays (static per transition)
    log_m_obs_list = [jnp.array(tr['log_m_obs']) for tr in transitions]

    @eqx.filter_jit
    def step(model, opt_state, key):
        def loss_fn(model):
            total_loss = 0.0
            total_mmd = 0.0
            total_mass = 0.0
            total_sparse = 0.0

            keys = jax.random.split(key, len(transitions))
            for i, (tr, lfn) in enumerate(zip(transitions, loss_fns)):
                loss_i, (mmd_i, mass_i, sparse_i) = lfn(
                    model, tr['X0'], tr['X1'], log_m_obs_list[i], keys[i]
                )
                total_loss = total_loss + loss_i
                total_mmd = total_mmd + mmd_i
                total_mass = total_mass + mass_i
                total_sparse = total_sparse + sparse_i

            return total_loss, (total_mmd, total_mass, total_sparse)

        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss, aux

    return step


def train(
    model,
    transitions,
    optimizer,
    dt,
    loss_config,
    n_epochs=500,
    key=None,
    use_fresh_keys=False,
    print_every=50,
    patience=100,
):
    """
    Multi-timepoint training with short-time transition matching.

    Args:
        model: WaddingtonModel
        transitions: list of dicts {'t0', 't1', 'X0', 'X1', 'n_steps', 'log_m_obs'}
        optimizer: optax optimizer
        dt: integration step size
        loss_config: dict with 'lam_mass', 'lam_sparse'
        n_epochs: max iterations
        key: PRNG key
        use_fresh_keys: resplit key each epoch
        print_every: logging interval
        patience: early stopping patience

    Returns:
        model: trained WaddingtonModel
        history: dict of loss lists
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    step_fn = make_multi_step(optimizer, transitions, dt, loss_config)

    n_trans = len(transitions)
    interval_str = " + ".join(
        f"[{tr['t0']:.2f}->{tr['t1']:.2f}]({tr['n_steps']})" for tr in transitions
    )
    print(f"Training on {n_trans} transitions: {interval_str}")
    for tr in transitions:
        print(f"  log_m_obs[{tr['t0']:.1f}->{tr['t1']:.1f}] = {tr['log_m_obs']:.4f}")

    history = {'loss': [], 'mmd': [], 'mass': [], 'sparse': []}
    best_loss = float('inf')
    wait = 0
    sim_key = key

    for epoch in range(n_epochs):
        if use_fresh_keys:
            key, sim_key = jax.random.split(key)

        model, opt_state, loss, (l_mmd, l_mass, l_sparse) = step_fn(
            model, opt_state, sim_key
        )

        loss_val = float(loss)
        history['loss'].append(loss_val)
        history['mmd'].append(float(l_mmd))
        history['mass'].append(float(l_mass))
        history['sparse'].append(float(l_sparse))

        if epoch % print_every == 0:
            sigma_str = ""
            if hasattr(model.noise, 'log_sigma'):
                sigma_str = f", sigma={float(jnp.exp(model.noise.log_sigma)):.3f}"
            print(
                f"Epoch {epoch:4d}: Loss={loss_val:.5f} "
                f"(MMD={float(l_mmd):.5f}, Mass={float(l_mass):.5f}, "
                f"Sparse={float(l_sparse):.5f}){sigma_str}"
            )

        # Early stopping
        if loss_val < best_loss - 1e-6:
            best_loss = loss_val
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if loss_val < 1e-4:
            print(f"Converged at epoch {epoch}")
            break

    return model, history
