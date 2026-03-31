import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from simulator.sde_solver import simulate_transition
from simulator.weighted_mmd import weighted_mmd_loss, conservation_loss, gradient_reg_loss


# ======================================================================
# Multi-timepoint short-time transition matching
# Howe & Mani style: for each (t_i, X_i) -> (t_{i+1}, X_{i+1}),
# simulate from X_i, compare against X_{i+1} via weighted MMD.
# ======================================================================

def make_transition_loss(n_steps, dt, lam_cons, lam_reg):
    """
    Build a loss function for a single transition (t_i, X_i) -> (t_{i+1}, X_{i+1}).

    n_steps is static (fixed per transition segment) so JAX can trace.
    """

    @eqx.filter_jit
    def transition_loss(model, X0, X1, key):
        """
        Simulate from X0 for n_steps, compare result against X1.

        Returns:
            total: scalar loss
            aux: (l_mmd, l_cons, l_reg)
        """
        final_z, alpha = simulate_transition(model, X0, n_steps, dt, key)

        l_mmd = weighted_mmd_loss(final_z, alpha, X1)
        l_cons = conservation_loss(model, final_z, alpha)
        l_reg = gradient_reg_loss(model, final_z)

        total = l_mmd + lam_cons * l_cons + lam_reg * l_reg
        return total, (l_mmd, l_cons, l_reg)

    return transition_loss


def make_multi_step(optimizer, transitions, dt, loss_config):
    """
    Build a JIT-compiled training step that sums loss over all transitions.

    Args:
        optimizer: optax optimizer
        transitions: list of dicts with 'X0', 'X1', 'n_steps'
        dt: integration step size
        loss_config: dict with 'lam_cons', 'lam_reg'
    """
    lam_cons = loss_config['lam_cons']
    lam_reg = loss_config['lam_reg']

    # Pre-build a loss fn for each transition segment (different n_steps => different traces)
    unique_steps = {}
    for tr in transitions:
        ns = tr['n_steps']
        if ns not in unique_steps:
            unique_steps[ns] = make_transition_loss(ns, dt, lam_cons, lam_reg)

    # Map each transition to its loss fn
    loss_fns = [unique_steps[tr['n_steps']] for tr in transitions]

    @eqx.filter_jit
    def step(model, opt_state, key):
        def loss_fn(model):
            total_loss = 0.0
            total_mmd = 0.0
            total_cons = 0.0
            total_reg = 0.0

            keys = jax.random.split(key, len(transitions))
            for i, (tr, lfn) in enumerate(zip(transitions, loss_fns)):
                loss_i, (mmd_i, cons_i, reg_i) = lfn(
                    model, tr['X0'], tr['X1'], keys[i]
                )
                total_loss = total_loss + loss_i
                total_mmd = total_mmd + mmd_i
                total_cons = total_cons + cons_i
                total_reg = total_reg + reg_i

            return total_loss, (total_mmd, total_cons, total_reg)

        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, new_opt_state = optimizer.update(grads, opt_state)
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
        transitions: list of dicts {'t0', 't1', 'X0', 'X1', 'n_steps'}
        optimizer: optax optimizer
        dt: integration step size
        loss_config: dict with 'lam_cons', 'lam_reg'
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

    history = {'loss': [], 'mmd': [], 'cons': [], 'reg': []}
    best_loss = float('inf')
    wait = 0
    sim_key = key

    for epoch in range(n_epochs):
        if use_fresh_keys:
            key, sim_key = jax.random.split(key)

        model, opt_state, loss, (l_mmd, l_cons, l_reg) = step_fn(
            model, opt_state, sim_key
        )

        loss_val = float(loss)
        history['loss'].append(loss_val)
        history['mmd'].append(float(l_mmd))
        history['cons'].append(float(l_cons))
        history['reg'].append(float(l_reg))

        if epoch % print_every == 0:
            sigma_str = ""
            if hasattr(model.noise, 'log_sigma'):
                sigma_str = f", sigma={float(jnp.exp(model.noise.log_sigma)):.3f}"
            print(
                f"Epoch {epoch:4d}: Loss={loss_val:.5f} "
                f"(MMD={float(l_mmd):.5f}, Cons={float(l_cons):.5f}, "
                f"Reg={float(l_reg):.5f}){sigma_str}"
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
