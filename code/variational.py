import jax
import jax.numpy as jnp
import equinox as eqx

import jax_tqdm
import optax
from optax._src.linear_algebra import global_norm

from paramax import NonTrainable


def fit(key, flow, log_target, clip=True, lr=1e-1, steps=1_000, batch_size=1):
    """ Train loop wrapped with some default choices for population inference
        in terms of learning rate, schedule, and optimizer.
    """
    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=lr,
        peak_value=lr,
        warmup_steps=0,
        decay_steps=steps,
        end_value=0
    )

    if clip:
        optimizer = optax.chain(
            optax.clip_by_global_norm(1),
            optax.adam(learning_rate=learning_rate)
        )
    else:
        optimizer = optax.adam(learning_rate=learning_rate) 

    flow, losses = train(
        key,
        flow,
        lambda x, _ : log_target(x),
        steps,
        optimizer,
        batch_size=batch_size
    )
    return flow, losses


def save(filename, model):
    # TODO: maybe redundant and could be accomplished with filter_spec and
    # is_leaf kwargs of serialise ?
    params, _ = eqx.partition(
        pytree=model,
        filter_spec=eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, NonTrainable),
    )
    eqx.tree_serialise_leaves(filename, params)


def load(filename, like_model):
    params, static = eqx.partition(
        pytree=like_model,
        filter_spec=eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, NonTrainable),
    )
    params = eqx.tree_deserialise_leaves(filename, params)
    return eqx.combine(params, static)


def reverse_kl(log_p, log_q):
    return jnp.mean(log_q - log_p)


def estimate_convergence(log_p, log_q):
    import numpy as np
    from arviz import psislw

    n = len(log_p)
    log_weights = np.array(log_p - log_q)
    log_evidence = jax.scipy.special.logsumexp(log_weights) - jnp.log(n)

    log_norm_weights = log_weights - log_evidence - jnp.log(n)
    neff = jnp.exp(-jax.scipy.special.logsumexp(2 * log_norm_weights))

    var_log_evidence = 1 / neff - 1 / n

    smoothed_log_weights, kss = psislw(log_weights, normalize=False)
    eff = neff / n

    return dict(
        smoothed_log_weights=smoothed_log_weights,
        log_weights=log_norm_weights,
        log_evidence=log_evidence,
        log_evidence_variance=var_log_evidence,
        eff=eff,
        kss=kss
    )


def sample_and_log_prob(key, flow, n=10_000, batch_size=10_000):
    keys = jax.random.split(key, n)
    return jax.lax.map(flow.sample_and_log_prob, keys, batch_size=batch_size)


def train(
    key,
    flow,
    log_target,
    steps,
    optimizer=None,
    batch_size=1,
    map_batch_size='all',
    reject_non_finite=False,
    state=None,
    return_state=False
):
    params, static = eqx.partition(
        pytree=flow,
        filter_spec=eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, NonTrainable),
    )

    """if map_batch_size == 'all':
        def batch_log_target(step):
            return jax.vmap(lambda x: log_target(x, step))
    else:
        def batch_log_target(step):
            return lambda xs: jax.lax.map(
                lambda x: log_target(x, step),
                xs,
                batch_size=map_batch_size
            )
    """

    if map_batch_size == 'all':
        def loss_fn(params, key, step):
            flow = eqx.combine(params, static)
            samples, log_flows = flow.sample_and_log_prob(key, (batch_size,))
            log_targets = jax.vmap(lambda x: log_target(x, step))(samples)
            return reverse_kl(log_targets, log_flows)
    else:
        def loss_fn(params, key, step):
            flow = eqx.combine(params, static)
            samples, log_flows = flow.sample_and_log_prob(key, (batch_size,))
            log_targets = jax.lax.map(
                lambda x: log_target(x, step),
                samples,
                batch_size=map_batch_size
            )
            return reverse_kl(log_targets, log_flows)

    if state is None:
        state = optimizer.init(params)

    if reject_non_finite:
        @jax_tqdm.scan_tqdm(steps, desc='train')
        @eqx.filter_jit
        def update(carry, step):
            key, params, state = carry
            key, _key = jax.random.split(key)
            loss, grad = eqx.filter_value_and_grad(loss_fn)(params, _key)

            def do(grad, params, state):
                updates, state = optimizer.update(grad, state, params)
                params = eqx.apply_updates(params, updates)
                return (params, state)

            grad_norm = global_norm(grad)

            (params, state) = jax.lax.cond(
                jnp.isfinite(loss) & jnp.isfinite(grad_norm),
                do,
                lambda grad, params, state: (params, state),
                grad, params, state
            )

            return (key, params, state), (loss, grad_norm)
    else:
        @jax_tqdm.scan_tqdm(steps, desc='train')
        @eqx.filter_jit
        def update(carry, step):
            key, params, state = carry
            key, _key = jax.random.split(key)
            loss, grad = eqx.filter_value_and_grad(loss_fn)(params, _key, step)
            updates, state = optimizer.update(grad, state, params)
            params = eqx.apply_updates(params, updates)
            return (key, params, state), loss

    (key, params, state), losses = jax.lax.scan(
        update, (key, params, state), jnp.arange(steps),
    )
    flow = eqx.combine(params, static)

    if return_state:
        return flow, losses, state
    else:
        return flow, losses
