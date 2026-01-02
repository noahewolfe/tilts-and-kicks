import jax
import jax.numpy as jnp
import equinox as eqx

import jax_tqdm
import optax
from optax._src.linear_algebra import global_norm

from paramax import NonTrainable


def fit(
    key,
    flow,
    log_target,
    clip=True,
    lr=1e-1,
    steps=1_000,
    batch_size=1,
    final_lr=0
):
    """ Train loop wrapped with some default choices for population inference
        in terms of learning rate, schedule, and optimizer.
    """

    if lr == final_lr:
        # NOTE: small numerics mean that even using same init_value, peak_value,
        # and end_value in warmup_cosine_decay_schedule, we don't get
        # precisely the same learning rate. doesn't matter much in practice.
        learning_rate = lr
    else:
        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value=lr,
            peak_value=lr,
            warmup_steps=0,
            decay_steps=steps,
            end_value=final_lr
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
    return_state=False
):
    params, static = eqx.partition(
        pytree=flow,
        filter_spec=eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, NonTrainable),
    )

    def loss_fn(params, key, step):
        flow = eqx.combine(params, static)
        samples, log_flows = flow.sample_and_log_prob(key, (batch_size,))
        log_targets = jax.vmap(lambda x: log_target(x, step))(samples)

        # debug
        # bad = (~jnp.isfinite(log_flows)).any() | (~jnp.isfinite(log_targets)).any()
        bad_flow = ~jnp.isfinite(log_flows).any()
        bad_target = ~jnp.isfinite(log_targets).any() 

        jax.debug.print("step {} bad_flow={} bad_target={} log_flows[min,max]={},{} log_t[min,max]={},{}",
                        step, bad_flow, bad_target,
                        jnp.nanmin(log_flows), jnp.nanmax(log_flows),
                        jnp.nanmin(log_targets), jnp.nanmax(log_targets))

        return reverse_kl(log_targets, log_flows)

    state = optimizer.init(params)

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
