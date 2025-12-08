import jax
import jax.numpy as jnp
import equinox as eqx

import jax_tqdm
import optax
from optax._src.linear_algebra import global_norm

from flowjax.bijections import (
    Affine as AffinePositiveScale,
    Chain,
    Exp,
    Identity,
    Stack,
    Tanh,
)
from flowjax.distributions import Uniform, StandardNormal, Transformed

from flowjax.flows import coupling_flow
from flowjax.flows import block_neural_autoregressive_flow
from flowjax.flows import triangular_spline_flow
from flowjax.bijections import RationalQuadraticSpline

from paramax import NonTrainable
from paramax.wrappers import non_trainable


def bounded_to_reals(x, bounds):
    y = (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    return jax.scipy.special.logit(y)


def Affine(loc=0, scale=1):
    affine = AffinePositiveScale(loc, scale)
    loc, scale = jnp.broadcast_arrays(
        affine.loc, jnp.asarray(scale, dtype=float),
    )
    affine = eqx.tree_at(lambda tree: tree.scale, affine, scale)
    return affine


def Logistic(shape=()):
    loc = jnp.ones(shape) * 0.5
    scale = jnp.ones(shape) * 0.5
    return Chain([Tanh(shape), Affine(loc, scale)])


def UnivariateBounder(bounds=None):
    # no bounds
    if (bounds is None) or all(bound is None for bound in bounds):
        return Identity()

    # bounded on one side
    elif any(bound is None for bound in bounds):
        # bounded on right-hand side
        if bounds[0] is None:
            loc = bounds[1]
            scale = -1
        # bounded on left-hand side
        elif bounds[1] is None:
            loc = bounds[0]
            scale = 1
        return Chain([Exp(), Affine(loc, scale)])

    # bounded on both sides
    else:
        loc = bounds[0]
        scale = bounds[1] - bounds[0]
        return Chain([Logistic(), Affine(loc, scale)])


def Bounder(bounds):
    return Stack(list(map(UnivariateBounder, bounds)))


def bound_from_unbound(flow, bounds=None):
    bounder = Bounder(bounds)

    if all(type(b) is Identity for b in bounder.bijections):
        bijection = flow.bijection
    else:
        bijection = Chain([flow.bijection, non_trainable(bounder)])

    return Transformed(non_trainable(flow.base_dist), bijection)


def bound_from_bound(flow, bounds=None, interval=(0, 1)):
    """ add a bijection that maps from [0, 1] -> [a, b] """
    if (bounds is None) or all(bound is None for bound in bounds):
        bijection = flow.bijection
    else:
        affines = []
        for b in bounds:
            # to go from x\in [x0, x1] to z \in [a, b] we first map from [x0, x1] to y \in [0, 1]
            # y = (x - x0) / (x1 - x0)
            # z = y * (b - a) + a

            # so
            # z = (x - x0) / (x1 - x0) * (b - a) + a
            #   = x * (b - a) / (x1 - x0) - x0 * (b - a) / (x1 - x0) + a

            scale = (b[1] - b[0]) / (interval[1] - interval[0])
            loc = b[0] - interval[0] * scale
            affines.append(Affine(loc=loc, scale=scale))
        _bijection = Stack(affines)
        bijection = Chain([flow.bijection, non_trainable(_bijection)])
    return Transformed(non_trainable(flow.base_dist), bijection)


def default_flow(key, bounds, **kwargs):
    default_kwargs = dict(
        key=key,
        base_dist=StandardNormal(shape=(len(bounds),)),
        invert=False,
        nn_depth=1,
        nn_block_dim=8,
        flow_layers=1,
    )

    for arg in kwargs:
        default_kwargs[arg] = kwargs[arg]

    flow = block_neural_autoregressive_flow(**default_kwargs)

    return bound_from_unbound(flow, bounds)


def default_spline_flow(
    key, bounds, base='normal', init_identity=False, interval=(-5, 5), **kwargs
):
    ndim = len(bounds)

    if base == 'normal':
        base_dist = StandardNormal((ndim,))
    elif base == 'uniform':
        base_dist = Uniform(
            minval=interval[0] * jnp.ones(ndim),
            maxval=interval[1] * jnp.ones(ndim)
        )
    else:
        raise ValueError(f'bad base dist {base}')

    key, subkey = jax.random.split(key)

    default_kwargs = dict(
        key=subkey,
        base_dist=base_dist,
        invert=False,
        nn_depth=1,
        nn_width=50,
        flow_layers=3,
    )

    for arg in kwargs:
        default_kwargs[arg] = kwargs[arg]

    transformer = RationalQuadraticSpline(knots=5, interval=interval)
    flow = coupling_flow(transformer=transformer, **default_kwargs)

    if init_identity:
        # TODO: more elegant way to do all this?
        def get_last_layer(flow):
            conditioner = flow.bijection.bijection.bijections[0].conditioner
            return conditioner.layers[-1]

        eps = 1e-3

        def normal_like(key, x):
            return jax.random.normal(key, x.shape) * eps + eps

        key, subkey = jax.random.split(key)
        flow = eqx.tree_at(
            lambda flow: get_last_layer(flow).weight,
            flow,
            replace_fn=lambda x: normal_like(subkey, x)
        )

        key, subkey = jax.random.split(key)
        flow = eqx.tree_at(
            lambda flow: get_last_layer(flow).bias,
            flow,
            replace_fn=lambda x: normal_like(subkey, x)
        )

    if base == 'normal':
        return bound_from_unbound(flow, bounds)
    elif base == 'uniform':
        # TODO: handling if we give some [a, inf] bounds
        return bound_from_bound(flow, bounds, interval=interval)


def default_triangular_spline_flow(key, bounds, **kwargs):
    default_kwargs = dict(
        key=key,
        base_dist=StandardNormal(shape=(len(bounds),)),
        invert=False,
        flow_layers=1,
        knots=8
    )

    for arg in kwargs:
        default_kwargs[arg] = kwargs[arg]

    flow = triangular_spline_flow(**default_kwargs)

    return bound_from_unbound(flow, bounds)


def default_bernstein_flow(key, bounds, **kwargs):
    from bernsteinbijectors.flowjax import make_coupling_bernstein_flow

    default_kwargs = dict(
        key=key,
        base_dist=Uniform(
            minval=jnp.zeros(len(bounds)),
            maxval=jnp.ones(len(bounds))
        ),
        invert=False,
        nn_depth=1,
        nn_width=50,
        flow_layers=3,
        bernstein_degree=3,
        inverse_solver='bisection'
    )

    for arg in kwargs:
        default_kwargs[arg] = kwargs[arg]

    flow = make_coupling_bernstein_flow(**default_kwargs)
    return bound_from_bound(flow, bounds)




def get_params(model, filter_spec=eqx.is_inexact_array):
    return eqx.filter(
        pytree=model,
        filter_spec=filter_spec,
        is_leaf=lambda leaf: isinstance(leaf, NonTrainable),
    )


def params_to_array(params):
    return jax.flatten_util.ravel_pytree(params)[0]


def count_params(model, filter_spec=eqx.is_inexact_array):
    params = get_params(model, filter_spec)
    return params_to_array(params).size


def partition(model, **kwargs):
    """ wrap kwargs typical when partitioning stuff """
    default_kwargs = dict(
        filter_spec=eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, NonTrainable),
    )
    default_kwargs.update(**kwargs)
    return eqx.partition(pytree=model, **default_kwargs)

def save(filename, model):
    # TODO: maybe redundant and could be accomplished with filter_spec and
    # is_leaf kwargs of serialise ?
    params, _ = partition(model)
    eqx.tree_serialise_leaves(filename, params)


def load(filename, like_model):
    params, static = partition(like_model)
    params = eqx.tree_deserialise_leaves(filename, params)
    return eqx.combine(params, static)