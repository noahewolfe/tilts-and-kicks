import os
import json


def plot_corner(xs, fname=None, trim=None, **kwargs):
    import numpy as np
    import corner
    from jax import Array

    """ Make corner plots, bilby-style. """

    if isinstance(xs, dict):
        labels = kwargs.get('labels', list(xs.keys()))
        kwargs['labels'] = labels
        unpacked = np.column_stack([xs[k] for k in xs.keys()])
        return plot_corner(unpacked, fname=fname, trim=trim, **kwargs)

    if isinstance(xs, Array):
        xs = np.array(xs)

    defaults_kwargs = dict(
        bins=50, smooth=0.9,
        title_kwargs=dict(fontsize=16),
        color='#0072C1',
        truth_color='black',
        quantiles=None,  # [0.16, 0.84],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
        plot_density=False,
        plot_datapoints=True,
        fill_contours=True,
        max_n_ticks=3,
        truths=None,
        hist_kwargs=dict(density=True)
    )
    defaults_kwargs.update(kwargs)

    defaults_kwargs['hist_kwargs']['color'] = defaults_kwargs['color']

    if trim is not None:
        fig = corner.corner(xs[:, :trim], **defaults_kwargs) 
    else:
        fig = corner.corner(xs, **defaults_kwargs)

    if fname is not None:
        fig.savefig(fname)

    return fig


def plot_multiple(
    xs_list, colors=None, fname=None, frameon=True, fig=None,
    linestyles=None, linewidths=None, xs_labels_fontsize='large', weights=None, **kwargs
):
    """ Make bilby-style corner plots with multiple data sets. """
    from matplotlib.lines import Line2D

    if colors is None:
        colors = ['#0072C1', '#FF8C00'] + [
            f'C{i}' for i in range(2, len(xs_list))
        ]
    if linestyles is None:
        linestyles = ['-' for _ in range(len(xs_list))]
    if linewidths is None:
        linewidths = [1 for _ in range(len(xs_list))]

    xs_labels = kwargs.pop('xs_labels', None)
    default_kwargs = dict(color=colors[0])
    default_kwargs.update(kwargs)

    default_kwargs['weights'] = None if weights is None else weights[0]
    default_kwargs['contour_kwargs'] = dict(
        linestyles=[linestyles[0]],
        linewidths=[linewidths[0]],
    )

    fig = plot_corner(xs_list[0], fname=None, fig=fig, **default_kwargs)

    for i in range(1, len(xs_list)):
        default_kwargs['color'] = colors[i]
        default_kwargs['weights'] = None if weights is None else weights[i]
        default_kwargs['contour_kwargs'] = dict(
            linestyles=[linestyles[i]],
            linewidths=[linewidths[i]],
        )
        plot_corner(xs_list[i], fname=None, fig=fig, **default_kwargs)

    lines = []
    if xs_labels is not None:
        for color in colors:
            lines.append(Line2D([0], [0], color=color, linestyle='-'))

    fig.legend(
        handles=lines,
        labels=xs_labels,
        frameon=frameon,
        fontsize=xs_labels_fontsize
    )

    if fname is not None:
        fig.savefig(fname)

    return fig


def get_git_revision_short_hash() -> str:
    """ https://stackoverflow.com/a/21901260 """
    import subprocess
    return subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']
    ).decode('ascii').strip()


def compile(func, shape, static=None):
    from time import time
    from jax import jit, ShapeDtypeStruct
    from jax.numpy import float64

    shape = (shape,) if isinstance(shape, int) else shape

    # func = log_prior, likelihood, or posterior ... anything that takes in 
    # a vector of {dim} length, really!
    # dim = number of parameters in pop. model
    hollow = ShapeDtypeStruct(shape, float64)

    if static is None:
        t1 = time()
        f = jit(func).trace(hollow).lower().compile()
        t2 = time()
    else:
        t1 = time()
        f = jit(func, static_argnums=[i for i in range(len(static))])
        f = f.trace(*static, hollow).lower()
        t2 = time()

    return t2 - t1, f


def make_dir_and_subdirs(outdir, subdirs):
    for p in subdirs:
        os.makedirs(f'{outdir}/{p}', exist_ok=True)


def write_config(args, outdir=None):
    outdir = outdir if outdir is not None else args.outdir
    with open(f'{outdir}/config.json', 'w') as f:
        d = dict(**args.__dict__, commit=get_git_revision_short_hash())
        f.write(json.dumps(d, indent=4, sort_keys=False))


def scan(fn, desc=None):
    """ A simple jax.lax.scan wrapper with progress bar. """
    import jax
    import jax.numpy as jnp
    from jax_tqdm import scan_tqdm

    def step(_, d):
        _, x = d
        return None, fn(x)

    def tracked(xs):
        n = len(xs)
        _, ys = jax.lax.scan(
            scan_tqdm(n, desc=desc)(step),
            None,
            (jnp.arange(n), xs)
        )
        return ys

    return tracked


def plot_loss(loss, log=False, zoom=False, outpath=None):
    import numpy as np
    import matplotlib.pyplot as plt

    if not np.isfinite(loss).any():
        print('no valid loss values')
        return None, None

    fig, ax = plt.subplots()
    ax.plot(loss)

    if log:
        ax.set_yscale('log')

    if zoom:
        ax.set_ylim(np.nanmin(loss) - 5, np.nanmin(loss) + 50)

    ax.set_xlabel('step')
    ax.set_ylabel('loss')

    if outpath is not None:
        fig.savefig(outpath)

    return fig, ax


def plot_comparison(
    key, *distributions, trim=None, n=1_000, **kwargs
):
    from jax import vmap
    from jax.random import split

    default_kwargs = dict(xs_labels=['target', 'flow'])
    default_kwargs.update(kwargs)

    samples = []
    for d in distributions:
        key, subkey = split(key)
        s = vmap(d.sample)(split(subkey, n))
        samples.append(s)

    if trim is not None:
        samples = [s[:, :trim] for s in samples]

    return plot_multiple(samples, **default_kwargs)


def logtrapz(log_y, x=None, dx=1.0, axis=-1):
    import math
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp
    """
    Integrate y values that are given in log space.

    Parameters
    ----------
    log_y : array_like
        Natural logarithm of input `n`-dimensional array to integrate.
    x : array_like, optional
        The sample points corresponding to the `log_y` values.
        If `x` is None, the sample point are assumed to be evenly spaced `dx` apart.
        The default is None.
    dx : scalar, optional
        The spacing between the sample points when `x` is None.
        The default is 1.
    axis : int, optional
        The axis along which to integrate.

    Returns
    -------
    float or ndarray
        Natural logarithm of the definite integral of `exp(log_y)`.
        If `log_y` is a 1-dimensional array, then the result is a float.
        If `n` is greater than 1, then the result is an `n`-1 dimensional array.
    """

    # This first part is taken from np.trapz
    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/function_base.py#L4773-L4884

    log_y = jnp.asarray(log_y)
    nd = log_y.ndim

    if x is None:
        d = dx
    else:
        x = jnp.asarray(x)
        if x.ndim == 1:
            d = jnp.diff(x)
            shape = [1] * nd
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = jnp.diff(x, axis=axis)

    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)

    # Now we write the trapezoidal rule in log space

    log_integrand = logsumexp(
        jnp.array([log_y[slice1], log_y[slice2]]),
        axis=0,
        ) + jnp.log(d)
    log_integral = logsumexp(log_integrand, axis=axis) - math.log(2)

    return log_integral


def monotonic_select(thresholds, values, right_closed_left_open=True):
    import jax.numpy as jnp

    thresholds = jnp.asarray(thresholds)
    values = jnp.asarray(values)
    assert values.shape[0] == thresholds.shape[0] + 1

    side = 'right' if right_closed_left_open else 'left'

    def func(x):
        # x can be scalar or array; searchsorted broadcasts over x
        idx = jnp.searchsorted(thresholds, x, side=side)  # in [0, len(thresholds)]
        return values[idx]  # jnp.take handles array idx too
    return func


def calc_chieff(q, a1, a2, ct1, ct2):
    numer = a1 * ct1 + q * a2 * ct2
    denom = 1 + q
    return numer / denom