""" utils for population models that bin part of the source parameter space and employ an ICAR prior on the log merger rates in each bin (i.e. pixelpop) """
import numpy as np
from astropy.cosmology import Planck15
from astropy import units

from bilby.core.prior import ConditionalPriorDict

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from scipy.stats import spearmanr
import scipy.special as sps

from pixelpop.models.car import initialize_ICAR
from pixelpop.utils.nearest_neighbor import create_CAR_coupling_matrix
from pixelpop.utils.data import place_in_bins
from pixelpop.utils.data import clean_par

from util import logtrapz
from bilby_util import convert_bilby_uniform_prior


def fuse_priors(
    prior,
    log_rate_prior,
    nbins,
    dimension,
    ignore_keys=[],
    outdir=None
):
    """ combine prior on parametric model parameters plus CAR prior
        (jointly with or marginalized over \\sigma)
    """
    prior = ConditionalPriorDict(prior)
    for k in list(prior.keys()):
        if k in ignore_keys:
            prior.pop(k)

    if outdir is not None:
        prior.to_file(outdir, 'init')

    (
        param_keys,
        param_bounds,
        log_param_prior
    ) = convert_bilby_uniform_prior(prior)

    def log_prior(parameters):
        return log_param_prior(parameters) + log_rate_prior(parameters)

    total_nbins = nbins**dimension
    bounds = [list(b) for b in param_bounds] + [None for _ in range(total_nbins)]

    return param_keys, bounds, log_prior


def unravel(param_keys, nbins, dimension, x):
    """ repackage a vector `x` in terms of a dict of parametric model
        parameters and (nbins, nbins) log_merger_rate_density array """
    npar = len(param_keys)
    parameters = dict(zip(param_keys, x[:npar]))
    parameters['log_merger_rate_density'] = x[npar:].reshape(
        *[nbins for _ in range(dimension)]
    )
    return parameters


def get_comoving_factors(posteriors, injections):
    """ If a parametric model for redshift, like PowerlawRedshift, is not
    included, we compute the factors of

    \frac{d V_c}{d z} \frac{1}{1 + z}

    such that the merger rates we compute in the binned part of the model
    are the detector-frame rates.

    TODO: not clearly explained

    stolen from: `pixelpop.models.probabilistic.setup_probabilistic_model`
    TODO: implement in terms of jax, wcosmo, for consistency?
    """
    max_z = np.maximum(
        np.max(injections['redshift']), np.max(posteriors['redshift'])
    )
    zs = np.linspace(1e-6, max_z, 10000)
    dvs = Planck15.differential_comoving_volume(zs) * 4 * np.pi * units.sr
    ln_dvc = np.log(dvs.to(units.Gpc**3).value) - np.log(1 + zs)
    event_z = posteriors['redshift']
    inj_z = injections['redshift']
    event_ln_dvc = jnp.array(np.interp(event_z, zs, ln_dvc))
    inj_ln_dvc = jnp.array(np.interp(inj_z, zs, ln_dvc))
    return event_ln_dvc, inj_ln_dvc


def build_pixelpop(posteriors, injections, parameters, nbins):
    """ return bins, comoving factors, and (log) CAR prior method """
    dimension = len(parameters)
    bins = [nbins] * dimension

    adj_matrices = [
        create_CAR_coupling_matrix(bins[ii], 1, isVisible=False)
        for ii in range(dimension)
    ]

    ICAR_model = initialize_ICAR(dimension, length_scales=False)

    def log_car_prior(parameters):
        log_rates = parameters['log_merger_rate_density']
        dist = ICAR_model(
            log_sigmas=parameters['log_sigma'],
            single_dimension_adj_matrices=adj_matrices,
            is_sparse=True
        )
        return dist.log_prob(log_rates)

    event_bins, inj_bins, bin_axes, _ = place_in_bins(
        list(parameters.keys()),
        posteriors,
        injections,
        bins,
        minima={k : v[0] for k, v in parameters.items()},
        maxima={k : v[1] for k, v in parameters.items()},
    )

    if 'redshift' in parameters.keys():
        event_ln_dvc, inj_ln_dvc = get_comoving_factors(posteriors, injections)
    else:
        event_ln_dvc = jnp.zeros_like(event_bins[0])
        inj_ln_dvc = jnp.zeros_like(inj_bins[0])

    return (
        bin_axes, event_bins, inj_bins, event_ln_dvc, inj_ln_dvc, log_car_prior
    )


def log_rate_prior(log_car_prior, parameters):
    """ log of p(\\ln R) ... marginalized over sigma """
    log_sigma = jnp.linspace(-3, 3, 1_000)

    def fn(log_sigma):
        x = dict(log_sigma=log_sigma, **parameters)
        return log_car_prior(x)

    log_ys = jax.vmap(fn)(log_sigma)
    return logtrapz(log_ys, jnp.exp(log_sigma))


def rate_likelihood_and_variance(
    live_time,
    posteriors,
    injections,
    log_density,
    event_bins,
    inj_bins,
    event_ln_dvc,
    inj_ln_dvc,
    parameters
):
    """ log-likelihood and variance of log-likelihood estimator """
    log_rates = parameters['log_merger_rate_density']

    log_pe_weights = (
        log_rates[event_bins]
        + log_density(posteriors, parameters)
        - posteriors['log_prior']
        + event_ln_dvc
    )
    log_vt_weights = (
        log_rates[inj_bins]
        + log_density(injections, parameters)
        - injections['log_prior']
        + inj_ln_dvc
    )

    event_weights = log_pe_weights
    denominator_weights = log_vt_weights

    ninj = injections['total_generated']

    nobs, npe = event_weights.shape
    numerators = logsumexp(event_weights, axis=1) - jnp.log(npe)
    denominator = logsumexp(denominator_weights) - jnp.log(ninj)

    pe_ln_likelihood = jnp.sum(numerators)

    nexp = live_time * jnp.exp(denominator)
    vt_ln_likelihood = nobs * jnp.log(live_time) - nexp
    ln_likelihood = pe_ln_likelihood + vt_ln_likelihood

    square_sums = logsumexp(2 * event_weights, axis=1) - 2 * jnp.log(npe)
    square_sum = logsumexp(2 * denominator_weights) - 2 * jnp.log(ninj)

    pe_var = jnp.sum(jnp.exp(square_sums - 2 * numerators) - 1 / npe)
    vt_var = live_time**2 * (
        jnp.exp(square_sum) - jnp.exp(2 * denominator) / ninj
    )

    ln_likelihood_variance = pe_var + vt_var

    return ln_likelihood, ln_likelihood_variance


def clean_data(data, min_m=3, max_m=150, max_z=1.45, remove=False):
    """ clean data (pe or vt samples) by setting the log-prior to -jnp.inf
    where those data fall outside [min_m, max_m] and [0, max_z].
    """
    data['log_prior'] = jnp.log(data.pop('prior'))

    if 'log_mass_1' in data.keys():
        min_lnm = jnp.log(min_m)
        max_lnm = jnp.log(max_m)
        data = clean_par(data, 'log_mass_1', min_lnm, max_lnm, remove=remove)
        data = clean_par(data, 'log_mass_2', min_lnm, max_lnm, remove=remove)
    elif 'mass_1' in data.keys():
        data = clean_par(data, 'mass_1', min_m, max_m, remove=remove)
        data = clean_par(data, 'mass_2', min_m, max_m, remove=remove)

    data = clean_par(data, 'redshift', 0., max_z, remove=remove)
    return data


def effective_log_likelihood(scale, parameters):
    """ an effective likelihood to form a proper product distribution with
        the improper CAR prior
    """
    return -jnp.sum(parameters['log_merger_rate_density'])**2 / scale**2


def flat_bin_indices(desired_bins, nbins):
    """
    Only works for 2D pixelpop!
    desired_bins: list of tuples being (i, j) bin indices
    nbins: number of bins along each axis
    """
    return [np.ravel_multi_index(b, (nbins, nbins)) for b in desired_bins]


default_parameters = dict(
    log_mass_1=[1.09861228867, 5.01063529413],
    redshift=[0, 1.45]
)


def get_bin_axes(nbins, parameters=default_parameters):
    """ return just the bin axes """
    _, _, bin_axes, _ = place_in_bins(
        list(parameters.keys()),
        dict(log_mass_1=[], redshift=[]),
        dict(log_mass_1=[], redshift=[]),
        nbins,
        minima={k : v[0] for k, v in parameters.items()},
        maxima={k : v[1] for k, v in parameters.items()},
    )
    return bin_axes


def draw_2d_grid(xbins, ybins, pxy, size=1):
    dx = xbins[1] - xbins[0]
    dy = ybins[1] - ybins[0]
    which = np.random.choice(
        np.arange(pxy.size),
        size=size,
        replace=True,
        p=pxy.reshape(pxy.size)/np.sum(pxy)
    )
    xlocale = np.array(np.floor(which / (len(ybins) - 1)), dtype=int)
    ylocale = which % (len(ybins) - 1)
    xrand = xbins[xlocale] + dx * np.random.uniform(size=size)
    yrand = ybins[ylocale] + dy * np.random.uniform(size=size)
    return xrand, yrand


def Spearman(pxyarr, xarr=None, yarr=None):
    '''
    pxyarr should be in logspace

    \textit{With thanks to Jack Heinzel}
    '''
    if len(pxyarr.shape) != 2:
        print('Wrong shape for the array lol')
        return None

    norm = sps.logsumexp(pxyarr)
    pxyarr -= norm

    cum_xy = np.cumsum(np.cumsum(
        np.insert(np.insert(np.exp(pxyarr), 0, 0., axis=0), 0, 0., axis=1),
        axis=0), axis=1)

    marginal_x = sps.logsumexp(pxyarr, axis=1)
    marginal_y = sps.logsumexp(pxyarr, axis=0)

    cum_x = np.insert(np.cumsum(np.exp(marginal_x)), 0, 0.)
    cum_y = np.insert(np.cumsum(np.exp(marginal_y)), 0, 0.)

    integrand = (cum_xy - np.multiply.outer(cum_x, cum_y))
    integrand = (integrand[1:] + integrand[:-1]) / 2
    integrand = (integrand[:, 1:] + integrand[:, :-1]) / 2
    # integral is accomplished by average between bin edges. This is exact!

    return 12 * np.sum(
        integrand * np.exp(np.add.outer(marginal_x, marginal_y))
    )


def Spearman_Sample(
    pxyarr,
    xarr,
    yarr,
    broadening=False,
    precision=1e6,
    xwindow=[0, 1],
    ywindow=[0, 1]
):
    '''
    pxyarr should be in logspace

    \textit{With thanks to Jack Heinzel}
    '''
    if len(pxyarr.shape) != 2:
        print('Wrong shape for the array lol')
        return None

    norm = sps.logsumexp(pxyarr)
    pxyarr -= norm
    x_min = int(xwindow[0] * len(xarr))
    x_max = int(xwindow[1] * len(xarr))
    y_min = int(ywindow[0] * len(yarr))
    y_max = int(ywindow[1] * len(yarr))
    redpxy = np.exp(pxyarr)[x_min:x_max - 1, y_min:y_max - 1]

    xsample, ysample = draw_2d_grid(
        xarr[x_min:x_max],
        yarr[y_min:y_max],
        redpxy,
        size=int(precision)
    )

    if broadening:
        ymean = np.mean(ysample)
        return spearmanr(xsample, (ysample - ymean)**2)[0]
    else:
        return spearmanr(xsample, ysample)[0]
