""" variational inference of 2d pixelpop parameters """

import json
import argparse
from functools import partial

import jax
jax.config.update('jax_debug_nans', True)
import jax.numpy as jnp

import h5ify

from data import get_posteriors
from data import get_injections

from icar import unravel
from icar import clean_data
from icar import fuse_priors
from icar import build_pixelpop
from icar import log_rate_prior
from icar import rate_likelihood_and_variance

from likelihood import taper

from flows import estimate_convergence
from flows import count_params
from flows import save
from flows import default_spline_flow
from flows import default_flow
from flows import default_triangular_spline_flow

from train import fit

from util import write_config
from util import save_key

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str)
parser.add_argument('--flow', type=str, default='bnaf')
parser.add_argument('--flow-kwargs', type=json.loads, default=dict())
parser.add_argument('--train-kwargs', type=json.loads, default=dict())
parser.add_argument('--nbins', type=int, help='number of pixelpop bins')
parser.add_argument('--cut', type=int)

parameters_for_pixelpop = dict(
    log_mass_1=[1.09861228867, 5.703782474656201],  # [3, 300] Msun
    cos_tilt_1=[-1, 1],
    cos_tilt_2=[-1, 1]
)
maximum_variance = 5

taper = partial(taper, maximum_variance)


def parse_args():
    """ parse command-line arguments """
    args = parser.parse_args()
    write_config(args)

    flow_type = args.flow

    train_kwargs = args.train_kwargs
    if 'log10_lr' in train_kwargs.keys():
        train_kwargs['lr'] = 10**(train_kwargs.pop('log10_lr'))
    if 'log10_final_lr' in train_kwargs.keys():
        train_kwargs['final_lr'] = 10**(train_kwargs.pop('log10_final_lr'))

    if flow_type == 'rqs':
        def build_flow(key, bounds):
            return default_spline_flow(
                key, bounds, **args.flow_kwargs
            )
    elif flow_type == 'bnaf':
        def build_flow(key, bounds):
            return default_flow(
                key, bounds, **args.flow_kwargs
            )
    elif flow_type == 'tsf':
        def build_flow(key, bounds):
            return default_triangular_spline_flow(
                key, bounds, **args.flow_kwargs
            )

    return args.outdir, args.nbins, build_flow, args.train_kwargs, args.cut


def log_density(mmin, dataset, parameters):
    """ `log_plq_betamag_igtilt` which fixes mmin; meant to be closed over
        mmin prior to inference
    """
    from models import truncnorm
    from models import log_powerlaw_redshift
    from pixelpop.models.gwpop_models import PowerlawPlusPeak_MassRatio
 
    p_q_given_m1 = PowerlawPlusPeak_MassRatio(
        dataset,
        slope=parameters['beta'],
        minimum=mmin,
        delta_m=parameters['delta_m_1']
    )

    pl_z = log_powerlaw_redshift(dataset, parameters)

    p_a1 = jnp.log(truncnorm(
        dataset['a_1'],
        parameters['mu_chi'],
        parameters['sigma_chi'],
        high=1,
        low=0
    ))
    p_a2 = jnp.log(truncnorm(
        dataset['a_2'],
        parameters['mu_chi'],
        parameters['sigma_chi'],
        high=1,
        low=0
    ))

    return p_q_given_m1 + pl_z + p_a1 + p_a2


def set_log_mass_1(data):
    data['prior'] *= data['mass_1']
    data['log_mass_1'] = jnp.log(data.pop('mass_1'))
    return data



key = jax.random.key(1701)

if __name__ == '__main__':
    outdir, nbins, build_flow, train_kwargs, cut = parse_args()

    dimension = len(parameters_for_pixelpop)

    posteriors, _ = get_posteriors(load=True)
    injections = get_injections(load=True)

    posteriors['mass_1'] = posteriors.pop('mass_1_source')

    injections['total_generated'] = injections.pop('total')
    injections['mass_1'] = injections.pop('mass_1_source')

    if 'log_mass_1' in parameters_for_pixelpop.keys():
        posteriors = set_log_mass_1(posteriors)
        injections = set_log_mass_1(injections)

    posteriors = clean_data(posteriors, max_m=300)
    injections = clean_data(injections, max_m=300)


    ntot = len(posteriors['log_prior'])

    (
        bin_axes, event_bins, inj_bins, event_ln_dvc, inj_ln_dvc, log_car_prior
    ) = build_pixelpop(
        posteriors,
        injections,
        parameters_for_pixelpop,
        nbins
    )

    m1_min = bin_axes[0][0]

    log_density = partial(log_density, m1_min)
    log_rate_prior = partial(log_rate_prior, log_car_prior)

    param_keys, bounds, log_prior = fuse_priors(
        prior='./priors/lvk.prior',
        log_rate_prior=log_rate_prior,
        nbins=nbins,
        dimension=dimension,
        ignore_keys=[
            'alpha_1',
            'alpha_2',
            'break_mass',
            'mpp_1',
            'sigpp_1',
            'mpp_2',
            'sigpp_2',
            'mlow_1',
            'lam_0',
            'lam_1'
        ]
    )

    unravel = partial(unravel, param_keys, nbins, dimension)

    ttot = injections.pop('time')
    rate_likelihood_and_variance = partial(
        rate_likelihood_and_variance,
        ttot,
        posteriors,
        injections,
        log_density,
        event_bins,
        inj_bins,
        event_ln_dvc,
        inj_ln_dvc,
    )

    def wrapped_likelihood_and_prior(x):
        """ wrap the likelihood and prior with a pre-ravel """
        parameters = unravel(x)
        ln_lkl, variance = rate_likelihood_and_variance(parameters)
        return ln_lkl, variance, log_prior(parameters)

    def log_posterior(x):
        """ log posterior with tapered likelihood """
        ln_lkl, variance, lpr = wrapped_likelihood_and_prior(x)
        return ln_lkl + taper(variance) + lpr

    def log_test(x):
        """ log posterior w/o tapered likelihood """
        ln_lkl, _, lpr = wrapped_likelihood_and_prior(x)
        return ln_lkl + lpr

    key, subkey = jax.random.split(key)
    save_key(f'{outdir}/flow_init_key.npy', subkey)
    flow_init = build_flow(subkey, bounds)

    with open(f'{outdir}/num_flow_params.txt', 'w') as f:
        f.write(str(count_params(flow_init)))

    key, subkey = jax.random.split(key)
    flow, metrics = fit(
        subkey,
        flow_init,
        log_posterior,
        outdir,
        log_test=log_test,
        **train_kwargs
    )

    save(f'{outdir}/flow.eqx', flow)

    key, subkey = jax.random.split(key)
    save_key(f'{outdir}/sample_key.npy', subkey)

    samples, log_q = flow.sample_and_log_prob(subkey, (10_000,))

    lkl, var, lpr = jax.lax.map(
        wrapped_likelihood_and_prior, samples, batch_size=1_000
    )
    log_post = lkl + lpr

    samples = jax.vmap(unravel)(samples)

    stats = estimate_convergence(log_post, log_q)

    print(
        f"eff : {stats['eff']}, kss : {stats['kss']}"
    )

    res = dict(
        log_likelihood=lkl,
        variance=var,
        log_posterior=log_post,
        log_q=log_q,
        **stats,
        **samples
    )

    h5ify.save(f'{outdir}/result.h5', res, mode='w')

    print('done.')