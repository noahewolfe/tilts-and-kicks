""" variational inference of 2d pixelpop parameters """

import json
import argparse
from functools import partial

import jax
#jax.config.update('jax_debug_nans', True)
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import h5ify
from jax_tqdm import scan_tqdm

from data import get_posteriors
from data import get_injections

from icar import unravel
from icar import clean_data
from icar import fuse_priors
from icar import build_pixelpop
from icar import log_rate_prior
from icar import rate_likelihood_and_variance

from likelihood import taper

from variational import estimate_convergence
from flows import count_params
from flows import save
from flows import default_spline_flow
from flows import default_flow
from flows import default_triangular_spline_flow

from variational import fit

from util import write_config
from util import save_key

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str)
parser.add_argument('--flow', type=str, default='bnaf')
parser.add_argument('--flow-kwargs', type=json.loads, default=dict())
parser.add_argument('--train-kwargs', type=json.loads, default=dict())
parser.add_argument('--nbins', type=int, help='number of pixelpop bins')
parser.add_argument('--cut', type=int)
parser.add_argument('--parameters', type=json.loads)
parser.add_argument('--maximum-variance', type=float)


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

    return (
        args.outdir,
        args.nbins,
        build_flow,
        args.train_kwargs,
        args.cut,
        args.parameters,
        args.maximum_variance
    )


def get_ignore_keys(parameters_for_pixelpop):
    ignore_keys = []
    if 'log_mass_1' in parameters_for_pixelpop.keys():
        ignore_keys += [
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

    if (
        'cos_tilt_1' in parameters_for_pixelpop.keys()
        and 'cos_tilt_2' in parameters_for_pixelpop.keys()
    ):
        ignore_keys += [
            'mu_spin',
            'sigma_spin',
            'xi_spin'
        ]

    return ignore_keys


def get_model(parameters_for_pixelpop):
    # options: either m_1 and cos_tilt_1 or cos_tilt_1 and cos_tilt_2

    if 'log_mass_1' in parameters_for_pixelpop.keys() and 'cos_tilt_1' in parameters_for_pixelpop.keys() and 'cos_tilt_2' in parameters_for_pixelpop.keys():
        from models import log_powerlaw_redshift

        from pixelpop.models.gwpop_models import PowerlawPlusPeak_MassRatio
        from models import truncnorm
        
        def log_density(dataset, parameters):
            log_p_q = PowerlawPlusPeak_MassRatio(
                dataset,
                slope=parameters['beta'],
                minimum=parameters['mlow_1'],#parameters['mmin'],
                delta_m=parameters['delta_m_1'],#parameters['delta_m']
            )

            log_p_z = log_powerlaw_redshift(dataset, parameters)

            p_a1 = truncnorm(
                dataset['a_1'],
                parameters['mu_chi'],
                parameters['sigma_chi'],
                high=1,
                low=0
            )
            p_a2 = truncnorm(
                dataset['a_2'],
                parameters['mu_chi'],
                parameters['sigma_chi'],
                high=1,
                low=0
            )

            return log_p_m1 + log_p_q + log_p_z + jnp.log(p_a1) + jnp.log(p_a2)
    elif 'log_mass_1' in parameters_for_pixelpop.keys() and 'redshift' in parameters_for_pixelpop.keys():
        raise NotImplementedError('deleted this')
    elif 'cos_tilt_1' in parameters_for_pixelpop.keys() and 'cos_tilt_2' in parameters_for_pixelpop.keys():
        #from models import BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth
        #from models import log_iid_spin_mag_truncnorm
        #from models import log_truncated_powerlaw
        from models import log_powerlaw_redshift

        from pixelpop.models.gwpop_models import BrokenPowerlawPlusTwoPeaks_PrimaryMass
        from pixelpop.models.gwpop_models import PowerlawPlusPeak_MassRatio
        from models import truncnorm

        def log_density(dataset, parameters):
            #if 'lam_2' not in parameters.keys():
            #    if (
            #        'lam_0' in parameters.keys()
            #        and 'lam_1' in parameters.keys()
            #    ):
            #        parameters['lam_2'] = (
            #            1 - parameters['lam_0'] - parameters['lam_1']
            #        )

            log_p_m1 = BrokenPowerlawPlusTwoPeaks_PrimaryMass(
                dataset,
                alpha_1=parameters['alpha_1'],
                alpha_2=parameters['alpha_2'],
                mmin=parameters['mlow_1'],
                break_mass=parameters['break_mass'],
                delta_m_1=parameters['delta_m_1'],
                lam_fractions=(
                    parameters['lam_0'],
                    parameters['lam_1'],
                    parameters['lam_2']
                ),
                mpp_1=parameters['mpp_1'],
                sigpp_1=parameters['sigpp_1'],
                mpp_2=parameters['mpp_2'],
                sigpp_2=parameters['sigpp_2'],
                mmax=300.0,
                gaussian_mass_maximum=100.0
            )

            log_p_q = PowerlawPlusPeak_MassRatio(
                dataset,
                slope=parameters['beta'],
                minimum=parameters['mlow_1'],#parameters['mmin'],
                delta_m=parameters['delta_m_1'],#parameters['delta_m']
            )

            log_p_z = log_powerlaw_redshift(dataset, parameters)

            p_a1 = truncnorm(
                dataset['a_1'],
                parameters['mu_chi'],
                parameters['sigma_chi'],
                high=1,
                low=0
            )
            p_a2 = truncnorm(
                dataset['a_2'],
                parameters['mu_chi'],
                parameters['sigma_chi'],
                high=1,
                low=0
            )

            return log_p_m1 + log_p_q + log_p_z + jnp.log(p_a1) + jnp.log(p_a2)

        """
        def log_density(dataset, parameters):
            if 'lam_2' not in parameters.keys():
                if (
                    'lam_0' in parameters.keys()
                    and 'lam_1' in parameters.keys()
                ):
                    parameters['lam_2'] = (
                        1 - parameters['lam_0'] - parameters['lam_1']
                    )

            log_prob = BrokenPowerlawPlusTwoPeaks_PrimaryMass(
                dataset,
                alpha_1=parameters['alpha_1'],
                alpha_2=parameters['alpha_2'],
                mmin=parameters['mlow_1'],
                break_mass=parameters['break_mass'],
                delta_m_1=parameters['delta_m_1'],
                lam_fractions=(
                    parameters['lam_0'],
                    parameters['lam_1'],
                    parameters['lam_2']
                ),
                mpp_1=parameters['mpp_1'],
                sigpp_1=parameters['sigpp_1'],
                mpp_2=parameters['mpp_2'],
                sigpp_2=parameters['sigpp_2'],
                mmax=300.0,
                gaussian_mass_maximum=100.0
            )

            log_prob += PowerlawPlusPeak_MassRatio(
                dataset,
                slope=parameters['beta'],
                minimum=parameters['mlow_1'],
                delta_m=parameters['delta_m_1']
            )

            log_prob += log_powerlaw_redshift(dataset, parameters)

            log_prob += log_iid_spin_mag_truncnorm(dataset, parameters)

            return log_prob
        """
    else:
        raise ValueError(
            'Bad combination of parameters for pixelpop:'
            f'{parameters_for_pixelpop}'
        )

    return log_density


def set_log_mass_1(data):
    data['prior'] *= data['mass_1']
    data['log_mass_1'] = jnp.log(data.pop('mass_1'))
    return data


key = jax.random.key(1701)

if __name__ == '__main__':
    (
        outdir, nbins, build_flow, train_kwargs, cut, parameters_for_pixelpop,
        maximum_variance
    ) = parse_args()

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
        bin_axes, event_bins, inj_bins, event_ln_dvc, inj_ln_dvc, log_car_prior, log_dV
    ) = build_pixelpop(
        posteriors,
        injections,
        parameters_for_pixelpop,
        nbins
    )

    print(bin_axes)
    print(event_ln_dvc.shape)
    print(inj_ln_dvc.shape)
    print(log_dV.shape)

    log_rate_prior = partial(log_rate_prior, log_car_prior)

    log_density = get_model(parameters_for_pixelpop)
    ignore_keys = get_ignore_keys(parameters_for_pixelpop)

    print('assuming an iid model so using dimension - 1')
    param_keys, bounds, log_prior, fold = fuse_priors(
        prior='./priors/lvk.prior',
        log_rate_prior=log_rate_prior,
        nbins=nbins,
        dimension=dimension - 1,
        ignore_keys=ignore_keys
    )

    def pack(x):
        return fold(unravel(param_keys, nbins, dimension - 1, x))

    taper = partial(taper, maximum_variance)

    ttot = injections.pop('time')

    from icar import log_binned_rates_cond
    log_binned_rates = partial(log_binned_rates_cond, log_dV)

#    test_parameters=dict(
#        log_merger_rate_density=jax.random.normal(jax.random.key(1), (10, 10))
#    )
#    print(log_binned_rates(test_parameters, event_bins, event_ln_dvc))

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
        log_binned_rates=log_binned_rates
    )

    def wrapped_likelihood_and_prior(x):
        """ wrap the likelihood and prior with a pre-ravel """
        parameters = pack(x)
        ln_lkl, variance = rate_likelihood_and_variance(parameters)
        return ln_lkl, variance, log_prior(parameters)

    def log_posterior(x):
        """ log posterior with tapered likelihood """
        ln_lkl, variance, lpr = wrapped_likelihood_and_prior(x)
        return ln_lkl + taper(variance) + lpr

    key, subkey = jax.random.split(key)
    save_key(f'{outdir}/flow_init_key.npy', subkey)
    flow_init = build_flow(subkey, bounds)

    with open(f'{outdir}/num_flow_params.txt', 'w') as f:
        f.write(str(count_params(flow_init)))

    key, subkey = jax.random.split(key)
    flow, loss = fit(
        subkey,
        flow_init,
        log_posterior,
        outdir,
        **train_kwargs
    )

    jnp.save(f'{outdir}/loss.npy', loss)
    print(loss)

    save(f'{outdir}/flow.eqx', flow)

    key, subkey = jax.random.split(key)
    save_key(f'{outdir}/sample_key.npy', subkey)

    samples, log_q = flow.sample_and_log_prob(subkey, (10_000,))

    @scan_tqdm(10_000, desc='log_p')
    def step(carry, x):
        _, parameters = x
        return None, wrapped_likelihood_and_prior(parameters)

    _, (lkl, var, lpr) = jax.lax.scan(
        step, None, (jnp.arange(10_000), samples)
    )
    log_post = lkl + lpr

    samples = jax.vmap(pack)(samples)

    mask = var < maximum_variance

    if jnp.sum(mask) == 0:
        stats = dict()
        print('warning! no samples found below variance cut!')
    else:
        stats = estimate_convergence(log_post[mask], log_q[mask])

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
