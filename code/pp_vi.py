""" variational inference of 2d pixelpop parameters """

import json
import argparse
from functools import partial

import jax
#jax.config.update('jax_debug_nans', True)
jax.config.update('jax_enable_x64', True)
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

    if (
        'log_mass_1' in parameters_for_pixelpop.keys()
        and 'cos_tilt_1' in parameters_for_pixelpop.keys()
    ):
        raise NotImplementedError('m1 and cos_tilt_1 not yet implemented')
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
            lam_tilde_0 = parameters['lam_tilde_0']
            lam_tilde_1 = parameters['lam_tilde_1']
            lam_tilde_2 = parameters['lam_tilde_2']

            norm = lam_tilde_0 + lam_tilde_1 + lam_tilde_2
            parameters['lam0'] = lam_tilde_0 / norm
            parameters['lam1'] = lam_tilde_1 / norm
            parameters['lam2'] = lam_tilde_2 / norm

            #p_m1qzmag = bplm1q_plz_truncnormmag(dataset, parameters)

            log_p_m1 = BrokenPowerlawPlusTwoPeaks_PrimaryMass(
                dataset,
                alpha_1=parameters['alpha1'],
                alpha_2=parameters['alpha2'],
                mmin=parameters['mmin'],
                break_mass=parameters['mbreak'],
                delta_m_1=parameters['delta_m'],
                lam_fractions=[
                    parameters['lam0'],
                    parameters['lam1'],
                    parameters['lam2']
                ],
                mpp_1=parameters['mpp1'],
                sigpp_1=parameters['sigpp1'],
                mpp_2=parameters['mpp2'],
                sigpp_2=parameters['sigpp2'],
            )
            #p_m1 = jnp.exp(log_p_m1)

            log_p_q = PowerlawPlusPeak_MassRatio(
                dataset,
                slope=parameters['beta'],
                minimum=parameters['mmin'],
                delta_m=parameters['delta_m']
            )
            #p_q = jnp.exp(log_p_q)

            log_p_z = log_powerlaw_redshift(dataset, parameters)
            #p_z = jnp.exp(log_p_z)

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

            #p_tilt = full_tilt_model(dataset, parameters)

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

            log_prob = BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth(
                dataset,
                alpha_1=parameters['alpha_1'],
                alpha_2=parameters['alpha_2'],
                mlow_1=parameters['mlow_1'],
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

            if 'log_mass_1' in dataset.keys():
                m1 = jnp.exp(dataset['log_mass_1'])
            else:
                m1 = dataset['mass_1']

            log_prob += log_truncated_powerlaw(
                dataset['mass_ratio'],
                parameters['beta'],
                parameters['mlow_1'] / m1,
                1.0
            )

            log_prob += log_powerlaw_redshift(dataset, parameters)

            log_prob += log_iid_spin_mag_truncnorm(dataset, parameters)

            return log_prob
        """
    else:
        raise ValueError(f'Bad combination of parameters for pixelpop: {parameters_for_pixelpop}')

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
        bin_axes, event_bins, inj_bins, event_ln_dvc, inj_ln_dvc, log_car_prior
    ) = build_pixelpop(
        posteriors,
        injections,
        parameters_for_pixelpop,
        nbins
    )

    print(bin_axes)

    log_rate_prior = partial(log_rate_prior, log_car_prior)

    log_density = get_model(parameters_for_pixelpop)
    ignore_keys = get_ignore_keys(parameters_for_pixelpop)

    param_keys, bounds, log_prior = fuse_priors(
        prior='./priors/test.prior',  # TODO: test
        log_rate_prior=log_rate_prior,
        nbins=nbins,
        dimension=dimension,
        ignore_keys=ignore_keys
    )

    unravel = partial(unravel, param_keys, nbins, dimension)
    taper = partial(taper, maximum_variance)

    '''test_parameters = {
        "alpha_1": 1.81,
        "alpha_2": 4.16,
        "beta": 1.78,
        "break_mass": 32.51,
        "delta_m_1": 2.51,
        "lam_0": 0.11,
        "lam_1": 0.88,
        "lam_2": 0.01,
        "lamb": 2.61,
        "mlow_1": 3.25,
        "mmax": 300.0,
        "mpp_1": 9.2,
        "mpp_2": 33.83,
        "mu_chi": 0.1,
        "mu_spin": 0.23,
        "sigma_chi": 0.34,
        "sigma_spin": 0.53,
        "sigpp_1": 0.79,
        "sigpp_2": 2.65,
        "xi_spin": 0.94,
    }'''
    test_parameters = {'alpha1': 4.605874906846779, 'alpha2': 8.835278710477295, 'mbreak': 32.43731335596479, 'mpp1': 14.595596069166115, 'sigpp1': 0.25951732739587285, 'mpp2': 27.705466772233606, 'sigpp2': 4.890859773474901, 'delta_m': 3.657104208444623, 'mmin': 5.06341043483552, 'lam_tilde_0': 0.9548639995981757, 'lam_tilde_1': 0.7619448230244875, 'lam_tilde_2': 0.7284467719182006, 'beta': 0.5176796465274656, 'lamb': 2.1223505768482376, 'mu_chi': 0.7853617729755967, 'sigma_chi': 0.9595484591320099}
    test_parameters['log_merger_rate_density'] = jax.random.normal(
        jax.random.key(18), (nbins, nbins)
    )

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

    print('grad of lnl:', jax.grad(lambda x: rate_likelihood_and_variance(x)[0])(test_parameters))

    def wrapped_likelihood_and_prior(x):
        """ wrap the likelihood and prior with a pre-ravel """
        parameters = unravel(x)
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
