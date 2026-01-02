import json
import numpy as np
import argparse

import jax.numpy as jnp

import pixelpop

from data import get_injections
from data import get_posteriors

from priors import bilby_prior_to_pixelpop_prior

from models import log_powerlaw_redshift as log_pl_z
from pixelpop.models.gwpop_models import trunc_gaussian
from pixelpop.models.gwpop_models import PowerlawPlusPeak_MassRatio

from util import write_config

maximum_variance = 1

parser = argparse.ArgumentParser()
parser.add_argument('--parentdir', type=str, default='./runs/pixelpop/hmc')
parser.add_argument('--name', type=str)
parser.add_argument('--warmup', type=int)
parser.add_argument('--tot-samples', type=int)
parser.add_argument('--parallel', type=int)
parser.add_argument('--thinning', type=int)
parser.add_argument(
    '--binned-parameters',
    type=json.loads,
    default=dict()
)
parser.add_argument('--nbins', type=int)


def parse_args():
    args = parser.parse_args()
    parentdir = args.parentdir
    name = args.name
    write_config(args, outdir=f'{parentdir}/{name}')
    return (
        parentdir,
        name,
        args.warmup,
        args.tot_samples,
        args.parallel,
        args.thinning,
        args.nbins,
        args.binned_parameters,
    )


print_keys = ['Nexp', 'log_likelihood', 'log_likelihood_variance', 'lnsigma']


def set_log_mass_1(data):
    data['prior'] *= data['mass_1']
    data['log_mass_1'] = jnp.log(data.pop('mass_1'))
    return data


def log_spin_magnitude(dataset, mu_chi, sigma_chi):
    # TODO: cursed! should use sigma_chi as std. but we set a prior
    # following LVK on `sigma_chi^2`
    log_p_a1 = trunc_gaussian(
        dataset['a_1'],
        mu_chi,
        jnp.sqrt(sigma_chi),
        lower=0,
        upper=1
    )

    log_p_a2 = trunc_gaussian(
        dataset['a_2'],
        mu_chi,
        jnp.sqrt(sigma_chi),
        lower=0,
        upper=1
    )

    return log_p_a1 + log_p_a2


def log_powerlaw_redshift(dataset, lamb):
    return log_pl_z(dataset, dict(lamb=lamb))


def clean_data(data, min_m=3, max_m=150, max_z=1.45, remove=False):
    # TODO: unify w/ parameters_for_pixelpop dict
    if 'log_mass_1' in data.keys():
        data = pixelpop.utils.data.clean_par(
            data, 'log_mass_1', jnp.log(min_m), jnp.log(max_m), remove=remove
        )
        data = pixelpop.utils.data.clean_par(
            data, 'log_mass_2', jnp.log(min_m), jnp.log(max_m), remove=remove
        )
    elif 'mass_1' in data.keys():
        data = pixelpop.utils.data.clean_par(
            data, 'mass_1', min_m, max_m, remove=remove
        )
        data = pixelpop.utils.data.clean_par(
            data, 'mass_2', min_m, max_m, remove=remove
        )

    data = pixelpop.utils.data.clean_par(
        data, 'redshift', 0., max_z, remove=remove
    )
    return data


def get_ignore_keys(parameters):
    ignore_keys = []
    if 'log_mass_1' in parameters.keys():
        ignore_keys += [
            'alpha_1',
            'alpha_2',
            'break_mass',
            'mpp_1',
            'sigpp_1',
            'mpp_2',
            'sigpp_2',
            'lam_0',
            'lam_1',
            'mlow_1'
        ]

    if (
        'cos_tilt_1' in parameters.keys()
        and 'cos_tilt_2' in parameters.keys()
    ):
        ignore_keys += [
            'mu_spin',
            'sigma_spin',
            'xi_spin'
        ]

    return ignore_keys


if __name__ == '__main__':
    (
        parentdir,
        name,
        warmup,
        tot_samples,
        parallel,
        thinning,
        nbins,
        binned_parameters,
    ) = parse_args()

    parameters = list(binned_parameters.keys())

    print('parameters =', parameters)

    other_parameters = ['mass_ratio', 'redshift', 'a']
    other_parameters = [k for k in other_parameters if k not in parameters]

    if 'mass_1' not in parameters and 'log_mass_1' not in parameters:
        raise NotImplementedError('Not pulled in m1 parametric model here')

    posteriors, _ = get_posteriors(load=True)
    injections = get_injections(load=True)

    posteriors['mass_1'] = posteriors.pop('mass_1_source')

    injections['analysis_time'] = injections.pop('time')
    injections['total_generated'] = injections.pop('total')
    injections['mass_1'] = injections.pop('mass_1_source')

    if 'log_mass_1' in binned_parameters.keys():
        posteriors = set_log_mass_1(posteriors)
        injections = set_log_mass_1(injections)

    posteriors['log_prior'] = jnp.log(posteriors.pop('prior'))
    injections['log_prior'] = jnp.log(injections.pop('prior'))

    posteriors = clean_data(
        posteriors,
        max_m=jnp.exp(binned_parameters['log_mass_1'][1]),
        max_z=(
            binned_parameters['redshift'][1]
            if 'redshift' in binned_parameters
            else 1.45
        )
    )
    injections = clean_data(
        injections,
        max_m=jnp.exp(binned_parameters['log_mass_1'][1]),
        max_z=(
            binned_parameters['redshift'][1]
            if 'redshift' in binned_parameters
            else 1.45
        )
    )

    ignore_keys = get_ignore_keys(binned_parameters)

    m1_min = (
        binned_parameters['mass_1'][0]
        if 'mass_1' in parameters
        else jnp.exp(binned_parameters['log_mass_1'][0])
    )

    def log_mass_ratio_given_primary_mass(dataset, beta, delta_m_1):
        return PowerlawPlusPeak_MassRatio(
            dataset,
            slope=beta,
            minimum=m1_min,
            delta_m=delta_m_1,
        )

    priors = bilby_prior_to_pixelpop_prior(
        './priors/pp-m1-t1-t2.prior',
        ignore_keys
    )

    parametric_models = dict(
        mass_ratio=log_mass_ratio_given_primary_mass,
        redshift=log_powerlaw_redshift,
        a=log_spin_magnitude,
    )

    hyperparameters = dict(
        mass_ratio=['beta', 'delta_m_1'],
        redshift=['lamb'],
        a=['mu_chi', 'sigma_chi'],
    )

    for k in parameters:
        if k in list(parametric_models.keys()):
            parametric_models.pop(k)
        if k in list(hyperparameters.keys()):
            hyperparameters.pop(k)

    assert (
        len([
            w for v in hyperparameters.values() for w in v
        ]) == len(priors.keys())
    )

    probabilistic_model, initial_value = \
        pixelpop.models.probabilistic.setup_probabilistic_model(
            posteriors,
            injections,
            parameters,
            other_parameters,
            nbins,
            parametric_models=parametric_models,
            minima={k : v[0] for k, v in binned_parameters.items()},
            maxima={k : v[1] for k, v in binned_parameters.items()},
            priors=priors,
            UncertaintyCut=np.sqrt(maximum_variance),
            hyperparameters=hyperparameters,
            length_scales=False,
            random_initialization=True,
            iid_tilt=True
        )

    print_keys += list(priors.keys())

    output, mcmc = pixelpop.models.probabilistic.inference_loop(
        probabilistic_model,
        model_kwargs={'posteriors': posteriors, 'injections': injections},
        initial_value=initial_value,
        warmup=warmup,
        tot_samples=tot_samples,
        thinning=thinning,
        pacc=0.45,
        maxtreedepth=5,
        num_samples=10,
        run_dir=parentdir,
        name=name,
        print_keys=print_keys,
        dense_mass=False,
        parallel=parallel,
    )

    print('done.')
