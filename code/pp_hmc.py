import json
import numpy as np
import argparse

import jax
import jax.numpy as jnp

import pixelpop

from data import load_data
from bilby_util import bilby_prior_to_pixelpop_prior

from models import log_truncated_powerlaw
from models import log_powerlaw_redshift as log_pl_z
from models import beta_dist
from models import iso_gauss_spin_tilt

from util import write_config

maximum_variance = 4

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
parser.add_argument('--cut', type=float, default=30)


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
        args.cut
    )


print_keys = ['Nexp', 'log_likelihood', 'log_likelihood_variance', 'lnsigma']


def log_spin_magnitude(dataset, alpha_chi, beta_chi):
    p_a1 = beta_dist(
        dataset['a_1'],
        alpha_chi,
        beta_chi
    )
    p_a2 = beta_dist(
        dataset['a_2'],
        alpha_chi,
        beta_chi
    )

    return jnp.log(p_a1) + jnp.log(p_a2)


def log_iso_gauss_spin_tilt(dataset, xi_spin, sigma_spin):
    return jnp.log(iso_gauss_spin_tilt(dataset, xi_spin, sigma_spin))


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
        cut
    ) = parse_args()

    parameters = list(binned_parameters.keys())
    other_parameters = ['mass_ratio', 'redshift', 'a', 'cos_tilt_2']
    other_parameters = [k for k in other_parameters if k not in parameters]

    if 'mass_1' not in parameters and 'log_mass_1' not in parameters:
        raise NotImplementedError('Not pulled in m1 parametric model here')

    posteriors, injections = load_data(
        catalog='noah',
        cut=cut,
        return_events=False
    )

    if 'log_mass_1' in parameters:
        posteriors['prior'] *= posteriors['mass_1']
        injections['prior'] *= injections['mass_1']
        posteriors['log_mass_1'] = jnp.log(posteriors.pop('mass_1'))
        injections['log_mass_1'] = jnp.log(injections.pop('mass_1'))

    posteriors['log_prior'] = jnp.log(posteriors.pop('prior'))
    injections['log_prior'] = jnp.log(injections.pop('prior'))

    injections['analysis_time'] = 10

    posteriors = clean_data(posteriors, max_z=binned_parameters['redshift'][1])
    injections = clean_data(injections, max_z=binned_parameters['redshift'][1])

    ignore_keys = []
    if 'mass_1' in parameters or 'log_mass_1' in parameters:
        ignore_keys += [
            'mmin', 'mmax', 'alpha', 'lam', 'mpp', 'sigpp', 'delta_m',
            'delta_max'
        ]
    if 'mass_ratio' in parameters:
        ignore_keys += ['beta']
    if 'redshift' in parameters:
        ignore_keys += ['lamb']

    m1_min = (
        binned_parameters['mass_1'][0]
        if 'mass_1' in parameters
        else jnp.exp(binned_parameters['log_mass_1'][0])
    )

    def log_mass_ratio_given_primary_mass(dataset, beta):
        if 'log_mass_1' in parameters:
            mass_1 = jnp.exp(dataset['log_mass_1'])
        else:
            mass_1 = dataset['mass_1']

        qmin = m1_min / mass_1
        qmax = 1.0
        return log_truncated_powerlaw(
            x=dataset['mass_ratio'],
            alpha=beta,
            xmin=qmin,
            xmax=qmax
        )

    priors = bilby_prior_to_pixelpop_prior(
        './priors/wide-bandpass-snr30.prior',
        ignore_keys
    )

    parametric_models = dict(
        mass_ratio=log_mass_ratio_given_primary_mass,
        redshift=log_powerlaw_redshift,
        a=log_spin_magnitude,
        t=log_iso_gauss_spin_tilt
    )

    hyperparameters = dict(
        mass_ratio=['beta'],
        redshift=['lamb'],
        a=['alpha_chi', 'beta_chi'],
        t=['xi_spin', 'sigma_spin']
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
            random_initialization=True
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