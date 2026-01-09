import argparse

import numpy as np

import jax.numpy as jnp

from numpyro.distributions import Delta

from pixelpop.utils.data import clean_par
from pixelpop.utils.data import convert_m1m2_to_lm1lm2
from pixelpop.models.probabilistic import setup_probabilistic_model
from pixelpop.models.probabilistic import inference_loop

from data import get_posteriors
from data import get_injections

from util import write_config

parser = argparse.ArgumentParser()
parser.add_argument(
    '--parentdir',
    type=str,
    default='../../data/inference/pixelpop/hmc/m1m2t1t2'
)
parser.add_argument('--name', type=str)
parser.add_argument('--marginalize-sigma', action='store_true')
parser.add_argument('--maximum-variance', type=float, default=1)
parser.add_argument('--parallel', type=int, default=1)

mmin = 3
mmax = 300
z_max = 2.3

parameters = ['log_mass_1', 'log_mass_2', 'cos_tilt_1', 'cos_tilt_2']
other_parameters = ['redshift', 'a']


def parse_args():
    args = parser.parse_args()
    parentdir = args.parentdir
    name = args.name
    write_config(args, outdir=f'{parentdir}/{name}')
    return (
        parentdir,
        name,
        args.maximum_variance,
        args.marginalize_sigma,
        args.parallel
    )


def load_data():
    posteriors, _ = get_posteriors(load=True)
    injections = get_injections(load=True)

    posteriors['mass_1'] = posteriors.pop('mass_1_source')

    injections['analysis_time'] = injections.pop('time')
    injections['total_generated'] = injections.pop('total')
    injections['mass_1'] = injections.pop('mass_1_source')

    posteriors['mass_2'] = posteriors.pop('mass_ratio') * posteriors['mass_1']
    injections['mass_2'] = injections.pop('mass_ratio') * injections['mass_1']

    posteriors['prior'] /= posteriors['mass_1']
    injections['prior'] /= injections['mass_1']

    return posteriors, injections


def clean_data(data, min_m=mmin, max_m=mmax, max_z=z_max, remove=False):
    log_mmin = jnp.log(min_m)
    log_mmax = jnp.log(max_m)
    clean_par(data, 'log_mass_1', log_mmin, log_mmax, remove=remove)
    clean_par(data, 'log_mass_2', log_mmin, log_mmax, remove=remove)
    clean_par(data, 'redshift', 0., max_z, remove=remove)


if __name__ == '__main__':
    parentdir, name, maximum_variance, marginalize_sigma, parallel = parse_args()

    posteriors, injections = load_data()

    posteriors = convert_m1m2_to_lm1lm2(posteriors)
    injections = convert_m1m2_to_lm1lm2(injections)

    clean_data(posteriors)
    clean_data(injections, remove=True)

    priors = {
        'max_z': [[z_max], Delta]
    }

    probabilistic_model, initial_value = setup_probabilistic_model(
        posteriors,  # individual GW parameters
        injections,  # injections to estimate selection effects
        parameters,  # parameters to infer with PixelPop ICAR model
        other_parameters,  # nuisance parameters
        [40, 40, 10, 10],  # number of bins along each axis
        minima={
            'log_mass_1': np.log(mmin),
            'log_mass_2': np.log(mmin),
            'cos_tilt_1': -1.0,
            'cos_tilt_2': -1.0,
        },
        maxima={
            'log_mass_1': np.log(mmax),
            'log_mass_2': np.log(mmax),
            'cos_tilt_1': 1.0,
            'cos_tilt_2': 1.0,
        },
        priors=priors,  # priors which differ from defaults
        UncertaintyCut=np.sqrt(maximum_variance),  # convergence criteria for likelihood estimator
        parametric_models={}, # parametric models for nuisance parameters are set to defaults
        length_scales=False, # same ICAR Gaussian coupling strength in all directions
        random_initialization=True, # initialize ICAR model from random Gaussian draw
        lower_triangular=True, # Restrict domain to m1 > m2
        marginalize_sigma=marginalize_sigma
    )

    print_keys = [
        'Nexp',
        'log_likelihood',
        'log_likelihood_variance',
        'lamb',
        'mu_spin',
        'var_spin'
    ]

    if not marginalize_sigma:
        print_keys += ['lnsigma']

    output, mcmc = inference_loop(
        probabilistic_model,
        model_kwargs={'posteriors': posteriors, 'injections': injections},
        initial_value=initial_value,
        warmup=250_000,
        tot_samples=1_000,
        thinning=500,
        pacc=0.45,
        maxtreedepth=5,
        num_samples=10,
        parallel=parallel,
        run_dir=parentdir,
        name=name,
        print_keys=print_keys,
        dense_mass=False
    )

    print('done.')
