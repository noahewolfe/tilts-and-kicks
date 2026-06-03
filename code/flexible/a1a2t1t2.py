import argparse

import numpy as np

import jax
import jax.numpy as jnp

from numpyro.distributions import Delta, Dirichlet
import numpyro.distributions as dist

from pixelpop.utils.data import clean_par
from pixelpop.models.probabilistic import setup_probabilistic_model
from pixelpop.models.probabilistic import inference_loop

from data import get_posteriors
from data import get_injections

from util import write_config

from models import BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth
from pixelpop.models.gwpop_models import PowerlawPlusPeak_MassRatio

parser = argparse.ArgumentParser()
parser.add_argument(
    '--parentdir',
    type=str,
    default='../../data/inference/pixelpop/hmc/a1a2t1t2'
)
parser.add_argument('--name', type=str)
parser.add_argument('--marginalize-sigma', action='store_true')
parser.add_argument('--maximum-variance', type=float, default=1)
parser.add_argument('--parallel', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)

mmin = 3
mmax = 300
z_max = 2.3

parameters = ['a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2']
other_parameters = ['mass_1', 'mass_ratio',  'redshift']


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
        args.parallel,
        args.seed
    )


def load_data():
    posteriors, _ = get_posteriors(load=True)
    injections = get_injections(load=True)

    posteriors['mass_1'] = posteriors.pop('mass_1_source')

    injections['analysis_time'] = injections.pop('time')
    injections['total_generated'] = injections.pop('total')
    injections['mass_1'] = injections.pop('mass_1_source')

    return posteriors, injections


def clean_data(data, min_m=mmin, max_m=mmax, max_z=z_max, remove=False):
    clean_par(data, 'mass_1', mmin, mmax, remove=remove)
    clean_par(data, 'mass_ratio', 0.1, 1.0, remove=remove)
    clean_par(data, 'redshift', 0., max_z, remove=remove)


if __name__ == '__main__':
    parentdir, name, maximum_variance, marginalize_sigma, parallel, seed = parse_args()

    posteriors, injections = load_data()

    posteriors['log_prior'] = jnp.log(posteriors['prior'])
    injections['log_prior'] = jnp.log(injections['prior'])
    
    clean_data(posteriors)
    clean_data(injections, remove=True)

    priors = {
        'max_z':        [[z_max],          Delta],
        'alpha_1':      ([-4, 12],         dist.Uniform),
        'alpha_2':      ([-4, 12],         dist.Uniform),
        'mlow_1':       ([2, 10],          dist.Uniform),
        'break_mass':   ([10, 100],        dist.Uniform),
        'delta_m_1':    ([0, 10],          dist.Uniform),
        'lam_fractions': ([jnp.ones(3)],   Dirichlet),
        'mpp_1':        ([20, 60],         dist.Uniform),
        'sigpp_1':      ([1, 10],          dist.Uniform),
        'mpp_2':        ([40, 100],        dist.Uniform),
        'sigpp_2':      ([1, 20],          dist.Uniform),
        'beta':         ([-2, 7],          dist.Uniform),
        'lamb':         ([-2, 10],         dist.Uniform),
    }

    probabilistic_model, initial_value = setup_probabilistic_model(
        posteriors,  # individual GW parameters
        injections,  # injections to estimate selection effects
        parameters,  # parameters to infer with PixelPop ICAR model
        other_parameters,  # nuisance parameters
        [10, 10, 10, 10],  # number of bins along each axis
        minima={
            'a_1': 0.0,
            'a_2': 0.0,
            'cos_tilt_1': -1.0,
            'cos_tilt_2': -1.0,
        },
        maxima={
            'a_1': 1.0,
            'a_2': 1.0,
            'cos_tilt_1': 1.0,
            'cos_tilt_2': 1.0,
        },
        priors=priors,
        UncertaintyCut=np.sqrt(maximum_variance),  # convergence criteria for likelihood estimator
        parametric_models={
            'mass_1':     BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth,
            'mass_ratio': PowerlawPlusPeak_MassRatio,
        },
        hyperparameters={
            'mass_1':     ['alpha_1', 'alpha_2', 'mlow_1', 'break_mass', 'delta_m_1',
                           'lam_fractions', 'mpp_1', 'sigpp_1', 'mpp_2', 'sigpp_2'],
            'mass_ratio': ['beta', 'mlow_1', 'delta_m_1'],
        },
        length_scales=False, # same ICAR Gaussian coupling strength in all directions
        random_initialization=True, # initialize ICAR model from random Gaussian draw
        marginalize_sigma=marginalize_sigma
    )

    print_keys = [
        'Nexp',
        'log_likelihood',
        'log_likelihood_variance',
        'lamb',
        'alpha_1', 'alpha_2', 'mlow_1', 'break_mass', 'delta_m_1',
        'lam_fractions',
        'mpp_1', 'sigpp_1', 'mpp_2', 'sigpp_2',
        'beta',
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
        dense_mass=False,
        rng_key=jax.random.PRNGKey(seed)
    )

    print('done.')
