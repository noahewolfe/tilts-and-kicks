import os
import json
from argparse import ArgumentParser
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import h5ify
import numpy as np

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from jax_tqdm import scan_tqdm

# TODO: monkey patch because bilby dynesty wrapper
# is probably not compatible with numpy version < 2
if not hasattr(np.linalg, "linalg"):
    np.linalg.linalg = np.linalg

from bilby import run_sampler
from bilby.core.prior import Uniform
from bilby.core.prior import ConditionalPriorDict

from pixelpop.models.gwpop_models import PowerlawPlusPeak_MassRatio

from data import resample_and_reshape_posteriors

from likelihood import get_bilby_likelihood

from models import log_powerlaw_redshift
from models import log_iid_spin_mag_truncnorm
from models import log_nid_iso_gauss_tilt
from models import BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth

parser = ArgumentParser()
parser.add_argument('--outdir', type=str)
parser.add_argument('--posteriors', type=str, help='path to posteriors')
parser.add_argument('--injections', type=str, help='path to vt file')
parser.add_argument('--truths', help='path to json file with true parameters')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--nobs', default=70, type=int)
parser.add_argument('--maximum-variance', default=5, type=int)
parser.add_argument('--deltas', action='store_true')


def parse_args():
    """ return command-line arguments """
    args = parser.parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    with open(args.truths, 'r') as f:
        truths = json.loads(f.read())
        truths = {k : np.asarray(v).item() for k, v in truths.items()}
    return (
        outdir,
        args.injections,
        args.posteriors,
        truths,
        args.seed,
        args.nobs,
        args.maximum_variance,
        args.deltas
    )


def log_model(dataset, parameters):
    """ log-density of the population model """
    if 'lam_2' not in parameters:
        lam_2 = 1 - parameters['lam_1'] - parameters['lam_0']
    else:
        lam_2 = parameters['lam_2']

    log_p_m1 = BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth(
        dataset['mass_1'],
        alpha_1=parameters['alpha_1'],
        alpha_2=parameters['alpha_2'],
        mlow_1=parameters['mlow_1'],
        break_mass=parameters['break_mass'],
        delta_m_1=parameters['delta_m_1'],
        lam_fractions=(
            parameters['lam_0'], parameters['lam_1'], lam_2
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
        minimum=parameters['mlow_1'],
        delta_m=parameters['delta_m_1']
    )

    log_p_z = log_powerlaw_redshift(dataset, parameters)

    log_p_chi = log_iid_spin_mag_truncnorm(dataset, parameters)

    parameters['sigma_spin'] = jnp.exp(parameters['log_sigma_spin'])
    parameters['mu_spin'] = jnp.exp(parameters['log_mu_spin'])
    log_p_tau = log_nid_iso_gauss_tilt(dataset, parameters)

    return log_p_m1 + log_p_q + log_p_z + log_p_chi + log_p_tau


def get_posteriors(key, outdir, path, nobs, deltas=False):

    if deltas:
        posteriors = h5ify.load(path)
        if 'mass_1_source' in posteriors:
            posteriors['mass_1'] = posteriors.pop('mass_1_source')
        posteriors = {
            k : posteriors[k].reshape(-1, 1)
            for k in [
                'mass_1',
                'mass_ratio',
                'redshift',
                'a_1',
                'a_2',
                'cos_tilt_1',
                'cos_tilt_2',
                'log_prior'
            ]
        }
    else:
        data = h5ify.load(path)
        posteriors = list(data.values())
        for p in posteriors:
            p.pop('attrs')
        posteriors = resample_and_reshape_posteriors(posteriors)
        posteriors['log_prior'] = np.log(posteriors.pop('prior'))

    idxs = jax.random.choice(
        key,
        len(posteriors['log_prior']),
        shape=(nobs,),
        replace=False
    )
    idxs = jnp.sort(idxs)

    np.save(f'{outdir}/idxs.npy', idxs)

    posteriors = {k : v[idxs] for k, v in posteriors.items()}

    h5ify.save(f'{outdir}/posteriors.h5', posteriors, mode='w')

    return posteriors


def get_injections(path, cut=15):
    injections = h5ify.load(path)
    injections['mass_1'] = injections.pop('mass_1_source')
    print(f'applying an snr cut of {cut}')
    mask = injections['network_matched_filter_snr'] > cut
    for k in list(injections.keys()):
        if k not in ['attrs', 'total_generated', 'model', 'parameters']:
            injections[k] = injections[k][mask]
    return injections


def taper(maximum_variance, v):
    """ its actually a hard cut! """
    return jnp.nan_to_num(-1e10 * (v >= maximum_variance), nan=0)


if __name__ == '__main__':
    (
        outdir,
        injections,
        posteriors,
        truths,
        seed,
        nobs,
        maximum_variance,
        deltas
    ) = parse_args()
    truths['log_sigma_spin'] = np.log(truths.pop('sigma_spin')).item()
    truths['log_mu_spin'] = np.log(truths.pop('mu_spin')).item()

    injections = get_injections(injections)

    key = jax.random.key(seed)
    posteriors = get_posteriors(key, outdir, posteriors, nobs, deltas=deltas)

    likelihood = get_bilby_likelihood(
        log_model,
        posteriors,
        injections,
        taper=lambda v: taper(maximum_variance, v),
        rate=False
    )

    print('lnl at truths: ', likelihood.log_likelihood(parameters=truths))
    print('extras at truths: ', likelihood.generate_extra_statistics(truths))

    priors = ConditionalPriorDict('./priors/lvk-lnsigma.prior')

    # tight prior or we run into variance issues
    priors['log_sigma_spin'].minimum = np.log(1e-2 * 0.5).item()
    priors['log_sigma_spin'].maximum = np.log(1e-2 * 1.5).item()

    # tight prior here as well, for testing
    priors.pop('mu_spin')
    priors['log_mu_spin'] = Uniform(
        minimum=np.log(0.9).item(),
        maximum=0.0
    )

    result = run_sampler(
        likelihood=likelihood,
        priors=priors,
        outdir=outdir,
        label='run',
        sampler='dynesty',
        sample='acceptance-walk',
        naccept=5,
        nlive=250,
        # need enough live points to resolve w/in and w/o variance cut ...
    )

    posterior = result.posterior.to_dict('list')
    posterior = {k : jnp.array(v) for k, v in posterior.items()}

    n = len(result.posterior)

    @scan_tqdm(n)
    def step(carry, d):
        _, x = d
        return carry, likelihood.generate_extra_statistics(x)

    _, extras = jax.lax.scan(step, None, (jnp.arange(n), posterior))
    posterior = dict(**posterior, **extras)
    h5ify.save(f'{outdir}/extras.h5', posterior, mode='w')

    result.posterior['variance'] = posterior['variance']

    for k in list(truths.keys()):
        if k not in result.posterior:
            truths.pop(k)

    result.plot_corner(truths=truths)
