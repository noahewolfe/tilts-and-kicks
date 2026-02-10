import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import matplotlib.pyplot as plt

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import h5ify
from jax_tqdm import scan_tqdm

from bilby import run_sampler
from bilby.core.prior import ConditionalPriorDict

from data import get_posteriors
from data import get_injections

from pixelpop.models.gwpop_models import PowerlawPlusPeak_MassRatio

from models import log_powerlaw_redshift
from models import BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth
from models import log_nid_iso_gauss_tilt
from models import log_iid_spin_mag_truncnorm

from likelihood import get_bilby_likelihood


def cut_data(event_data, injections, snr=10, far=1):
    posteriors = event_data[0]
    event_snrs = event_data[2]
    event_fars = event_data[3]
    found = (event_fars < far) | (event_snrs > snr)
    posts = {k : v[found] for k, v in posteriors.items()}

    found = (injections['far'] < far) | (injections['snr'] > snr)
    injs = {k : v[found] for k, v in injections.items() if k not in ['time', 'total']}
    injs['total'] = injections['total']
    injs['time'] = injections['time']

    return posts, injs


def get_data(snr=10, far=1):
    event_data = get_posteriors(load=True)
    injections = get_injections(load=True)
    posteriors, injections = cut_data(event_data, injections, snr=snr, far=far)

    posteriors['mass_1'] = posteriors.pop('mass_1_source')

    injections['mass_1'] = injections.pop('mass_1_source')
    injections['total_generated'] = injections.pop('total')

    posteriors['log_prior'] = np.log(posteriors['prior'])
    injections['log_prior'] = np.log(injections['prior'])

    return posteriors, injections


post_snr10, injs_snr10 = get_data(snr=10, far=1)
post_snr15, injs_snr15 = get_data(snr=15, far=1e-5)


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

    log_p_tau = log_nid_iso_gauss_tilt(dataset, parameters)

    return log_p_m1 + log_p_q + log_p_z + log_p_chi + log_p_tau


priors = ConditionalPriorDict('./priors/lvk.prior')

maximum_variance = 1


def taper(v):
    """ its actually a hard cut! """
    return jnp.nan_to_num(-1e10 * (v >= maximum_variance), nan=0)


likelihood = get_bilby_likelihood(
    log_model,
    post_snr15,
    injs_snr15,
    taper=taper,
    rate=False
)


outdir = '../../data/inference/cuts/snr15-far1e-5'


"""
npri = 5_000


@scan_tqdm(npri)
def step(carry, d):
    _, x = d
    extras = likelihood.generate_extra_statistics(x)
    return carry, extras


prior_samples = {k : jnp.array(v) for k, v in priors.sample(npri).items()}
_, prior_samples = jax.lax.scan(
    step,
    None,
    (jnp.arange(npri), prior_samples)
)

os.makedirs(outdir, exist_ok=True)
h5ify.save(f'{outdir}/prior.h5', prior_samples)
"""

result = run_sampler(
    likelihood=likelihood,
    priors=priors,
    outdir=outdir,
    label='test',
    sampler='dynesty',
    sample='acceptance-walk',
    naccept=5,
    nlive=100,
    # need enough live points to resolve w/in and w/o variance cut ...
)
