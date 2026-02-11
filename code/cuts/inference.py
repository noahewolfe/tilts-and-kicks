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

from data import get_data

from pixelpop.models.gwpop_models import PowerlawPlusPeak_MassRatio

from models import log_powerlaw_redshift
from models import BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth
from models import log_nid_iso_gauss_tilt
from models import log_iid_spin_mag_truncnorm

from likelihood import get_bilby_likelihood

npri = 100_000
maximum_variance = 1

snr = 15
far = 1e-5
nlive = 100

outdir = f'../../data/inference/cuts/xphm-snr{snr}-far{far:.0e}-nlive{nlive}'
os.makedirs(outdir, exist_ok=True)

events, posteriors, injections = get_data(
    snr_thresh=snr,
    far_thresh=far,
    prefer_xphm=True
)
np.savetxt(f'{outdir}/events.txt', events, fmt='%s')


if 'log_prior' in posteriors:
    print('nobs, npe =', posteriors['log_prior'].shape)
    ct1_samps = posteriors['cos_tilt_1'].flatten()
else:
    print('nobs = ', len(posteriors))
    rng = np.random.default_rng(1)

    # downsample if there are not a lot of unique samples...
    for i, (e, p) in enumerate(zip(events, posteriors)):
        npe = len(p['a_1'])
        nun = len(np.unique(p['a_1']))
        frac = nun / npe
        if frac < 0.99:
            print(
                f'Downsampling {e} since {frac} < 0.99 samples unique'
            )
            idxs = rng.choice(npe, size=6913, replace=False)
            posteriors[i] = {k : v[idxs] for k, v in p.items()}

    ct1_samps = np.concatenate([p['cos_tilt_1'] for p in posteriors])

fig, ax = plt.subplots()
ax.hist(ct1_samps, histtype='step', bins=50, density=True)
ax.set_xlabel('cos_tilt_1')
ax.set_ylabel('density')
fig.savefig(f'{outdir}/concat-ct1.png')


def taper(v):
    """ its actually a hard cut! """
    return jnp.nan_to_num(-1e10 * (v >= maximum_variance), nan=0)


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


likelihood = get_bilby_likelihood(
    log_model,
    posteriors,
    injections,
    taper=taper,
    rate=False
)


def step(carry, d):
    _, x = d
    extras = likelihood.generate_extra_statistics(x)
    return carry, extras


prior_samples = {k : jnp.array(v) for k, v in priors.sample(npri).items()}
_, extras = jax.lax.scan(
    scan_tqdm(npri)(step),
    None,
    (jnp.arange(npri), prior_samples)
)
extras['samples'] = prior_samples
h5ify.save(f'{outdir}/prior.h5', extras, mode='w')

result = run_sampler(
    likelihood=likelihood,
    priors=priors,
    outdir=outdir,
    label='test2',
    sampler='dynesty',
    sample='acceptance-walk',
    naccept=5,
    nlive=nlive,
)

result.plot_corner()

nsamps = len(result.posterior)
samples = result.posterior.to_dict('list')
samples = {k : jnp.array(v) for k, v in samples.items()}

_, extras = jax.lax.scan(
    scan_tqdm(nsamps)(step),
    None,
    (jnp.arange(nsamps), samples)
)
extras['samples'] = samples
h5ify.save(f'{outdir}/posterior.h5', extras, mode='w')
