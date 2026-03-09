import os
import sys

import jax
import jax.numpy as jnp
from jax.scipy.stats import gaussian_kde
from jax_tqdm import scan_tqdm

import h5ify

from pixelpop.models.gwpop_models import trunc_gaussian
from pixelpop.models.gwpop_models import PowerlawPlusPeak_MassRatio
from models import BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth

from models import build_interp_sampler

from util import logtrapz
from util import calc_chieff


def log_p_q(parameters):
    """ log of density of q marginalized over m1 """
    if 'lam_2' not in parameters:
        lam_2 = 1 - parameters['lam_1'] - parameters['lam_0']
    else:
        lam_2 = parameters['lam_2']

    test_m1 = jnp.linspace(3, 300, 500)
    test_q = jnp.linspace(0.1, 1, 500)
    mm, qq = jnp.meshgrid(test_m1, test_q, indexing='ij')

    dataset = dict(mass_1=mm, mass_ratio=qq)

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

    return test_q, logtrapz(log_p_m1 + log_p_q, test_m1, axis=0)


def sample_iso_gauss(key, branching_ratio, sample_gauss):
    key, _key = jax.random.split(key)
    u = jax.random.uniform(_key)
    keys = jax.random.split(key, 2)
    return jax.lax.select(
        u < branching_ratio,
        jax.vmap(sample_gauss)(keys),
        jax.vmap(lambda k: jax.random.uniform(k, minval=-1, maxval=1))(keys)
    )


def build_samplers(parameters, eps=1e-10):
    """ build inverse CDF samplers for mass ratio and spin mag, and gaussian
        component of the cosine-spin tilt distributions
    """
    # gaussian-component of cos tilt
    xs = jnp.linspace(-1 + eps, 1 - eps, 10_000)
    log_p = trunc_gaussian(
        xs,
        mean=parameters['mu_spin'],
        sig=parameters['sigma_spin'],
        lower=-1,
        upper=1
    )
    sample_gauss = build_interp_sampler(jnp.exp(log_p), xs)

    # mass ratio marginalized over m1
    xs, log_p = log_p_q(parameters)
    sample_q = build_interp_sampler(jnp.exp(log_p), xs)

    # chi1, chi2
    xs = jnp.linspace(-1 + eps, 1 - eps, 500)
    log_p = trunc_gaussian(
        xs,
        mean=parameters['mu_chi'],
        sig=parameters['sigma_chi'],
        lower=0,
        upper=1
    )
    sample_chi = build_interp_sampler(jnp.exp(log_p), xs)

    return sample_q, sample_chi, sample_gauss


def monte_carlo_sample_chieff_and_kde(posterior, nmc=10_000):
    nsamples = len(posterior['xi_spin'])

    @scan_tqdm(nsamples)
    def step(key, d):
        _, parameters = d

        sample_q, sample_chi, sample_gauss = build_samplers(parameters)

        def sample(key):
            key, _key = jax.random.split(key)
            q = sample_q(_key)

            key, _key = jax.random.split(key)
            a1, a2 = jax.vmap(sample_chi)(jax.random.split(_key, 2))

            key, _key = jax.random.split(key)
            ct1, ct2 = sample_iso_gauss(
                _key, parameters['xi_spin'], sample_gauss
            )

            chi_eff = calc_chieff(q, a1, a2, ct1, ct2)

            return dict(
                mass_ratio=q,
                a_1=a1,
                a_2=a2,
                cos_tilt_1=ct1,
                cos_tilt_2=ct2,
                chi_eff=chi_eff,
            )

        key, _key = jax.random.split(key)
        samples = jax.vmap(sample)(jax.random.split(_key, nmc))

        test_chi = jnp.linspace(-1, 1, 500)
        log_prob = gaussian_kde(samples['chi_eff']).logpdf(test_chi)

        samples['log_prob'] = log_prob

        return key, samples

    _, samples = jax.lax.scan(
        step,
        jax.random.key(1),
        (jnp.arange(nsamples), posterior)
    )
    return samples


if __name__ == '__main__':
    outdir = os.path.abspath(sys.argv[1])
    posterior = h5ify.load(f'{outdir}/extras.h5')
    posterior = {k : jnp.array(v) for k, v in posterior.items()}
    samples = monte_carlo_sample_chieff_and_kde(posterior)
    h5ify.save(f'{outdir}/mcppd.h5', samples, mode='w')
