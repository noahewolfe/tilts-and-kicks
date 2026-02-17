""" implementations in here are sometimes inspired by pixelpop """
from typing import Callable
import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import wcosmo
import unxt

from pixelpop.models.gwpop_models import trunc_gaussian

MPC3_TO_GPC3 = 1e-9


def log_powerlaw_redshift(dataset, parameters, return_norm=False):
    lamb = parameters['lamb']
    z = dataset['redshift']
    z_max = parameters.get('z_max', 1.45)

    zs_fixed = jnp.linspace(1e-5, z_max, 1000)
    dvc_dz = wcosmo.Planck15.differential_comoving_volume(zs_fixed)
    if isinstance(dvc_dz, unxt.quantity.Quantity):
        # dVc/dz is in Mpc^3/sr; convert value-only to Gpc^3/sr.
        dvc_dz = 4 * jnp.pi * MPC3_TO_GPC3 * dvc_dz.value
    else:
        dvc_dz = 4 * jnp.pi * MPC3_TO_GPC3 * dvc_dz
    fixed_ln_dvc_dz = jnp.log(dvc_dz)

    dz = zs_fixed[1] - zs_fixed[0]
    test_ln_p = fixed_ln_dvc_dz + (lamb - 1) * jnp.log1p(zs_fixed)
    ln_norm = logsumexp(test_ln_p) + jnp.log(dz)

    if return_norm:
        return ln_norm

    ln_dvc_dz = jnp.interp(z, zs_fixed, fixed_ln_dvc_dz)
    ln_p = ln_dvc_dz + (lamb - 1) * jnp.log1p(z)
    ln_p -= ln_norm

    window = jnp.logical_and(z >= 0., z <= z_max)
    p = jnp.where(window, ln_p, -jnp.inf)
    return p


def BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth(
    data,
    alpha_1,
    alpha_2,
    mlow_1,
    break_mass,
    delta_m_1,
    lam_fractions,
    mpp_1,
    sigpp_1,
    mpp_2,
    sigpp_2,
    mmax=300.0,
    gaussian_mass_maximum=100.0
):
    """
    Primary mass distribution: broken power-law + two Gaussian peaks.

    Implements the default GWTC-4.0 primary mass population model:
    a mixture of (1) a smoothed broken power-law, and (2–3) two
    truncated Gaussians representing additional features.

    Parameters
    ----------
    data : dict or jnp.ndarray
        Either a dict with key 'mass_1' or 'log_mass_1',
        or a direct array of primary masses.
    alpha_1 : float
        Low-mass slope of the power-law.
    alpha_2 : float
        High-mass slope of the power-law.
    mmin : float
        Minimum primary mass cutoff.
    break_mass : float
        Break mass separating the two slopes.
    delta_m_1 : float
        Smoothing width at the low-mass cutoff.
    lam_fractions : tuple of floats
        Mixture fractions (lam_0, lam_1, lam_2) for
        {power-law, first Gaussian, second Gaussian}.
    mpp_1 : float
        Mean of the first Gaussian peak.
    sigpp_1 : float
        Std. deviation of the first Gaussian peak.
    mpp_2 : float
        Mean of the second Gaussian peak.
    sigpp_2 : float
        Std. deviation of the second Gaussian peak.
    mmax : float, optional
        Maximum primary mass cutoff (default 300).
    gaussian_mass_maximum : float, optional
        Upper truncation for Gaussian peaks (default 100).

    Returns
    -------
    jnp.ndarray
        Log-probability density of the normalized mass distribution.
    """

    from pixelpop.models.gwpop_models import BrokenPowerLaw
    from pixelpop.models.gwpop_models import m_smoother

    isLogMass = True
    if isinstance(data, dict):
        try:
            m1 = jnp.exp(data['log_mass_1'])
        except KeyError:
            isLogMass = False
            m1 = data['mass_1']
    else:
        isLogMass = False
        m1 = data

    lam_0, lam_1, lam_2 = lam_fractions
    break_fraction = (break_mass - mlow_1) / (mmax - mlow_1)

    def shape(m1):
        p_pow = BrokenPowerLaw(
            m1, -alpha_1, -alpha_2, mlow_1, mmax, break_fraction
        )
        p_norm1 = trunc_gaussian(
            m1, mpp_1, sigpp_1, mlow_1, gaussian_mass_maximum
        )
        p_norm2 = trunc_gaussian(
            m1, mpp_2, sigpp_2, mlow_1, gaussian_mass_maximum
        )

        pm1 = logsumexp(jnp.array([
            jnp.log(lam_0) + p_pow,
            jnp.log(lam_1) + p_norm1,
            jnp.log(lam_2) + p_norm2
        ]), axis=0)

        pm1 += m_smoother(m1, mlow_1, delta_m_1)

        return pm1

    log_prob = shape(m1)

    xs = jnp.linspace(3, 300, 2_000)
    dx = xs[1] - xs[0]
    ys = shape(xs)
    norm = logsumexp(ys) + jnp.log(dx)  # simple Riemann rule.

    log_prob -= norm

    if isLogMass:  # include jacobian
        log_prob = log_prob + data['log_mass_1']

    return log_prob


def build_interp_sampler(density, xs, xp=jnp):
    """ factory-function for inverse CDF sampling by interpolated a density
        over points xp. """
    if xp == jnp:
        from util import cumulative_trapezoid
    else:
        from scipy.integrate import cumulative_trapezoid

    if isinstance(density, Callable):
        prob = density(xs)
        norm = xp.trapezoid(prob, xs)
        prob /= norm
    else:
        prob = density

    cdf = cumulative_trapezoid(y=prob, x=xs, initial=0)

    if xp == jnp:
        def func(key):
            u = jax.random.uniform(key)
            return xp.interp(u, cdf, xs)
    elif xp == np:
        def func(rng, size=()):
            u = rng.uniform(size=size)
            return xp.interp(u, cdf, xs)

    return func


def log_iid_spin_mag_truncnorm(dataset, parameters):
    """ iid spin magnitudes """
    log_p_a1 = trunc_gaussian(
        dataset['a_1'],
        parameters['mu_chi'],
        parameters['sigma_chi'],
        lower=0,
        upper=1
    )

    log_p_a2 = trunc_gaussian(
        dataset['a_2'],
        parameters['mu_chi'],
        parameters['sigma_chi'],
        lower=0,
        upper=1
    )

    return log_p_a1 + log_p_a2


def log_nid_iso_gauss_tilt(dataset, parameters):
    """ non-identical iso+Gauss mixture for cosine-spin tilts """
    cos_tilt_1, cos_tilt_2 = dataset['cos_tilt_1'], dataset['cos_tilt_2']

    xi_spin = parameters['xi_spin']
    sigma_spin = parameters['sigma_spin']
    mu_spin = parameters.get('mu_spin', 1)

    iso = jnp.log((1 - xi_spin) / 4)

    gauss = (
        jnp.log(xi_spin)
        + trunc_gaussian(cos_tilt_1, mu_spin, sigma_spin, lower=-1, upper=1)
        + trunc_gaussian(cos_tilt_2, mu_spin, sigma_spin, lower=-1, upper=1)
    )

    return jnp.logaddexp(iso, gauss)


def log_marg_iso_gauss_spin_tilt(cos_tau, xi_spin, sigma_spin, mu_spin=1):
    """ marginal iso+Gauss mixture for cosine-spin tilts """
    iso = jnp.log((1 - xi_spin) / 2)
    gauss = (
        jnp.log(xi_spin)
        + trunc_gaussian(cos_tau, mu_spin, sigma_spin, lower=-1, upper=1)
    )
    return jnp.logaddexp(iso, gauss)
