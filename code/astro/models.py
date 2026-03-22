import numpy as np

import jax
import jax.numpy as jnp
from jax.random import split
from jax.scipy.stats import gaussian_kde
from jax.scipy.special import logsumexp

import wcosmo
import unxt


def build_interp_sampler(density, xs, xp=jnp):
    """ factory-function for inverse CDF sampling by interpolated a density
        over points xp. """
    if xp == jnp:
        from quadax import cumulative_trapezoid
    else:
        from scipy.integrate import cumulative_trapezoid

    prob = density(xs)
    norm = xp.trapezoid(prob, xs)
    prob /= norm

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


def monotonic_select(thresholds, values, right_closed_left_open=True):
    import jax.numpy as jnp

    thresholds = jnp.asarray(thresholds)
    values = jnp.asarray(values)
    assert values.shape[0] == thresholds.shape[0] + 1

    side = 'right' if right_closed_left_open else 'left'

    def func(x):
        # x can be scalar or array; searchsorted broadcasts over x
        idx = jnp.searchsorted(thresholds, x, side=side)  # in [0, len(thresholds)]
        return values[idx]  # jnp.take handles array idx too
    return func


def remnant_mass_golomb2023(mprog, mbhmax, mturnover):
    """ remnant mass function from Golomb+2023"""
    return monotonic_select(
        jnp.array([mturnover, 2 * mbhmax - mturnover]),
        jnp.array([
            mprog,
            (
                mbhmax
                + (
                    (mprog - 2 * mbhmax + mturnover)**2
                    / 4
                    / (mturnover - mbhmax)
                )
            ),
            0.0
        ])
    )(mprog)


def log_bpl_progenitor_mass(mprog, alpha_1, alpha_2, mmin, mmax, mbreak):
    """ broken powerlaw in progenitor masses """
    from pixelpop.models.gwpop_models import BrokenPowerLaw

    break_fraction = (mbreak - mmin) / (mmax - mmin)
    return BrokenPowerLaw(
        mprog, -alpha_1, -alpha_2, mmin, mmax, break_fraction
    )


def get_density_estimator_mbh(parameters):
    mprog_min = parameters.get('mprog_min')
    mprog_max = parameters.get('mprog_max')
    mprog_break = parameters.get('mprog_break')

    ms = jnp.linspace(
        mprog_min,
        mprog_max,
        500
    )

    sample_mprog = build_interp_sampler(
        lambda x: jnp.exp(log_bpl_progenitor_mass(
            x,
            parameters.get('alpha_prog_1'),
            parameters.get('alpha_prog_2'),
            mprog_min,
            mprog_max,
            mprog_break,
        )),
        ms
    )

    mbhmax = parameters.get('mbhmax')
    mturnover = parameters.get('mturnover')

    def sample(key):
        mprog = sample_mprog(key)
        return remnant_mass_golomb2023(mprog, mbhmax, mturnover)

    n = 500
    key = jax.random.key(7)
    mbh = jax.vmap(sample)(split(key, n))
    return gaussian_kde(mbh, weights=mbh > 0)


#kde_mbh = get_density_estimator_mbh(dict(
#    mprog_min=3,
#    mprog_max=100,
#   mprog_break=50,
#    alpha_prog_1=2.1,
#    alpha_prog_2=1.1,
#))


def get_density_estimators_dynamical(parameters):
    kde_mbh = get_density_estimator_mbh(parameters)

    beta = parameters.get('beta_1g1g')

    def sample_1g1g(key):
        """ return samples m1 >= m2, unweighted """
        return kde_mbh.resample(key, (2,)).squeeze()

    n = 1_000
    key = jax.random.key(8)

    ms_1g1g = jax.vmap(sample_1g1g)(split(key, n))
    mtots = jnp.sum(ms_1g1g, axis=1)
    weights = jnp.clip(mtots, 0)**beta

    kde_1g1g = gaussian_kde(ms_1g1g.T, weights=weights)
    kde_mtot = gaussian_kde(mtots, weights=weights)

    def sample_1g2g(key):
        key, subkey = jax.random.split(key)
        m1g = kde_mbh.resample(key).squeeze()

        key, subkey = jax.random.split(key)
        m2g = kde_mtot.resample(subkey).squeeze()

        return jnp.array([m1g, m2g])

    beta = parameters.get('beta_1g2g')

    key = jax.random.key(9)
    ms_1g2g = jax.vmap(sample_1g2g)(split(key, n))
    mtots_1g2g = jnp.sum(ms_1g2g, axis=1)
    weights = jnp.clip(mtots_1g2g, 0)**beta

    kde_1g2g = gaussian_kde(ms_1g2g.T, weights=weights)

    return kde_1g1g, kde_1g2g


def order_wrap(kde):

    def prob(x):
        """ x = [m1, m2] """
        prob = kde.pdf(x) + kde.pdf(x[::-1])
        return prob * (x[0] >= x[1])

    return prob


def bilinear_interp_2d(m1, m2, m1_grid, m2_grid, values):
    """Bilinear interpolation on a regular 2D grid."""
    n1 = len(m1_grid)
    n2 = len(m2_grid)

    i = jnp.searchsorted(m1_grid, m1) - 1
    j = jnp.searchsorted(m2_grid, m2) - 1
    i = jnp.clip(i, 0, n1 - 2)
    j = jnp.clip(j, 0, n2 - 2)

    s = (m1 - m1_grid[i]) / (m1_grid[i + 1] - m1_grid[i])
    t = (m2 - m2_grid[j]) / (m2_grid[j + 1] - m2_grid[j])

    return (
        values[i, j] * (1 - s) * (1 - t)
        + values[i + 1, j] * s * (1 - t)
        + values[i, j + 1] * (1 - s) * t
        + values[i + 1, j + 1] * s * t
    )


def _kde_grid(kde, m_grid):
    """Evaluate a 2D KDE on a grid with order symmetrization."""
    pts = jnp.stack(
        jnp.meshgrid(m_grid, m_grid, indexing='ij'), axis=0
    ).reshape(2, -1)
    pdf = kde.pdf(pts).reshape(len(m_grid), len(m_grid))
    pdf_sym = pdf + pdf.T
    mask = m_grid[:, None] >= m_grid[None, :]
    return pdf_sym * mask


def mass_dynamical(mass_1, mass_2, parameters, n_grid=200):
    kde_1g1g, kde_1g2g = get_density_estimators_dynamical(parameters)

    m_grid = jnp.linspace(1, 150, n_grid)

    pdf_1g1g = _kde_grid(kde_1g1g, m_grid)
    pdf_1g2g = _kde_grid(kde_1g2g, m_grid)

    shape = mass_1.shape
    m1 = mass_1.ravel()
    m2 = mass_2.ravel()

    prob_1g1g = bilinear_interp_2d(m1, m2, m_grid, m_grid, pdf_1g1g)
    prob_1g2g = bilinear_interp_2d(m1, m2, m_grid, m_grid, pdf_1g2g)

    return prob_1g1g.reshape(shape), prob_1g2g.reshape(shape)


def mass_dynamical_kde(mass_1, mass_2, parameters):
    """Original KDE-based evaluation (for comparison/testing)."""
    kde_1g1g, kde_1g2g = get_density_estimators_dynamical(parameters)
    prob_1g1g = order_wrap(kde_1g1g)
    prob_1g2g = order_wrap(kde_1g2g)

    shape = mass_1.shape
    x = jnp.stack([mass_1.ravel(), mass_2.ravel()], axis=-1)

    prob_1g1g = jax.vmap(prob_1g1g)(x)
    prob_1g2g = jax.vmap(prob_1g2g)(x)

    return prob_1g1g.reshape(shape), prob_1g2g.reshape(shape)


def dynamical(dataset, parameters):
    from pixelpop.models.gwpop_models import trunc_gaussian

    zeta = parameters.get('zeta')

    mu_chi = parameters.get('mu_chi')
    sigma_chi = parameters.get('sigma_chi')

    mu_spin_1g1g = parameters.get('mu_spin_0')
    sigma_spin_1g1g = parameters.get('sigma_spin_0')

    mu_spin_1g2g = parameters.get('mu_spin_1')
    sigma_spin_1g2g = parameters.get('sigma_spin_1')

    mass_1 = dataset.get('mass_1_source', dataset.get('mass_1'))

    if 'mass_ratio' in dataset:
        mass_2 = mass_1 * dataset['mass_ratio']
    else:
        mass_2 = dataset.get('mass_2_source', dataset.get('mass_2'))

    p_m1m2_1g1g, p_m1m2_1g2g = mass_dynamical(
        mass_1,
        mass_2,
        parameters
    )

    p_chi_1g1g = jnp.exp(
        trunc_gaussian(dataset['a_1'], mu_chi, sigma_chi, 0, 1) +
        trunc_gaussian(dataset['a_2'], mu_chi, sigma_chi, 0, 1)
    )

    p_chi_1g2g = jnp.exp(
        trunc_gaussian(dataset['a_1'], 0.69, 0.1, 0, 1) +
        trunc_gaussian(dataset['a_2'], mu_chi, sigma_chi, 0, 1)
    )

    p_tau_1g1g = jnp.exp(
        trunc_gaussian(dataset['cos_tilt_1'], mu_spin_1g1g, sigma_spin_1g1g, -1, 1) +
        trunc_gaussian(dataset['cos_tilt_2'], mu_spin_1g1g, sigma_spin_1g1g, -1, 1)
    )

    p_tau_1g2g = jnp.exp(
        trunc_gaussian(dataset['cos_tilt_1'], mu_spin_1g2g, sigma_spin_1g2g, -1, 1) +
        trunc_gaussian(dataset['cos_tilt_2'], mu_spin_1g2g, sigma_spin_1g2g, -1, 1)
    )

    p_1g1g = p_m1m2_1g1g * p_chi_1g1g * p_tau_1g1g
    p_1g2g = p_m1m2_1g2g * p_chi_1g2g * p_tau_1g2g

    return (1 - zeta) * p_1g1g + zeta * p_1g2g


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