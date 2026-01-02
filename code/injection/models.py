import numpy as np

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

import wcosmo
from astropy import units

def truncnorm(xx, mu, sigma, high, low):
    """ ripped from gwpop v1.1.1 -- Colm's later version has a nan switch
        that breaks autodiff!
    """
    norm = 2**0.5 / jnp.pi**0.5 / sigma
    norm /= jsp.erf((high - mu) / 2**0.5 / sigma) + jsp.erf(
        (mu - low) / 2**0.5 / sigma
    )
    prob = jnp.exp(-jnp.power(xx - mu, 2) / (2 * sigma**2))
    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return prob


def truncated_powerlaw(x, alpha, xmin, xmax):
    cut = (xmin <= x) * (x <= xmax)
    shape = x**alpha
    norm = (xmax**(alpha + 1) - xmin**(alpha + 1)) / (alpha + 1)
    return (shape / norm) * cut


def broken_powerlaw(x, alpha1, alpha2, xbreak, xmin, xmax):
    y = x / xbreak
    prob = y**(-alpha1) * (xmin <= x) * (x < xbreak)
    prob *= y**(-alpha2) * (xbreak <= x) * (x < xmax)
    norm = xbreak * (
        (1 - (xmin / xbreak)**(1 - alpha1)) / (1 - alpha1)
        + ((xmax / xbreak)**(1 - alpha2) - 1) / (1 - alpha2)
    )
    return prob * norm


def highpass_sigmoid(x, xmin, xmax, dmin):
    y = (x - xmin) / dmin
    y = jnp.clip(y, 1e-6, 1 - 1e-6)
    y = 1 / y - 1 / (1 - y)
    y = jsp.expit(-y) * (x <= xmax) * (x >= xmin)
    return jnp.where(dmin > 0, y, 1)


def highpass_broken_powerlaw_twopeaks_shape(
    x, alpha1, alpha2, xbreak, xmin, xmax, mpp1, sigpp1, mpp2, sigpp2,
    lam0, lam1, dmin, norm_xmax=300
):
    lam2 = 1 - lam0 - lam1
    prob = lam0 * broken_powerlaw(x, alpha1, alpha2, xbreak, xmin, xmax)
    prob += lam1 * truncnorm(x, mpp1, sigpp1, high=1e100, low=xmin)
    prob += lam2 * truncnorm(x, mpp2, sigpp2, high=1e100, low=xmin)

    highpass = highpass_sigmoid(x, xmin, norm_xmax, dmin)
    prob *= highpass

    return prob


def highpass_broken_powerlaw_twopeaks_norm(
    alpha1, alpha2, xbreak, xmin, xmax, mpp1, sigpp1, mpp2, sigpp2,
    lam0, lam1, dmin, norm_xmin=3, norm_xmax=300
):
    xs = jnp.linspace(norm_xmin, norm_xmax, 1_000)
    shape = highpass_broken_powerlaw_twopeaks_shape(
        xs,
        alpha1,
        alpha2,
        xbreak,
        xmin,
        xmax,
        mpp1,
        sigpp1,
        mpp2,
        sigpp2,
        lam0,
        lam1,
        dmin,
        norm_xmax=norm_xmax
    )
    return jnp.clip(jnp.trapezoid(shape, xs), 1e-100)


def highpass_broken_powerlaw_twopeaks(
    x, alpha1, alpha2, xbreak, xmin, xmax, mpp1, sigpp1, mpp2, sigpp2,
    lam0, lam1, dmin, norm_xmin=3, norm_xmax=300
):
    """ very similar to the BPL+2Pk model in GWTC-4 pop paper;
        there might be small differences. i should probably enumerate those.
    """
    shape = highpass_broken_powerlaw_twopeaks_shape(
        x, alpha1, alpha2, xbreak, xmin, xmax, mpp1, sigpp1, mpp2, sigpp2,
        lam0, lam1, dmin, norm_xmax=norm_xmax 
    )
    norm = highpass_broken_powerlaw_twopeaks_norm(
        alpha1, alpha2, xbreak, xmin, xmax, mpp1, sigpp1, mpp2, sigpp2,
        lam0, lam1, dmin, norm_xmin=norm_xmin, norm_xmax=norm_xmax
    )
    return jnp.where(norm > 0, shape / norm, 0)


def log_powerlaw_redshift(dataset, parameters, max_z=1.45, return_norm=False):
    lamb = parameters['lamb']
    z = dataset['redshift']

    zs_fixed = np.linspace(1e-5, max_z, 1000)
    fixed_ln_dvc_dz = jnp.log(
        4 * jnp.pi * wcosmo.Planck15.differential_comoving_volume(zs_fixed).to(
            units.Gpc**3 / units.sr
        ).value
    )

    dz = zs_fixed[1] - zs_fixed[0]
    test_ln_p = fixed_ln_dvc_dz + (lamb - 1) * jnp.log(1. + zs_fixed)
    ln_norm = jsp.logsumexp(test_ln_p) + jnp.log(dz)

    if return_norm:
        return ln_norm

    ln_dvc_dz = jnp.interp(z, zs_fixed, fixed_ln_dvc_dz)
    ln_p = ln_dvc_dz + (lamb - 1) * jnp.log(1. + z)
    ln_p -= ln_norm

    window = jnp.logical_and(z >= 0., z <= max_z)
    p = jnp.where(window, ln_p, -jnp.inf)
    return p


def plp_q_shape(mass_ratio, mass_1, beta, mmin, dmin):
    prob = truncated_powerlaw(mass_ratio, beta, mmin / mass_1, 1)
    smooth = highpass_sigmoid(mass_1 * mass_ratio, mmin, mass_1, dmin)
    return prob * smooth


def plp_q_norm(
    mass_1, beta, mmin, dmin, norm_mmin=1, norm_mmax=200,
    interp_method='linear'
):
    # note: interp_method e.g. `cubic` will probably be slow because I don't
    # do any fancy caching.
    m1s = jnp.linspace(norm_mmin, norm_mmax, 1_000)
    qs = jnp.linspace(1e-3, 1, 500)
    mm, qq = jnp.meshgrid(m1s, qs, indexing='ij')

    shapes = plp_q_shape(qq, mm, beta, mmin, dmin)
    norms = jnp.nan_to_num(jnp.trapezoid(shapes, qs, axis=1))

    return jnp.clip(jnp.interp(mass_1, m1s, norms), min=1e-100)


def plp_q(
    mass_ratio, mass_1, beta, mmin, dmin, norm_mmin=1, norm_mmax=200,
    interp_method='linear'
):
    shape = plp_q_shape(mass_ratio, mass_1, beta, mmin, dmin)
    norm = plp_q_norm(
        mass_1, beta, mmin, dmin, norm_mmin, norm_mmax, interp_method
    )
    return jnp.where(norm > 0, shape / norm, 0)


def iso_gauss_spin_tilt(dataset, xi_spin, sigma_spin, mu_spin=1):
    cos_tilt_1, cos_tilt_2 = dataset['cos_tilt_1'], dataset['cos_tilt_2']
    return (
        (1 - xi_spin) / 4
        + (
            xi_spin
            * truncnorm(cos_tilt_1, mu_spin, sigma_spin, high=1, low=-1)
            * truncnorm(cos_tilt_2, mu_spin, sigma_spin, high=1, low=-1)
        )
    )


def marg_iso_gauss_spin_tilt(cos_tau, xi_spin, sigma_spin, mu_spin=1):
    return (
        (1 - xi_spin) / 2
        + (
            xi_spin
            * truncnorm(cos_tau, mu_spin, sigma_spin, high=1, low=-1)
        )
    )


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
    from pixelpop.models.gwpop_models import trunc_gaussian

    import jax.scipy.special as scs

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

        pm1 = scs.logsumexp(jnp.array([
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
    norm = scs.logsumexp(ys) + jnp.log(dx)  # simple Riemann rule.

    log_prob -= norm

    if isLogMass:  # include jacobian
        log_prob = log_prob + data['log_mass_1']

    return log_prob


def bplm1q_plz_truncnormmag(
    dataset, parameters, norm_mmin=3, norm_mmax=300, interp_method='linear'
):
    p_m1 = highpass_broken_powerlaw_twopeaks(
        x=dataset['mass_1_source'],
        alpha1=parameters['alpha1'],
        alpha2=parameters['alpha2'],
        xbreak=parameters['mbreak'],
        xmin=parameters['mmin'],
        xmax=norm_mmax,
        mpp1=parameters['mpp1'],
        sigpp1=parameters['sigpp1'],
        mpp2=parameters['mpp2'],
        sigpp2=parameters['sigpp2'],
        lam0=parameters['lam0'],
        lam1=parameters['lam1'],
        dmin=parameters['delta_m'],
        norm_xmin=norm_mmin,
        norm_xmax=norm_mmax
    )

    p_q = plp_q(
        mass_ratio=dataset['mass_ratio'],
        mass_1=dataset['mass_1_source'],
        beta=parameters['beta'],
        mmin=parameters['mmin'],
        dmin=parameters['delta_m'],
        norm_mmin=norm_mmin,
        norm_mmax=norm_mmax,
        interp_method=interp_method
    )

    log_pl_z = log_powerlaw_redshift(dataset, parameters)
    p_z = jnp.exp(log_pl_z)

    p_a1 = truncnorm(
        dataset['a_1'],
        parameters['mu_chi'],
        parameters['sigma_chi'],
        high=1,
        low=0
    )
    p_a2 = truncnorm(
        dataset['a_2'],
        parameters['mu_chi'],
        parameters['sigma_chi'],
        high=1,
        low=0
    )

    return p_m1 * p_q * p_z * p_a1 * p_a2


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


def skewtruncnorm_shape(x, mu, sigma, skew, high, low):
    branch1 = truncnorm(x, mu, sigma * (1 + skew), high, low) * (1 + skew)
    branch2 = truncnorm(x, mu, sigma * (1 - skew), high, low) * (1 - skew) 
    return jnp.where(x <= 0, branch1, branch2)


def skewtruncnorm_norm(mu, sigma, skew, high, low):
    xs = jnp.linspace(low, high, 1_000)
    shape = skewtruncnorm_shape(xs, mu, sigma, skew, high, low)
    return jnp.trapezoid(shape, xs)


def skewtruncnorm(x, mu, sigma, skew, high, low):
    shape = skewtruncnorm_shape(x, mu, sigma, skew, high, low)
    norm = skewtruncnorm_norm(mu, sigma, skew, high, low)
    return shape / norm


def bpl2p_plz_truncnormmag_isogausstilt(dataset, parameters):
    from pixelpop.models.gwpop_models import PowerlawPlusPeak_MassRatio

    if 'lam_2' not in parameters.keys():
        if 'lam_0' in parameters.keys() and 'lam_1' in parameters.keys():
            parameters['lam_2'] = 1 - parameters['lam_0'] - parameters['lam_1']

    log_p_m1 = BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth(
        dataset,
        alpha_1=parameters['alpha_1'],
        alpha_2=parameters['alpha_2'],
        mlow_1=parameters['mlow_1'],
        break_mass=parameters['break_mass'],
        delta_m_1=parameters['delta_m_1'],
        lam_fractions=(
            parameters['lam_0'], parameters['lam_1'], parameters['lam_2']
        ),
        mpp_1=parameters['mpp_1'],
        sigpp_1=parameters['sigpp_1'],
        mpp_2=parameters['mpp_2'],
        sigpp_2=parameters['sigpp_2'],
        mmax=300.0,
        gaussian_mass_maximum=100.0
    )
    p_m1 = jnp.exp(log_p_m1)

    log_p_q_given_m1 = PowerlawPlusPeak_MassRatio(
        dataset,
        slope=parameters['beta'],
        minimum=parameters['mlow_1'],
        delta_m=parameters['delta_m_1']
    )
    p_q_given_m1 = jnp.exp(log_p_q_given_m1)

    log_pl_z = log_powerlaw_redshift(dataset, parameters)
    p_z = jnp.exp(log_pl_z)

    p_a1 = truncnorm(
        dataset['a_1'],
        parameters['mu_chi'],
        parameters['sigma_chi'],
        high=1,
        low=0
    )
    p_a2 = truncnorm(
        dataset['a_2'],
        parameters['mu_chi'],
        parameters['sigma_chi'],
        high=1,
        low=0
    )

    p_cos_tilts = iso_gauss_spin_tilt(
        dataset,
        parameters['xi_spin'],
        parameters['sigma_spin'],
        parameters['mu_spin']
    )

    return p_m1 * p_q_given_m1 * p_z * p_a1 * p_a2 * p_cos_tilts


def log_truncated_powerlaw(x, alpha, xmin, xmax):
    cut = (xmin <= x) * (x <= xmax)

    norm = (xmax**(alpha + 1) - xmin**(alpha + 1)) / (alpha + 1)

    log_prob = jnp.where(
        cut,
        alpha * jnp.log(x),
        -jnp.inf
    )

    return log_prob - norm


def log_plq_given_m1(dataset, parameters):
    """ simple powerlaw in q, no smoothing """
    return log_truncated_powerlaw(
        dataset['mass_ratio'],
        parameters['beta'],
        parameters['mmin'] / jnp.exp(dataset['log_mass_1']),
        1.0
    )


def log_iid_spin_mag_truncnorm(dataset, parameters):
    from pixelpop.models.gwpop_models import trunc_gaussian

    log_p_a1 = trunc_gaussian(
        dataset['a_1'],
        parameters['mu_chi'],
        jnp.sqrt(parameters['sigma_chi']),
        lower=0,
        upper=1
    )

    log_p_a2 = trunc_gaussian(
        dataset['a_2'],
        parameters['mu_chi'],
        jnp.sqrt(parameters['sigma_chi']),
        lower=0,
        upper=1
    )

    return log_p_a1 + log_p_a2


def log_nid_iso_gauss_tilt(dataset, parameters):
    from pixelpop.models.gwpop_models import trunc_gaussian

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
