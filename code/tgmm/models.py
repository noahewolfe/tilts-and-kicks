import jax.numpy as jnp

from pixelpop.models.gwpop_models import PowerlawPlusPeak_MassRatio

from gravpop.models.generic import SampledPopulationModel
from gravpop.models.generic import MassPopulationModel


def log_bpl2pk_primary_mass(
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

    from pixelpop.models.gwpop_models import trunc_gaussian
    from pixelpop.models.gwpop_models import BrokenPowerLaw
    from pixelpop.models.gwpop_models import m_smoother

    import jax.scipy.special as scs

    m1 = None
    isLogMass = False
    possible_keys = [
        'log_mass_1_source', 'mass_1_source', 'log_mass_1', 'mass_1_source'
    ]
    for k in possible_keys:
        if k in data:
            m1 = data[k]
            if 'log' in k:
                isLogMass = True
                m1 = jnp.exp(m1)
            break

    if m1 is None:
        raise ValueError(f'No m1 key among {possible_keys} found in data.')

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


def generate_mass_hyper_var_names(component):
    m1_hyper_var_names = [
        'alpha_1',
        'alpha_2',
        'mlow_1',
        'break_mass',
        'delta_m_1',
        'lam_fractions',
        'mpp_1',
        'sigpp_1',
        'mpp_2',
        'sigpp_2'
    ]
    q_hyper_var_names = [
        'beta',
        'mlow_1',
        'delta_m_1'
    ]
    m1_hyper_var_names = [f'{k}_{component}' for k in m1_hyper_var_names]
    q_hyper_var_names = [f'{k}_{component}' for k in q_hyper_var_names]


class SmoothedBrokenPowerLawTwoPeaks(SampledPopulationModel, MassPopulationModel):
    def __init__(
        self,
        mmax=300.0,
        gaussian_mass_maximum=100.0,
        m1_hyper_var_names=[
            'alpha_1',
            'alpha_2',
            'mlow_1',
            'break_mass',
            'delta_m_1',
            'lam_fractions',
            'mpp_1',
            'sigpp_1',
            'mpp_2',
            'sigpp_2'
        ],
        q_hyper_var_names=[
            'beta',
            'mlow_1',
            'delta_m_1'
        ]
    ):
        self.mmax = mmax
        self.gaussian_mass_maximum = gaussian_mass_maximum

        self.m1_hyper_var_names = m1_hyper_var_names
        self.q_hyper_var_names = q_hyper_var_names

        self.hyper_var_names = set(
            self.m1_hyper_var_names + self.q_hyper_var_names
        )
        self.var_names = ['mass_1_source', 'mass_ratio']

    def __call__(self, data, params):
        params['lam_fractions'] = (
            params['lam_0'], params['lam_1'], params['lam_2']
        )
        log_p_m1 = log_bpl2pk_primary_mass(
            data,
            *[params[k] for k in self.m1_hyper_var_names],
            mmax=self.mmax,
            gaussian_mass_maximum=self.gaussian_mass_maximum
        )
        log_q_given_m1 = PowerlawPlusPeak_MassRatio(
            data,
            *[params[k] for k in self.q_hyper_var_names]
        )
        return jnp.exp(log_p_m1 + log_q_given_m1)


