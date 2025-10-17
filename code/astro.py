import jax
from jax.random import split
import jax.numpy as jnp

from pixelpop.models.gwpop_models import BrokenPowerLaw
from models import build_interp_sampler
from util import monotonic_select

seconds_per_day = 86_400 # [ s / day ]
days_per_year = 365

grav_constant = 1.3e20 #* seconds_per_day**2 # [meters^3 / sec^2 / Msun]
speed_of_light = 3e8 #* seconds_per_day # [meters / sec]


def calc_chieff(q, a1, a2, ct1, ct2):
    chieff = a1 * ct1 + q * a2 * ct2
    return chieff / (1 + q)


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


def sample_mbh_given_mprog_golomb2023(
    key, mprog, mbhmax, mturnover, sigma_mbh
):
    """ sample black hole masses from log normal where mean follows
        Golomb+2023 remnant mass prescription
    """
    mean_mbh = remnant_mass_golomb2023(mprog, mbhmax, mturnover)
    ln_mbh = jax.random.normal(key) * sigma_mbh + jnp.log(mean_mbh)
    return jnp.exp(ln_mbh)


def log_bpl_progenitor_mass(mprog, alpha_1, alpha_2, mmin, mmax, mbreak):
    """ broken powerlaw in progenitor masses """
    break_fraction = (mbreak - mmin) / (mmax - mmin)
    return BrokenPowerLaw(
        mprog, -alpha_1, -alpha_2, mmin, mmax, break_fraction
    )


def draw_maxwellian_velocity(key, rms):
    x = jax.random.normal(key, (3,))
    return x * rms


def beefy_term(beta, u_vec):
    ux, uy, uz = u_vec 
    return 2 * beta - ux**2 - (uy + 1)**2 - uz**2


def calc_relative_semimajoraxis(beta, u_vec):
    numer = beta
    denom = beefy_term(beta, u_vec)
    return numer / denom


def calc_e2(beta, u_vec):
    """ returns square of eccentricity """
    _, uy, uz = u_vec
    numer = (uz**2 + (uy + 1)**2) * beefy_term(beta, u_vec)
    denom = beta**2
    return 1 - numer / denom


def calc_cos_theta(u_vec):
    _, uy, uz = u_vec
    numer = uy + 1
    denom = jnp.sqrt(uz**2 + (uy + 1)**2)
    return numer / denom


def calc_orbital_decay_time(a, m1, m2):
    """ From Peters (1964), Eq. (5.10) -- time [days] for orbit to decay """
    prefactor = 64 / 5 * grav_constant**3 / speed_of_light**5
    beta = prefactor * m1 * m2 * (m1 + m2)
    return a**4 / 4 / beta


# from: https://arxiv.org/abs/2010.16333
period_min = 0.4  # [ days ]
period_max = 10**(5.5)  # [ days ]

mprog_min = 3
mprog_max = 80
mprog_break = 20


def draw_kick_and_compute_new_orbit(key, beta, vorb, sigma_kick):
    vkick_vec = draw_maxwellian_velocity(key, sigma_kick)
    u_vec = vkick_vec / vorb

    alpha = calc_relative_semimajoraxis(beta, u_vec)
    ecc2 = calc_e2(beta, u_vec)
    ecc = jnp.sqrt(ecc2)
    cos_theta = calc_cos_theta(u_vec)

    return vkick_vec, u_vec, alpha, ecc, cos_theta


def get_binary(key, parameters):
    ms = jnp.linspace(
        mprog_min,
        mprog_max,
        500
    )

    sample_mprog = build_interp_sampler(
        lambda x: jnp.exp(log_bpl_progenitor_mass(
            x,
            parameters['alpha_prog_1'],
            parameters['alpha_prog_2'],
            mprog_min,
            mprog_max,
            mprog_break,
        )),
        ms
    )

    # sample mprog
    key, subkey = split(key)
    mprog1 = sample_mprog(subkey)

    key, subkey = split(key)
    mbh1 = sample_mbh_given_mprog_golomb2023(
        subkey,
        mprog1,
        parameters['mbhmax'],
        parameters['mturnover'],
        parameters['sigma_mbh']
    )

    key, subkey = split(key)
    mprog2 = sample_mprog(subkey)

    key, subkey = split(key)
    mbh2 = sample_mbh_given_mprog_golomb2023(
        subkey,
        mprog2,
        parameters['mbhmax'],
        parameters['mturnover'],
        parameters['sigma_mbh']
    )

    mtot_i = mbh1 + mprog2
    mtot_f = mbh1 + mbh2

    beta = mtot_f / mtot_i

    gmi = grav_constant * mtot_i

    key, subkey = split(key)
    log_period_days = jax.random.uniform(
        subkey, minval=jnp.log(period_min), maxval=jnp.log(period_max)
    )  # [days]
    period_sec_squared = jnp.exp(
        2 * (log_period_days + jnp.log(seconds_per_day))
    )

    # kepler's 3rd law! in units of [m]
    ai = (period_sec_squared * gmi / 4 / jnp.pi**2)**(1 / 3)

    # TODO: here we assume r = a...but instead we should marginalize over
    # orbital phase! or at least compare!
    # well...oth...if we assume a circular orbit, r = a always
    # vorb^2 = G * mtot * (2 / r - 1 / a)
    vorb = jnp.sqrt(gmi / ai)

    key, subkey = split(key)
    vkick_vec, u_vec, alpha, ecc, cos_theta = draw_kick_and_compute_new_orbit(
        subkey, beta, vorb, parameters['sigma_kick']
    )

    survive = (alpha > 0) & (ecc < 1) & (mbh2 > 0)

    af = ai / alpha
    decay_time = calc_orbital_decay_time(af, mbh1, mbh2)

    merge = decay_time / seconds_per_day / days_per_year < 14e9

    return dict(
        mprog1=mprog1,
        mprog2=mprog2,
        mbh1=mbh1,
        mbh2=mbh2,
        beta=beta,
        log_period=log_period_days,
        alpha=alpha,
        ai=ai,
        af=af,
        vorb=vorb,
        vkick_vec=vkick_vec,
        u_vec=u_vec,
        ecc=ecc,
        cos_theta=cos_theta,
        decay_time=decay_time,
        survive=survive,
        merge=merge,
    )


def get_merging_binary(key, parameters):
    """ sample binaries until we get one that actually survived & merged """
    def cond(carry):
        _, binary = carry
        return ~jnp.logical_and(binary['survive'], binary['merge'])

    def body(carry):
        key, _ = carry
        key, subkey = split(key)
        return key, get_binary(subkey, parameters)

    init = body((key, None))
    _, binary = jax.lax.while_loop(cond, body, init)
    return binary
