import jax.numpy as jnp
from jax.random import split
from jax.random import uniform

from astro import period_min, period_max
from astro import au_per_meter
from astro import seconds_per_day
from astro import days_per_year

from astro import calc_beta
from astro import calc_orbital_velocity
from astro import calc_semimajor_axis
from astro import calc_orbital_decay_time
from astro import calc_spin_orbit_misalignment
from astro import draw_kick_and_compute_new_orbit
from astro import sample_mbh_given_mprog_golomb2023
from astro import log_bpl_progenitor_mass

from models import build_interp_sampler


def sample_progenitor_masses(key, parameters):
    ms = jnp.linspace(
        parameters['prog_mmin'],
        parameters['prog_mmax'],
        500
    )

    sample_mprog = build_interp_sampler(
        lambda x: jnp.exp(log_bpl_progenitor_mass(
            x,
            parameters['alpha_prog_1'],
            parameters['alpha_prog_2'],
            parameters['prog_mmin'],
            parameters['prog_mmax'],
            parameters['prog_break_mass'],
        )),
        ms
    )

    # sample mprog
    key, subkey = split(key)
    mprog1 = sample_mprog(subkey)

    key, subkey = split(key)
    mprog2 = sample_mprog(subkey)

    return mprog1, mprog2


def initialize_stellar_binary(key, parameters):
    """
    initialize the binary progneitor masses, seperation, and orbital angles
    """
    key, subkey = split(key)
    ma, mb = sample_progenitor_masses(subkey, parameters)
    mtot_0 = ma + mb

    # sort by more/less massive star!
    mprog_a = jnp.where(ma >= mb, ma, mb)
    mprog_b = jnp.where(ma >= mb, mb, ma)

    # initial angles
    # TODO: come in here with a crude "triple" prescription?
    cos_theta_a_0 = 1.0
    cos_theta_b_0 = 1.0

    key, subkey = split(key)
    phi_a, phi_b = uniform(subkey, (2,), minval=0, maxval=2 * jnp.pi)

    # initial orbital properties
    key, subkey = split(key)
    log_period_days = uniform(
        subkey,
        minval=jnp.log(period_min),
        maxval=jnp.log(period_max)
    )  # [days]
    a0 = calc_semimajor_axis(jnp.exp(log_period_days), mtot_0)  # [ AU ]
    vorb_0 = calc_orbital_velocity(mtot_0, a0)  # [ km / s ]

    return key, dict(
        mprog_a=mprog_a,
        mprog_b=mprog_b,
        mtot_0=mtot_0,
        log_period_0=log_period_days,
        semimajor_axis_0=a0,
        orbital_velocity_0=vorb_0,
        ecc_0=0.0,
        cos_theta_a_0=cos_theta_a_0,
        cos_theta_b_0=cos_theta_b_0,
        phi_a=phi_a,
        phi_b=phi_b
    )


def first_supernova(key, parameters, binary):
    mprog_a = binary['mprog_a']
    mtot_0 = binary['mtot_0']

    key, subkey = split(key)
    mbh_a = sample_mbh_given_mprog_golomb2023(
        subkey,
        mprog_a,
        parameters['mbhmax'],
        parameters['mturnover'],
        parameters['sigma_mbh']
    )

    mtot_1 = mbh_a + binary['mprog_b']
    beta_1 = calc_beta(mtot_0, mtot_1)

    key, subkey = split(key)
    (
        v_kick_1,
        u_kick_1,
        alpha_1,
        ecc_1,
        cos_gamma_1
    ) = draw_kick_and_compute_new_orbit(
        subkey,
        beta_1,
        binary['orbital_velocity_0'],
        parameters['sigma_kick']
    )
    gamma_1 = jnp.arccos(cos_gamma_1)

    if parameters['align_a_post_sn1'] is True:
        cos_theta_a_1 = 1.0
    else:
        cos_theta_a_1 = calc_spin_orbit_misalignment(
            gamma_1,
            jnp.arccos(binary['cos_theta_a_0']),
            binary['phi_a']
        )

    if parameters['align_b_post_sn1'] is True:
        cos_theta_b_1 = 1.0
    else:
        cos_theta_b_1 = calc_spin_orbit_misalignment(
            gamma_1,
            jnp.arccos(binary['cos_theta_b_0']),
            binary['phi_b']
        )

    a1 = alpha_1 * binary['semimajor_axis_0']
    vorb_1 = calc_orbital_velocity(mtot_1, a1)

    return key, dict(
        **binary,
        mbh_a=mbh_a,
        mtot_1=mtot_1,
        #beta_1=beta_1,
        v_kick_1=v_kick_1,
        u_kick_1=u_kick_1,
        cos_theta_a_1=cos_theta_a_1,
        cos_theta_b_1=cos_theta_b_1,
        #alpha_1=alpha_1,
        semimajor_axis_1=a1,
        orbital_velocity_1=vorb_1,
        cos_gamma_1=cos_gamma_1,
        eccentricity_1=ecc_1,
    )


def second_supernova(key, parameters, binary):
    key, subkey = split(key)
    mbh_b = sample_mbh_given_mprog_golomb2023(
        subkey,
        binary['mprog_b'],
        parameters['mbhmax'],
        parameters['mturnover'],
        parameters['sigma_mbh']
    )

    mtot_1 = binary['mtot_1']
    mtot_2 = binary['mbh_a'] + mbh_b

    beta_2 = calc_beta(mtot_1, mtot_2)

    key, subkey = split(key)
    (
        v_kick_2,
        u_kick_2,
        alpha_2,
        ecc_2,
        cos_gamma_2
    ) = draw_kick_and_compute_new_orbit(
        subkey,
        beta_2,
        binary['orbital_velocity_1'],
        parameters['sigma_kick']
    )
    gamma_2 = jnp.arccos(cos_gamma_2)

    cos_theta_a_2 = calc_spin_orbit_misalignment(
        gamma_2,
        jnp.arccos(binary['cos_theta_a_1']),
        binary['phi_a']
    )

    cos_theta_b_2 = calc_spin_orbit_misalignment(
        gamma_2,
        jnp.arccos(binary['cos_theta_b_1']),
        binary['phi_b']
    )

    a2 = alpha_2 * binary['semimajor_axis_1']
    vorb_2 = calc_orbital_velocity(mtot_2, a2)

    return key, dict(
        **binary,
        mbh_b=mbh_b,
        mtot_2=mtot_2,
        #beta_2=beta_2,
        v_kick_2=v_kick_2,
        u_kick_2=u_kick_2,
        cos_theta_a_2=cos_theta_a_2,
        cos_theta_b_2=cos_theta_b_2,
        #alpha_2=alpha_2,
        semimajor_axis_2=a2,
        orbital_velocity_2=vorb_2,
        cos_gamma_2=cos_gamma_2,
        eccentricity_2=ecc_2
    )


def determine_survival(binary):
    survive = binary['semimajor_axis_2'] > 0
    survive &= binary['eccentricity_2'] < 1
    survive &= binary['mbh_b'] > 0
    return survive


def determine_merger(binary):
    decay_time = calc_orbital_decay_time(
        binary['semimajor_axis_2'] / au_per_meter,
        binary['mbh_a'],
        binary['mbh_b']
    )  # [ seconds ]

    log_decay_time = (
        jnp.log(decay_time) - jnp.log(seconds_per_day) - jnp.log(days_per_year)
    )
    return log_decay_time < jnp.log(14e9)


def get_binary(key, parameters):
    key, binary = initialize_stellar_binary(key, parameters)
    key, binary = first_supernova(key, parameters, binary)
    key, binary = second_supernova(key, parameters, binary)
    survive = determine_survival(binary)
    merge = determine_merger(binary)
    return dict(**binary, survive=survive, merge=merge)
