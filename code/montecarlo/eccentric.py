import jax.numpy as jnp

from astro import grav_constant
from astro import draw_maxwellian_velocity


def calc_new_semimajor_axis(total_mass_final, vkick_vec, vorb, seperation):
    """ arXiv:2202.05892, Eq. 40 """
    vkick_sq = jnp.sum(vkick_vec**2)
    vkick_y = vkick_vec[1]

    af = 2 / seperation
    af -= (
        (vkick_sq + vorb**2 + 2 * vorb * vkick_y)
        / grav_constant
        / total_mass_final
    )
    af = 1 / af

    return af


def calc_new_eccentricity(
    total_mass_initial,
    semimajor_axis_initial,
    eccentricity,
    seperation,
    orbital_velocity,
    vkick_vec,
    semimajor_axis_final
):
    """ arXiv:2202.05892, Eqs. 41 and 42 """
    sin_psi = jnp.sqrt(
        grav_constant
        * total_mass_initial
        * (1 - eccentricity)**2
        * semimajor_axis_initial
    )
    sin_psi = sin_psi / orbital_velocity / seperation
    cos_psi = jnp.cos(jnp.arcsin(sin_psi))

    vx, vy, vz = vkick_vec

    ecc = vz**2 + (
        sin_psi * (orbital_velocity + vy) - cos_psi * vx
    )**2
    ecc *= seperation**2
    ecc = ecc / grav_constant / total_mass_initial / semimajor_axis_final
    ecc = jnp.sqrt(1 - ecc)

    return ecc


def estimate_eccentric_anomaly(mean_anomaly):
    """
    TODO: numerical solve. start with mikkola's solution
    and then pass to `optimistix.Newton`
    """


def calc_


def draw_kick_and_compute_new_orbit(
    key,
    total_mass_initial,
    total_mass_final,
    semimajor_axis_initial,
    orbital_velocity,
    eccentricity,
    seperation,
    sigma_kick,
):
    vkick_vec = draw_maxwellian_velocity(key, sigma_kick)
    u_vec = vkick_vec / orbital_velocity

    af = calc_new_semimajor_axis(
        total_mass_final,
        vkick_vec,
        orbital_velocity,
        seperation
    )

    ef = calc_new_eccentricity(
        total_mass_initial,
        semimajor_axis_initial,
        eccentricity,
        seperation,
        orbital_velocity,
        vkick_vec,
        af
    )

    # TODO: cos_theta;
    # https://github.com/POSYDON-code/POSYDON/blob/a277ecae7cadadb98c6b6902d8f7a48c13adee67/posydon/binary_evol/SN/step_SN.py#L1816

    return vkick_vec, u_vec, af, ef
