import os
from argparse import ArgumentParser

import jax.numpy as jnp

from bilby.core.prior import Uniform, TruncatedGaussian

from models import log_stegmann_spin
from models import log_powerlaw_redshift
from pixelpop.models.gwpop_models import powerlaw
from pixelpop.models.gwpop_models import trunc_gaussian

from inference import run

from util import write_config

parser = ArgumentParser()
parser.add_argument('--outdir')
parser.add_argument('--nlive', default=150, type=int)
parser.add_argument('--nprior', default=0, type=int)
parser.add_argument('--maximum-variance', default=1, type=float)


def parse_args():
    args = parser.parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    write_config(args, outdir=outdir)
    return outdir, args.nprior, args.maximum_variance, args.nlive


def log_powerlaw_primary_mass(dataset, parameters):
    m1 = dataset.get('mass_1_source', dataset.get('mass_1'))
    lam = parameters['lam']
    mmin = parameters['mmin']
    slope = -parameters['alpha']

    pow = powerlaw(
        data=m1,
        slope=slope,
        minimum=mmin,
        maximum=parameters['mmax']
    )
    pow += jnp.log(1 - lam)

    norm = trunc_gaussian(
        data=m1,
        mean=parameters['mu_m'],
        sig=parameters['sigma_m'],
        lower=mmin,
        upper=100

    )
    norm += jnp.log(lam)

    return jnp.logaddexp(pow, norm)


def log_powerlaw_mass_ratio(dataset, parameters):
    m1 = dataset.get('mass_1_source', dataset.get('mass_1'))
    return powerlaw(
        data=dataset['mass_ratio'],
        slope=parameters['beta'],
        minimum=parameters['mmin'] / m1,
        maximum=jnp.ones_like(m1)
    )


def log_model(dataset, parameters):
    log_p_z = log_powerlaw_redshift(dataset, parameters)
    log_p_m1 = log_powerlaw_primary_mass(dataset, parameters)
    log_p_q = log_powerlaw_mass_ratio(dataset, parameters)
    log_spin = log_stegmann_spin(dataset, parameters)
    return log_p_z + log_p_m1 + log_p_q + log_spin


priors = dict(
    lam=[0, 1],
    alpha=[-2, 4],
    mmin=[2, 2.5],
    mmax=[80, 100],
    mu_m=[10, 50],
    sigma_m=[1, 10],
    beta=[-4, 12],
    lamb=[-1, 10],

    mu_chi=Uniform(minimum=0, maximum=1),
    mu_chi_iso=Uniform(minimum=0, maximum=1),
    mu_chi_high_iso=Uniform(minimum=0, maximum=1),

    sigma_chi=Uniform(minimum=0.1, maximum=1),
    sigma_chi_iso=Uniform(minimum=0.1, maximum=1),
    sigma_chi_high_iso=Uniform(minimum=0.1, maximum=1),

    xi_spin=Uniform(minimum=0, maximum=1),
    mu_spin=Uniform(minimum=-1, maximum=1),

    # Tab.1, row 9
    sigma_spin=TruncatedGaussian(mu=0, sigma=0.5, minimum=0.1, maximum=4),

    transition_mass=Uniform(minimum=10, maximum=100)
)


if __name__ == '__main__':
    from data import get_data

    outdir, nprior, maximum_variance, nlive = parse_args()

    for k, v in priors.items():
        if isinstance(v, list):
            priors[k] = Uniform(minimum=v[0], maximum=v[1])

    events, posteriors, injections = get_data(
        snr_thresh=10,
        far_thresh=1,
        prefer_xphm=False,
        prefer_xphm_gwtc3=True
    )

    run(
        outdir,
        priors,
        log_model,
        posteriors,
        injections,
        nlive,
        nprior=nprior,
        maximum_variance=maximum_variance
    )
