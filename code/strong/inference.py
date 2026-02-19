import os
from argparse import ArgumentParser

import h5ify

import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm

from bilby import run_sampler
from bilby.core.prior import ConditionalPriorDict

#from pixelpop.models.gwpop_models import PowerlawPlusPeak_MassRatio

from data import get_data

#from models import log_stegmann_spin
from models import log_powerlaw_redshift
#from models import BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth

from models import log_threemass_stegmann_spin

from likelihood import get_bilby_likelihood
from util import write_config
from util import scan

parser = ArgumentParser()
parser.add_argument('--outdir')
parser.add_argument('--priors', type=str, required=True)
parser.add_argument('--nlive', default=150, type=int)
parser.add_argument('--nprior', default=0, type=int)
parser.add_argument('--maximum-variance', default=1, type=float)


def taper(maximum_variance, v):
    """ its actually a hard cut! """
    return jnp.nan_to_num(-1e6 * (v >= maximum_variance), nan=0)


def log_model(dataset, parameters):
    """ log-density of the population model """
    log_m1q_chi_tau = log_threemass_stegmann_spin(dataset, parameters)
    log_p_z = log_powerlaw_redshift(dataset, parameters)
    return log_m1q_chi_tau + log_p_z


def parse_args():
    args = parser.parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    write_config(args, outdir=outdir)
    return outdir, args.priors, args.nprior, args.maximum_variance, args.nlive


if __name__ == '__main__':
    outdir, priors, npri, maximum_variance, nlive = parse_args()

    priors = ConditionalPriorDict(priors)
    priors.to_file(outdir, 'run')

    events, posteriors, injections = get_data(
        snr_thresh=10,
        far_thresh=1,
        prefer_xphm=False
    )

    likelihood = get_bilby_likelihood(
        log_model,
        posteriors,
        injections,
        taper=lambda v: taper(maximum_variance, v),
        rate=False
    )

    if npri > 0:
        prior_samples = {k : jnp.array(v) for k, v in priors.sample(npri).items()}
        extras = scan(likelihood.generate_extra_statistics)(prior_samples)
        extras['samples'] = prior_samples
        h5ify.save(f'{outdir}/prior.h5', extras, mode='w')

    result = run_sampler(
        likelihood=likelihood,
        priors=priors,
        outdir=outdir,
        label='run',
        sampler='dynesty',
        sample='acceptance-walk',
        naccept=5,
        nlive=nlive,
    )

    result.plot_corner()

    nsamps = len(result.posterior)
    samples = result.posterior.to_dict('list')
    samples = {k : jnp.array(v) for k, v in samples.items()}

    extras = scan(likelihood.generate_extra_statistics)(samples)
    h5ify.save(f'{outdir}/posterior.h5', extras, mode='w')
