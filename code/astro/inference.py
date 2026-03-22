import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from jax_tqdm import scan_tqdm

import h5ify

from bilby import run_sampler
from bilby.core.prior import ConditionalPriorDict

from data import get_data
from likelihood import get_bilby_likelihood
from likelihood import taper
from models import log_powerlaw_redshift
from models import dynamical


priors = ConditionalPriorDict('./priors/test.prior')


def model(dataset, parameters):
    return (
        dynamical(dataset, parameters)
        * jnp.exp(log_powerlaw_redshift(dataset, parameters))
    )


maximum_variance = 1
npri = 1_000

sampler_kwargs = dict(
    sample="acceptance-walk",
    naccept=5,
    nlive=200
)

outdir = './test'

if __name__ == '__main__':
    _, posteriors, injections, ln_evidences = get_data(
        snr_thresh=10,
        far_thresh=1,
        prefer_xphm=False,
        prefer_xphm_gwtc3=True,
        return_ln_evidence=True
    )

    if 'log_prior' in posteriors:
        posteriors['prior'] = jnp.exp(posteriors['log_prior'])
    if 'log_prior' in injections:
        injections['prior'] = jnp.exp(injections['log_prior'])

    likelihood = get_bilby_likelihood(
        model,
        posteriors,
        injections,
        taper=lambda v: taper(maximum_variance, v),
        rate=False,
    )

    def step(carry, d):
        _, x = d
        extras = likelihood.generate_extra_statistics(x)
        return carry, extras

    prior_samples = {k : jnp.array(v) for k, v in priors.sample(npri).items()}
    _, extras = jax.lax.scan(
        scan_tqdm(npri)(step),
        None,
        (jnp.arange(npri), prior_samples)
    )
    extras['samples'] = prior_samples
    h5ify.save(f'{outdir}/prior.h5', extras, mode='w')

    result = run_sampler(
        likelihood=likelihood,
        priors=priors,
        outdir=outdir,
        label='test',
        sampler='dynesty',
        save="hdf5",
        **sampler_kwargs
    )

    result.plot_corner()

    nsamps = len(result.posterior)
    samples = result.posterior.to_dict('list')
    samples = {k : jnp.array(v) for k, v in samples.items()}

    _, extras = jax.lax.scan(
        scan_tqdm(nsamps)(step),
        None,
        (jnp.arange(nsamps), samples)
    )

    for k, v in extras.items():
        result.posterior[k] = v

    result.save_to_file(overwrite=True, extension='hdf5')
