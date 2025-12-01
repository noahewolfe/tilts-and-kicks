import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import h5ify
from jax_tqdm import scan_tqdm

from data import get_posteriors
from data import get_injections

import bilby
from bilby.core.prior import ConditionalPriorDict
from bilby_util import convert_bilby_uniform_prior
from bilby_util import LikelihoodWrapper

from likelihood import unravel
from likelihood import get_log_probs
from likelihood import taper

from flows import save
from flows import default_flow

from variational import fit
from variational import estimate_convergence

from util import write_config
from util import plot_loss

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str)
parser.add_argument('--priors', type=os.path.abspath)
parser.add_argument('--model', type=str, default='o4a-strong')
parser.add_argument('--seed', type=int, default=1)


def get_model(name):
    if name == 'o4a-strong':
        from models import bpl2p_plz_truncnormmag_isogausstilt
        return bpl2p_plz_truncnormmag_isogausstilt
    else:
        raise NotImplementedError(f'unknown model {name}')


def parse_args():
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    write_config(args)
    model = get_model(args.model)
    return args.outdir, args.priors, model, args.seed


def vi(outdir, priors, model, seed):
    posteriors, events = get_posteriors(load=True)
    injections = get_injections(load=True)

    with open(f'{outdir}/events.txt', 'w') as f:
        for e in events:
            f.write(f'{e}\n')

    priors = ConditionalPriorDict(priors)
    priors.to_file(outdir, 'run')
    param_keys, bounds, log_prior = convert_bilby_uniform_prior(priors)

    log_likelihood, log_target = get_log_probs(
        param_keys, posteriors, injections, model, log_prior
    )

    key = jax.random.key(seed)

    key, subkey = jax.random.split(key)
    flow_init = default_flow(subkey, bounds)

    key, subkey = jax.random.split(key)
    flow, loss_prior = fit(
        subkey,
        flow_init,
        log_prior,
        clip=False,
        lr=1,
        batch_size=10_000
    )

    plot_loss(loss_prior, log=True, outpath=f'{outdir}/loss-prior.png')

    key, subkey = jax.random.split(key)
    flow, loss = fit(
        subkey,
        flow,
        log_target,
        clip=True,
        lr=1e-1,
        batch_size=10,
        steps=1_000
    )

    plot_loss(loss, outpath=f'{outdir}/loss-target.png')
    save(f'{outdir}/flow.eqx', flow)

    key, subkey = jax.random.split(key)
    samples, log_q = flow.sample_and_log_prob(subkey, (n,))
    samples = jax.vmap(lambda x: unravel(param_keys, x))(samples)

    lkl, var, _, _, _ = jax.lax.map(log_likelihood, samples, batch_size=1_000)
    lpr = jax.vmap(log_prior)(samples)
    log_p = lkl + lpr

    samples['log_q'] = log_q
    samples['log_posterior'] = log_p
    samples['log_likelihood'] = lkl
    samples['log_prior'] = lpr
    samples['variance'] = var

    stats = estimate_convergence(log_p, log_q)

    h5ify.save(f'{outdir}/result.h5', dict(samples=samples, **stats))


def nest(outdir, priors, model, seed, maximum_variance=1):
    posteriors, events = get_posteriors(load=True)

    if 'mass_1_source' in posteriors.keys():
        posteriors['mass_1'] = posteriors.pop('mass_1_source')

    injections = get_injections(load=True)
    injections['total_generated'] = injections.pop('total')
    if 'mass_1_source' in injections.keys():
        injections['mass_1'] = injections.pop('mass_1_source')

    with open(f'{outdir}/events.txt', 'w') as f:
        for e in events:
            f.write(f'{e}\n')

    priors = ConditionalPriorDict(priors)
    priors.to_file(outdir, 'run')

    likelihood_kwargs = dict(
        taper=lambda v: taper(maximum_variance, v),
        rate=False,
        tobs=None
    )

    likelihood = LikelihoodWrapper(
        posteriors,
        injections,
        model,
        **likelihood_kwargs
    )

    likelihood.parameters.update(priors.sample())
    print('log like ratio: ', likelihood.log_likelihood_ratio())

    bilby.core.utils.random.seed(seed)

    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        outdir=outdir,
        sampler='dynesty',
        nlive=50,
        label='dynesty',
        save='hdf5',
        resume=False
    )

    nsamples = len(result.posterior)
    samples = {
        k : jnp.array(v)
        for k, v in result.posterior.items()
    }

    @scan_tqdm(nsamples)
    def step(_, x):
        _, d = x
        extras = likelihood.generate_extra_statistics(d)
        return None, extras

    _, extras = jax.lax.scan(
        step,
        None,
        (jnp.arange(nsamples), samples)
    )

    for key in extras.keys():
        samples[key] = extras[key]

    h5ify.save(f'{outdir}/extras.h5', samples, mode='w')


if __name__ == '__main__':
    outdir, priors, model, seed = parse_args()
    nest(outdir, priors, model, seed)
    print('done.')
