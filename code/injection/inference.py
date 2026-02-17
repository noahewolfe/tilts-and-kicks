import os
import json
from argparse import ArgumentParser
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import h5ify
import numpy as np
import matplotlib.pyplot as plt

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from jax_tqdm import scan_tqdm

# TODO: monkey patch because bilby dynesty wrapper
# is probably not compatible with numpy version < 2
if not hasattr(np.linalg, "linalg"):
    np.linalg.linalg = np.linalg

from bilby import run_sampler
from bilby.core.prior import ConditionalPriorDict

from pixelpop.models.gwpop_models import PowerlawPlusPeak_MassRatio
from pixelpop.models.gwpop_models import trunc_gaussian

from data import load_posteriors
from data import load_salvo_posteriors

from likelihood import get_bilby_likelihood

from models import log_powerlaw_redshift
from models import log_iid_spin_mag_truncnorm
from models import log_nid_iso_gauss_tilt
from models import BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth
from models import log_marg_iso_gauss_spin_tilt

from chieff_ppd import monte_carlo_sample_chieff_and_kde

from util import calc_chieff

parser = ArgumentParser()
parser.add_argument('--outdir', type=str)
parser.add_argument('--posteriors', type=str, help='path to posteriors')
parser.add_argument('--injections', type=str, help='path to vt file')
parser.add_argument(
    '--truths',
    help='path to json file with true parameters or a histogram of injections'
)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--nobs', default=70, type=int)
parser.add_argument('--cut', default=15, type=int)
parser.add_argument('--maximum-variance', default=5, type=int)
parser.add_argument('--deltas', action='store_true')


def parse_args():
    """ return command-line arguments """
    args = parser.parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(f'{outdir}/ppds', exist_ok=True)

    return (
        outdir,
        args.cut,
        args.injections,
        args.posteriors,
        args.truths,
        args.seed,
        args.nobs,
        args.maximum_variance,
        args.deltas
    )


def load_astro_distribution(path):
    """ load either the hyperparameters or draws from the astro. dist. """
    kind = 'draws'
    _, ext = os.path.splitext(path)
    if ext == '.json':
        with open(path, 'r') as f:
            truths = json.loads(f.read())
            truths = {k : np.asarray(v).item() for k, v in truths.items()}
        kind = 'hyperparameters'
    elif ext == '.dat' or ext in ['.h5', '.hdf5']:
        from pandas import read_csv

        if ext == '.dat':   # Salvo-style
            truths = read_csv(path, header=0, sep='\t')
            truths = {
                k : np.array(v)
                for k, v in truths.to_dict(orient='list').items()
            }
        elif ext in ['.h5', '.hdf5']:
            truths = h5ify.load(path)

        if 'mass_1_source' in truths:
            truths['mass_1'] = truths.pop('mass_1_source')

        for i in [1, 2]:
            if f'tilt_{i}' in truths and f'cos_tilt_{i}' not in truths:
                truths[f'cos_tilt_{i}'] = np.cos(truths.pop(f'tilt_{i}'))

        if 'chi_eff' not in truths:
            truths['chi_eff'] = calc_chieff(
                truths['mass_ratio'],
                truths['a_1'],
                truths['a_2'],
                truths['cos_tilt_1'],
                truths['cos_tilt_2']
            )
    else:
        raise ValueError(f'Unknown extension {ext} on truths file')

    return kind, truths


def log_model(dataset, parameters):
    """ log-density of the population model """
    if 'lam_2' not in parameters:
        lam_2 = 1 - parameters['lam_1'] - parameters['lam_0']
    else:
        lam_2 = parameters['lam_2']

    log_p_m1 = BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth(
        dataset['mass_1'],
        alpha_1=parameters['alpha_1'],
        alpha_2=parameters['alpha_2'],
        mlow_1=parameters['mlow_1'],
        break_mass=parameters['break_mass'],
        delta_m_1=parameters['delta_m_1'],
        lam_fractions=(
            parameters['lam_0'], parameters['lam_1'], lam_2
        ),
        mpp_1=parameters['mpp_1'],
        sigpp_1=parameters['sigpp_1'],
        mpp_2=parameters['mpp_2'],
        sigpp_2=parameters['sigpp_2'],
        mmax=300.0,
        gaussian_mass_maximum=100.0
    )

    log_p_q = PowerlawPlusPeak_MassRatio(
        dataset,
        slope=parameters['beta'],
        minimum=parameters['mlow_1'],
        delta_m=parameters['delta_m_1']
    )

    log_p_z = log_powerlaw_redshift(dataset, parameters)

    log_p_chi = log_iid_spin_mag_truncnorm(dataset, parameters)

    if 'log_sigma_spin' in parameters:
        parameters['sigma_spin'] = jnp.exp(parameters['log_sigma_spin'])
    log_p_tau = log_nid_iso_gauss_tilt(dataset, parameters)

    return log_p_m1 + log_p_q + log_p_z + log_p_chi + log_p_tau


def get_posteriors(key, outdir, path, nobs, deltas=False):
    """ load noah or salvo posteriors, inferring kind based on ext """
    _, ext = os.path.splitext(path)

    if ext in ['.hdf5', '.h5']:
        posteriors = load_posteriors(path, deltas=deltas)
    elif ext in ['.pkl']:
        if deltas:
            raise NotImplementedError('No delta functions at salvo posts yet')
        posteriors = load_salvo_posteriors(path)
    else:
        raise ValueError(f'unknown posteriors extension {ext}')

    if nobs == len(posteriors['log_prior']):
        return posteriors
    else:
        idxs = jax.random.choice(
            key,
            len(posteriors['log_prior']),
            shape=(nobs,),
            replace=False
        )
        idxs = jnp.sort(idxs)

        np.save(f'{outdir}/idxs.npy', idxs)

        posteriors = {k : v[idxs] for k, v in posteriors.items()}

        h5ify.save(f'{outdir}/posteriors.h5', posteriors, mode='w')

    return posteriors


def get_injections(path, cut=15):
    injections = h5ify.load(path)
    injections['mass_1'] = injections.pop('mass_1_source')
    print(f'applying an snr cut of {cut}')
    mask = injections['network_matched_filter_snr'] > cut
    for k in list(injections.keys()):
        if k not in ['attrs', 'total_generated', 'model', 'parameters']:
            injections[k] = injections[k][mask]
    if 'prior' in injections and 'log_prior' not in injections:
        injections['log_prior'] = np.log(injections.pop('prior'))
    return injections


def taper(maximum_variance, v):
    """ its actually a hard cut! """
    return jnp.nan_to_num(-1e10 * (v >= maximum_variance), nan=0)


def make_ppds(outdir, truths, posterior, kind):
    """ calculate and plot in m1, q (marg over m1), a1, a2, ct1, ct2 """
    mmin, mmax = 3.0, 300.0
    m1_grid = jnp.linspace(mmin, mmax, 1000)

    m1_grid_for_q = jnp.linspace(mmin, mmax, 500)
    q_grid = jnp.linspace(0.1, 1.0, 500)
    mm, qq = jnp.meshgrid(m1_grid_for_q, q_grid, indexing='ij')

    a_grid = jnp.linspace(0.0, 1.0, 500)
    ct_grid = jnp.linspace(-1.0, 1.0, 500)

    def ppd_for_sample(parameters):
        lam_2 = 1 - parameters['lam_0'] - parameters['lam_1']

        log_p_m1 = BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth(
            dict(mass_1=m1_grid),
            alpha_1=parameters['alpha_1'],
            alpha_2=parameters['alpha_2'],
            mlow_1=parameters['mlow_1'],
            break_mass=parameters['break_mass'],
            delta_m_1=parameters['delta_m_1'],
            lam_fractions=(parameters['lam_0'], parameters['lam_1'], lam_2),
            mpp_1=parameters['mpp_1'],
            sigpp_1=parameters['sigpp_1'],
            mpp_2=parameters['mpp_2'],
            sigpp_2=parameters['sigpp_2'],
            mmax=300.0,
            gaussian_mass_maximum=100.0
        )
        p_m1 = jnp.exp(log_p_m1)

        log_p_m1_for_q = BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth(
            dict(mass_1=mm),
            alpha_1=parameters['alpha_1'],
            alpha_2=parameters['alpha_2'],
            mlow_1=parameters['mlow_1'],
            break_mass=parameters['break_mass'],
            delta_m_1=parameters['delta_m_1'],
            lam_fractions=(parameters['lam_0'], parameters['lam_1'], lam_2),
            mpp_1=parameters['mpp_1'],
            sigpp_1=parameters['sigpp_1'],
            mpp_2=parameters['mpp_2'],
            sigpp_2=parameters['sigpp_2'],
            mmax=300.0,
            gaussian_mass_maximum=100.0
        )
        log_p_q_given_m1 = PowerlawPlusPeak_MassRatio(
            dict(mass_1=mm, mass_ratio=qq),
            slope=parameters['beta'],
            minimum=parameters['mlow_1'],
            delta_m=parameters['delta_m_1']
        )
        p_q = jnp.trapezoid(
            y=jnp.exp(log_p_q_given_m1 + log_p_m1_for_q),
            x=m1_grid_for_q,
            axis=0
        )

        if 'sigma_spin' in parameters:
            sigma_spin = parameters['sigma_spin']
        else:
            sigma_spin = jnp.exp(parameters['log_sigma_spin'])

        log_p_a = trunc_gaussian(
            a_grid,
            parameters['mu_chi'],
            parameters['sigma_chi'],
            lower=0,
            upper=1
        )
        p_a = jnp.exp(log_p_a)

        log_p_ct = log_marg_iso_gauss_spin_tilt(
            ct_grid,
            parameters['xi_spin'],
            sigma_spin,
            mu_spin=parameters['mu_spin']
        )
        p_ct = jnp.exp(log_p_ct)

        return dict(
            mass_1=p_m1,
            mass_ratio=p_q,
            a_1=p_a,
            a_2=p_a,
            cos_tilt_1=p_ct,
            cos_tilt_2=p_ct
        )

    n = len(next(iter(posterior.values())))

    @scan_tqdm(n)
    def step(carry, d):
        _, x = d
        return carry, ppd_for_sample(x)

    _, ppds = jax.lax.scan(step, None, (jnp.arange(n), posterior))

    mc_ppd = monte_carlo_sample_chieff_and_kde()
    ppds['chi_eff'] = jnp.exp(mc_ppd['log_prob'])
    ppds['monte_carlo_chi_eff_samples'] = {
        k : v for k, v in mc_ppd.items() if k not in ['log_prob']
    }

    xs = dict(
        mass_1=np.array(m1_grid),
        mass_ratio=np.array(q_grid),
        a_1=np.array(a_grid),
        a_2=np.array(a_grid),
        cos_tilt_1=np.array(ct_grid),
        cos_tilt_2=np.array(ct_grid),
        chi_eff=np.linspace(-1, 1, 500)
    )
    ppds = {k: np.array(v) for k, v in ppds.items()}
    medians = {k: np.median(v, axis=0) for k, v in ppds.items()}
    q05 = {k : np.quantile(v, 0.05, axis=0) for k, v in ppds.items()}
    q95 = {k : np.quantile(v, 0.95, axis=0) for k, v in ppds.items()} 

    data = dict(xs=xs, ppd=ppds, medians=medians, q05=q05, q95=q95)
    h5ify.save(f'{outdir}/ppds.h5', data, mode='w')

    if kind == 'hyperparameters':
        p_true = ppd_for_sample(truths)
        # TODO: we could do MC chieff here for just the truths and KDE result

    plot_order = [
        ('mass_1', r'$m_1$'),
        ('mass_ratio', r'$q$'),
        ('a_1', r'$a_1$'),
        ('a_2', r'$a_2$'),
        ('cos_tilt_1', r'$\cos\theta_1$'),
        ('cos_tilt_2', r'$\cos\theta_2$'),
        ('chi_eff', r'$\chi_\mathrm{eff}$')
    ]
    for (k, label) in plot_order:
        fig, ax = plt.subplots()

        if kind == 'hyperparameters' and k != 'chi_eff':
            ax.plot(xs[k], p_true[k], lw=1.7, color='black', linestyle='--')
        else:
            ax.hist(
                truths[k],
                histtype='step',
                bins=50,
                density=True,
                lw=1.7,
                color='black',
                linestyle='--'
            )

        ax.fill_between(xs[k], q05[k], q95[k], alpha=0.25, color='C0')
        ax.plot(xs[k], medians[k], lw=1.7, color='C0')
        ax.set_xlabel(label)
        ax.set_ylabel('density')

        if k == 'mass_1':
            ax.loglog()
            ax.set_ylim(1e-6, 5e0)
        elif 'cos_tilt' in k:
            ax.semilogy()
            ax.set_ylim(2e-1, 3e1)

        fig.tight_layout()
        fig.savefig(f'{outdir}/ppds/{k}.png', dpi=200)
        plt.close(fig)


if __name__ == '__main__':
    (
        outdir,
        cut,
        injections,
        posteriors,
        truths,
        seed,
        nobs,
        maximum_variance,
        deltas
    ) = parse_args()

    priors = ConditionalPriorDict('./priors/lvk.prior')
    kind, truths = load_astro_distribution(truths)

    injections = get_injections(injections, cut=cut)

    key = jax.random.key(seed)
    posteriors = get_posteriors(key, outdir, posteriors, nobs, deltas=deltas)

    likelihood = get_bilby_likelihood(
        log_model,
        posteriors,
        injections,
        taper=lambda v: taper(maximum_variance, v),
        rate=False
    )

    if kind == 'hyperparameters':
        truths['variance'] = 0
        if 'log_sigma_spin' in priors:
            truths['log_sigma_spin'] = np.log(truths.pop('sigma_spin')).item()
        print(
            'lnl at truths: ', likelihood.log_likelihood(parameters=truths)
        )
        print(
            'extras at truths: ', likelihood.generate_extra_statistics(truths)
        )

    result = run_sampler(
        likelihood=likelihood,
        priors=priors,
        outdir=outdir,
        label='run',
        sampler='dynesty',
        sample='acceptance-walk',
        naccept=5,
        nlive=250,
        # need enough live points to resolve w/in and w/o variance cut ...
    )

    posterior = result.posterior.to_dict('list')
    posterior = {k : jnp.array(v) for k, v in posterior.items()}

    n = len(result.posterior)

    @scan_tqdm(n)
    def step(carry, d):
        _, x = d
        return carry, likelihood.generate_extra_statistics(x)

    _, extras = jax.lax.scan(step, None, (jnp.arange(n), posterior))
    posterior = dict(**posterior, **extras)
    h5ify.save(f'{outdir}/extras.h5', posterior, mode='w')

    result.posterior['variance'] = posterior['variance']

    if kind == 'hyperparameters':
        for k in list(truths.keys()):
            if k not in result.posterior:
                truths.pop(k)
        result.plot_corner(truths=truths)
    else:
        result.plot_corner(parameters=list(result.posterior.keys()))

    make_ppds(outdir, truths, posterior, kind)

    print('done.')
