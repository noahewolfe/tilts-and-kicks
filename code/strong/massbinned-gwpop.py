"""Inference script for the mass-binned spin population model."""
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import bilby as bb
from bilby.hyper.model import Model
from bilby.core.prior import ConditionalPriorDict
from bilby.core.prior import Uniform
from bilby.core.prior import TruncatedNormal
from bilby.core.prior import DirichletElement

import gwpopulation as gwpop
from gwpopulation.experimental.jax import JittedLikelihood
gwpop.set_backend("jax")

xp = gwpop.utils.xp
import jax
import jax.numpy as jnp

from jax_tqdm import scan_tqdm

import h5ify
from util import scan
from util import plot_corner
from util import plot_multiple
from util import write_config
from util import calc_chieff
from likelihood import likelihood_extras

from models import bpl2p_m1q
from models import make_massbinned_spin_model
from models import BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth
from pixelpop.models.gwpop_models import PowerlawPlusPeak_MassRatio
from pixelpop.models.gwpop_models import trunc_gaussian

label = 'run'

parser = ArgumentParser()
parser.add_argument('--outdir', type=str, required=True)
parser.add_argument('--which-data', type=str, required=True)
parser.add_argument('--sampling-seed', type=int, default=1701)
parser.add_argument('--maximum-uncertainty', required=True)
parser.add_argument('--priors', type=str, required=True)
parser.add_argument('--sampler-settings', type=str, default='fast')
parser.add_argument('--nlive', type=int, default=100)
parser.add_argument(
    '--bin-edges', type=str, required=True,
    help='Comma-separated mass bin edges, e.g. 3,14,45,300'
)
parser.add_argument(
    '--dynamic',
    action='store_true',
    help='Use DynamicDynesty instead of static Dynesty'
)

args = parser.parse_args()
outdir = args.outdir
os.makedirs(f'{outdir}/ppds', exist_ok=True)
os.makedirs(f'{outdir}/corners', exist_ok=True)
write_config(args)

which_data = args.which_data
sampling_seed = args.sampling_seed
maximum_uncertainty = args.maximum_uncertainty
sampler_settings = args.sampler_settings
nlive = args.nlive
dynamic = args.dynamic

bin_edges = [float(x) for x in args.bin_edges.split(',')]
n_bins = len(bin_edges) - 1
interior_edges = jnp.array(bin_edges[1:-1])
print(f'mass bins: {n_bins} bins with edges {bin_edges}')

### --- Data --- ###

ln_evidences = None

if which_data == 'stegmann':
    print('Using stegmann data')
    datadir = '../../data/stegmann'
    posteriors = pd.read_pickle(f"{datadir}/gwtc4_posteriors.pkl")
    injections = pd.read_pickle(f"{datadir}/gwtc4_injections_dict.pkl")

elif which_data == 'noah':
    print('Using noah data')
    from data import get_data
    _, posteriors, injections, ln_evidences = get_data(
        snr_thresh=10,
        far_thresh=1,
        prefer_xphm=False,
        prefer_xphm_gwtc3=True,
        return_ln_evidence=True
    )

    if 'log_prior' in posteriors:
        posteriors['prior'] = xp.exp(posteriors['log_prior'])
    if 'log_prior' in injections:
        injections['prior'] = xp.exp(injections['log_prior'])

    posteriors = [
        pd.DataFrame.from_dict(
            {k: v[i] for k, v in posteriors.items()},
            orient='columns'
        )
        for i in range(posteriors['prior'].shape[0])
    ]
else:
    raise ValueError(f'bad data {which_data}')

### --- Model --- ###

if maximum_uncertainty == 'inf':
    maximum_uncertainty = xp.inf
else:
    maximum_uncertainty = int(maximum_uncertainty)
print(f'using variance cut : {maximum_uncertainty}')

if sampler_settings == 'fast':
    sampler_kwargs = dict(
        sample="acceptance-walk",
        naccept=5,
    )
elif sampler_settings == 'robust':
    sampler_kwargs = dict()

massbinned_spin = make_massbinned_spin_model(bin_edges)


def get_model():
    model_functions = [
        bpl2p_m1q,
        massbinned_spin,
        gwpop.models.redshift.PowerLawRedshift(cosmo_model="Planck15"),
    ]
    return Model(model_functions=model_functions, cache=False)


vt = gwpop.vt.ResamplingVT(
    model=get_model(),
    data=injections,
    n_events=len(posteriors)
)

np.random.seed(42)

likelihood = gwpop.hyperpe.HyperparameterLikelihood(
    posteriors=posteriors,
    hyper_prior=get_model(),
    selection_function=vt,
    maximum_uncertainty=maximum_uncertainty,
    ln_evidences=ln_evidences,
)

### --- Priors --- ###

priors = ConditionalPriorDict(args.priors)

# per-bin spin priors
for k in range(n_bins):
    priors[f'mu_chi_{k}'] = Uniform(
        minimum=0, maximum=1,
        latex_label=f"$\\mu_{{\\chi,{k}}}$"
    )
    priors[f'sigma_chi_{k}'] = Uniform(
        minimum=0.1, maximum=1,
        latex_label=f"$\\sigma_{{\\chi,{k}}}$"
    )
    priors[f'mu_tilt_{k}'] = Uniform(
        minimum=-1, maximum=1,
        latex_label=f"$\\mu_{{t,{k}}}$"
    )
    priors[f'sigma_tilt_{k}'] = Uniform(
        minimum=0.1,
        maximum=5,
        latex_label=f"$\\sigma_{{t,{k}}}$"
    )

# redshift
priors["lamb"] = Uniform(minimum=-1, maximum=10, latex_label="$\\lambda_{z}$")

priors.to_file(outdir, label)

jit_likelihood = JittedLikelihood(likelihood)
ll = jit_likelihood.log_likelihood_ratio(priors.sample())
print('log likelihood ratio :', ll)

### --- Sample --- ###

result = bb.run_sampler(
    likelihood=jit_likelihood,
    priors=priors,
    sampler="dynesty" if not dynamic else "DynamicDynesty",
    label=label,
    nlive=nlive,
    save="hdf5",
    outdir=outdir,
    seed=sampling_seed,
    **sampler_kwargs
)

### --- Post-process --- ###

if ln_evidences is not None:
    sum_ln_ev = np.sum(ln_evidences)
    result.log_noise_evidence = sum_ln_ev
    result.log_evidence = result.log_bayes_factor + sum_ln_ev
    print(f'log noise evidence : {result.log_noise_evidence:.3f}')
    print(f'log Bayes factor   : {result.log_bayes_factor:.3f}')
    print(f'log evidence       : {result.log_evidence:.3f}')
result.save_to_file(overwrite=True, extension='hdf5')

# corner plot -- first call early in case later code fails
result.plot_corner()

samples = result.posterior.to_dict('list')
samples = {k: xp.array(v) for k, v in samples.items()}

extras = scan(
    lambda parameters: likelihood_extras(
        jax.random.key(1), parameters, likelihood
    )
)(samples)
extras['samples'] = samples
h5ify.save(f'{outdir}/posterior.h5', extras, mode='w')

result.posterior['variance'] = extras['variance']
samples['variance'] = np.asarray(extras['variance'])
corner_params = list(result.posterior.keys())
result.plot_corner(parameters=corner_params)

### --- PPDs --- ###

n_mc = 5000

m1_grid     = jnp.linspace(3.0, 300.0, 1000)
q_grid      = jnp.linspace(0.05, 1.0, 500)
a_grid      = jnp.linspace(0.0, 1.0, 500)
ct_grid     = jnp.linspace(-1.0, 1.0, 500)
chieff_grid = jnp.linspace(-1.0, 1.0, 500)

m1_for_q = jnp.linspace(3.0, 300.0, 500)
mm_q, qq = jnp.meshgrid(m1_for_q, q_grid, indexing='ij')  # (500, 500)


def _icdf_sample(key, p_unnorm, xs, n):
    dx = xs[1] - xs[0]
    cdf = jnp.cumsum(p_unnorm) * dx
    cdf = cdf / cdf[-1]
    return jnp.interp(jax.random.uniform(key, (n,)), cdf, xs)


def _m1_density(params, masses):
    lam_fractions = (
        params['lam_0'], params['lam_1'],
        1 - params['lam_0'] - params['lam_1']
    )
    return jnp.exp(BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth(
        masses, params['alpha_1'], params['alpha_2'],
        params['mlow_1'], params['break_mass'], params['delta_m_1'],
        lam_fractions, params['mpp_1'], params['sigpp_1'],
        params['mpp_2'], params['sigpp_2'],
    ))


def _q_density(params, m1, q):
    return jnp.exp(PowerlawPlusPeak_MassRatio(
        {'mass_1': m1, 'mass_ratio': q},
        slope=params['beta'], minimum=params['mlow_1'],
        delta_m=params['delta_m_1'],
    ))


def ppd_for_sample(params, key):
    # --- primary mass ---
    p_m1 = _m1_density(params, m1_grid)

    # --- mass ratio (marginalize over m1) ---
    p_m1q = _m1_density(params, mm_q) * _q_density(params, mm_q, qq)
    p_q = jnp.trapezoid(p_m1q, m1_for_q, axis=0)
    p_q = p_q / jnp.trapezoid(p_q, q_grid)

    # --- bin weights (fraction of p(m1) in each bin) ---
    bin_weights = jnp.array([
        jnp.trapezoid(
            p_m1 * ((m1_grid >= bin_edges[k]) & (m1_grid < bin_edges[k + 1])),
            m1_grid
        )
        for k in range(n_bins)
    ])
    bin_weights = bin_weights / jnp.sum(bin_weights)

    # --- per-bin spin distributions ---
    p_a_per_bin = jnp.stack([
        jnp.exp(trunc_gaussian(
            a_grid, params[f'mu_chi_{k}'], params[f'sigma_chi_{k}'], 0.0, 1.0
        ))
        for k in range(n_bins)
    ])
    p_ct_per_bin = jnp.stack([
        jnp.exp(trunc_gaussian(
            ct_grid, params[f'mu_tilt_{k}'], params[f'sigma_tilt_{k}'],
            -1.0, 1.0
        ))
        for k in range(n_bins)
    ])

    # marginal spin magnitude and tilt
    p_a = jnp.sum(bin_weights[:, None] * p_a_per_bin, axis=0)
    p_ct = jnp.sum(bin_weights[:, None] * p_ct_per_bin, axis=0)

    # --- chi_eff via MC ---
    keys = jax.random.split(key, 7)

    m1_s = _icdf_sample(keys[0], p_m1, m1_grid, n_mc)
    q_s = _icdf_sample(keys[1], p_q, q_grid, n_mc)
    m2_s = m1_s * q_s

    bin1 = jnp.searchsorted(interior_edges, m1_s, side='right')
    bin2 = jnp.searchsorted(interior_edges, m2_s, side='right')

    # pre-compute ICDF samples per bin, then index by assignment
    a1_per_bin = jnp.stack([
        _icdf_sample(keys[2], p_a_per_bin[k], a_grid, n_mc)
        for k in range(n_bins)
    ])
    a2_per_bin = jnp.stack([
        _icdf_sample(keys[3], p_a_per_bin[k], a_grid, n_mc)
        for k in range(n_bins)
    ])
    ct1_per_bin = jnp.stack([
        _icdf_sample(keys[4], p_ct_per_bin[k], ct_grid, n_mc)
        for k in range(n_bins)
    ])
    ct2_per_bin = jnp.stack([
        _icdf_sample(keys[5], p_ct_per_bin[k], ct_grid, n_mc)
        for k in range(n_bins)
    ])

    idx = jnp.arange(n_mc)
    a1_s = a1_per_bin[bin1, idx]
    a2_s = a2_per_bin[bin2, idx]
    ct1_s = ct1_per_bin[bin1, idx]
    ct2_s = ct2_per_bin[bin2, idx]

    chieff_s = calc_chieff(q_s, a1_s, a2_s, ct1_s, ct2_s)
    p_chieff = jax.scipy.stats.gaussian_kde(chieff_s)(chieff_grid)

    result = dict(
        mass_1=p_m1,
        mass_ratio=p_q,
        spin_magnitude=p_a,
        cos_tilt=p_ct,
        chi_eff=p_chieff,
        mean_chi_eff=jnp.mean(chieff_s),
        std_chi_eff=jnp.std(chieff_s),
        bin_weights=bin_weights,
    )
    for k in range(n_bins):
        result[f'spin_magnitude_bin{k}'] = p_a_per_bin[k]
        result[f'cos_tilt_bin{k}'] = p_ct_per_bin[k]

    return result


n_samples = len(next(iter(samples.values())))
print(f'computing PPDs for {n_samples} posterior samples')

nmax_ppd = 5_000
if n_samples > nmax_ppd:
    rng = np.random.default_rng(43)
    idxs = rng.choice(n_samples, size=nmax_ppd, replace=False)
    ppd_samples = {k: v[idxs] for k, v in samples.items()}
    n_ppd = nmax_ppd
else:
    ppd_samples = samples
    n_ppd = n_samples


@scan_tqdm(n_ppd)
def ppd_step(key, d):
    _, params = d
    key, subkey = jax.random.split(key)
    return key, ppd_for_sample(params, subkey)


init_key = jax.random.key(42)
_, ppds_jax = jax.lax.scan(
    ppd_step, init_key, (jnp.arange(n_ppd), ppd_samples)
)

xs = dict(
    mass_1=np.array(m1_grid),
    mass_ratio=np.array(q_grid),
    spin_magnitude=np.array(a_grid),
    cos_tilt=np.array(ct_grid),
    chi_eff=np.array(chieff_grid),
)
ppds = {k: np.array(v) for k, v in ppds_jax.items()}
medians = {k: np.median(v, axis=0) for k, v in ppds.items()}
q05 = {k: np.quantile(v, 0.05, axis=0) for k, v in ppds.items()}
q95 = {k: np.quantile(v, 0.95, axis=0) for k, v in ppds.items()}

h5ify.save(
    f'{outdir}/ppds.h5',
    dict(xs=xs, ppd=ppds, medians=medians, q05=q05, q95=q95,
         samples={k: np.array(v) for k, v in ppd_samples.items()}),
    mode='w',
)

### --- PPD Plots --- ###

PPD_SETTINGS = {
    'mass_1': dict(
        xlabel=r'$m_1$ [$\mathrm{M}_\odot$]',
        ylabel=r'$\mathrm{PPD}(m_1)$ [$\mathrm{M}_\odot^{-1}$]',
        loglog=True, xlim=(3, 300), ylim=(1e-6, 1e0),
    ),
    'mass_ratio': dict(
        xlabel=r'$q$',
        ylabel=r'$\mathrm{PPD}(q)$',
        semilogy=True, xlim=(0.1, 1.0), ylim=(1e-6, 1e1),
    ),
    'spin_magnitude': dict(
        xlabel=r'$\chi$',
        ylabel=r'$p(\chi)$',
        xlim=(0, 1), ylim=(-0.1, 3.0),
    ),
    'cos_tilt': dict(
        xlabel=r'$\cos \theta_{1,2}$',
        ylabel=r'$p(\cos \theta)$',
        xlim=(-1, 1), ylim=(0, 1.5),
    ),
    'chi_eff': dict(
        xlabel=r'$\chi_\mathrm{eff}$',
        ylabel=r'$p(\chi_\mathrm{eff})$',
        xlim=(-1, 1), ylim=(0, 5.5),
    ),
}

BIN_COLORS = [f'C{i}' for i in range(n_bins)]
BIN_LABELS = [
    f'Bin {k}: [{bin_edges[k]:.0f}, {bin_edges[k+1]:.0f})'
    + r' $M_\odot$'
    for k in range(n_bins)
]


def _apply_ppd_axes(ax, var):
    s = PPD_SETTINGS[var]
    if s.get('loglog'):
        ax.set_xscale('log')
        ax.set_yscale('log')
    elif s.get('semilogy'):
        ax.set_yscale('log')
    ax.set_xlim(s['xlim'])
    ax.set_ylim(s['ylim'])
    ax.set_xlabel(s['xlabel'])
    ax.set_ylabel(s['ylabel'])


def plot_ppd_median_ci(x, ppd_total, fname, var,
                       component_ppds=None, component_weights=None):
    fig, ax = plt.subplots()

    if component_ppds is not None:
        for p_ci, w_ci, lbl, clr in zip(
            component_ppds, component_weights, BIN_LABELS, BIN_COLORS
        ):
            weighted = w_ci[:, None] * p_ci
            med = np.median(weighted, axis=0)
            lo = np.quantile(weighted, 0.05, axis=0)
            hi = np.quantile(weighted, 0.95, axis=0)
            ax.plot(x, med, color=clr, lw=2, label=lbl)
            ax.fill_between(x, lo, hi, alpha=0.4, color=clr, lw=0)

    med = np.median(ppd_total, axis=0)
    lo = np.quantile(ppd_total, 0.05, axis=0)
    hi = np.quantile(ppd_total, 0.95, axis=0)
    ax.plot(x, med, color='black', lw=2, label='Total')
    ax.fill_between(x, lo, hi, alpha=0.15, color='black', lw=0)

    _apply_ppd_axes(ax, var)
    ax.legend()
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)


def plot_ppd_lightning(x, ppd_total, fname, var,
                       component_ppds=None, component_weights=None,
                       alpha=0.08, max_draws=200):
    fig, ax = plt.subplots()

    n = len(ppd_total)
    n_draw = min(n, max_draws)
    draw_rng = np.random.default_rng(0)
    idxs = draw_rng.choice(n, size=n_draw, replace=False)

    if component_ppds is not None:
        for p_ci, w_ci, lbl, clr in zip(
            component_ppds, component_weights, BIN_LABELS, BIN_COLORS
        ):
            weighted = w_ci[:, None] * p_ci
            for i in idxs:
                ax.plot(x, weighted[i], color=clr, alpha=alpha, lw=0.5)
            med = np.median(weighted, axis=0)
            lo = np.quantile(weighted, 0.05, axis=0)
            hi = np.quantile(weighted, 0.95, axis=0)
            ax.plot(x, med, color=clr, lw=1, label=lbl)
            ax.plot(x, lo, color=clr, lw=0.8, ls='--')
            ax.plot(x, hi, color=clr, lw=0.8, ls='--')
    else:
        for i in idxs:
            ax.plot(x, ppd_total[i], color='C0', alpha=alpha, lw=0.5)

    med = np.median(ppd_total, axis=0)
    lo = np.quantile(ppd_total, 0.05, axis=0)
    hi = np.quantile(ppd_total, 0.95, axis=0)
    ax.plot(x, med, color='black', lw=1, label='Median')
    ax.plot(x, lo, color='black', lw=0.8, ls='--', label='5th/95th')
    ax.plot(x, hi, color='black', lw=0.8, ls='--')

    _apply_ppd_axes(ax, var)
    ax.legend()
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)


# overall distributions
for var in ['mass_1', 'mass_ratio', 'spin_magnitude', 'cos_tilt', 'chi_eff']:
    x = xs[var]
    p = ppds[var]
    plot_ppd_median_ci(x, p, f'{outdir}/ppds/{var}_ci.png', var)
    plot_ppd_lightning(x, p, f'{outdir}/ppds/{var}_lightning.png', var)

# per-bin decompositions for spin distributions
for var, bin_key_fmt in [
    ('spin_magnitude', 'spin_magnitude_bin{}'),
    ('cos_tilt', 'cos_tilt_bin{}'),
]:
    x = xs[var]
    comp_list = [ppds[bin_key_fmt.format(k)] for k in range(n_bins)]
    w_list = [ppds['bin_weights'][:, k] for k in range(n_bins)]
    plot_ppd_median_ci(
        x, ppds[var], f'{outdir}/ppds/{var}_bins_ci.png', var,
        component_ppds=comp_list, component_weights=w_list,
    )
    plot_ppd_lightning(
        x, ppds[var], f'{outdir}/ppds/{var}_bins_lightning.png', var,
        component_ppds=comp_list, component_weights=w_list,
    )

### --- Corner Plots --- ###

MASS_PARAMS = [
    'alpha_1', 'alpha_2', 'break_mass',
    'mpp_1', 'sigpp_1', 'mpp_2', 'sigpp_2',
    'delta_m_1', 'mlow_1', 'lam_0', 'lam_1', 'beta',
]
DIAG = ['variance']

CORNER_KWARGS = dict(levels=(0.5, 0.9), plot_datapoints=False)


def _corner(keys, fname):
    plot_corner({k: samples[k] for k in keys}, fname=fname, **CORNER_KWARGS)
    plt.close('all')


# mass parameters
_corner(MASS_PARAMS + DIAG, f'{outdir}/corners/mass.png')

# per-bin spin parameters
for k in range(n_bins):
    spin_keys = [
        f'mu_chi_{k}', f'sigma_chi_{k}',
        f'mu_tilt_{k}', f'sigma_tilt_{k}',
    ]
    _corner(spin_keys + DIAG, f'{outdir}/corners/spin_bin{k}.png')

# all spin parameters together
all_spin = []
for k in range(n_bins):
    all_spin.extend([
        f'mu_chi_{k}', f'sigma_chi_{k}',
        f'mu_tilt_{k}', f'sigma_tilt_{k}',
    ])
_corner(all_spin + DIAG, f'{outdir}/corners/all_spin.png')

# cross-bin: spin magnitude comparison
spin_mag_per_bin = [
    np.column_stack([samples[f'mu_chi_{k}'], samples[f'sigma_chi_{k}']])
    for k in range(n_bins)
]
plot_multiple(
    spin_mag_per_bin,
    colors=BIN_COLORS,
    fname=f'{outdir}/corners/spin_mag_comparison.png',
    xs_labels=BIN_LABELS,
    labels=[r'$\mu_\chi$', r'$\sigma_\chi$'],
    **CORNER_KWARGS,
)
plt.close('all')

# cross-bin: spin tilt comparison
spin_tilt_per_bin = [
    np.column_stack([samples[f'mu_tilt_{k}'], samples[f'sigma_tilt_{k}']])
    for k in range(n_bins)
]
plot_multiple(
    spin_tilt_per_bin,
    colors=BIN_COLORS,
    fname=f'{outdir}/corners/spin_tilt_comparison.png',
    xs_labels=BIN_LABELS,
    labels=[r'$\mu_t$', r'$\sigma_t$'],
    **CORNER_KWARGS,
)
plt.close('all')

# diagnostics
_corner(DIAG + ['lamb'], f'{outdir}/corners/diagnostics.png')

print('done.')
