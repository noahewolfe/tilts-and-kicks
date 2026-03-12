"""Inference script for the identifiable (mass-split) population model."""
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

from models import identifiable_model
from models import BrokenPowerlawPlusTwoPeaks_PrimaryMass_LowHighSmooth
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
    '--constrain-mu-order', choices=['none', 'ascending', 'descending'],
    default='none',
    help="Enforce ordering on spin magnitude means: "
         "ascending => mu_1 <= mu_2, descending => mu_1 >= mu_2."
)
parser.add_argument(
    '--sample-log-sigma',
    action='store_true',
    help='Sample in log_sigma_i instead of sigma_i.'
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
constrain_mu_order = args.constrain_mu_order
sample_log_sigma = args.sample_log_sigma
dynamic = args.dynamic

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


def get_model():
    model_functions = [
        identifiable_model,
        gwpop.models.redshift.PowerLawRedshift(cosmo_model="Planck15"),
    ]
    return Model(model_functions=model_functions, cache=False)


def make_conversion(order, sample_log_sigma):
    """Return conversion_function for HyperparameterLikelihood."""

    def convert(parameters):
        added = []

        if sample_log_sigma:
            for i in range(1, 4):
                key = f'sigma_{i}'
                parameters[key] = xp.exp(parameters[f'log_{key}'])
                added.append(key)

        if order != 'none':
            g0, g1 = parameters['g_0'], parameters['g_1']
            if order == 'ascending':
                parameters['mu_1'], parameters['mu_2'] = g0, g0 + g1
            elif order == 'descending':
                parameters['mu_1'], parameters['mu_2'] = g0 + g1, g0
            added += ['mu_1', 'mu_2']

        return parameters, added

    return convert


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
    conversion_function=make_conversion(constrain_mu_order, sample_log_sigma),
)

### --- Priors --- ###

priors = ConditionalPriorDict(args.priors)

# spin magnitude
if constrain_mu_order == 'none':
    priors["mu_1"] = Uniform(minimum=0, maximum=1, latex_label="$\\mu_1$")
    priors["mu_2"] = Uniform(minimum=0, maximum=1, latex_label="$\\mu_2$")
else:
    priors["g_0"] = DirichletElement(order=0, n_dimensions=3, label='g_')
    priors["g_1"] = DirichletElement(order=1, n_dimensions=3, label='g_')

if sample_log_sigma:
    for i in range(1, 4):
        priors[f'log_sigma_{i}'] = Uniform(
            minimum=np.log(0.1), maximum=0,
            latex_label="$\\ln \\sigma_" + f"{i}$"
        )
else:
    priors["sigma_1"] = Uniform(minimum=0.1, maximum=1, latex_label="$\\sigma_1$")
    priors["sigma_2"] = Uniform(minimum=0.1, maximum=1, latex_label="$\\sigma_2$")
    priors["sigma_3"] = Uniform(minimum=0.1, maximum=1, latex_label="$\\sigma_3$")

# spin tilt (truncnorm tilts component)
priors["mu_tilt_1"] = Uniform(minimum=-1, maximum=1, latex_label="$\\mu_{t,1}$")
priors["sigma_tilt_1"] = TruncatedNormal(
    minimum=0.1, maximum=4, sigma=1/2, mu=0, latex_label="$\\sigma_{t,1}$"
)

# mixing
priors["weight_a"] = Uniform(minimum=0, maximum=1, latex_label="$w_a$")
priors["mu_3"] = Uniform(minimum=0, maximum=1, latex_label="$\\mu_3$")

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

# corner plot — first call early in case later code fails
result.plot_corner()

delta_keys = ['mmax_low', 'mmax_high_iso']

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
corner_params = [k for k in result.posterior.keys() if k not in delta_keys]
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


def _low_m1(params, masses):
    lam_fractions = (
        params['lam_0'], params['lam_1'],
        1 - params['lam_0'] - params['lam_1']
    )
    return jnp.exp(BrokenPowerlawPlusTwoPeaks_PrimaryMass_LowHighSmooth(
        masses, params['alpha_1'], params['alpha_2'],
        params['mlow_1'], params['break_mass'], params['delta_m_1'],
        lam_fractions, params['mpp_1'], params['sigpp_1'],
        params['mpp_2'], params['sigpp_2'],
        params['delta_max'], mmax=params['mmax_low'],
    ))


def _high_m1(params, masses):
    return jnp.exp(BrokenPowerlawPlusTwoPeaks_PrimaryMass_LowHighSmooth(
        masses, params['alpha_high_iso'], params['alpha_high_iso'],
        params['mmax_low'], params['mmax_low'] + 1,
        params['delta_m_1_high_iso'],
        (1.0, 0.0, 0.0), 50.0, 1.0, 50.0, 1.0,
        0.0, mmax=params['mmax_high_iso'],
    ))


def _low_q(params, m1, q):
    return jnp.exp(PowerlawPlusPeak_MassRatio(
        {'mass_1': m1, 'mass_ratio': q},
        slope=params['beta'], minimum=params['mlow_1'],
        delta_m=params['delta_m_1'],
    ))


def _high_q(params, m1, q):
    return jnp.exp(PowerlawPlusPeak_MassRatio(
        {'mass_1': m1, 'mass_ratio': q},
        slope=params['beta_high_iso'],
        minimum=params['mmax_low'],
        delta_m=params['delta_m_1_high_iso'],
    ))


def ppd_for_sample(params, key):
    zeta = params['zeta']
    weight_a = params['weight_a']

    # constant component weights (sum to 1)
    w1 = (1 - zeta) * weight_a       # low-mass truncnorm tilts
    w2 = (1 - zeta) * (1 - weight_a) # low-mass iso
    w3 = zeta                         # high-mass iso

    # --- primary mass ---
    p_m1_low  = _low_m1(params, m1_grid)
    p_m1_high = _high_m1(params, m1_grid)

    p_m1_c1 = p_m1_low
    p_m1_c2 = p_m1_low
    p_m1_c3 = p_m1_high
    p_m1 = w1 * p_m1_c1 + w2 * p_m1_c2 + w3 * p_m1_c3

    # --- mass ratio (marginalize over m1 via 2D trapezoid) ---
    p_low_m1q  = _low_m1(params, mm_q) * _low_q(params, mm_q, qq)
    p_high_m1q = _high_m1(params, mm_q) * _high_q(params, mm_q, qq)

    p_q_c1 = jnp.trapezoid(p_low_m1q, m1_for_q, axis=0)
    p_q_c2 = p_q_c1  # same mass model, different spin (irrelevant for q)
    p_q_c3 = jnp.trapezoid(p_high_m1q, m1_for_q, axis=0)

    # normalize each component's marginal q
    p_q_c1_norm = p_q_c1 / jnp.trapezoid(p_q_c1, q_grid)
    p_q_c3_norm = p_q_c3 / jnp.trapezoid(p_q_c3, q_grid)

    p_q = w1 * p_q_c1_norm + w2 * p_q_c1_norm + w3 * p_q_c3_norm

    # --- spin magnitude ---
    mu_1, sig_1 = params['mu_1'], params['sigma_1']
    mu_2, sig_2 = params['mu_2'], params['sigma_2']
    mu_3, sig_3 = params['mu_3'], params['sigma_3']

    p_a_c1 = jnp.exp(trunc_gaussian(a_grid, mu_1, sig_1, 0.0, 1.0))
    p_a_c2 = jnp.exp(trunc_gaussian(a_grid, mu_2, sig_2, 0.0, 1.0))
    p_a_c3 = jnp.exp(trunc_gaussian(a_grid, mu_3, sig_3, 0.0, 1.0))
    p_a = w1 * p_a_c1 + w2 * p_a_c2 + w3 * p_a_c3

    # --- spin tilt ---
    mu_t, sig_t = params['mu_tilt_1'], params['sigma_tilt_1']

    p_ct_c1 = jnp.exp(trunc_gaussian(ct_grid, mu_t, sig_t, -1.0, 1.0))
    p_ct_c2 = 0.5 * jnp.ones_like(ct_grid)
    p_ct_c3 = 0.5 * jnp.ones_like(ct_grid)
    p_ct = w1 * p_ct_c1 + w2 * p_ct_c2 + w3 * p_ct_c3

    # --- chi_eff via MC ---
    keys = jax.random.split(key, 13)

    log_w = jnp.log(jnp.array([w1, w2, w3]))
    c = jax.random.categorical(keys[0], log_w, shape=(n_mc,))

    a1_s = jnp.stack([
        _icdf_sample(keys[1], p_a_c1, a_grid, n_mc),
        _icdf_sample(keys[2], p_a_c2, a_grid, n_mc),
        _icdf_sample(keys[3], p_a_c3, a_grid, n_mc),
    ], axis=1)[jnp.arange(n_mc), c]

    a2_s = jnp.stack([
        _icdf_sample(keys[4], p_a_c1, a_grid, n_mc),
        _icdf_sample(keys[5], p_a_c2, a_grid, n_mc),
        _icdf_sample(keys[6], p_a_c3, a_grid, n_mc),
    ], axis=1)[jnp.arange(n_mc), c]

    pct = jnp.exp(trunc_gaussian(ct_grid, mu_t, sig_t, -1.0, 1.0))
    ct1_s = jnp.stack([
        _icdf_sample(keys[7], pct, ct_grid, n_mc),
        jax.random.uniform(keys[8], (n_mc,), minval=-1.0, maxval=1.0),
        jax.random.uniform(keys[9], (n_mc,), minval=-1.0, maxval=1.0),
    ], axis=1)[jnp.arange(n_mc), c]

    ct2_s = jnp.stack([
        _icdf_sample(keys[10], pct, ct_grid, n_mc),
        jax.random.uniform(keys[11], (n_mc,), minval=-1.0, maxval=1.0),
        jax.random.uniform(keys[12], (n_mc,), minval=-1.0, maxval=1.0),
    ], axis=1)[jnp.arange(n_mc), c]

    key_q = jax.random.fold_in(key, 999)
    kq1, kq2 = jax.random.split(key_q, 2)
    q_s = jnp.stack([
        _icdf_sample(kq1, p_q_c1, q_grid, n_mc),
        _icdf_sample(kq1, p_q_c1, q_grid, n_mc),  # c2 same q as c1
        _icdf_sample(kq2, p_q_c3, q_grid, n_mc),
    ], axis=1)[jnp.arange(n_mc), c]

    chieff_s = calc_chieff(q_s, a1_s, a2_s, ct1_s, ct2_s)
    p_chieff = jax.scipy.stats.gaussian_kde(chieff_s)(chieff_grid)

    return dict(
        w_c1=w1, w_c2=w2, w_c3=w3,
        mass_1=p_m1,
        mass_1_c1=p_m1_c1, mass_1_c2=p_m1_c2, mass_1_c3=p_m1_c3,
        mass_ratio=p_q,
        mass_ratio_c1=p_q_c1_norm, mass_ratio_c3=p_q_c3_norm,
        spin_magnitude=p_a,
        spin_magnitude_c1=p_a_c1, spin_magnitude_c2=p_a_c2,
        spin_magnitude_c3=p_a_c3,
        cos_tilt=p_ct,
        cos_tilt_c1=p_ct_c1, cos_tilt_c2=p_ct_c2, cos_tilt_c3=p_ct_c3,
        chi_eff=p_chieff,
        mean_chi_eff=jnp.mean(chieff_s),
        std_chi_eff=jnp.std(chieff_s),
    )


# Apply conversion to posterior samples before PPD computation
if sample_log_sigma:
    for i in range(1, 4):
        samples[f'sigma_{i}'] = jnp.exp(samples[f'log_sigma_{i}'])

if constrain_mu_order != 'none':
    g0, g1 = samples['g_0'], samples['g_1']
    if constrain_mu_order == 'ascending':
        samples['mu_1'], samples['mu_2'] = g0, g0 + g1
    else:
        samples['mu_1'], samples['mu_2'] = g0 + g1, g0

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
_, ppds_jax = jax.lax.scan(ppd_step, init_key, (jnp.arange(n_ppd), ppd_samples))

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

COMP_COLORS = ['C0', 'C1', 'C2']
COMP_LABELS = ['Comp A (truncnorm tilts)', 'Comp B (low iso)', 'Comp C (high iso)']


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
            component_ppds, component_weights, COMP_LABELS, COMP_COLORS
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
            component_ppds, component_weights, COMP_LABELS, COMP_COLORS
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

# per-component decompositions
DECOMP_VARS = {
    'mass_1': ['mass_1_c1', 'mass_1_c2', 'mass_1_c3'],
    'mass_ratio': ['mass_ratio_c1', 'mass_ratio_c1', 'mass_ratio_c3'],
    'spin_magnitude': [
        'spin_magnitude_c1', 'spin_magnitude_c2', 'spin_magnitude_c3'
    ],
    'cos_tilt': ['cos_tilt_c1', 'cos_tilt_c2', 'cos_tilt_c3'],
}
weight_keys = ['w_c1', 'w_c2', 'w_c3']

for var, ci_keys in DECOMP_VARS.items():
    x = xs[var]
    comp_list = [ppds[k] for k in ci_keys]
    w_list = [ppds[k] for k in weight_keys]
    plot_ppd_median_ci(
        x, ppds[var], f'{outdir}/ppds/{var}_components_ci.png', var,
        component_ppds=comp_list, component_weights=w_list,
    )
    plot_ppd_lightning(
        x, ppds[var], f'{outdir}/ppds/{var}_components_lightning.png', var,
        component_ppds=comp_list, component_weights=w_list,
    )

### --- Corner Plots --- ###

LOW_MASS_PARAMS = [
    'alpha_1', 'alpha_2', 'break_mass',
    'mpp_1', 'sigpp_1', 'mpp_2', 'sigpp_2',
    'delta_m_1', 'mlow_1', 'delta_max', 'lam_0', 'lam_1', 'beta',
]
HIGH_MASS_PARAMS = [
    'alpha_high_iso', 'delta_m_1_high_iso', 'beta_high_iso',
]
SPIN_A = ['mu_1', 'sigma_1']
SPIN_B = ['mu_2', 'sigma_2']
SPIN_C = ['mu_3', 'sigma_3']
TILT_A = ['mu_tilt_1', 'sigma_tilt_1']
MIXING = ['weight_a', 'zeta']


CORNER_KWARGS = dict(levels=(0.5, 0.9), plot_datapoints=False)


def _corner(keys, fname):
    plot_corner({k: samples[k] for k in keys}, fname=fname, **CORNER_KWARGS)
    plt.close('all')


# per-sub-population, all params
_corner(
    LOW_MASS_PARAMS + SPIN_A + TILT_A + ['weight_a'],
    f'{outdir}/corners/comp_a_all.png',
)
_corner(
    LOW_MASS_PARAMS + SPIN_B + ['weight_a'],
    f'{outdir}/corners/comp_b_all.png',
)
_corner(
    HIGH_MASS_PARAMS + SPIN_C + ['zeta'],
    f'{outdir}/corners/comp_c_all.png',
)

# per-sub-population, per sector (with mixing params)
_corner(LOW_MASS_PARAMS + MIXING, f'{outdir}/corners/low_mass.png')
_corner(HIGH_MASS_PARAMS + MIXING, f'{outdir}/corners/high_mass.png')
_corner(SPIN_A + MIXING, f'{outdir}/corners/spin_mag_a.png')
_corner(SPIN_B + MIXING, f'{outdir}/corners/spin_mag_b.png')
_corner(SPIN_C + MIXING, f'{outdir}/corners/spin_mag_c.png')
_corner(TILT_A + MIXING, f'{outdir}/corners/spin_tilt_a.png')

# cross-population: spin magnitude comparison
spin_common = [
    np.column_stack([samples[k] for k in keys])
    for keys in [SPIN_A, SPIN_B, SPIN_C]
]
plot_multiple(
    spin_common,
    colors=['C0', 'C1', 'C2'],
    fname=f'{outdir}/corners/spin_mag_comparison.png',
    xs_labels=['Comp A (truncnorm tilts)', 'Comp B (low iso)', 'Comp C (high iso)'],
    labels=[r'$\mu_\chi$', r'$\sigma_\chi$'],
    **CORNER_KWARGS,
)
plt.close('all')

# cross-population: shared mass params comparison
mass_shared_low = np.column_stack([
    samples['alpha_1'], samples['beta'],
])
mass_shared_high = np.column_stack([
    samples['alpha_high_iso'], samples['beta_high_iso'],
])
plot_multiple(
    [mass_shared_low, mass_shared_high],
    colors=['C0', 'C2'],
    fname=f'{outdir}/corners/mass_shared_comparison.png',
    xs_labels=['Low-mass', 'High-mass'],
    labels=[r'$\alpha$', r'$\beta$'],
    **CORNER_KWARGS,
)
plt.close('all')

# joint cross-population corners (all params on separate axes)
_corner(
    SPIN_A + SPIN_B + SPIN_C,
    f'{outdir}/corners/spin_mag_joint.png',
)
_corner(
    ['alpha_1', 'mpp_1', 'sigpp_1', 'beta',
     'alpha_high_iso', 'beta_high_iso'],
    f'{outdir}/corners/mass_shared_joint.png',
)

# mixing
_corner(MIXING, f'{outdir}/corners/mixing.png')

print('done.')
