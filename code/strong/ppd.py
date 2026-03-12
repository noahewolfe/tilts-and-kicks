"""
Compute posterior predictive distributions (PPDs) from a bilby result file.

PPDs computed:
  1. primary mass p(m1)
  2. mass ratio p(q), marginalized over m1
  3. spin magnitude p(a)
  4. chi_eff p(χ_eff), via MC sampling + Gaussian KDE

Supports models: default-spin-simple-power-law-mass, default-spin-bpl2p-mass,
                 twomass, threemass
"""
import os
from argparse import ArgumentParser

import numpy as np
import h5ify

import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')

import bilby
import gwpopulation as gwpop
gwpop.set_backend('jax')
from gwpopulation.models.mass import two_component_single
from gwpopulation.models.mass import two_component_primary_mass_ratio

from jax_tqdm import scan_tqdm
from pixelpop.models.gwpop_models import trunc_gaussian
from pixelpop.models.gwpop_models import PowerlawPlusPeak_MassRatio

from models import BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth
from util import calc_chieff

parser = ArgumentParser()
parser.add_argument('--result', type=str, required=True,
                    help='path to bilby result .hdf5 file')
parser.add_argument('--outdir', type=str, required=True)
parser.add_argument('--model', type=str, required=True,
                    choices=[
                        'default-spin-simple-power-law-mass',
                        'default-spin-bpl2p-mass',
                        'twomass',
                        'threemass',
                    ])
parser.add_argument('--n-mc', type=int, default=5000,
                    help='MC samples per posterior draw for chi_eff PPD')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--constrain-mu-order',
                    choices=['none', 'ascending', 'descending'], default='none',
                    help='must match the inference run')
parser.add_argument('--sample-log-sigma', action='store_true',
                    help='must match the inference run')
parser.add_argument('--stable-expit', action='store_true',
                    help='must match the inference run')

args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

model        = args.model
n_mc         = args.n_mc
stable_expit = args.stable_expit

# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

m1_grid     = jnp.linspace(3.0, 300.0, 1000)
q_grid      = jnp.linspace(0.05, 1.0, 500)
a_grid      = jnp.linspace(0.0, 1.0, 500)
chieff_grid = jnp.linspace(-1.0, 1.0, 500)
ct_grid     = jnp.linspace(-1.0, 1.0, 500)

# coarser m1 grid for 2D mass-ratio integral (500 x 500 = 150k points)
m1_for_q = jnp.linspace(3.0, 300.0, 500)
mm, qq   = jnp.meshgrid(m1_for_q, q_grid, indexing='ij')  # (500, 500)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zeta(params, m1):
    """Mass-dependent sigmoid: probability of being in the high-mass component."""
    if stable_expit:
        return jax.scipy.special.expit(m1 - params['m_cut'])
    return 1.0 / (1.0 + jnp.exp(-m1 + params['m_cut']))


def _bpl2p_m1(params, masses, suffix=''):
    """
    BPL+2P primary mass density (linear scale) for one model component.
    suffix: '' | '_iso' | '_high_iso'
    """
    s    = suffix
    lam0 = params[f'lam{s}_0']
    lam1 = params[f'lam{s}_1']
    return jnp.exp(BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth(
        masses,
        alpha_1=params[f'alpha_1{s}'],
        alpha_2=params[f'alpha_2{s}'],
        mlow_1=params[f'mlow_1{s}'],
        break_mass=params[f'break_mass{s}'],
        delta_m_1=params[f'delta_m_1{s}'],
        lam_fractions=(lam0, lam1, 1.0 - lam0 - lam1),
        mpp_1=params[f'mpp_1{s}'],
        sigpp_1=params[f'sigpp_1{s}'],
        mpp_2=params[f'mpp_2{s}'],
        sigpp_2=params[f'sigpp_2{s}'],
        mmax=300.0,
        gaussian_mass_maximum=100.0,
    ))


def _bpl2p_q(params, m1, q, suffix=''):
    """BPL+2P mass ratio density p(q | m1), linear scale."""
    s = suffix
    return jnp.exp(PowerlawPlusPeak_MassRatio(
        {'mass_1': m1, 'mass_ratio': q},
        slope=params[f'beta{s}'],
        minimum=params[f'mlow_1{s}'],
        delta_m=params[f'delta_m_1{s}'],
    ))


def _icdf_sample(key, p_unnorm, xs, n):
    """Draw n samples from a 1D density on a uniform grid via inverse CDF."""
    dx  = xs[1] - xs[0]
    cdf = jnp.cumsum(p_unnorm) * dx
    cdf = cdf / cdf[-1]
    return jnp.interp(jax.random.uniform(key, (n,)), cdf, xs)


# ---------------------------------------------------------------------------
# Per-component 2D joint densities (m1, q)
# ---------------------------------------------------------------------------

def _joint_m1q(params):
    """
    Returns per-component joint densities on (m1_for_q × q_grid) and
    per-component mass densities on m1_grid, plus the sigmoid weight arrays.

    Returns (p1_m1, p2_m1, p3_m1, p1_m1q, p2_m1q, p3_m1q, w1_m1, w2_m1, w3_m1)
    All mass densities are LINEAR (not log).
    """
    weight_a = params['weight_a']
    z_m1  = _zeta(params, m1_grid)    # (1000,)
    z_m1q = _zeta(params, m1_for_q)  # (300,)

    if model == 'default-spin-simple-power-law-mass':
        kw = dict(
            alpha=params['alpha'], mmin=params['mmin'], mmax=params['mmax'],
            lam=params['lam'], mpp=params['mpp'], sigpp=params['sigpp'],
            gaussian_mass_maximum=100.0,
        )
        p1_m1 = two_component_single(m1_grid, **kw)
        p1_m1q = two_component_primary_mass_ratio(
            {'mass_1': mm, 'mass_ratio': qq},
            beta=params['beta'], **kw,
        )
        p2_m1, p3_m1 = p1_m1, p1_m1
        p2_m1q, p3_m1q = p1_m1q, p1_m1q

    elif model == 'default-spin-bpl2p-mass':
        p1_m1  = _bpl2p_m1(params, m1_grid)
        p1_m1q = _bpl2p_m1(params, mm) * _bpl2p_q(params, mm, qq)
        p2_m1, p3_m1     = p1_m1, p1_m1
        p2_m1q, p3_m1q   = p1_m1q, p1_m1q

    elif model == 'twomass':
        p1_m1  = _bpl2p_m1(params, m1_grid)
        p2_m1  = _bpl2p_m1(params, m1_grid, suffix='_iso')
        p3_m1  = p2_m1
        p1_m1q = _bpl2p_m1(params, mm) * _bpl2p_q(params, mm, qq)
        p2_m1q = (_bpl2p_m1(params, mm, suffix='_iso')
                  * _bpl2p_q(params, mm, qq, suffix='_iso'))
        p3_m1q = p2_m1q

    else:  # threemass
        p1_m1  = _bpl2p_m1(params, m1_grid)
        p2_m1  = _bpl2p_m1(params, m1_grid, suffix='_iso')
        p3_m1  = _bpl2p_m1(params, m1_grid, suffix='_high_iso')
        p1_m1q = _bpl2p_m1(params, mm) * _bpl2p_q(params, mm, qq)
        p2_m1q = (_bpl2p_m1(params, mm, suffix='_iso')
                  * _bpl2p_q(params, mm, qq, suffix='_iso'))
        p3_m1q = (_bpl2p_m1(params, mm, suffix='_high_iso')
                  * _bpl2p_q(params, mm, qq, suffix='_high_iso'))

    w1_m1 = (1 - z_m1) * weight_a
    w2_m1 = (1 - z_m1) * (1 - weight_a)
    w3_m1 = z_m1

    return (
        p1_m1, p2_m1, p3_m1, p1_m1q, p2_m1q, p3_m1q, w1_m1, w2_m1, w3_m1, z_m1q, weight_a, z_m1
    )


# ---------------------------------------------------------------------------
# PPD functions
# ---------------------------------------------------------------------------

def ppd_for_sample(params, key):
    (p1_m1, p2_m1, p3_m1,
     p1_m1q, p2_m1q, p3_m1q,
     w1_m1, w2_m1, w3_m1,
     z_m1q, weight_a, z_m1) = _joint_m1q(params)

    # 1. compute unnormalized marginal densities in q
    # weight functions on m1_for_q grid, shape (500,) → broadcast to (500, 500)
    w1_q = ((1 - z_m1q) * weight_a)[:, None]
    w2_q = ((1 - z_m1q) * (1 - weight_a))[:, None]
    w3_q = z_m1q[:, None]

    p_q_c1_unnorm = jnp.trapezoid(w1_q * p1_m1q, m1_for_q, axis=0)
    p_q_c2_unnorm = jnp.trapezoid(w2_q * p2_m1q, m1_for_q, axis=0)
    p_q_c3_unnorm = jnp.trapezoid(w3_q * p3_m1q, m1_for_q, axis=0)

    # 2. compute the normalizations of each marginal in q
    norm1 = jnp.trapezoid(p_q_c1_unnorm, q_grid, axis=0)
    norm2 = jnp.trapezoid(p_q_c2_unnorm, q_grid, axis=0)
    norm3 = jnp.trapezoid(p_q_c3_unnorm, q_grid, axis=0)

    p_q_c1 = p_q_c1_unnorm / norm1
    p_q_c2 = p_q_c2_unnorm / norm2
    p_q_c3 = p_q_c3_unnorm / norm3

    norm = norm1 + norm2 + norm3

    p_q = (p_q_c1_unnorm + p_q_c2_unnorm + p_q_c3_unnorm) / norm

    # 3. compute per-component weights for q, chi, tau
    w1 = norm1 / norm
    w2 = norm2 / norm
    w3 = norm3 / norm

    # --- Primary mass PPD ---
    p_m1 = w1_m1 * p1_m1 + w2_m1 * p2_m1 + w3_m1 * p3_m1
    p_m1 = p_m1 / jnp.trapezoid(p_m1, m1_grid)

    # --- Spin magnitude PPD ---
    mu_1, sig_1 = params['mu_1'], params['sigma_1']
    mu_2, sig_2 = params['mu_2'], params['sigma_2']
    mu_3, sig_3 = params['mu_3'], params['sigma_3']

    p_a_c1 = w1 * jnp.exp(trunc_gaussian(a_grid, mu_1, sig_1, 0.0, 1.0))
    p_a_c2 = w2 * jnp.exp(trunc_gaussian(a_grid, mu_2, sig_2, 0.0, 1.0))
    p_a_c3 = w3 * jnp.exp(trunc_gaussian(a_grid, mu_3, sig_3, 0.0, 1.0))

    p_a = p_a_c1 + p_a_c2 + p_a_c3
    p_a = p_a / jnp.trapezoid(p_a, a_grid)

    # --- chi_eff PPD via MC ---
    chieff_samples, p_chieff = _ppd_chieff(
        params, key, w1, w2, w3,
        mu_1, sig_1, mu_2, sig_2, mu_3, sig_3,
        p_q_c1, p_q_c2, p_q_c3,
    )

    mean_chi_eff = np.mean(chieff_samples)
    std_chi_eff = np.std(chieff_samples)

    return dict(
        w_c1=w1,
        w_c2=w2,
        w_c3=w3,
        z_m1=z_m1,
        mass_1=p_m1,
        mass_1_c1=p1_m1,
        mass_1_c2=p2_m1,
        mass_1_c3=p3_m1,
        mass_ratio=p_q,
        mass_ratio_c1=p_q_c1,
        mass_ratio_c2=p_q_c2,
        mass_ratio_c3=p_q_c3,
        spin_magnitude_c1=p_a_c1,
        spin_magnitude_c2=p_a_c2,
        spin_magnitude_c3=p_a_c3,
        spin_magnitude=p_a,
        chi_eff=p_chieff,
        mean_chi_eff=mean_chi_eff,
        std_chi_eff=std_chi_eff
    )


def _ppd_chieff(
    params, key,
    w, w_iso, w_hi,
    mu_1, sig_1, mu_2, sig_2, mu_3, sig_3,
    p_q_c1, p_q_c2, p_q_c3,
):
    """
    MC chi_eff PPD.  For each of n_mc samples:
      1. draw spin component c ~ Categorical([w, w_iso, w_hi])
         # NW: see note below
      2. draw (a1, a2) from the component's spin magnitude distribution
      3. draw (ct1, ct2) from the component's tilt distribution
      4. draw q from the component's marginal mass-ratio distribution
      5. compute chi_eff = (a1*ct1 + q*a2*ct2) / (1 + q)
    Then KDE-estimate p(chi_eff) on chieff_grid.
    """
    mu_tilt = params['mu_tilt_1']
    sig_tilt = params['sigma_tilt_1']

    keys = jax.random.split(key, 13)

    # Component assignment
    log_w = jnp.log(jnp.array([w, w_iso, w_hi]))
    c = jax.random.categorical(keys[0], log_w, shape=(n_mc,))  # (n_mc,)

    # Spin magnitude samples from each component (same for a1 and a2)
    # NW: this is correct. no extra weights here
    pa1 = jnp.exp(trunc_gaussian(a_grid, mu_1, sig_1, 0.0, 1.0))
    pa2 = jnp.exp(trunc_gaussian(a_grid, mu_2, sig_2, 0.0, 1.0))
    pa3 = jnp.exp(trunc_gaussian(a_grid, mu_3, sig_3, 0.0, 1.0))

    a1_s = jnp.stack([
        _icdf_sample(keys[1], pa1, a_grid, n_mc),
        _icdf_sample(keys[2], pa2, a_grid, n_mc),
        _icdf_sample(keys[3], pa3, a_grid, n_mc),
    ], axis=1)[jnp.arange(n_mc), c]   # (n_mc,)

    a2_s = jnp.stack([
        _icdf_sample(keys[4], pa1, a_grid, n_mc),
        _icdf_sample(keys[5], pa2, a_grid, n_mc),
        _icdf_sample(keys[6], pa3, a_grid, n_mc),
    ], axis=1)[jnp.arange(n_mc), c]

    # Tilt samples: comp1 = TruncNorm, comp2/3 = Uniform
    # NW: also correct, no extra weights.
    pct = jnp.exp(trunc_gaussian(ct_grid, mu_tilt, sig_tilt, -1.0, 1.0))
    ct_gauss_1 = _icdf_sample(keys[7], pct, ct_grid, n_mc)
    ct_gauss_2 = _icdf_sample(keys[8], pct, ct_grid, n_mc)
    ct_iso_1 = jax.random.uniform(keys[9],  (n_mc,), minval=-1.0, maxval=1.0)
    ct_iso_2 = jax.random.uniform(keys[10], (n_mc,), minval=-1.0, maxval=1.0)
    ct_hiso_1 = jax.random.uniform(keys[11], (n_mc,), minval=-1.0, maxval=1.0)
    ct_hiso_2 = jax.random.uniform(keys[12], (n_mc,), minval=-1.0, maxval=1.0)

    ct1_s = jnp.stack([ct_gauss_1, ct_iso_1, ct_hiso_1], axis=1)[jnp.arange(n_mc), c]
    ct2_s = jnp.stack([ct_gauss_2, ct_iso_2, ct_hiso_2], axis=1)[jnp.arange(n_mc), c]

    # Mass ratio samples per component, then select by c
    # Re-use keys[1..3] split from a secondary key for q to avoid correlation
    key_q = jax.random.fold_in(key, 999)
    kq1, kq2, kq3 = jax.random.split(key_q, 3)
    q_s = jnp.stack([
        _icdf_sample(kq1, p_q_c1, q_grid, n_mc),
        _icdf_sample(kq2, p_q_c2, q_grid, n_mc),
        _icdf_sample(kq3, p_q_c3, q_grid, n_mc),
    ], axis=1)[jnp.arange(n_mc), c]

    chieff_s = calc_chieff(q_s, a1_s, a2_s, ct1_s, ct2_s)  # (n_mc,)

    return chieff_s, jax.scipy.stats.gaussian_kde(chieff_s)(chieff_grid)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

try:
    result = bilby.read_in_result(args.result)
    samples = result.posterior.to_dict('list')
except:
    samples = h5ify.load(args.result)

samples = {k: jnp.array(v) for k, v in samples.items()}

# Mirror make_conversion from multimass-gwpop.py
if args.sample_log_sigma:
    for i in range(1, 4):
        samples[f'sigma_{i}'] = jnp.exp(samples[f'log_sigma_{i}'])

if args.constrain_mu_order != 'none':
    g0, g1 = samples['g_0'], samples['g_1']
    if args.constrain_mu_order == 'ascending':
        samples['mu_1'], samples['mu_2'] = g0, g0 + g1
    else:
        samples['mu_1'], samples['mu_2'] = g0 + g1, g0

n = len(next(iter(samples.values())))
print(f'computing PPDs for {n} posterior samples, model={model}')

# randomly reshuffle the samples *just in case* they are ordered by log-likelihood
# (occurs in some versions of bilby)
rng = np.random.default_rng(43)
idxs = rng.choice(n, size=n, replace=False)

samples = {k : v[idxs] for k, v in samples.items()}

nmax = 5_000
if nmax < n:
    samples = {k : v[:nmax] for k, v in samples.items()}
    n = nmax

@scan_tqdm(n)
def step(key, d):
    _, params = d
    key, subkey = jax.random.split(key)
    return key, ppd_for_sample(params, subkey)

init_key = jax.random.key(args.seed)
_, ppds_jax = jax.lax.scan(step, init_key, (jnp.arange(n), samples))

xs = dict(
    mass_1=np.array(m1_grid),
    mass_ratio=np.array(q_grid),
    spin_magnitude=np.array(a_grid),
    chi_eff=np.array(chieff_grid),
)
ppds = {k: np.array(v) for k, v in ppds_jax.items()}
medians = {k: np.median(v, axis=0) for k, v in ppds.items()}
q05 = {k: np.quantile(v, 0.05, axis=0) for k, v in ppds.items()}
q95 = {k: np.quantile(v, 0.95, axis=0) for k, v in ppds.items()}

# we also save a copy of the samples used, b/c of the rng above
h5ify.save(
    f'{args.outdir}/ppds.h5',
    dict(xs=xs, ppd=ppds, medians=medians, q05=q05, q95=q95, samples=samples),
    mode='w',
)
print('done.')
