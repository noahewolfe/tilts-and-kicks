""" ripped from: https://github.com/stegmaja/black-hole-spin-orbit-tilts/blob/main/main.ipynb """
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import bilby as bb
from bilby.hyper.model import Model
from bilby.core.prior import PriorDict
from bilby.core.prior import ConditionalPriorDict
from bilby.core.prior import Uniform
from bilby.core.prior import TruncatedNormal
from bilby.core.prior import DirichletElement

import gwpopulation as gwpop
from gwpopulation.experimental.jax import JittedLikelihood
gwpop.set_backend("jax")

# TODO: cleanup
xp = gwpop.utils.xp
import jax
import jax.numpy as jnp

import h5ify
from util import scan
from util import plot_corner
from util import write_config

from pixelpop.models.gwpop_models import PowerlawPlusPeak_MassRatio
from models import BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth
from models import bpl2p_m1q

label = 'run'

parser = ArgumentParser()
parser.add_argument('--outdir', type=str, required=True)
parser.add_argument('--which-data', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--sampling-seed', type=int, default=1701)
parser.add_argument('--maximum-uncertainty', required=True)
parser.add_argument('--priors', type=str)
parser.add_argument('--sampler-settings', type=str, default='fast')
parser.add_argument('--nlive', type=int, default=100)
parser.add_argument('--stable-expit', action='store_true')
parser.add_argument('--nest-result', type=str, required=True,
    help='Path to bilby result file to draw tempered initial samples from.')
parser.add_argument(
    '--nest-result-mu-order', choices=['ascending', 'descending'], default='ascending',
    help='Ordering used in the source nested sampling run, if it has g_0/g_1 instead '
         'of mu_1/mu_2. ascending => mu_1=g_0, mu_2=g_0+g_1; '
         'descending => mu_1=g_0+g_1, mu_2=g_0.'
)
parser.add_argument('--nwalkers', type=int, default=200)
parser.add_argument(
    '--constrain-mu-order', choices=['none', 'ascending', 'descending'],
    default='none',
    help="Enforce ordering on spin magnitude means: "
         "ascending => mu_1 <= mu_2, descending => mu_1 >= mu_2."
)

args = parser.parse_args()
outdir = args.outdir
os.makedirs(f'{outdir}/ppds', exist_ok=True)
write_config(args)

which_data = args.which_data
model = args.model

if model != 'default-spin-simple-power-law-mass':
    priors = args.priors

sampling_seed = args.sampling_seed
maximum_uncertainty = args.maximum_uncertainty
sampler_settings = args.sampler_settings
nlive = args.nlive
stable_expit = args.stable_expit
constrain_mu_order = args.constrain_mu_order

### --- Priors --- ###

if model == 'default-spin-simple-power-law-mass':
    priors = PriorDict()
    priors["alpha"] = Uniform(minimum=-2, maximum=4, latex_label="$\\alpha$")
    priors["beta"] = Uniform(minimum=-4, maximum=12, latex_label="$\\beta$")
    priors["mmin"] = Uniform(minimum=2, maximum=2.5, latex_label="$m_{\\min}$")
    priors["mmax"] = Uniform(minimum=80, maximum=100, latex_label="$m_{\\max}$")
    priors["lam"] = Uniform(minimum=0, maximum=1, latex_label="$\\lambda_{m}$")
    priors["mpp"] = Uniform(minimum=10, maximum=50, latex_label="$\\mu_{m}$")
    priors["sigpp"] = Uniform(minimum=1, maximum=10, latex_label="$\\sigma_{m}$")
    priors["gaussian_mass_maximum"] = 100
else:
    priors = ConditionalPriorDict(priors)

# spin
if constrain_mu_order == 'none':
    priors["mu_1"] = Uniform(minimum=0, maximum=1, latex_label="$\\mu_1$")
    priors["mu_2"] = Uniform(minimum=0, maximum=1, latex_label="$\\mu_2$")
else:
    priors["g_0"] = DirichletElement(order=0, n_dimensions=3, label='g_')
    priors["g_1"] = DirichletElement(order=1, n_dimensions=3, label='g_')

priors["sigma_1"] = Uniform(minimum=0.1, maximum=1, latex_label="$\\sigma_1$")
priors["mu_tilt_1"] = Uniform(minimum=-1, maximum=1, latex_label="$\\mu_{t,1}$")
priors["sigma_tilt_1"] = TruncatedNormal(minimum=0.1, maximum=4, sigma=1/2, mu=0, latex_label="$\\sigma_{t,1}$")
priors["sigma_2"] = Uniform(minimum=0.1, maximum=1, latex_label="$\\sigma_2$")
priors["weight_a"] = Uniform(minimum=0, maximum=1, latex_label="$w_a$")
priors["mu_3"] = Uniform(minimum=0, maximum=1, latex_label="$\\mu_3$")
priors["sigma_3"] = Uniform(minimum=0.1, maximum=1, latex_label="$\\sigma_3$")
priors["m_cut"] = Uniform(minimum=10, maximum=100, latex_label="$m_{\\rm cut}$")

# redshift
priors["lamb"] = Uniform(minimum=-1, maximum=10, latex_label="$\\lambda_{z}$")

priors.to_file(outdir, label)


### --- Initialization --- ###
from bilby.core.result import read_in_result
nest_result = read_in_result(args.nest_result)
ns = nest_result.nested_samples

if 'g_0' in ns.columns and 'g_1' in ns.columns and 'mu_1' not in ns.columns:
    if args.nest_result_mu_order == 'descending':
        ns['mu_1'] = ns['g_0'] + ns['g_1']
        ns['mu_2'] = ns['g_0']
    else:
        ns['mu_1'] = ns['g_0']
        ns['mu_2'] = ns['g_0'] + ns['g_1']

missing = set(priors.keys()) - set(ns.columns)
if missing:
    raise ValueError(f'Nested sampling result missing keys required by prior: {missing}')

def tempered_weights(nested_samples, beta):
    """

    Tempers the posterior weights from a nested sampling analysis. See Section B and
    Eq.~10 of https://arxiv.org/abs/2208.12872.

    Notes
    =====
    We use the notation of https://arxiv.org/abs/2208.12872. First we read in or compute
    the log prior volume weights, :math:`w_i`, up to an overall normalization. If those
    are not directly provided, we compute it as

    .. math::
        \ln w_i \propto \ln p_i - \ln \cal L_i.

    To get tempered posterior weights at an inverse temperature :math:`\beta_T`, we have

    .. math::
        \ln p_{i, \beta_T} \propto \ln w_i + \beta_T \ln \cal L_i.

    Finally, we normalize these tempered posterior weights by the tempered evidence,
    which is computed as

    .. math::
        \ln \cal Z_{\beta_T} = \ln \sum_i \exp \left( \ln w_i + \beta_T \ln \cal L_i
        \right).

    Note that we return :math:`p_{i, \beta_T}`, not :math:`\ln p_{i, \beta_T}`.

    Parameters
    ==========
    nested_samples: pandas.DataFrame
        Resultant dataframe from a `dynesty` run, or generally, the `nested_samples`
        member of a `bilby.core.result` object. (This code has not yet been tested
        with other nested samplers.)
    beta: float or array_like
        Inverse temperature(s) at which to compute tempered posterior weights.

    Returns
    =======
    array_like: tempered posterior weights, shaped like (number of nested samples, )
    or (number of nested samples, number of inverse temperatures).

    """

    from scipy.special import logsumexp

    if "weights" in nested_samples:
        ln_weights = (
            np.log(nested_samples["weights"]) - nested_samples["log_likelihood"]
        ).values
    else:
        ln_weights = nested_samples["log_prior_volume"].values

    if isinstance(beta, (list, np.ndarray)):
        ln_weights = np.tile(ln_weights, (len(beta), 1))
        ln_weights = ln_weights + (
            np.tile(nested_samples["log_likelihood"].values, (len(beta), 1)).T * beta
        ).T
    else:
        ln_weights = ln_weights + nested_samples["log_likelihood"].values * beta

    return np.exp(ln_weights - logsumexp(ln_weights))


def set_tempered_nested_samples(
    samples, nested_samples, parameter_keys, nwalkers, temperatures, set_idxs=None
):
    """

    Return an array (or dictionary) of tempered nested samples.

    In particular, return an array (or dictionary) representing a subsampling of
    nested samples, where the probability of including a particular sample
    is determined by the associated nested sampling weights which have been
    raised to the power of a "temperature" beta.

    Parameters
    ==========
    samples: dict
        Dictionary of samples; the value at each key is an np.ndarray array
        shaped like (ntemps, nwalkers)
    nested_samples: pandas.DataFrame
        pandas DataFrame of nested samples, indexed by parameter names.
    parameter_keys: list
        List of strings, of the names of the parameters that we want to sample.
        This may represent a subset of the keys of the samples dictionary.
    nwalkers: int
        Number of walkers in our eventual ensemble of parallel-tempered MCMC chains.
    temperatures: np.ndarray or list
        List or array of temperatures beta (of length we call "ntemps").
    set_idxs: tuple
        Numpy-comptaible tuple of arrays for indexing the values at each key of samples;
        if we only want to set the value of samples at certain (temperature, walker)
        positions in the arrays at each key of samples.

    Returns
    =======
    None (this modifies the dictionary samples).

    """

    ndim = len(parameter_keys)
    ntemps = len(temperatures)

    if set_idxs is None:
        # indices for each position in an array (ntemps, nwalkers)
        set_idxs = tuple(np.mgrid[0:ntemps, 0:nwalkers])

    # Draw one sample index per (temperature, walker); all parameters for a
    # walker share the same index so joint constraints (e.g. Dirichlet sums)
    # are preserved.
    tempered_sample_idxs = np.array(
        [
            np.random.choice(
                len(nested_samples),
                nwalkers,
                p=tempered_weights(nested_samples.copy(), temp),
                replace=True,
            )
            for temp in temperatures
        ]
    )
    # shape: (ntemps, nwalkers)

    for key in parameter_keys:
        samples[key][set_idxs] = nested_samples[key].values[
            tempered_sample_idxs
        ]

nested_samples = ns

from ptemcee import default_beta_ladder
ntemps = 10
nwalkers = args.nwalkers
pos0 = priors.sample((ntemps, nwalkers))
temperatures = default_beta_ladder(
    ndim=len(priors.keys()),
    ntemps=ntemps,
    Tmax=None
)
set_tempered_nested_samples(
    pos0, nested_samples, list(priors.keys()), nwalkers, temperatures
)

np.save(f'{outdir}/pos0.npy', pos0)

from util import plot_corner

plot_corner(np.column_stack([ pos0[k].flatten() for k in pos0.keys() ]), labels=list(pos0.keys()), fname=f'{outdir}/pos0.jpg')

sampler_kwargs = dict(
    ntemps=ntemps,
    nwalkers=nwalkers,
    pos0=pos0,
    use_ratio=True,
)


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

    # format expected by gwpop
    posteriors = [
        pd.DataFrame.from_dict(
            {k : v[i] for k, v in posteriors.items()},
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
    maximum_uncertainty = float(maximum_uncertainty)
print(f'using variance cut : {maximum_uncertainty}')

#if sampler_settings == 'fast':
#    sampler_kwargs = dict(
#        sample="acceptance-walk",
#        naccept=5,
#    )
#elif sampler_settings == 'robust':
#    # bilby dynesty defaults
#    sampler_kwargs = dict()


def get_model(model):
    from models import default_stegmann_spin_model

    def spin_model(
        dataset,
        mu_1,
        sigma_1,
        mu_tilt_1,
        sigma_tilt_1,
        mu_2,
        sigma_2,
        mu_3,
        sigma_3,
        weight_a,
        m_cut,
    ):
        return default_stegmann_spin_model(
            dataset,
            mu_1,
            sigma_1,
            mu_tilt_1,
            sigma_tilt_1,
            mu_2,
            sigma_2,
            mu_3,
            sigma_3,
            weight_a,
            m_cut,
            stable_expit=stable_expit
        )

    if model == 'default-spin-simple-power-law-mass':
        model_functions = [
            gwpop.models.mass.two_component_primary_mass_ratio,
            spin_model,
        ]
    elif model == 'default-spin-bpl2p-mass':
        model_functions = [
            bpl2p_m1q,
            spin_model,
        ]
    elif model == 'twomass':
        from models import twomass_and_spin_model
        model_functions = [twomass_and_spin_model]
    elif model == 'threemass':
        from models import threemass_and_spin_model
        model_functions = [threemass_and_spin_model]
    else:
        raise ValueError(f'bad model {model}')

    model_functions += [
        gwpop.models.redshift.PowerLawRedshift(cosmo_model="Planck15")
    ]

    return Model(
        model_functions=model_functions,
        cache=False,
    )


def make_mu_conversion(order):
    """Return conversion_function for HyperparameterLikelihood."""
    if order == 'none':
        return lambda params: (params, [])
    def convert(parameters):
        g0, g1 = parameters['g_0'], parameters['g_1']
        if order == 'ascending':
            parameters['mu_1'], parameters['mu_2'] = g0, g0 + g1
        else:  # descending
            parameters['mu_1'], parameters['mu_2'] = g0 + g1, g0
        return parameters, ['mu_1', 'mu_2']
    return convert


vt = gwpop.vt.ResamplingVT(
    model=get_model(model),
    data=injections,
    n_events=len(posteriors)
)

# set random state for re-sampling the single-event PE
np.random.seed(42)

likelihood = gwpop.hyperpe.HyperparameterLikelihood(
    posteriors=posteriors,
    hyper_prior=get_model(model),
    selection_function=vt,
    maximum_uncertainty=maximum_uncertainty,
    ln_evidences=ln_evidences,
    conversion_function=make_mu_conversion(constrain_mu_order),
)

jit_likelihood = JittedLikelihood(likelihood)
ll = jit_likelihood.log_likelihood_ratio(priors.sample())
print('log likelihood ratio :', ll)

### ll at pos0
#nested_samples = nested_samples.to_dict('list')
#nested_samples = {k : xp.array(v) for k, v in nested_samples.items()}
#pos0_extras = scan(likelihood.generate_extra_statistics)(nested_samples)
#h5ify.save(f'{outdir}/pos0.h5', {**pos0_extras, **nested_samples}, mode='w')

### --- Sample --- ###

result = bb.run_sampler(
    likelihood=jit_likelihood,
    priors=priors,
    sampler="ptemcee",
    label=label,
    save="hdf5",
    outdir=outdir,
    seed=sampling_seed,
    **sampler_kwargs
)

### --- Post-process --- ###

# result.log_evidence is nan because JittedLikelihood.noise_log_likelihood() inherits
# bilby's base which returns nan. result.log_bayes_factor = dynesty's logz is correct.
# Recompute log_evidence by scaling the nested sampling evidence by sum(ln_evidences).
if ln_evidences is not None:
    sum_ln_ev = np.sum(ln_evidences)
    result.log_noise_evidence = sum_ln_ev
    result.log_evidence = result.log_bayes_factor + sum_ln_ev
    print(f'log noise evidence : {result.log_noise_evidence:.3f}')
    print(f'log Bayes factor   : {result.log_bayes_factor:.3f}')
    print(f'log evidence       : {result.log_evidence:.3f}')
result.save_to_file(overwrite=True)

# call here first, in case the later code fails
# ---we want to get at least something!
result.plot_corner()


def components_and_weights(parameters, *, model, stable_expit, constrain_mu_order):
    m_cut = parameters['m_cut']
    weight_a = parameters['weight_a']

    if constrain_mu_order == 'none':
        mu_1 = parameters['mu_1']
        mu_2 = parameters['mu_2']
    elif constrain_mu_order == 'ascending':
        mu_1 = parameters['g_0']
        mu_2 = parameters['g_0'] + parameters['g_1']
    elif constrain_mu_order == 'descending':
        mu_1 = parameters['g_0'] + parameters['g_1']
        mu_2 = parameters['g_0']

    test_chi = xp.linspace(0, 1, 500)
    test_tau = xp.linspace(-1, 1, 500)

    # Free Gaussian component
    chi = gwpop.utils.truncnorm(test_chi, mu_1, parameters['sigma_1'], 1, 0)
    chi_iso = gwpop.utils.truncnorm(test_chi, mu_2, parameters['sigma_2'], 1, 0)
    chi_high_iso = gwpop.utils.truncnorm(test_chi, parameters['mu_3'], parameters['sigma_3'], 1, 0)

    cos_tilt = gwpop.utils.truncnorm(test_tau, parameters['mu_tilt_1'], parameters['sigma_tilt_1'], 1, -1)

    mm = xp.linspace(3, 300, 500)
    dataset = dict(mass_1=mm)

    if stable_expit:
        zeta = jax.scipy.special.expit(mm - m_cut)
    else:
        zeta = 1 / (1 + jnp.exp(-mm + m_cut))


    weight = (1 - zeta) * weight_a
    weight_iso = (1 - zeta) * (1 - weight_a)
    weight_high_iso = zeta

    if model == 'default-spin-simple-power-law-mass':
        from gwpopulation.models.mass import two_component_single
        p_m1 = two_component_single(
            dataset['mass_1'],
            alpha=parameters['alpha'],
            mmin=parameters['mmin'],
            mmax=parameters['mmax'],
            lam=parameters['lam'],
            mpp=parameters['mpp'],
            sigpp=parameters['sigpp'],
            gaussian_mass_maximum=parameters.get('gaussian_mass_maximum', 100),
        )

        p_m1_comp1 = p_m1
        p_m1_comp2 = p_m1
        p_m1_comp3 = p_m1
    elif model == 'default-spin-bpl2p-mass':
        lam_fractions = (
            parameters['lam_0'],
            parameters['lam_1'],
            1 - parameters['lam_0'] - parameters['lam_1']
        )

        p_m1 = xp.exp(
            BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth(
                dataset,
                parameters["alpha_1"],
                parameters["alpha_2"],
                parameters["mlow_1"],
                parameters["break_mass"],
                parameters["delta_m_1"],
                lam_fractions,
                parameters["mpp_1"],
                parameters["sigpp_1"],
                parameters["mpp_2"],
                parameters["sigpp_2"],
            )
        )

        p_m1_comp1 = p_m1
        p_m1_comp2 = p_m1
        p_m1_comp3 = p_m1
    elif model == 'twomass':
        lam_fractions = (
            parameters['lam_0'],
            parameters['lam_1'],
            1 - parameters['lam_0'] - parameters['lam_1']
        )

        p_m1_comp1 = xp.exp(
            BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth(
                dataset,
                parameters["alpha_1"],
                parameters["alpha_2"],
                parameters["mlow_1"],
                parameters["break_mass"],
                parameters["delta_m_1"],
                lam_fractions,
                parameters["mpp_1"],
                parameters["sigpp_1"],
                parameters["mpp_2"],
                parameters["sigpp_2"],
            )
        )

        lam_fractions = (
            parameters['lam_iso_0'],
            parameters['lam_iso_1'],
            1 - parameters['lam_iso_0'] - parameters['lam_iso_1']
        )

        p_m1_comp2 = xp.exp(
            BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth(
                dataset,
                parameters["alpha_1_iso"],
                parameters["alpha_2_iso"],
                parameters["mlow_1_iso"],
                parameters["break_mass_iso"],
                parameters["delta_m_1_iso"],
                lam_fractions,
                parameters["mpp_1_iso"],
                parameters["sigpp_1_iso"],
                parameters["mpp_2_iso"],
                parameters["sigpp_2_iso"],
            )
        )

        p_m1_comp3 = p_m1_comp2
    elif model == 'threemass':
        lam_fractions = (
            parameters['lam_0'],
            parameters['lam_1'],
            1 - parameters['lam_0'] - parameters['lam_1']
        )

        p_m1_comp1 = xp.exp(
            BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth(
                dataset,
                parameters["alpha_1"],
                parameters["alpha_2"],
                parameters["mlow_1"],
                parameters["break_mass"],
                parameters["delta_m_1"],
                lam_fractions,
                parameters["mpp_1"],
                parameters["sigpp_1"],
                parameters["mpp_2"],
                parameters["sigpp_2"],
            )
        )

        lam_fractions = (
            parameters['lam_iso_0'],
            parameters['lam_iso_1'],
            1 - parameters['lam_iso_0'] - parameters['lam_iso_1']
        )

        p_m1_comp2 = xp.exp(
            BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth(
                dataset,
                parameters["alpha_1_iso"],
                parameters["alpha_2_iso"],
                parameters["mlow_1_iso"],
                parameters["break_mass_iso"],
                parameters["delta_m_1_iso"],
                lam_fractions,
                parameters["mpp_1_iso"],
                parameters["sigpp_1_iso"],
                parameters["mpp_2_iso"],
                parameters["sigpp_2_iso"],
            )
        )

        lam_fractions = (
            parameters['lam_high_iso_0'],
            parameters['lam_high_iso_1'],
            1 - parameters['lam_high_iso_0'] - parameters['lam_high_iso_1']
        )

        p_m1_comp3 = xp.exp(
            BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth(
                dataset,
                parameters["alpha_1_high_iso"],
                parameters["alpha_2_high_iso"],
                parameters["mlow_1_high_iso"],
                parameters["break_mass_high_iso"],
                parameters["delta_m_1_high_iso"],
                lam_fractions,
                parameters["mpp_1_high_iso"],
                parameters["sigpp_1_high_iso"],
                parameters["mpp_2_high_iso"],
                parameters["sigpp_2_high_iso"],
            )
        )
    else:
        raise ValueError(f'bad model {model}')

    weight = xp.trapezoid(
        weight * p_m1_comp1,
        mm
    )

    weight_iso = xp.trapezoid(
        weight_iso * p_m1_comp2,
        mm
    )

    weight_high_iso = xp.trapezoid(
        weight_high_iso * p_m1_comp3,
        mm
    )

    return dict(
        zeta=zeta,
        cos_tilt=cos_tilt,
        chi=chi,
        chi_iso=chi_iso,
        chi_high_iso=chi_high_iso,
        weight=weight,
        weight_iso=weight_iso,
        weight_high_iso=weight_high_iso,
        mass_1=p_m1_comp1,
        mass_1_iso=p_m1_comp2,
        mass_1_high_iso=p_m1_comp3
    )


samples = result.posterior.to_dict('list')
samples = {k : xp.array(v) for k, v in samples.items()}

extras = scan(likelihood.generate_extra_statistics)(samples)
extras['samples'] = samples
h5ify.save(f'{outdir}/posterior.h5', extras, mode='w')

result.posterior['variance'] = extras['variance']

if 'gaussian_mass_maximum' in result.posterior:
    result.posterior.pop('gaussian_mass_maximum')

result.plot_corner(parameters=list(result.posterior.keys()))

cw = scan(
    lambda p: components_and_weights(
        p,
        model=model,
        stable_expit=stable_expit,
        constrain_mu_order=constrain_mu_order,
    )
)(samples)
h5ify.save(f'{outdir}/components-and-weights.h5', cw, mode='w')

# truncnorm component
# TODO: fix `chi`
for param in ['cos_tilt']:
    fig, ax = plt.subplots()

    if param == 'cos_tilt':
        xs = jnp.linspace(-1, 1, 500)
        ax.set_ylim(-0.1, 1.0)
        ax.set_xlim(-1, 1)
        ax.set_xlabel(r'$\cos \theta_{1,2}$')
    elif param == 'chi':
        xs = jnp.linspace(0, 1, 500)
        ax.set_ylim(-0.1, 3.0)
        ax.set_xlim(0, 1)
        ax.set_xlabel(r'$\chi_{1,2}$')

    prob1 = cw[param] * cw['weight'][:, None]
    med = jnp.median(prob1, axis=0)
    q05 = jnp.quantile(prob1, 0.05, axis=0)
    q95 = jnp.quantile(prob1, 0.95, axis=0)
    ax.plot(xs, med, color=f'C0', label='truncnorm tilts')
    ax.fill_between(xs, q05, q95, alpha=0.4, color=f'C0', lw=0)

    # iso component
    prob2 = 0.5 * jnp.ones_like(cw[param]) * cw['weight_iso'][:, None]
    med = jnp.median(prob2, axis=0, )
    q05 = jnp.quantile(prob2, 0.05, axis=0)
    q95 = jnp.quantile(prob2, 0.95, axis=0)
    ax.plot(xs, med, color=f'C1', label='Low-mass iso tilts')
    ax.fill_between(xs, q05, q95, alpha=0.4, color=f'C1', lw=0)

    # high-iso component
    prob3 = 0.5 * jnp.ones_like(cw[param]) * cw['weight_high_iso'][:, None]
    med = jnp.median(prob3, axis=0)
    q05 = jnp.quantile(prob3, 0.05, axis=0)
    q95 = jnp.quantile(prob3, 0.95, axis=0)
    ax.plot(xs, med, color=f'C2', label='High-mass iso tilts')
    ax.fill_between(xs, q05, q95, alpha=0.4, color=f'C2', lw=0)

    prob = prob1 + prob2 + prob3
    med = jnp.median(prob, axis=0, )
    q05 = jnp.quantile(prob, 0.05, axis=0)
    q95 = jnp.quantile(prob, 0.95, axis=0)
    ax.plot(xs, med, color=f'C3', label='Weighted sum')
    ax.fill_between(xs, q05, q95, alpha=0.4, color=f'C3', lw=0)

    ax.axhline(y=0, lw=0.5, color='grey')
    ax.legend()

    fig.savefig(f'{outdir}/ppds/{param}.png')

print('done.')
