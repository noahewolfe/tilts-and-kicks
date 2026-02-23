""" ripped from: https://github.com/stegmaja/black-hole-spin-orbit-tilts/blob/main/main.ipynb """
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import bilby as bb
from bilby.hyper.model import Model
from bilby.core.prior import PriorDict, Uniform, TruncatedNormal

import gwpopulation as gwpop
from gwpopulation.experimental.jax import JittedLikelihood
gwpop.set_backend("jax")

xp = gwpop.utils.xp

from util import write_config

label = 'run'

parser = ArgumentParser()
parser.add_argument('--outdir', type=str, required=True)
parser.add_argument('--which-data', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--sampling-seed', type=int, default=1701)
parser.add_argument('--maximum-uncertainty', required=True)
parser.add_argument('--mass-prior', type=str)

args = parser.parse_args()
write_config(args)
outdir = args.outdir
which_data = args.which_data
model = args.model

if model != 'default-spin-simple-power-law-mass':
    mass_prior = args.mass_prior

sampling_seed = args.sampling_seed
maximum_uncertainty = args.maximum_uncertainty

# stegmann data
if which_data == 'stegmann':
    datadir = '../../data/stegmann'
    posteriors = pd.read_pickle(f"{datadir}/gwtc4_posteriors.pkl")
    injections = pd.read_pickle(f"{datadir}/gwtc4_injections_dict.pkl")

### load noah data
elif which_data == 'noah':
    from data import get_data
    _, posteriors, injections = get_data(
        snr_thresh=10,
        far_thresh=1,
        prefer_xphm=False,
        prefer_xphm_gwtc3=True
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
###

# We are considering the default Gaussian_Isotropic_Cut spin model from Stegmann et al. (2025)

################## IMPORTANT SETTINGS ##################

# Control maximum uncertainty in selection function estimation
# For production runs (as used in Stegmann et al. (2025)), I recommend setting maximum_uncertainty = 1, naccept = 5, nlive = 1000 (which should take several hours)
# For quick tests, you can set maximum_uncertainty = xp.inf, naccept = 5, nlive = 100 (which should take several minutes)

if maximum_uncertainty == 'inf':
    maximum_uncertainty = xp.inf
else:
    maximum_uncertainty = int(maximum_uncertainty)
print(f'using variance cut : {maximum_uncertainty}')
naccept = 5
nlive = 100

########################################################


def get_model(model):
    if model == 'default-spin-simple-power-law-mass':
        from models import default_stegmann_spin_model
        model_functions = [
            gwpop.models.mass.two_component_primary_mass_ratio,
            default_stegmann_spin_model,
        ]
    elif model == 'default-spin-bpl2p-mass':
        from models import default_stegmann_spin_model
        from models import BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth

        def mass_model(
            dataset,
            alpha_1,
            alpha_2,
            mlow_1,
            break_mass,
            delta_m_1,
            lam_0,
            lam_1,
            mpp_1,
            sigpp_1,
            mpp_2,
            sigpp_2,
        ):
            lam_fractions = (
                lam_0,
                lam_1,
                1 - lam_0 - lam_1
            )
            return xp.exp(BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth(
                dataset,
                alpha_1,
                alpha_2,
                mlow_1,
                break_mass,
                delta_m_1,
                lam_fractions,
                mpp_1,
                sigpp_1,
                mpp_2,
                sigpp_2,
            ))

        model_functions = [
            mass_model,
            default_stegmann_spin_model 
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
)

# Define priors for hyperparameters

# mass
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
    priors = ConditionalPriorDict(mass_prior)

# spin
priors["mu_1"] = Uniform(minimum=0, maximum=1, latex_label="$\\mu_1$")
priors["sigma_1"] = Uniform(minimum=0.1, maximum=1, latex_label="$\\sigma_1$")
priors["mu_tilt_1"] = Uniform(minimum=-1, maximum=1, latex_label="$\\mu_{t,1}$")
priors["sigma_tilt_1"] = TruncatedNormal(minimum=0.1, maximum=4, sigma=1/2, mu=0, latex_label="$\\sigma_{t,1}$")
priors["mu_2"] = Uniform(minimum=0, maximum=1, latex_label="$\\mu_2$")
priors["sigma_2"] = Uniform(minimum=0.1, maximum=1, latex_label="$\\sigma_2$")
priors["weight_a"] = Uniform(minimum=0, maximum=1, latex_label="$w_a$")
priors["mu_3"] = Uniform(minimum=0, maximum=1, latex_label="$\\mu_3$")
priors["sigma_3"] = Uniform(minimum=0.1, maximum=1, latex_label="$\\sigma_3$")
priors["m_cut"] = Uniform(minimum=10, maximum=100, latex_label="$m_{\\rm cut}$")

# redshift
priors["lamb"] = Uniform(minimum=-1, maximum=10, latex_label="$\\lambda_{z}$")

priors.to_file(outdir, label)

jit_likelihood = JittedLikelihood(likelihood)
jit_likelihood.log_likelihood_ratio(priors.sample())

# Run sampler
result = bb.run_sampler(
    likelihood=jit_likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=nlive,
    label=label,
    sample="acceptance-walk",
    naccept=naccept,
    save="hdf5",
    outdir=outdir,
    seed=sampling_seed
)
result.plot_corner()

# TODO: compute extras

# TODO: make ppds