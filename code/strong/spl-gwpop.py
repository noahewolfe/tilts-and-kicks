""" ripped from: https://github.com/stegmaja/black-hole-spin-orbit-tilts/blob/main/main.ipynb """
import os
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

parser = ArgumentParser()
parser.add_argument('--outdir', type=str, required=True)
parser.add_argument('--which-data', type=str, required=True)
parser.add_argument('--which-model', type=str, required=True)
parser.add_argument('--sampling-seed', type=int, default=1701)
parser.add_argument('--maximum-uncertainty')

args = parser.parse_args()
write_config(args)
outdir = args.outdir
which_data = args.which_data
which_model = args.which_model
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

if which_model == 'noah':
    from models import log_stegmann_spin

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

# Define custom spin model
def spin_model(dataset, mu_1, sigma_1, mu_tilt_1, sigma_tilt_1, 
               mu_2, sigma_2, mu_3, sigma_3,
               weight_a, m_cut):
    
    # Unpack variables from dataset
    a_1 = dataset["a_1"]
    a_2 = dataset["a_2"]
    cos_tilt_1 = dataset["cos_tilt_1"]
    cos_tilt_2 = dataset["cos_tilt_2"]
    m_1 = dataset["mass_1"]

    # STEGMANN model
    if which_model == 'stegmann':
        # Free Gaussian component
        comp1 = gwpop.utils.truncnorm(a_1, mu_1, sigma_1, 1, 0) * \
                gwpop.utils.truncnorm(a_2, mu_1, sigma_1, 1, 0) * \
                gwpop.utils.truncnorm(cos_tilt_1, mu_tilt_1, sigma_tilt_1, 1, -1) * \
                gwpop.utils.truncnorm(cos_tilt_2, mu_tilt_1, sigma_tilt_1, 1, -1)

        # Isotropic component
        comp2 = gwpop.utils.truncnorm(a_1, mu_2, sigma_2, 1, 0) * \
                gwpop.utils.truncnorm(a_2, mu_2, sigma_2, 1, 0) * \
                0.5 * 0.5 
        
        # High-mass isotropic component
        comp3 = gwpop.utils.truncnorm(a_1, mu_3, sigma_3, 1, 0) * \
                gwpop.utils.truncnorm(a_2, mu_3, sigma_3, 1, 0) * \
                0.5 * 0.5 
        
        # Mass-dependent transition function
        zeta = 1 / (1 + xp.exp(-m_1+m_cut))
        
        # Combine components with mass-dependent transition
        return (1 - zeta) * (weight_a * comp1 + (1 - weight_a) * comp2) + zeta * comp3

    # NOAH model
    elif which_model == 'noah':
        return xp.exp(log_stegmann_spin(
            dataset, dict(
                mu_chi=mu_1,
                sigma_chi=sigma_1,
                mu_chi_iso=mu_2,
                sigma_chi_iso=sigma_2,
                mu_chi_high_iso=mu_3,
                sigma_chi_high_iso=sigma_3,
                mu_spin=mu_tilt_1,
                sigma_spin=sigma_tilt_1,
                xi_spin=weight_a,
                transition_mass=m_cut
            )
        ))

    else:
        raise ValueError(f'bad model {which_model}')

# Define the full population model, combining mass, spin, and redshift models
model = Model(
    model_functions=[
        gwpop.models.mass.two_component_primary_mass_ratio,
        spin_model,
        gwpop.models.redshift.PowerLawRedshift(cosmo_model="Planck15"),
    ],
    cache=False,
)

vt = gwpop.vt.ResamplingVT(model=model, data=injections, n_events=len(posteriors))

likelihood = gwpop.hyperpe.HyperparameterLikelihood(
    posteriors=posteriors,
    hyper_prior=model,
    selection_function=vt,
    maximum_uncertainty=maximum_uncertainty,
)

# Define priors for hyperparameters
priors = PriorDict()

# mass
priors["alpha"] = Uniform(minimum=-2, maximum=4, latex_label="$\\alpha$")
priors["beta"] = Uniform(minimum=-4, maximum=12, latex_label="$\\beta$")
priors["mmin"] = Uniform(minimum=2, maximum=2.5, latex_label="$m_{\\min}$")
priors["mmax"] = Uniform(minimum=80, maximum=100, latex_label="$m_{\\max}$")
priors["lam"] = Uniform(minimum=0, maximum=1, latex_label="$\\lambda_{m}$")
priors["mpp"] = Uniform(minimum=10, maximum=50, latex_label="$\\mu_{m}$")
priors["sigpp"] = Uniform(minimum=1, maximum=10, latex_label="$\\sigma_{m}$")
priors["gaussian_mass_maximum"] = 100
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

# Test JittedLikelihood
parameters = dict(
    alpha=2.0,
    beta=1.1,
    mmin=2.25,
    mmax=85.0,
    lam=0.03,
    mpp=34.0,
    sigpp=3.4,
    gaussian_mass_maximum=100.0,
    mu_1=0.2,
    sigma_1=0.5,
    mu_2=0.2,
    sigma_2=0.5,
    mu_3=0.7,
    sigma_3=0.3,
    mu_tilt_1=0.15,
    sigma_tilt_1=0.4,
    weight_a=0.86,
    m_cut=45.0,
    lamb=2.73
)

print('testing likelihood with parameters:')
for k, v in parameters.items():
    print(f'\t{k} : {v}')

likelihood.log_likelihood_ratio(parameters)
jit_likelihood = JittedLikelihood(likelihood)
jit_likelihood.log_likelihood_ratio(parameters)
print(
    'log_likelihood:',
    jit_likelihood.log_likelihood_ratio(parameters)
)

# Run sampler
result = bb.run_sampler(
    likelihood=jit_likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=nlive,
    label='run',
    sample="acceptance-walk",
    naccept=naccept,
    save="hdf5",
    outdir=outdir,
    seed=sampling_seed
)
