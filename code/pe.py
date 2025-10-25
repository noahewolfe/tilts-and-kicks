# run PE. this is designed to be used with condor
# and for simplicity makes some assumptions: single-core runs, using dynesty
# IMRPhenomPv2, relative binning for all signals

import os
import ast
import pickle
import argparse
from shutil import move
from copy import deepcopy

import h5py
import numpy as np
import pandas as pd
from tqdm import trange

import bilby
from bilby.core.result import read_in_result
from bilby.gw.prior import BBHPriorDict
from bilby.gw.source import (
    lal_binary_black_hole,
    lal_binary_black_hole_relative_binning
)
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.likelihood import RelativeBinningGravitationalWaveTransient

from bilby_utils import get_network

from util import get_git_revision_short_hash, next_power_of_2

parameter_keys = [
    'chirp_mass',
    'mass_ratio',
    'luminosity_distance',
    'a_1',
    'a_2',
    'tilt_1',
    'tilt_2',
    'phi_12',
    'phi_jl',
    'theta_jn',
    'ra',
    'dec',
    'phase',
    'psi',
    'geocent_time'
]

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str)
parser.add_argument('--npool', type=int)
parser.add_argument('--prior-path', type=str)
parser.add_argument('--mchirp-width', type=float, default=-1)
parser.add_argument('--event-index', type=int)
parser.add_argument('--catalog-path', type=str)

parser.add_argument('--reference-frame', type=str, default="'H1L1'")
parser.add_argument('--time-reference', type=str, default='best')

parser.add_argument('--nlive', type=int, default=500)
parser.add_argument('--naccept', type=int, default=60)
parser.add_argument('--bound', type=str, default='live')
parser.add_argument('--proposals', type=list, default=['diff'])

parser.add_argument('--rerun-with-wider-priors', action='store_true')
parser.add_argument('--rerun-mass-delta', type=int, default=10)
parser.add_argument('--rerun-dl-delta', type=int, default=1000)
parser.add_argument('--prior-lowest-allowed-mchirp', type=float, default=3)

parser.add_argument('--outdir-extras', type=str, default='')


def digest_args(args):
    outdir = args.outdir
    npool = args.npool

    nlive = args.nlive
    naccept = args.naccept
    bound = args.bound
    proposals = args.proposals

    catalog_path = args.catalog_path
    event_index = args.event_index

    rerun_with_wider_priors = args.rerun_with_wider_priors
    rerun_mass_delta = args.rerun_mass_delta
    rerun_dl_delta = args.rerun_dl_delta
    prior_lowest_allowed_mchirp = args.prior_lowest_allowed_mchirp

    outdir_extras = args.outdir_extras
    if outdir_extras == '':
        outdir_extras = None

    network = 'H1L1V1O4_SALVO'
    minimum_frequency = 20
    sampling_frequency = 2048
    reference_frequency = 20

    catalog_ext = os.path.splitext(catalog_path)[1]
    if catalog_ext == '.dat':
        catalog = pd.read_csv(catalog_path, sep='\t')
    elif catalog_ext == '.hdf5':
        catalog = pd.read_hdf(catalog_path, 'detectable')

    # this preserves dtype for ints,
    # whereas catalog.iloc[event_index].to_dict() does not
    # ... if the file was originally saved s.t. the data_seed is an int
    noise_seed = catalog['data_seed'].iloc[event_index]

    event = catalog.iloc[event_index]
    injection_parameters = {
        k : event.get(k)
        for k in parameter_keys + ['redshift', 'mass_1', 'mass_2']
    }

    redshift = injection_parameters.pop('redshift')
    if injection_parameters['luminosity_distance'] is None:
        injection_parameters['luminosity_distance'] = (
            bilby.gw.conversion.redshift_to_luminosity_distance(
                redshift,
                cosmology='Planck15'
            )
        )

    true_m1 = injection_parameters.pop('mass_1')
    true_m2 = injection_parameters.pop('mass_2')

    raw_duration = bilby.gw.utils.calculate_time_to_merger(
        minimum_frequency,
        true_m1,
        true_m2
    )
    duration = next_power_of_2(int(np.ceil(raw_duration)))
    duration = 16 if duration < 16 else duration

    prior_path = args.prior_path
    priors = bilby.gw.prior.BBHPriorDict(prior_path)

    if not args.mchirp_width > 0:
        mchirp = injection_parameters['chirp_mass']

        # this is an approximate & conservative fit based off of salvo's runs
        # the maximum width of 46 reflects salvo's original PE script
        mchirp_width = 1 / 2 * min(46, 1e-2 * mchirp**(5 / 2))
    else:
        mchirp_width = np.inf

    mchirp_width = float(mchirp_width)
    if rerun_with_wider_priors:
        mchirp_width += rerun_mass_delta

    delta_dl_bound = rerun_dl_delta if rerun_with_wider_priors else 0

    if prior_lowest_allowed_mchirp == -1:
        print(
            'Computing the lowest allowed chirp mass, still consistent with '
            f'a duration of {duration} s.'
        )

        def compute_duration(mass_1):
            """ this is a conservative computation, as we assume q = 1 """

            try:
                raw_duration = bilby.gw.utils.calculate_time_to_merger(
                    minimum_frequency,
                    mass_1,
                    mass_1
                )
                return next_power_of_2(int(np.ceil(raw_duration)))
            except RuntimeError:
                return 0

        test_m1 = np.linspace(
            3, true_m1, 1000
        )
        test_durations = np.array([
            compute_duration(m1)
            for m1 in test_m1
        ])
        print(test_m1)
        lowest_allowed_m1 = test_m1[np.where(test_durations <= duration)[0][0]]
        print(
            f'Lowest allowed primary mass: {lowest_allowed_m1} '
            f'(compared to true primary mass {true_m1})'
        )
        prior_lowest_allowed_mchirp = (1 / 4)**(3 / 5) * 2 * lowest_allowed_m1

    print(f'Using {prior_lowest_allowed_mchirp} as the lowest allowed mchirp')
    priors['chirp_mass'].minimum = (
        max([
            prior_lowest_allowed_mchirp,
            injection_parameters['chirp_mass'] - mchirp_width
        ])
    )
    priors['chirp_mass'].maximum = (
        min([
            200,
            injection_parameters['chirp_mass'] + mchirp_width
        ])
    )

    priors['luminosity_distance'].minimum = (
        max([
            1,
            injection_parameters['luminosity_distance'] / 4 - delta_dl_bound
        ])
    )

    # Note: I copied the '4.2' from salvo's script for
    # extra safety/consistency..
    priors['luminosity_distance'].maximum = (
        injection_parameters['luminosity_distance'] * 4.2 + delta_dl_bound
    )

    injection_parameters['fiducial'] = 1
    fiducial_parameters = deepcopy(injection_parameters)

    likelihood_kwargs = dict()
    likelihood_kwargs['fiducial_parameters'] = fiducial_parameters
    likelihood_kwargs['epsilon'] = 0.025

    reference_frame = args.reference_frame
    if reference_frame != 'sky':
        reference_frame = ast.literal_eval(reference_frame)
    likelihood_kwargs['reference_frame'] = reference_frame

    if reference_frame != 'sky':
        if 'ra' in priors.keys():
            del priors['ra']

        if 'dec' in priors.keys():
            del priors['dec']

        priors["zenith"] = bilby.core.prior.Sine(latex_label="$\\kappa$")
        priors["azimuth"] = bilby.core.prior.Uniform(
            minimum=0,
            maximum=2 * np.pi,
            latex_label="$\\epsilon$",
            boundary="periodic"
        )

    time_reference = args.time_reference
    likelihood_kwargs['time_reference'] = time_reference
    if time_reference != 'geocent':
        if 'geocent_time' in priors.keys():
            del priors['geocent_time']

    optimal_network_snr = event.get('network_optimal_snr')
    matched_filter_network_snr = event.get('network_matched_filter_snr')

    rerun = rerun_with_wider_priors

    return (
        outdir, npool, catalog_path, event_index, priors, injection_parameters,
        noise_seed, duration,
        minimum_frequency, sampling_frequency, reference_frequency,
        network, likelihood_kwargs, nlive, naccept, bound, proposals,
        optimal_network_snr, matched_filter_network_snr, rerun, outdir_extras
    )


def get_waveform_generator(
    duration, minimum_frequency, sampling_frequency, reference_frequency,
    waveform_approximant, likelihood_class, phenomxprecversion
):
    waveform_arguments = dict(
        waveform_approximant=waveform_approximant,
        reference_frequency=reference_frequency,
        minimum_frequency=minimum_frequency,
        PhenomXPrecVersion=phenomxprecversion
    )

    print(f'waveform_arguments = {waveform_arguments}')

    frequency_domain_source_model = lal_binary_black_hole_relative_binning
    return waveform_arguments, bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=frequency_domain_source_model,
        parameter_conversion=convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=deepcopy(waveform_arguments),
    )


def combine(outdir, exclude=[], append=False):
    # TODO: save the actual bilby prior dict
    from wcosmo.wcosmo import dDLdz
    from astropy.cosmology import Planck15
    from gwpopulation_pipe.data_collection import (
        primary_mass_to_chirp_mass_jacobian
    )

    outdir = os.path.abspath(outdir)
    dirs = [
        f'{outdir}/{d}'
        for d in os.listdir(outdir)
        if (
            os.path.isdir(f'{outdir}/{d}')
            and 'rerun' not in d
            and 'old' not in d
        )
    ]
    for i, d in enumerate(dirs):
        if not os.path.isfile(f'{d}/dynesty_result.json'):
            exclude.append(i)
            print(f'no result found in {d}')

    def load(i):
        result = read_in_result(f'{dirs[i]}/dynesty_result.json')
        posterior = result.posterior
        injection_parameters = result.injection_parameters

        mf_net_snr = 0
        for ifo in ['H1', 'L1', 'V1']:
            mf_net_snr += np.abs(posterior[f'{ifo}_matched_filter_snr'])**2
        mf_net_snr = np.sqrt(mf_net_snr)
        posterior['mf_net_snr'] = mf_net_snr

        ifos = result.meta_data['likelihood']['interferometers']
        assert all([ifo in ifos.keys() for ifo in ['H1', 'L1', 'V1']])

        true_mf_net_snr = 0
        for ifo in ifos.keys():
            true_mf_net_snr += np.abs(ifos[ifo]['matched_filter_SNR'])**2
        true_mf_net_snr = np.sqrt(true_mf_net_snr)
        injection_parameters['network_matched_filter_snr'] = true_mf_net_snr

        injection_parameters.pop('reference_frequency')
        injection_parameters.pop('waveform_approximant')

        prior_dict = str(
            BBHPriorDict(f'{dirs[i]}/dynesty.prior')._get_json_dict()
        )

        return prior_dict, injection_parameters, posterior

    def compute_posterior_for_gwpop(posterior):
        posterior = deepcopy(posterior)

        # assumes uniform priors in a_{1,2}, cos_tilt_{1,2}
        # also assumes uniform in chirp_mass, mass_ratio, luminosity_distance
        prior = (
            # convert from (chirp mass, q) to (m1, q)
            primary_mass_to_chirp_mass_jacobian(posterior)**(-1)

            # convert from det. frame to src. frame primary mass
            * (1 + posterior['redshift'])

            # convert from lum. dist. to redshift
            * dDLdz(
                posterior['redshift'],
                H0=Planck15.H0.value,
                Om0=Planck15.Om0
            )
        )
        posterior['prior'] = prior

        posterior['mass_1'] = posterior.pop('mass_1_source')
        posterior = posterior[[
            'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2', 'mass_1', 'mass_ratio',
            'redshift', 'prior', 'mf_net_snr'
        ]]

        return posterior

    full_outpath = f'{outdir}/full.hdf5'
    post_outpath = f'{outdir}/posteriors.hdf5'

    mode = 'a' if append else 'w'

    with (
        h5py.File(full_outpath, mode) as f_full,
        h5py.File(post_outpath, mode) as f_post
    ):
        for i in trange(len(dirs)):
            if i in exclude:
                continue

            if append and str(i) in f_full.keys():
                continue

            try:
                prior_dict, injection_parameters, posterior = load(i)
                posterior_gwpop = compute_posterior_for_gwpop(posterior)
            except:
                print(f'error loading {i}')
                continue

            g_full = f_full.create_group(str(i))
            g_post = f_post.create_group(str(i))

            for key in injection_parameters.keys():
                val = injection_parameters[key]
                if isinstance(val, np.ndarray):
                    val = val.item()
                g_full.attrs[key] = val
                g_post.attrs[key] = val

            g_full.attrs['prior_dict'] = prior_dict
            g_post.attrs['prior_dict'] = prior_dict

            for key in posterior.keys():
                g_full.create_dataset(
                    key,
                    data=posterior[key].values
                )

            for key in posterior_gwpop.keys():
                g_post.create_dataset(
                    key,
                    data=posterior_gwpop[key].values
                )


precessing_waveforms = [
    'IMRPhenomPv2', 'IMRPhenomXP', 'IMRPhenomXPHM'
]
precession_only_keys = ['tilt_1', 'tilt_2', 'phi_jl', 'phi_12']

if __name__ == '__main__':
    print(f'using: {get_git_revision_short_hash()}')

    args = parser.parse_args()
    (
        outdir,
        npool,
        catalog_path,
        event_index,
        priors,
        injection_parameters,
        noise_seed,
        duration,
        minimum_frequency,
        sampling_frequency,
        reference_frequency,
        network,
        likelihood_kwargs,
        nlive,
        naccept,
        bound,
        proposals,
        optimal_network_snr,
        matched_filter_network_snr,
        rerun,
        outdir_extras
    ) = digest_args(args)

    likelihood_class = RelativeBinningGravitationalWaveTransient
    waveform_approximant = 'IMRPhenomXP'
    phenomxprecversion = 104

    outdir = f'{outdir}/{event_index}'
    if rerun:
        outdir += '_rerun'
    if outdir_extras is not None:
        outdir += f'_{outdir_extras}'
    os.makedirs(outdir, exist_ok=True)

    print('--- \\ waveform_generator for likelihood evaluations / ---')
    waveform_arguments, waveform_generator = get_waveform_generator(
        duration, minimum_frequency, sampling_frequency, reference_frequency,
        waveform_approximant, likelihood_class, phenomxprecversion
    )

    print('--- \\ waveform_generator for the injection / ---')
    _, injection_waveform_generator = get_waveform_generator(
        duration, minimum_frequency, sampling_frequency, reference_frequency,
        waveform_approximant, likelihood_class, phenomxprecversion
    )

    ifos = get_network(
        network,
        minimum_frequency=minimum_frequency,
        maximum_frequency=sampling_frequency / 2
    )

    start_time = injection_parameters['geocent_time'] + 2 - duration

    if isinstance(noise_seed, int) or isinstance(noise_seed, np.int64):
        bilby.core.utils.random.seed(noise_seed)

        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time
        )

        with open(f'{outdir}/seed.txt', 'w') as f:
            f.write(str(noise_seed))
    else:
        ifos.set_strain_data_from_zero_noise(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time
        )

    for ifo in ifos:
        np.save(
            f'{outdir}/{ifo.name}-noise.npy',
            ifo.whitened_frequency_domain_strain,
        )

    waveform_polarizations = ifos.inject_signal(
        waveform_generator=injection_waveform_generator,
        parameters=injection_parameters
    )

    opt_net_snr = np.sqrt(sum([
        ifo.meta_data['optimal_SNR']**2 for ifo in ifos
    ]))
    mf_net_snr = np.sqrt(sum([
        np.abs(ifo.meta_data['matched_filter_SNR'])**2 for ifo in ifos
    ]))

    if not np.isclose(opt_net_snr, optimal_network_snr):
        raise ValueError(
            'Optimal network snrs dont match for '
            f'{event_index} in {catalog_path}! '
            f'Got {opt_net_snr} vs. {optimal_network_snr} in catalog. '
            'Are you sure the snrs and injection parameters match?'
        )

    if not np.isclose(mf_net_snr, matched_filter_network_snr):
        raise ValueError(
            'Matched filter network snrs dont match for '
            f'{event_index} in {catalog_path}! '
            f'Got {mf_net_snr} vs. {matched_filter_network_snr} in catalog. '
            'Are you sure the snrs and noise seeds match? '
            'Do you have the correct bilby version?'
        )

    with open(f'{outdir}/pols.pkl', 'wb') as f:
        pickle.dump(waveform_polarizations, f)

    if likelihood_kwargs['reference_frame'] != 'sky':
        likelihood_kwargs['reference_frame'] = ifos

    if likelihood_kwargs['time_reference'] != 'geocent':
        ref_ifo = None

        if likelihood_kwargs['time_reference'] == 'best':
            best_snr = -np.inf
            ref_ifo = None

            for ifo in ifos:
                osnr = ifo.meta_data['optimal_SNR']
                if osnr > best_snr:
                    best_snr = osnr
                    ref_ifo = ifo

            likelihood_kwargs['time_reference'] = ref_ifo.name
            print(
                f'using {likelihood_kwargs["time_reference"]} for the time '
                'reference because it had the highest snr'
            )
        else:
            for ifo in ifos:
                if ifo.name == likelihood_kwargs['time_reference']:
                    ref_ifo = ifo

        if ref_ifo is None:
            raise ValueError(
                f'time reference is {likelihood_kwargs["time_reference"]} but '
                f'wasnt found in ifos: {ifos}'
            )

        time_delay = ref_ifo.time_delay_from_geocenter(
            injection_parameters["ra"],
            injection_parameters["dec"],
            injection_parameters["geocent_time"],
        )

        key = f'{ref_ifo.name}_time'
        priors[key] = bilby.core.prior.Uniform(
            minimum=injection_parameters["geocent_time"] + time_delay - 0.02,
            maximum=injection_parameters["geocent_time"] + time_delay + 0.02,
            name=key,
        )

    # TODO: ensure that updating priors (eg with key deletion) propogates
    # into likelihood_kwargs

    # dump the priors to a file in the outdir
    if os.path.isfile(f'{outdir}/dynesty.prior'):
        move(f'{outdir}/dynesty.prior', f'{outdir}/dynesty.prior.old')
    priors.to_file(outdir, 'dynesty')

    print('validating signal duration across prior... ', end='')
    priors.validate_prior(duration, minimum_frequency)
    print('done.')

    likelihood_kwargs['priors'] = priors
    likelihood_kwargs['distance_marginalization'] = True

    if likelihood_kwargs['distance_marginalization']:
        likelihood_kwargs['distance_marginalization_lookup_table'] = (
            f'{outdir}/.distance_marginalization_lookup.npz'
        )

    likelihood = likelihood_class(
        interferometers=ifos,
        waveform_generator=waveform_generator,
        **likelihood_kwargs
    )

    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=nlive,
        injection_parameters=injection_parameters,
        outdir=outdir,
        label='dynesty',
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        npool=npool,
        maxmcmc=50_000,
        resume=True
    )

    if 'fiducial' in injection_parameters.keys():
        injection_parameters.pop('fiducial')

    result.plot_corner()

    if isinstance(
        likelihood,
        bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient
    ):
        print('main run done-- reweighting!')

        # \/ lifted from https://git.ligo.org/lscsoft/bilby/-/blob/c62e678518c89f2a8e7c97a6591089d92c056845/examples/gw_examples/injection_examples/relative_binning.py

        alt_waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=lal_binary_black_hole,
            parameter_conversion=convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=deepcopy(waveform_arguments),
        )
        alt_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=ifos,
            waveform_generator=alt_waveform_generator,
        )
        likelihood.distance_marginalization = False
        weights = list()
        for ii in trange(len(result.posterior)):
            parameters = dict(result.posterior.iloc[ii])
            likelihood.parameters.update(parameters)
            alt_likelihood.parameters.update(parameters)
            weights.append(
                alt_likelihood.log_likelihood_ratio()
                - likelihood.log_likelihood_ratio()
            )
        weights = np.exp(weights)
        print(
            f'''
            Reweighting efficiency is
            {np.mean(weights)**2 / np.mean(weights**2) * 100:.5f}%'''
        )
        print(f'''
            Binned vs unbinned log Bayes factor is
            {np.log(np.mean(weights)):.5f}''')

        # Generate result object with the posterior for the regular likelihood
        # using rejection sampling
        alt_result = deepcopy(result)
        keep = weights > np.random.uniform(0, max(weights), len(weights))
        alt_result.posterior = result.posterior.iloc[keep]

        # Make a comparison corner plot.
        bilby.core.result.plot_multiple(
            [result, alt_result],
            labels=["Binned", "Reweighted"],
            filename=f"{outdir}/reweighted_corner.png",
        )

        alt_result.save_to_file(
            filename=f'{outdir}/dynesty_result_reweighted.json',
            extension='json',
            outdir=result.outdir,
            overwrite=False,
            gzip=False
        )