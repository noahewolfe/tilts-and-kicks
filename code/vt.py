import os
import json
import pickle
import argparse
from copy import deepcopy

import h5py
import h5ify
import numpy as np
from tqdm import tqdm

import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp

import bilby
bilby.core.utils.setup_logger(log_level='WARNING')

from bilby.gw.detector import PowerSpectralDensity
from bilby.core.prior import Uniform, Sine, Cosine, Interped
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.conversion import generate_all_bbh_parameters

from pixelpop.models.gwpop_models import PowerlawPlusPeak_MassRatio

from models import truncnorm
from models import iso_gauss_spin_tilt
from models import marg_iso_gauss_spin_tilt
from models import log_powerlaw_redshift
from models import BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth

from util import list_to_dict
from util import write_config
from util import get_git_revision_short_hash
from util import next_power_of_2
from util import concat_dicts


parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str)
parser.add_argument('--ninj', type=int, default=1_000)
parser.add_argument('--sample-prior', action='store_true')
parser.add_argument('--snr-threshold', type=int, default=11)
parser.add_argument('--zero-noise', action='store_true')
parser.add_argument('--seed', '--index', type=int)
parser.add_argument('--init-seed', type=int, default=0)
parser.add_argument('--model', type=str)
parser.add_argument('--parameters', default=None)
parser.add_argument('--extra-kwargs', type=json.loads, default='{}')
parser.add_argument(
    '--injection-waveform-approximant', type=str, default='IMRPhenomXPHM'
)
parser.add_argument(
    '--recovery-waveform-approximant', type=str, default='IMRPhenomXP'
)
parser.add_argument(
    '--interp-net-path',
    type=os.path.abspath,
    default='../data/interp_net.pkl'
)
parser.add_argument(
    '--not-fast',
    action='store_true'
)
parser.add_argument(
    '--overwrite',
    action='store_true',
    help='overwrite existing vt result'
)


def parse_args():
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    write_config(args)
    seed = args.seed + args.init_seed
    model = args.model
    with open(model, 'r') as f:
        model = json.loads(f.read())
    return (
        args.outdir,
        args.ninj,
        args.sample_prior,
        args.snr_threshold,
        args.zero_noise,
        seed,
        model,
        args.parameters,
        args.extra_kwargs,
        args.injection_waveform_approximant,
        args.recovery_waveform_approximant,
        args.interp_net_path,
        not args.not_fast,
        args.overwrite
    )


def write_hdf5(path, dic, total_generated, commit=None):
    with h5py.File(path, 'w') as f:
        f.attrs['total_generated'] = total_generated
        if commit is not None:
            f.attrs['commit'] = commit

        for k, v in dic.items():
            dtype = np.int64 if k == 'data_seed' else np.float64
            f.create_dataset(k, data=v, dtype=dtype)


def load_hdf5_as_dict(path):
    data = dict()
    with h5py.File(path, 'r') as f:
        total_generated = f.attrs['total_generated']
        for key in f.keys():
            data[key] = f[key][:]
    return total_generated, data


def get_parameters(path=None, outdir=None, **kwargs):
    parameters = dict()

    if path is not None:
        with open(path, 'r') as f:
            parameters.update(json.loads(f.read()))

    parameters.update(kwargs)

    if outdir is not None:
        with open(f'{outdir}/parameters.json', 'w') as f:
            f.write(json.dumps(parameters, indent=4, sort_keys=False))

        return parameters


def get_inj_priors(model, parameters, outdir=None):
    inj_priors = bilby.core.prior.PriorDict(dict(
        dec=Cosine(name='dec'),
        ra=Uniform(
            name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic'
        ),
        theta_jn=Sine(name='theta_jn'),
        psi=Uniform(
            name='psi', minimum=0, maximum=np.pi, boundary='periodic'
        ),
        phase=Uniform(
            name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic'
        ),
        phi_12=Uniform(
            name='phi_12', minimum=0, maximum=2 * np.pi, boundary='periodic'
        ),
        phi_jl=Uniform(
            name='phi_jl', minimum=0, maximum=2 * np.pi, boundary='periodic'
        ),
        geocent_time=Uniform(
            name='geocent_time',
            minimum=1126259642.413,
            maximum=1126259642.413 + 86_400
        )
    ))

    if model['cos_tilt'] != 'iso_gauss':
        if model['cos_tilt'] == 'uniform':
            ct_low = parameters['cos_tilt_min']
            ct_high = parameters['cos_tilt_max']

            inj_priors['cos_tilt_1'] = Uniform(
                ct_low, ct_high, name='cos_tilt_1'
            )
            inj_priors['cos_tilt_2'] = Uniform(
                ct_low, ct_high, name='cos_tilt_2'
            )
        else:
            raise ValueError(
                f"unknown tilt model {model['cos_tilt']}"
            )

    if model['a_1'] == 'iid_truncnorm' and model['a_2'] == 'iid_truncnorm':
        mu_chi = parameters['mu_chi']
        sigma_chi = parameters['sigma_chi']
        mags = np.linspace(0, 1, 500)
        pchi = truncnorm(mags, mu_chi, sigma_chi, high=1, low=0)

        chi_prior = Interped(
            mags,
            pchi,
            minimum=min(mags),
            maximum=max(mags),
            name='chi'
        )
        inj_priors['a_1'] = chi_prior
        inj_priors['a_2'] = chi_prior
    elif model['a_1'] == 'uniform' and model['a_2'] == 'uniform':
        inj_priors['a_1'] = Uniform(0, 1, name='a_1')
        inj_priors['a_2'] = Uniform(0, 1, name='a_2')
    else:
        raise ValueError(f"unknown spin models {model['a_1'], model['a_2']}")

    if model['redshift'] == 'powerlaw':
        z_max = parameters['z_max']
        zs = np.linspace(1e-5, z_max, 1000)
        pz = np.exp(log_powerlaw_redshift(
            dict(redshift=zs),
            parameters
        ))
        z_prior = Interped(
            zs,
            np.array(pz),
            minimum=min(zs),
            maximum=max(zs),
            name='redshift'
        )
        inj_priors['redshift'] = z_prior
    else:
        raise ValueError(f"unknown redshift model {model['redshift']}")

    if model['mass_1_source'] == 'highpass_broken_powerlaw_two_peaks':
        m1s = np.linspace(3, 300, 1_000)
        p_m1 = np.exp(BrokenPowerlawPlusTwoPeaks_PrimaryMass_FullSmooth(
            dict(mass_1=m1s),
            alpha_1=parameters['alpha_1'],
            alpha_2=parameters['alpha_2'],
            mlow_1=parameters['mlow_1'],
            break_mass=parameters['break_mass'],
            delta_m_1=parameters['delta_m_1'],
            lam_fractions=(
                parameters['lam_0'], parameters['lam_1'], parameters['lam_2']
            ),
            mpp_1=parameters['mpp_1'],
            sigpp_1=parameters['sigpp_1'],
            mpp_2=parameters['mpp_2'],
            sigpp_2=parameters['sigpp_2'],
            mmax=300.0,
            gaussian_mass_maximum=100.0
        ))
        m1_prior = Interped(
            m1s,
            p_m1,
            minimum=min(m1s),
            maximum=max(m1s),
            name='mass_1_source'
        )
        inj_priors['mass_1_source'] = m1_prior
    else:
        raise ValueError(
            f"unknown mass_1_source model {model['mass_1_source']}"
        )

    if outdir is not None:
        inj_priors.to_file(outdir, 'inj')

    return inj_priors


def fill_parameters(injection_parameters, priors):
    """ fill in some extra parameters and log_prior column """
    injection_parameters = generate_all_bbh_parameters(injection_parameters)
    # pop nuisance parameters that bilby adds for some reason
    for k in [
        'reference_frequency', 'waveform_approximant', 'minimum_frequency'
    ]:
        injection_parameters.pop(k)

    for key in [
        'mass_1_source',
        'mass_ratio',
        'redshift',
        'a_1',
        'a_2',
        'cos_tilt_1',
        'cos_tilt_2'
    ]:
        name = f'log_prior_{key}'
        if name not in injection_parameters.keys():
            injection_parameters[name] = priors[key].ln_prob(
                injection_parameters[key]
            )

    ln_prior = 0
    for key in [
        'mass_1_source', 'mass_ratio', 'redshift', 'a_1', 'a_2', 'cos_tilt_1',
        'cos_tilt_2'
    ]:
        ln_prior += injection_parameters[f'log_prior_{key}']

    injection_parameters['log_prior'] = ln_prior

    return injection_parameters


def draw_injection(priors, model, parameters):
    qmin = 0.10
    qs = np.linspace(qmin, 1, 500)

    injection_parameters = priors.sample()
    this_m1 = injection_parameters['mass_1_source']

    if model['mass_ratio'] == 'highpass_powerlaw':
        if 'mmin' in parameters.keys():
            mmin = parameters['mmin']
        elif 'mlow_1' in parameters.keys():
            mmin = parameters['mlow_1']
        else:
            raise ValueError(
                'Could not identify population model parameter'
                'corresponding to minimum mass for mass_ratio distribution'
            )

        if 'delta_m' in parameters.keys():
            delta_m = parameters['delta_m']
        elif 'delta_m_1' in parameters.keys():
            delta_m = parameters['delta_m_1']
        else:
            raise ValueError(
                'Could not identify population model parameter'
                'corresponding to highpass taper of mass_ratio distribution'
            )

        @jax.jit
        def calc_p_q(this_m1):
            return jnp.exp(PowerlawPlusPeak_MassRatio(
                dict(
                    mass_1=this_m1 * jnp.ones(jnp.shape(qs)),
                    mass_ratio=qs
                ),
                slope=parameters['beta'],
                minimum=mmin,
                delta_m=delta_m
            ))
        pq = calc_p_q(this_m1)
        q_prior = Interped(qs, pq, minimum=qmin, maximum=1, name='q')
        injection_parameters['mass_ratio'] = q_prior.sample()
        priors['mass_ratio'] = q_prior
    else:
        raise ValueError(f"Unknown mass_ratio model {model['mass_ratio']}")

    if model['cos_tilt'] == 'iso_gauss':
        xi_spin = parameters['xi_spin']
        mu_spin = parameters['mu_spin']
        sigma_spin = parameters['sigma_spin']

        cts = np.linspace(-1, 1, 500)

        u = bilby.core.utils.random.rng.uniform()
        if u < xi_spin:
            p_ct = truncnorm(cts, mu_spin, sigma_spin, 1, -1)
            ct_prior = Interped(cts, p_ct, minimum=-1, maximum=1)
            ct1 = ct_prior.sample()
            ct2 = ct_prior.sample()
        else:
            ct1 = bilby.core.utils.random.rng.uniform(low=-1, high=1)
            ct2 = bilby.core.utils.random.rng.uniform(low=-1, high=1)

        injection_parameters['cos_tilt_1'] = ct1
        injection_parameters['cos_tilt_2'] = ct2

        log_p_ct1 = np.log(
            marg_iso_gauss_spin_tilt(ct1, xi_spin, sigma_spin, mu_spin)
        )
        log_p_both = np.log(iso_gauss_spin_tilt(
            dict(
                cos_tilt_1=ct1,
                cos_tilt_2=ct2
            ),
            xi_spin,
            sigma_spin,
            mu_spin
        ))

        log_p_ct2_given_ct1 = log_p_both - log_p_ct1

        injection_parameters['log_prior_cos_tilt_1'] = log_p_ct1
        injection_parameters['log_prior_cos_tilt_2'] = log_p_ct2_given_ct1

    return fill_parameters(injection_parameters, priors)


def estimate_duration(injection_parameters, minimum_frequency):
    raw_duration = bilby.gw.utils.calculate_time_to_merger(
        minimum_frequency,
        injection_parameters['mass_1'],
        injection_parameters['mass_2']
    )
    duration = next_power_of_2(int(np.ceil(raw_duration)))
    if duration < 16:
        duration = 16
    return raw_duration, duration


def is_hopeless(intrange_net, injection_parameters, make_fast=True):
    """ determine if an injection is hopeless """
    hopeless = False
    mtot_src = injection_parameters['total_mass_source']
    if mtot_src < 160:
        try:
            max_z = float(intrange_net(mtot_src))
        except ValueError as e:
            # if the total mass is outside the interpolation range
            # we just go ahead and compute SNRs. not the most
            # efficient; TODO
            if (
                len(e.args) > 0
                and (
                    "x_new is below the interpolation range's minimum value"
                    in e.args[0]
                )
            ):
                max_z = np.inf
            else:
                raise e

        if injection_parameters['redshift'] > max_z and make_fast is True:
            hopeless = True

    return hopeless


def get_ifos(minimum_frequency, sampling_frequency):
    ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
    for ifo in ifos:
        ifo.minimum_frequency = minimum_frequency
        ifo.maximum_frequency = sampling_frequency / 2
        if ifo.name == 'V1':
            ifo.power_spectral_density = PowerSpectralDensity(
                asd_file='../sensitivity_curves/avirgo_O4high_NEW.txt'
            )
        else:
            ifo.power_spectral_density = PowerSpectralDensity(
                asd_file='../sensitivity_curves/aligo_O4low.txt'
            )
    return ifos


def get_waveform_generator(
    duration, sampling_frequency, minimum_frequency, waveform_approximant
):
    return bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=lal_binary_black_hole,
        waveform_arguments=dict(
            reference_frequency=20,
            minimum_frequency=minimum_frequency,
            waveform_approximant=waveform_approximant,
            PhenomXPrecVersion=104,
        ),
        parameter_conversion=convert_to_lal_binary_black_hole_parameters
    )


def main(
    number,
    outdir,
    opt_net_snr_thre,
    intrange_net,
    sampleprior=False,
    zero_noise=False,
    seed=21,
    make_fast=True,
    model=dict(),
    commit=None,
    parameters=None,
    injection_waveform_approximant='IMRPhenomXPHM',
    recovery_waveform_approximant='IMRPhenomXP',
    **kwargs
):
    parameters = get_parameters(path=parameters, outdir=outdir, **kwargs)

    default_model = dict(
        mass_1_source='highpass_broken_powerlaw_two_peaks',
        mass_ratio='highpass_powerlaw',
        redshift='powerlaw',
        a_1='iid_truncnorm',
        a_2='iid_truncnorm',
        cos_tilt='uniform',
    )
    default_model.update(model)

    with open(f'{outdir}/model.json', 'w') as f:
        f.write(json.dumps(default_model, indent=4, sort_keys=False))

    priors = get_inj_priors(default_model, parameters)

    inj_list = []
    all_inj_list = []
    i = 0

    minimum_frequency = 20
    sampling_frequency = 2048

    pbar = tqdm(total=number)
    while i < number:
        injection_parameters = draw_injection(
            deepcopy(priors),
            default_model,
            parameters
        )
        raw_duration, duration = estimate_duration(
            injection_parameters,
            minimum_frequency
        )

        data_seed = bilby.core.utils.random.rng.integers(
            low=0, high=1e17 + seed, dtype=np.int64
        )
        start_time = injection_parameters['geocent_time'] + 2 - duration

        injection_parameters['data_seed'] = data_seed
        injection_parameters['raw_duration'] = raw_duration
        injection_parameters['duration'] = duration
        injection_parameters['start_time'] = start_time

        if sampleprior:
            injection_parameters['network_matched_filter_snr'] = 0
            injection_parameters['network_optimal_snr'] = 0
            i += 1
            pbar.update(1)
            all_inj_list.append(injection_parameters)
            continue

        hopeless = is_hopeless(
            intrange_net,
            injection_parameters,
            make_fast=make_fast
        )

        if hopeless:
            injection_parameters['injection_network_optimal_snr'] = 0
            injection_parameters['injection_network_matched_filter_snr'] = 0
            injection_parameters['network_optimal_snr'] = 0
            injection_parameters['network_matched_filter_snr'] = 0
            all_inj_list.append(injection_parameters)
            continue

        ifos = get_ifos(minimum_frequency, sampling_frequency)
        if zero_noise:
            ifos.set_strain_data_from_zero_noise(
                sampling_frequency=sampling_frequency,
                duration=duration,
                start_time=start_time
            )
        else:
            bilby.core.utils.random.seed(data_seed)
            ifos.set_strain_data_from_power_spectral_densities(
                sampling_frequency=sampling_frequency,
                duration=duration,
                start_time=start_time
            )

        inj_wavform_generator = get_waveform_generator(
            duration,
            sampling_frequency,
            minimum_frequency,
            injection_waveform_approximant
        )

        injection_parameters_without_m1_m2_src = deepcopy(
            injection_parameters
        )
        injection_parameters_without_m1_m2_src.pop('mass_1_source')
        injection_parameters_without_m1_m2_src.pop('mass_2_source')

        try:
            ifos.inject_signal(
                parameters=injection_parameters_without_m1_m2_src,
                waveform_generator=inj_wavform_generator
            )
        except IndexError as e:
            print(injection_parameters_without_m1_m2_src)
            raise e

        rho_opt_2 = 0
        rho_mf_2 = 0

        for ifo in ifos:
            rho_opt_2 += ifo.meta_data['optimal_SNR']**2
            rho_mf_2 += np.abs(ifo.meta_data['matched_filter_SNR'])**2

        injection_parameters['injection_network_optimal_snr'] = np.sqrt(
            rho_opt_2
        )
        injection_parameters['injection_network_matched_filter_snr'] = np.sqrt(
            rho_mf_2
        )

        if recovery_waveform_approximant != injection_waveform_approximant:
            injection_parameters_without_m1_m2_src = deepcopy(
                injection_parameters
            )
            injection_parameters_without_m1_m2_src.pop('mass_1_source')
            injection_parameters_without_m1_m2_src.pop('mass_2_source')

            rec_wavform_generator = get_waveform_generator(
                duration,
                sampling_frequency,
                minimum_frequency,
                recovery_waveform_approximant
            )

            rec_polarizations = rec_wavform_generator.frequency_domain_strain(
                injection_parameters_without_m1_m2_src
            )

            rho_opt_2 = 0
            rho_mf_2 = 0

            for ifo in ifos:
                signal_ifo = ifo.get_detector_response(
                    rec_polarizations, injection_parameters_without_m1_m2_src
                )
                rho_opt_2 += ifo.optimal_snr_squared(signal=signal_ifo).real
                rho_mf_2 += np.abs(
                    ifo.matched_filter_snr(signal=signal_ifo)
                )**2

        injection_parameters['network_optimal_snr'] = np.sqrt(
            rho_opt_2
        )
        injection_parameters['network_matched_filter_snr'] = np.sqrt(
            rho_mf_2
        )

        if zero_noise:
            if np.sqrt(rho_opt_2) >= opt_net_snr_thre:
                inj_list.append(injection_parameters)
                i += 1
                pbar.update(1)
        else:
            if np.sqrt(rho_mf_2) >= opt_net_snr_thre:
                inj_list.append(injection_parameters)
                i += 1
                pbar.update(1)

        all_inj_list.append(injection_parameters)

    total_generated = int(len(all_inj_list))
    all_injs = list_to_dict(all_inj_list)
    det_injs = list_to_dict(inj_list)

    if sampleprior is False:
        write_hdf5(
            f'{outdir}/detectable.hdf5',
            det_injs,
            total_generated,
            commit=commit
        )
        write_hdf5(
            f'{outdir}/all.hdf5',
            all_injs,
            total_generated,
            commit=commit
        )
    else:
        write_hdf5(
            f'{outdir}/prior.hdf5',
            all_injs,
            total_generated,
            commit=commit
        )


def concat(outdir, load_all=False):
    outdir = os.path.abspath(outdir)
    dirs = sorted([
        f'{outdir}/{d}'
        for d in os.listdir(outdir)
        if (
            os.path.isdir(f'{outdir}/{d}')
            and os.path.isfile(f'{outdir}/{d}/detectable.hdf5')
        )
    ])

    def load(dir, name):
        data = h5ify.load(f'{dir}/{name}.hdf5')

        extras = dict()

        for p in ['parameters', 'model']:
            with open(f'{dir}/{p}.json', 'r') as f:
                dic = json.loads(f.read())
                extras[p] = dic

        attrs = data.pop('attrs')
        total = attrs.pop('total_generated')

        with open(f'{dir}/config.json', 'r') as f:
            attrs.update(json.loads(f.read()))

        # pop keys we dont care to track
        # and will break our assertion that all keys in attrs
        # are the same (or enough so) between different vt files
        attrs.pop('seed')
        attrs.pop('ninj')
        attrs.pop('outdir')

        for k in list(attrs.keys()):
            if isinstance(attrs[k], dict):
                attrs.pop(k)

        extras['attrs'] = attrs

        return extras, total, data

    if load_all:
        extras, total_generated, detectable = load(outdir, 'detectable')

        with h5py.File(f'{outdir}/all.hdf5', 'w') as f:
            dsets = dict()
            for c in detectable.keys():
                dsets[c] = f.create_dataset(
                    c,
                    (total_generated,),
                    dtype=np.int64 if c == 'data_seed' else np.float64,
                    chunks=True
                )

            j = 0
            for dir in tqdm(dirs):
                _, _, data = load(dir, 'all')
                n = len(data[c])
                for c in detectable.keys():
                    dsets[c][j : j + n] = data[c]
                j += n

            for k, v in extras['attrs'].items():
                f.attrs[k] = v

            for k, v in extras.items():
                if k != 'attrs':
                    grp = f.create_group('k')
                    for s in v.keys():
                        grp.create_dataset(s, data=v[s])
    else:
        extras, total_generated, detectable = load(
            dirs[0], 'detectable'
        )

        for dir in tqdm(dirs[1:]):
            try:
                e, total, data = load(dir, 'detectable')
                assert e == extras
                total_generated += total
                detectable = concat_dicts(detectable, data)
            except AssertionError as err:
                print(f'Assertion error {err} with {dir}')
                continue

        detectable['total_generated'] = total_generated
        for k, v in extras.items():
            if k != 'attrs':
                detectable[k] = v

        detectable['mass_1_source'] = (
            detectable['mass_1'] / (1 + detectable['redshift'])
        )
        h5ify.save(
            f'{outdir}/detectable.hdf5',
            dict(**detectable, attrs=extras['attrs']),
            mode='w'
        )


if __name__ == '__main__':
    commit_hash = get_git_revision_short_hash()
    print(f'using: {commit_hash}')

    (
        outdir,
        ninj,
        sampleprior,
        snr_threshold,
        zero_noise,
        seed,
        model,
        parameters,
        kwargs,
        injection_waveform_approximant,
        recovery_waveform_approximant,
        interp_net_path,
        make_fast,
        overwrite
    ) = parse_args()

    print(f'will save injections to {outdir}')

    os.makedirs(outdir, exist_ok=True)
    det_path = f'{outdir}/detectable.hdf5'

    if not overwrite and os.path.isfile(det_path):
        print(f'{det_path} already exists; skipping!')
        pass
    else:
        with open(interp_net_path, 'rb') as f:
            intrange_net = pickle.load(f)

        bilby.core.utils.random.seed(seed)

        main(
            number=ninj,
            outdir=outdir,
            opt_net_snr_thre=snr_threshold,
            intrange_net=intrange_net,
            sampleprior=sampleprior,
            zero_noise=zero_noise,
            seed=seed,
            make_fast=make_fast,
            model=model,
            commit=commit_hash,
            parameters=parameters,
            injection_waveform_approximant=injection_waveform_approximant,
            recovery_waveform_approximant=recovery_waveform_approximant, 
            **kwargs
        )

    print('done.')
