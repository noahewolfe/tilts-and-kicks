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

from pixelpop.models.gwpop_models import PowerlawPlusPeak_MassRatio
from pixelpop.models.gwpop_models import BrokenPowerlawPlusTwoPeaks_PrimaryMass

from models import truncnorm
from models import log_powerlaw_redshift

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
parser.add_argument('--seed', type=int)
parser.add_argument('--model', type=str, default='o4a-strong-unif-tilts')
parser.add_argument('--parameters', default=None)
parser.add_argument('--extra-kwargs', type=json.loads, default='{}')


def parse_args():
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    write_config(args)
    add_noise = not args.zero_noise
    return (
        args.outdir,
        args.ninj,
        args.sample_prior,
        args.snr_threshold,
        add_noise,
        args.seed,
        args.model,
        args.parameters,
        args.extra_kwargs
    )


def write_hdf5(path, dic, total_generated, model, commit=None):
    # clean data types in df
    #for key in dic.keys():
    #    if df.dtypes[key] == object:
    #        df[key] = [v.item() for v in df[key].values]

    with h5py.File(path, 'w') as f:
        f.attrs['total_generated'] = total_generated
        f.attrs['model'] = model
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


def load_hdf5_as_df(path):
    total_generated, data = load_hdf5_as_dict(path)
    return total_generated, pd.DataFrame(data)


def get_parameters(name, path=None, outdir=None, **kwargs):
    if name == 'o4a-strong-unif-tilts':
        if path is None:
            path = os.path.abspath('./parameters/o4a-strong-maxl.json')

        print(
            'Loading mass, redshift and spin magnitude hyperparameters from'
            f'{path}...'
        )
        with open(path, 'r') as f:
            parameters = json.loads(f.read())

        for key in list(parameters.keys()):
            if key in ['mu_spin', 'sigma_spin', 'xi_spin']:
                parameters.pop(key)

        if 'lam_2' not in parameters.keys():
            parameters['lam_2'] = np.round(1 - parameters['lam_0'] - parameters['lam_1'], decimals=2)

        assert (parameters['lam_0'] + parameters['lam_1'] + parameters['lam_2']) == 1.0

        parameters.update(kwargs)

        if outdir is not None:
            with open(f'{outdir}/parameters.json', 'w') as f:
                f.write(json.dumps(parameters, indent=4, sort_keys=False))

        return parameters


def get_inj_priors(name, parameters):
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

    if name == 'o4a-strong-unif-tilts':
        ct_low, ct_high = parameters['cos_tilt_min'], parameters['cos_tilt_max']

        inj_priors['cos_tilt_1'] = Uniform(ct_low, ct_high, name='cos_tilt_1')
        inj_priors['cos_tilt_2'] = Uniform(ct_low, ct_high, name='cos_tilt_2')

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

        m1s = np.linspace(3, 300, 1_000)
        p_m1 = np.exp(BrokenPowerlawPlusTwoPeaks_PrimaryMass(
            dict(mass_1=m1s),
            alpha_1=parameters['alpha_1'],
            alpha_2=parameters['alpha_2'],
            mmin=parameters['mmin'],
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
        raise NotImplementedError(f'Model {name} not implemented!')

    return inj_priors


def main(
    number,
    outdir,
    opt_net_snr_thre,
    intrange_net,
    sampleprior=False,
    add_noise=False,
    seed=21,
    make_fast=True,
    model='o4a-strong-unif-tilts',
    commit=None,
    parameters=None,
    **kwargs
):
    parameters = get_parameters(
        model, path=parameters, outdir=outdir, **kwargs
    )
    priors = get_inj_priors(model, parameters)

    qmin = 0.10
    qs = np.linspace(qmin, 1, 500)

    inj_list = []
    all_inj_list = []
    i = 0

    flow = 20
    sampling_frequency = 2048
    duration = 16
    det_duration = 16

    pbar = tqdm(total=number)
    while i < number:
        inj_priors = deepcopy(priors)
        injection_parameters = inj_priors.sample()
        this_z = injection_parameters['redshift']
        this_m1 = injection_parameters['mass_1_source']

        if model == 'o4a-strong-unif-tilts':
            @jax.jit
            def calc_p_q(this_m1):
                return jnp.exp(PowerlawPlusPeak_MassRatio(
                    dict(
                        mass_1=this_m1 * jnp.ones(jnp.shape(qs)),
                        mass_ratio=qs
                    ),
                    slope=parameters['beta'],
                    minimum=parameters['mmin'],
                    delta_m=parameters['delta_m_1']
                ))
            pq = calc_p_q(this_m1) 
            q_prior = Interped(qs, pq, minimum=qmin, maximum=1, name='q')
            injection_parameters['mass_ratio'] = q_prior.sample()
            inj_priors['mass_ratio'] = q_prior

        injection_parameters['mass_1'] = this_m1 * (1 + this_z)
        injection_parameters['mass_2'] = (
            injection_parameters['mass_1']
            * injection_parameters['mass_ratio']
        )

        raw_duration = bilby.gw.utils.calculate_time_to_merger(
            flow,
            injection_parameters['mass_1'],
            injection_parameters['mass_2']
        )
        det_duration = duration = next_power_of_2(int(np.ceil(raw_duration)))
        if duration < 16:
            det_duration = duration = 16

        waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=lal_binary_black_hole,
            waveform_arguments=dict(
                reference_frequency=20,
                minimum_frequency=flow,
                waveform_approximant='IMRPhenomXPHM',
                PhenomXPrecVersion=104,
            ),
            parameter_conversion=convert_to_lal_binary_black_hole_parameters
        )

        mprod = injection_parameters['mass_1'] * injection_parameters['mass_2']
        mtot = injection_parameters['mass_1'] + injection_parameters['mass_2']
        injection_parameters['chirp_mass'] = mprod**(3 / 5) / mtot**(1 / 5)
        injection_parameters['mass_2_source'] = (
            injection_parameters['mass_2'] / (1 + this_z)
        )

        for key in ['tilt_1', 'tilt_2']:
            injection_parameters[key] = np.arccos(
                injection_parameters[f'cos_{key}']
            )

        injection_parameters['prior_m1s'] = inj_priors['mass_1_source'].prob(
            injection_parameters['mass_1_source']
        )
        for key in [
            'mass_ratio', 'redshift', 'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2'
        ]:
            injection_parameters[f'prior_{key}'] = inj_priors[key].prob(
                injection_parameters[key]
            )

        prior = 1
        ln_prior = 0
        for key in [
            'm1s', 'mass_ratio', 'redshift', 'a_1', 'a_2', 'cos_tilt_1',
            'cos_tilt_2'
        ]:
            prior *= injection_parameters[f'prior_{key}']
            ln_prior += np.log(injection_parameters[f'prior_{key}'])

        injection_parameters['prior'] = prior
        injection_parameters['ln_prior'] = ln_prior

        mtot_source = (
            injection_parameters['mass_1_source']
            + injection_parameters['mass_2_source']
        )
        hopeless = False
        if mtot_source < 160:
            try:
                max_z = float(intrange_net(mtot_source))
            except ValueError as e:
                # if the total mass is outside the interpolation range
                # we just go ahead and compute SNRs. not the most
                # efficient; TODO
                if (
                    len(e.args) > 0
                    and "x_new is below the interpolation range's minimum value" in e.args[0]
                ):
                    max_z = np.inf
                else:
                    raise e

            if injection_parameters['redshift'] > max_z and make_fast is True:
                hopeless = True

        zero_noise = not add_noise

        select_optimal = zero_noise
        data_seed = np.random.randint(1e17 + seed)
        start_time = injection_parameters['geocent_time'] + 2 - det_duration

        ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
        for ifo in ifos:
            ifo.minimum_frequency = flow
            ifo.maximum_frequency = sampling_frequency / 2
            if ifo.name == 'V1':
                ifo.power_spectral_density = PowerSpectralDensity(
                    asd_file='../sensitivity_curves/avirgo_O4high_NEW.txt'
                )
            else:
                ifo.power_spectral_density = PowerSpectralDensity(
                    asd_file='../sensitivity_curves/aligo_O4low.txt'
                )

        if sampleprior is False:
            if hopeless is True:
                injection_parameters['data_seed'] = data_seed
                injection_parameters['network_optimal_snr'] = 0
                injection_parameters['network_matched_filter_snr'] = 0
            elif zero_noise:
                ifos.set_strain_data_from_zero_noise(
                    sampling_frequency=sampling_frequency,
                    duration=det_duration,
                    start_time=start_time
                )
            else:
                bilby.core.utils.random.seed(data_seed)
                ifos.set_strain_data_from_power_spectral_densities(
                    sampling_frequency=sampling_frequency,
                    duration=det_duration,
                    start_time=start_time
                )

            if hopeless is False:
                injection_parameters_without_m1_m2_src = deepcopy(
                    injection_parameters
                )
                injection_parameters_without_m1_m2_src.pop('mass_1_source')
                injection_parameters_without_m1_m2_src.pop('mass_2_source')

                try:
                    ifos.inject_signal(
                        parameters=injection_parameters_without_m1_m2_src,
                        waveform_generator=waveform_generator
                    )
                except IndexError as e:
                    print(injection_parameters_without_m1_m2_src)
                    raise e

                rho_opt_2 = 0
                rho_mf_2 = 0

                for ifo in ifos:
                    rho_opt_2 += ifo.meta_data['optimal_SNR']**2
                    rho_mf_2 += np.abs(ifo.meta_data['matched_filter_SNR'])**2

                injection_parameters['data_seed'] = data_seed
                injection_parameters['network_optimal_snr'] = np.sqrt(
                    rho_opt_2
                )
                injection_parameters['network_matched_filter_snr'] = np.sqrt(
                    rho_mf_2
                )

                if select_optimal:
                    if np.sqrt(rho_opt_2) >= opt_net_snr_thre:
                        inj_list.append(injection_parameters)
                        i += 1
                        pbar.update(1)
                else:
                    if np.sqrt(rho_mf_2) >= opt_net_snr_thre:
                        inj_list.append(injection_parameters)
                        i += 1
                        pbar.update(1)
        else:
            injection_parameters['data_seed'] = data_seed
            injection_parameters['network_matched_filter_snr'] = 0
            injection_parameters['network_optimal_snr'] = 0
            i += 1
            pbar.update(1)

        injection_parameters['raw_duration'] = raw_duration
        injection_parameters['duration'] = duration
        injection_parameters['start_time'] = start_time

        all_inj_list.append(injection_parameters)
        del waveform_generator

    total_generated = int(len(all_inj_list))
    all_injs = list_to_dict(all_inj_list)
    det_injs = list_to_dict(inj_list)

    if sampleprior is False:
        write_hdf5(
            f'{outdir}/detectable.hdf5',
            det_injs,
            total_generated,
            model,
            commit=commit
        )
        write_hdf5(
            f'{outdir}/all.hdf5',
            all_injs,
            total_generated,
            model,
            commit=commit
        )
    else:
        write_hdf5(
            f'{outdir}/prior.hdf5',
            all_injs,
            total_generated,
            model,
            commit=commit
        )

    print('done.')


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
        with open(f'{dir}/parameters.json', 'r') as f:
            parameters = json.loads(f.read())
        data = h5ify.load(f'{dir}/{name}.hdf5')
        attrs = data.pop('attrs')
        total = attrs.pop('total_generated')
        return parameters, attrs, total, data

    if load_all:
        detectable = h5ify.load(f'{outdir}/detectable.hdf5')
        total_generated = detectable.pop('total_generated')
        parameters = detectable.pop('parameters')
        attrs = detectable.pop('attrs')

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
                _, _, _, data = load(dir, 'all')
                n = len(data[c])
                for c in detectable.keys():
                    dsets[c][j : j + n] = data[c]
                j += n

            for k, v in attrs.items():
                f.attrs[k] = v

            grp = f.create_group('parameters')
            for k in parameters.keys():
                grp.create_dataset(k, data=parameters[k])
    else:
        parameters, attrs, total_generated, detectable = load(
            dirs[0], 'detectable'
        )

        for dir in tqdm(dirs[1:]):
            _p, _a, total, data = load(dir, 'detectable')
            assert _p == parameters
            assert _a == attrs
            total_generated += total
            detectable = concat_dicts(detectable, data)

        detectable['total_generated'] = total_generated
        detectable['parameters'] = parameters
        detectable['mass_1_source'] = (
            detectable['mass_1'] / (1 + detectable['redshift'])
        )
        h5ify.save(
            f'{outdir}/detectable.hdf5',
            dict(**detectable, attrs=attrs),
            mode='w'
        )


def mix(models, injections):
    # TODO: under construction

    # TODO: decide on what type the injections should be
    total_generated = sum([
        (
            i['total_generated'][0]
            if isinstance(i['total_generated'], np.ndarray)
            else i['total_generated']
        )
        for i in injections
    ])

    new_injections = dict()
    #new_injections['total_generated'] = 


if __name__ == '__main__':
    commit_hash = get_git_revision_short_hash()
    print(f'using: {commit_hash}')

    (
        outdir,
        ninj,
        sampleprior,
        snr_threshold,
        add_noise,
        seed,
        model,
        parameters,
        kwargs
    ) = parse_args()

    print(f'will save injections to {outdir}')
    os.makedirs(outdir, exist_ok=True)

    np.random.seed(seed)
    bilby.core.utils.random.seed(seed)

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    if os.path.isfile(os.path.join(outdir, 'allinjs.dat')):
        pass
    else:
        with open('../data/interp_net.pkl', 'rb') as f:
            intrange_net = pickle.load(f)

        main(
            number=ninj,
            outdir=outdir,
            opt_net_snr_thre=snr_threshold,
            intrange_net=intrange_net,
            sampleprior=sampleprior,
            add_noise=add_noise,
            seed=seed,
            make_fast=True,
            model=model,
            commit=commit_hash,
            parameters=parameters,
            **kwargs
        )
