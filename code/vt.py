import os
import pickle
import argparse
import configparser
from ast import literal_eval
from copy import deepcopy

import h5py
import numpy as np
from tqdm import tqdm

import bilby
bilby.core.utils.setup_logger(log_level='WARNING')

from bilby.gw.detector import PowerSpectralDensity
from bilby.core.prior import Uniform, Sine, Cosine, Interped
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters

import gwpopulation
gwpopulation.set_backend('numpy')

from gwpopulation.models.redshift import PowerLawRedshift
from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution

from models import truncnorm

from util import write_config
from util import next_power_of_2


def write_hdf5(path, df, total_generated, model, commit=None):
    # clean data types in df
    for key in df.keys():
        if df.dtypes[key] == object:
            df[key] = [v.item() for v in df[key].values]

    with h5py.File(path, 'w') as f:
        f.attrs['total_generated'] = total_generated
        f.attrs['model'] = model
        if commit is not None:
            f.attrs['commit'] = commit

        for key in df.keys():
            dtype = np.int64 if key == 'data_seed' else np.float64
            print(key)
            print(df[key].values)
            f.create_dataset(key, data=df[key].values, dtype=dtype)


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


def get_inj_priors(name, **kwargs):
    inj_priors = bilby.gw.prior.BBHPriorDict(dict(
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
    ))

    if name == 'gwtc4_bpl2p_plz_trunc-norm-spin_unif-tilts':
        ct_low = kwargs.get('cos_tilt_min', -1)
        ct_high = kwargs.get('cos_tilt_max', 1)

        inj_priors['cos_tilt_1'] = Uniform(ct_low, ct_high, name='cos_tilt_1')
        inj_priors['cos_tilt_2'] = Uniform(ct_low, ct_high, name='cos_tilt_2')

        mu_chi = # TODO
        sigma_chi = # TODO
        mags = np.linspace(0, 1, 500)
        pchi = truncnorm(mags, mu_chi, sigma_chi, high=1, low=0)

        chi_prior = Interped(
            mags, pchi, minimum=min(mags), maximum=max(mags), name='chi'
        )

        inj_priors['a_1'] = chi_prior
        inj_priors['a_2'] = chi_prior

        z_model = PowerLawRedshift(z_max=# TODO)
        zs = z_model.zs
        z_lambda = # TODO
        pz = z_model.probability(dict(redshift=zs), lamb=z_lambda)
        z_prior = Interped(
            zs, np.array(pz), minimum=min(zs), maximum=max(zs), name='redshift'
        )
        inj_priors['redshift'] = z_prior

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
    model='gwtc4_bpl2p_plz_trunc-norm-spin_unif-tilts',
    **kwargs    
):

    mass_model = SinglePeakSmoothedMassDistribution(mmin=1, mmax=200)

    mass_hyperparameters = dict(
        alpha=3.4,
        mmin=5,
        mmax=87,
        lam=0.04,
        mpp=34,
        sigpp=3.6,
        delta_m=4.8,
        beta=1.1
    )
    mass_hyperparameters.update(mass_kwargs)
    print(f'using pl+p mass model hyperparameters: {mass_hyperparameters}')

    m1s = mass_model.m1s
    pm1 = mass_model.p_m1(
        dict(mass_1=m1s),
        **{
            k : v for k, v in mass_hyperparameters.items()
            if k in ['alpha', 'mmin', 'mmax', 'lam', 'mpp', 'sigpp', 'delta_m']
        }
    )
    m1_prior = Interped(
        m1s, pm1, minimum=min(m1s), maximum=max(m1s), name='m1_source'
    )
    inj_priors['mass_1_source'] = m1_prior

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
        tc = 1126259642.413 + 128 * i

        this_z = z_prior.sample()
        this_m1 = m1_prior.sample()

        pq = mass_model.p_q(
            dict(
                mass_1=this_m1 * np.ones(np.shape(qs)),
                mass_ratio=qs
            ),
            **{
                k : v for k, v in mass_hyperparameters.items()
                if k in ['beta', 'mmin', 'delta_m']
            }
        )
        q_prior = Interped(qs, pq, minimum=qmin, maximum=1, name='q')

        this_q = q_prior.sample()
        inj_priors['mass_ratio'] = q_prior

        # TODO: best to put this first! then fill in the missing parameters..
        # still need to add q, cos tilt, etc. priors to inj_priors
        # so we can evaluate probabilities later
        injection_parameters = inj_priors.sample()
        injection_parameters['geocent_time'] = tc

        # TODO: this is all becoming spaghetti
        if model == 'plp_plz_beta-spin_isogauss-tilt-condq':
            injection_parameters['cos_tilt_1'] = ct1
            injection_parameters['cos_tilt_2'] = ct2

        injection_parameters['redshift'] = this_z
        injection_parameters['mass_1_source'] = this_m1
        injection_parameters['mass_1'] = this_m1 * (1 + this_z)
        injection_parameters['mass_ratio'] = this_q
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
                waveform_approximant='IMRPhenomXP',
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

    if sampleprior is False:
        total_generated = int(len(all_inj_list))
        write_hdf5(
            f'{outdir}/detectable.hdf5',
            pd.DataFrame(inj_list),
            total_generated,
            model,
            commit=commit
        )
        write_hdf5(
            f'{outdir}/allinjs.hdf5',
            pd.DataFrame(all_inj_list),
            total_generated,
            model,
            commit=commit
        )
    else:
        # TODO: change to hdf5 as well
        inj_df = pd.DataFrame(all_inj_list)
        inj_df.to_csv(
            os.path.join(outdir, 'priordraws.dat'),
            index=False,
            sep='\t'
        )
    print('done.')


def read_config(args):
    config = configparser.ConfigParser()
    config.read(args.config)
    arrayid = args.arrayid

    outdir = config.get('job', 'outdir')

    ninj = config.getint('options', 'ninj', fallback=1000)
    sampleprior = config.getboolean('options', 'sampleprior', fallback=False)
    snr_threshold = config.getfloat('options', 'snr-threshold', fallback=11)
    add_noise = config.getboolean('options', 'add-noise', fallback=True)

    seed = literal_eval(config.get('seed', 'seed'))
    if isinstance(seed, tuple):
        seed = range(*seed)[arrayid]
    elif isinstance(seed, list):
        seed = seed[arrayid]

    model = config.get(
        'draw-population',
        'model',
        fallback='plp_plz_unif-spins_iso-tilts'
    )
    mass_kwargs = literal_eval(
        config.get('draw-population', 'mass-kwargs', fallback='{}')
    )

    return (
        outdir, ninj, sampleprior, snr_threshold, add_noise, seed,
        model, mass_kwargs
    )


def concat(outdir, all=False):
    outdir = os.path.abspath(outdir)
    dirs = [
        f'{outdir}/{d}'
        for d in os.listdir(outdir)
        if os.path.isdir(f'{outdir}/{d}')
    ]

    if all:
        outpath = f'{outdir}/all.hdf5'
        files = []
        for i in range(len(dirs)):
            fname = f'{dirs[i]}/allinjs'
            if os.path.isfile(f'{fname}.dat'):
                files.append(f'{fname}.dat')
            elif os.path.isfile(f'{fname}.hdf5'):
                files.append(f'{fname}.hdf5')
            else:
                print(f'No file found in {dirs[i]}')

        detectable = pd.read_hdf(
            f'{outdir}/detectable.hdf5',
            'detectable'
        )
        total_generated = detectable['total_generated'].iloc[0]
        columns = [c for c in detectable.columns if c != 'total_generated']

        def load(file):
            ext = os.path.splitext(file)[-1]
            if ext == '.dat':
                data = pd.read_csv(file, sep='\t')
                total = len(data)
            elif ext == '.hdf5':
                total, data = load_hdf5_as_dict(file)
            else:
                raise NotImplementedError(f'Cant load {file}')
            return total, data

        with h5py.File(outpath, 'w') as f:
            grp = f.create_group('all')
            dsets = dict()
            for c in columns:
                dsets[c] = grp.create_dataset(
                    c,
                    (total_generated,),
                    dtype=np.int64 if c == 'data_seed' else np.float64,
                    chunks=True
                )

            j = 0
            for file in tqdm(files):
                n, data = load(file)
                for c in columns:
                    dsets[c][j : j + len(data[c])] = data[c]
                j += n
    else:
        files = []
        for i in range(len(dirs)):
            fname = f'{dirs[i]}/detectable'
            if os.path.isfile(f'{fname}.dat'):
                files.append(f'{fname}.dat')
            elif os.path.isfile(f'{fname}.hdf5'):
                files.append(f'{fname}.hdf5')
            else:
                print(f'No file found in {dirs[i]}')
        # TODO: this could be much more efficient
        # namely, not concat all the pandas df into one big thing in memory

        def load(file):
            ext = os.path.splitext(file)[-1]
            if ext == '.dat':
                data = pd.read_csv(file, sep='\t')
                total = data['total_generated'].iloc[0]
            elif ext == '.hdf5':
                total, data = load_hdf5_as_df(file)
            else:
                raise NotImplementedError(f'Cant load {file}')
            return total, data

        total_generated, detectable = load(files[0])

        for file in tqdm(files[1:]):
            total, data = load(file)
            total_generated += total
            detectable = pd.concat((detectable, data), ignore_index=True)

        # TODO: ultimately, probably best not to save with pandas
        # pytable format, and instead work with my own hdf5 file
        # e.g. for storing meta data like the model, commit, total_generated
        detectable['total_generated'] = total_generated
        detectable['mass_1_source'] = (
            detectable['mass_1'] / (1 + detectable['redshift'])
        )
        detectable.to_hdf(
            f'{outdir}/detectable.hdf5',
            mode='w',
            key='detectable'
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


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument('--arrayid', type=int, default=0)

if __name__ == '__main__':
    commit_hash = get_git_revision_short_hash()
    print(f'using: {commit_hash}')

    args = parser.parse_args()
    (
        outdir, ninj, sampleprior, snr_threshold, add_noise, seed,
        model, mass_kwargs
    ) = read_config(args)

    outdir = f'{outdir}/{seed}'
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
            mass_kwargs=mass_kwargs,
            make_fast=True,
            model=model,
            commit=commit_hash
        )