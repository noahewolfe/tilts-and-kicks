import os
import re
from glob import glob

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from bilby.gw.prior import UniformSourceFrame

import h5ify

default_pars = [
    'mass_1_source',
    'mass_ratio',
    'redshift',
    'a_1',
    'a_2',
    'cos_tilt_1',
    'cos_tilt_2'
]

default_exclude = exclude = [
    'GW170817',  # BNS!
    'GW190425',  # BNS
    'GW190814',  # ???
    'GW190917',  # NSBH
    'GW200105',  # NSBH
    'GW200115',  # NSBH 
    'GW191219',  # NSBH
    'S230529ay',
    'S230518h',        
    'S190425z',
    'S200105ae',
    'S200115j',
    'S190426c',
    'S190814bv',
    'S190917u',
]


def get_datadir(datadir, catalog, pars):
    path = f"{datadir}/{catalog}-{'-'.join(pars)}"
    os.makedirs(path, exist_ok=True)
    return path


def load_and_reduce_pe(path, pars):
    # TODO: switches for prior choices?
    with h5py.File(path, 'r') as f:
        # GWTC-3
        if 'C01:Mixed' in f:
            data = f['C01:Mixed']['posterior_samples']
        # O4a
        elif 'C00:NRSur7dq4' in f:
            data = f['C00:NRSur7dq4']['posterior_samples']
        elif 'C00:Mixed' in f:
            data = f['C00:Mixed']['posterior_samples']
        # O4b
        else:
            keys = list(set(f) - {'history', 'version'})
            assert len(keys) == 1
            data = f[keys[0]]['posterior_samples']

        posterior = {par: data[par][:] for par in pars}
        posterior['prior'] = UniformSourceFrame(
            minimum = posterior['redshift'].min(),
            maximum = posterior['redshift'].max(),
            cosmology = 'Planck15_LAL',
            name = 'redshift',
        ).prob(posterior['redshift']) * (1 + posterior['redshift'])**2

    return posterior


def resample_and_reshape_posteriors(posteriors, seed=1):
    min_npe_samples = min([len(p['prior']) for p in posteriors])
    rng = np.random.default_rng(seed)

    for i, p in enumerate(posteriors):
        idxs = rng.choice(len(p['prior']), min_npe_samples, replace=False)
        posteriors[i] = {k : p[k][idxs] for k in p.keys()}

    return {
        k : np.stack([p[k] for p in posteriors])
        for k in posteriors[0].keys()
    }


def get_posteriors(
    pars = default_pars,
    catalog = 'GWTC-4',
    exclude = default_exclude,
    load = False,
    save = False,
    xp = np,
    datadir = '../data',
    seed = 1
):
    datadir = get_datadir(datadir, catalog, pars)
    datapath = f'{datadir}/posteriors.h5'

    if load and os.path.exists(datapath):
        data = h5ify.load(datapath)
        posteriors = data['posteriors']
        events = data['events']
        for par in posteriors:
            posteriors[par] = xp.array(posteriors[par])
        events = list(map(str, np.array(events).astype(str)))

        for event in events:
            if event in exclude:
                raise ValueError(
                    f'An excluded event ({event}) is included in the saved posteriors file!'
                )

        return posteriors, events

    confident = np.loadtxt('./events.txt', dtype = str, skiprows = 1)[:, 1]
    events = [
        str(event)
        for event in confident
        if len([ex for ex in exclude if ex in event]) == 0
    ]
    files = sorted([
        glob(f'/home/rp.o4/catalogs/GWTC-*/data-release/*{event}*_cosmo.h5')[0]
        for event in events
    ])

    if catalog in ['GWTC-4', 'GWTC-5']:
        files4a = sorted(glob(
            '/home/rp.o4/catalogs/GWTC-4/GWTC4-Stable_Release-1/4c4fd2cef_717/'
            'bbh_only/*.hdf5',
        ))
        files += files4a
        events += [
            'GW' + file.split('/')[-1].split('GW')[-1][:13] for file in files4a
        ]

    if catalog == 'GWTC-5':
        files4b = sorted(glob(
            '/home/rp.o4/catalogs/O4b_prelim/O4bPreliminaryPE20250404/*.h5',
        ))
        files += files4b
        events += [file.split('/')[-1].split('-')[0] for file in files4b]

    for event in exclude:
        if event in events:
            files.pop(events.index(event))
            events.remove(event)

    posteriors = [load_and_reduce_pe(path, pars) for path in tqdm(files)]
    posteriors = resample_and_reshape_posteriors(posteriors, seed)

    if 'mass_ratio' in pars:
        posteriors['prior'] *= posteriors['mass_1_source']

    if save:
        h5ify.save(
            datapath,
            dict(posteriors = posteriors, events = events),
            mode = 'w',
            compression = 'gzip',
            compression_opts = 9,
        )

    for par in posteriors:
        posteriors[par] = xp.array(posteriors[par])
    events = list(map(str, np.array(events).astype(str)))

    return posteriors, events


def get_injections(
    pars = default_pars,
    catalog = 'GWTC-4',
    far_cut = 1,
    snr_cut = 10,
    load = False,
    save = False,
    xp=np,
    datadir='../data'
):
    datadir = get_datadir(datadir, catalog, pars)
    datapath = f'{datadir}/injections.h5'

    if load and os.path.exists(datapath):
        injections = h5ify.load(datapath)
        for k in injections:
            injections[k] = xp.array(injections[k]).squeeze()
        return injections

    vt_path = '/home/rp.o4/offline-injections/mixtures/multirun-mixtures'
    if catalog == 'GWTC-3':
        vt_path = f'{vt_path}_20250503134659UTC/mixture-semi_o1_o2-real_o3/mixture-semi_o1_o2-real_o3-cartesian_spins_20250503134659UTC.hdf'
    elif catalog == 'GWTC-4':
        vt_path = f'{vt_path}_20250503134659UTC/mixture-semi_o1_o2-real_o3_o4a/mixture-semi_o1_o2-real_o3_o4a-cartesian_spins_20250503134659UTC.hdf'
    if catalog == 'GWTC-5':
        vt_path = f'{vt_path}_20250503134659UTC/mixture-semi_o1_o2_o4b-real_o3_o4a/mixture-semi_o1_o2_o4b-real_o3_o4a-cartesian_spins_20250503134659UTC.hdf'

    injections = {}

    with h5py.File(vt_path, 'r') as f:
        time = f.attrs['total_analysis_time'] / 60 / 60 / 24 / 365.25
        total = f.attrs['total_generated']

        d = f['events'][:]

        far = np.min([d[k] for k in d.dtype.names if 'far' in k], axis = 0)
        snr = d['semianalytic_observed_phase_maximized_snr_net']
        found = (far < far_cut) | (snr > snr_cut)

        prior = np.exp(d[
            'lnpdraw_mass1_source_mass2_source_redshift'
            '_spin1x_spin1y_spin1z_spin2x_spin2y_spin2z'
        ][found]) / d['weights'][found]

        m1 = d['mass1_source'][found]
        m2 = d['mass2_source'][found]
        s1x = d['spin1x'][found]
        s1y = d['spin1y'][found]
        s1z = d['spin1z'][found]
        s2x = d['spin2x'][found]
        s2y = d['spin2y'][found]
        s2z = d['spin2z'][found]
        z = d['redshift'][found]

    a1 = (s1x**2 + s1y**2 + s1z**2)**0.5
    a2 = (s2x**2 + s2y**2 + s2z**2)**0.5
    c1 = s1z / a1
    c2 = s2z / a2

    injections['mass_1_source'] = m1
    injections['mass_2_source'] = m2
    injections['mass_ratio'] = m2 / m1
    injections['a_1'] = a1
    injections['a_2'] = a2
    injections['cos_tilt_1'] = c1
    injections['cos_tilt_2'] = c2
    injections['redshift'] = z
    injections['chirp_mass'] = (m1 * m2) ** (3/5) / (m1 + m2) ** (1/5)
    injections['snr'] = snr[found]

    injections = {par: injections[par] for par in pars}

    injections['prior'] = prior * 4 * np.pi**2 * a1**2 * a2**2
    if 'mass_ratio' in pars:
        injections['prior'] *= m1

    injections['time'] = time
    injections['total'] = total

    for k in injections:
        injections[k] = np.atleast_1d(injections[k])

    if save:
        h5ify.save(
            datapath,
            injections,
            mode = 'w',
            compression = 'gzip',
            compression_opts = 9,
        )

    for k in injections:
        injections[k] = xp.array(injections[k]).squeeze()

    return injections


def get_gwtc4_bbh():
    '/work/submit/newolfe/lvk-data/GWTC-4/RP/events_list_bbh_only.txt'