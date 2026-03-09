import os
import re
from glob import glob

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from bilby.gw.prior import UniformSourceFrame

import h5ify

sec_to_yr = 1 / 60 / 60 / 24 / 365

# pulled from Reed's example code for querying O4a only, excluding ER15
o4a_start = 1368975618
o4a_end = 1389455118

# o1, o2, o3 times pulled from:
# https://gwosc.org/O1/
# https://gwosc.org/O2/
# https://gwosc.org/O3/O3a
# https://gwosc.org/O3/O3b
o1_start = 1126051217
o1_end = 1137254417

o2_start = 1164556817
o2_end = 1187733618

o3_start = 1238166018
o3_end = 1269363618

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


def _get_sample_set(f, prefer_xphm_gwtc3=False, prefer_xphm=False):
    keys = list(set(f) - {'history', 'version'})

    wvf = None

    if prefer_xphm:
        for k in keys:
            if 'XPHM' in k:
                wvf = k
                break
    else:
        # GWTC-3
        if 'C01:Mixed' in f:
            wvf = 'C01:Mixed'

            if prefer_xphm_gwtc3:
                for k in keys:
                    if 'XPHM' in k:
                        wvf = k
                        break

            if prefer_xphm_gwtc3 and 'XPHM' not in k:
                raise ValueError('Missing XPHM!')
        # O4a
        elif 'C00:NRSur7dq4' in f:
            wvf = 'C00:NRSur7dq4'
        elif 'C00:Mixed' in f:
            wvf = 'C00:Mixed'
        # O4b
        else:
            assert len(keys) == 1
            wvf = keys[0]
            print(f'No other keys found, using {wvf}')

    print('using sample set:', wvf)

    # f['C00:IMRPhenomXPHM-SpinTaylor']['meta_data']['sampler']

    grp = f[wvf]
    meta_data = grp['meta_data']['sampler'] 

    attrs = {
        k : meta_data[k][:] if k in meta_data else np.nan
        for k in ['ln_bayes_factor', 'ln_evidence', 'ln_evidence_error', 'ln_noise_evidence'] 
    }

    if wvf == 'C00:Mixed' and 'ln_evidence' not in meta_data:
        ln_z = 0
        for k in keys:
            if 'Mixed' not in k and 'XO4a' not in k:
                ln_z += f[k]['meta_data']['sampler']['ln_evidence'][:]
        attrs['ln_evidence'] = ln_z

    return grp['posterior_samples'], attrs 


def load_and_reduce_pe(path, pars, prefer_xphm_gwtc3, prefer_xphm=False):
    # TODO: switches for prior choices?
    with h5py.File(path, 'r') as f:
        data, attrs = _get_sample_set(
            f, prefer_xphm_gwtc3=prefer_xphm_gwtc3, prefer_xphm=prefer_xphm
        )

        posterior = {par: data[par][:] for par in pars}
        posterior['prior'] = UniformSourceFrame(
            minimum=posterior['redshift'].min(),
            maximum=posterior['redshift'].max(),
            cosmology='Planck15_LAL',
            name='redshift',
        ).prob(posterior['redshift']) * (1 + posterior['redshift'])**2

    for k, v in attrs.items():
        posterior[k] = v

    return posterior


def resample_and_reshape_posteriors(posteriors, seed=1):
    min_npe_samples = min([len(p['prior']) for p in posteriors])
    rng = np.random.default_rng(seed)

    for i, p in enumerate(posteriors):
        idxs = rng.choice(len(p['prior']), min_npe_samples, replace=False)
        posteriors[i] = {
            k : p[k][idxs] if len(np.atleast_1d(p[k])) > 1 else p[k]
            for k in p.keys()
        }

    return {
        k : np.stack([ np.atleast_1d(p[k]) for p in posteriors])
        for k in posteriors[0].keys()
    }


def get_posteriors(
    pars=default_pars,
    catalog='GWTC-4',
    exclude=default_exclude,
    load=False,
    save=False,
    xp=np,
    datadir='../../data/lvk',
    seed=1,
    prefer_xphm_gwtc3=False,
    prefer_xphm=False,
    resample=True
):
    gwtc3_list = f'{datadir}/gwtc3.txt'
    all_list = f'{datadir}/all.csv'

    datadir = get_datadir(datadir, catalog, pars)

    if prefer_xphm:
        datapath = f'{datadir}/posteriors-xphm.h5'
    elif prefer_xphm_gwtc3:
        datapath = f'{datadir}/posteriors-xphm-gwtc3.h5'
    else:
        datapath = f'{datadir}/posteriors.h5'

    if save or load:
        print(f'data will be saved to/loaded from {datapath}')

    if load and os.path.exists(datapath):
        print(f'loading posteriors from {datapath}')
        data = h5ify.load(datapath)
        posteriors = data['posteriors']
        events = data['events']

        if 'log_prior' in posteriors:
            for par in posteriors:
                posteriors[par] = xp.array(posteriors[par])
        elif not resample:
            # assumes posteriors is structured like:
            # { event_name : posterior dict }
            posteriors = [
                {k : xp.array(v) for k, v in p.items()}
                for p in posteriors.values()
            ]

        events = list(map(str, np.array(events).astype(str)))

        for event in events:
            if event in exclude:
                raise ValueError(
                    f'An excluded event ({event}) is included in the saved posteriors file!'
                )

        snrs = data['snrs']
        fars = data['fars']
        cats = data['catalogs']
        cats = list(map(str, np.array(cats).astype(str)))
        meta = data['meta']

        return posteriors, events, snrs, fars, cats, meta

    all_events = pd.read_csv(all_list, index_col=False)
    gwtc3 = pd.read_csv(gwtc3_list, delimiter=' ', index_col=False)

    events = []
    snrs = []
    fars = []
    cats = []

    for (event, snr, far, cat) in zip(gwtc3['commonName'].values, gwtc3['network_matched_filter_snr'].values, gwtc3['far'].values, gwtc3['catalog.shortName']):
        if len([ex for ex in exclude if ex in event]) == 0:
            events.append(str(event))
            snrs.append(snr)
            fars.append(far)
            cats.append(cat)

    files = sorted([
        glob(f'/home/rp.o4/catalogs/GWTC-*/data-release/*{event}*_cosmo.h5')[0]
        for event in events
    ])
    
    if catalog in ['GWTC-4', 'GWTC-5']:
        files4a = sorted(glob(
            '/home/rp.o4/catalogs/GWTC-4/GWTC4-Stable_Release-9/38214bd95_724/'
            'bbh_only/*.hdf5',
        ))

        for file in files4a:
            e = 'GW' + file.split('/')[-1].split('GW')[-1][:13]

            if e in exclude:
                continue

            files.append(file)
            events.append(e)

            info = all_events.query(f'name == "{e}"')
            info = info.iloc[np.argmax(info['version'])]
            snrs.append(info['network_matched_filter_snr'])
            fars.append(info['far'])
            cats.append('GWTC-4')

    if catalog == 'GWTC-5':
        raise NotImplementedError()
        #files4b = sorted(glob(
        #    '/home/rp.o4/catalogs/O4b_prelim/O4bPreliminaryPE20250404/*.h5',
        #))
        #files += files4b
        #events += [file.split('/')[-1].split('-')[0] for file in files4b]

    for event in exclude:
        if event in events:
            files.pop(events.index(event))
            events.remove(event)

    posteriors = [
        load_and_reduce_pe(
            path,
            pars,
            prefer_xphm_gwtc3=prefer_xphm_gwtc3,
            prefer_xphm=prefer_xphm
        )
        for path in tqdm(files)
    ]

    if resample:
        posteriors = resample_and_reshape_posteriors(posteriors, seed)

        if 'mass_ratio' in pars:
            posteriors['prior'] *= posteriors['mass_1_source']

    else:
        if 'mass_ratio' in pars:
            for i in range(len(posteriors)):
                posteriors[i]['prior'] *= posteriors[i]['mass_1_source']

        posteriors = {e : p for (e, p) in zip(events, posteriors)}

    # TODO: doesnt work if we dont do resample
    meta = { k : posteriors.pop(k).squeeze() for k in ['ln_bayes_factor', 'ln_evidence', 'ln_evidence_error', 'ln_noise_evidence'] }

    if save:
        h5ify.save(
            datapath,
            dict(
                posteriors=posteriors,
                events=events,
                snrs=snrs,
                fars=fars,
                catalogs=cats,
                meta=meta
            ),
            mode='w',
            compression='gzip',
            compression_opts=9,
        )

    for par in posteriors:
        posteriors[par] = xp.array(posteriors[par])
    events = list(map(str, np.array(events).astype(str)))

    return posteriors, events, snrs, fars, cats


def get_injections(
    pars=default_pars,
    catalog='GWTC-4',
    far_cut=1,
    snr_cut=10,
    load=False,
    save=False,
    xp=np,
    datadir='../../data/lvk',
    vt_path='/home/rp.o4/offline-injections/mixtures/multirun-mixtures'
):
    datadir = get_datadir(datadir, catalog, pars)
    datapath = f'{datadir}/injections.h5'

    if load and os.path.exists(datapath):
        injections = h5ify.load(datapath)
        for k in injections:
            injections[k] = xp.array(injections[k]).squeeze()
        return injections

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

        far = np.min([d[k] for k in d.dtype.names if 'far' in k], axis=0)
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
        tc = d['time_geocenter'][found]

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
    injections['chirp_mass'] = (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)
    injections['geocent_time'] = tc
    injections['snr'] = snr[found]
    injections['far'] = far[found]

    injections = {
        par: injections[par]
        for par in pars + ['snr', 'far', 'geocent_time']
    }

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
            mode='w',
            compression='gzip',
            compression_opts=9,
        )

    for k in injections:
        injections[k] = xp.array(injections[k]).squeeze()

    return injections


def cut_data(event_data, injections, snr_thresh=10, far_thresh=1):
    posteriors = event_data[0]

    found = []
    events = []
    for event, snr, far, catalog in zip(*event_data[1:]):
        fi = ('GWTC-1' in catalog) & (snr >= snr_thresh)
        fi |= ('GWTC-1' not in catalog) & (far <= far_thresh)
        found.append(fi)
        if fi:
            events.append(event)
    found = np.array(found)

    if isinstance(posteriors, dict):
        posts = {k : v[found] for k, v in posteriors.items()}
    elif isinstance(posteriors, list):
        posts = [p for (p, fi) in zip(posteriors, found) if fi]
    else:
        raise ValueError('posteriors need to be a dict or list')

    found = (
        (injections['geocent_time'] < o3_start)
        & (injections['snr'] >= snr_thresh)
    )
    found |= (
        (injections['geocent_time'] >= o3_start)
        & (injections['far'] <= far_thresh)
    )
    injs = {
        k : v[found]
        for k, v in injections.items()
        if k not in ['time', 'total']
    }
    injs['total'] = injections['total']
    injs['time'] = injections['time']

    return events, posts, injs


def get_data(
    snr_thresh=10, far_thresh=1, prefer_xphm=False, prefer_xphm_gwtc3=False, return_ln_evidence=False
):
    event_data = get_posteriors(
        load=True,
        prefer_xphm=prefer_xphm,
        prefer_xphm_gwtc3=prefer_xphm_gwtc3
    )
    injections = get_injections(load=True)
    events, posteriors, injections = cut_data(
        event_data[:1],
        injections,
        snr_thresh=snr_thresh,
        far_thresh=far_thresh
    )

    if isinstance(posteriors, dict):
        posteriors['mass_1'] = posteriors.pop('mass_1_source')
        posteriors['log_prior'] = np.log(posteriors['prior'])
    elif isinstance(posteriors, list):
        for i in range(len(posteriors)):
            posteriors[i]['mass_1'] = posteriors[i].pop('mass_1_source')
            posteriors[i]['log_prior'] = np.log(posteriors[i]['prior'])
    else:
        raise ValueError('posteriors need to be a dict or list')

    injections['mass_1'] = injections.pop('mass_1_source')
    injections['total_generated'] = injections.pop('total')
    injections['log_prior'] = np.log(injections['prior'])

    ret = (events, posteriors, injections,)
    if return_ln_evidence:
        ret += (event_data['meta']['ln_evidence'],)
    return ret


if __name__ == '__main__':
    get_posteriors(save=True, prefer_xphm=True, resample=False)
