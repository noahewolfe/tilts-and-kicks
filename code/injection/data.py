import h5ify
import pickle
import numpy as np


def resample_and_reshape_posteriors(posteriors, seed=1):
    """ downsample a list of posteriors and reshape into dict """
    min_npe_samples = min([len(p['prior']) for p in posteriors])
    rng = np.random.default_rng(seed)

    for i, p in enumerate(posteriors):
        idxs = rng.choice(len(p['prior']), min_npe_samples, replace=False)
        posteriors[i] = {k : p[k][idxs] for k in p.keys()}

    return {
        k : np.stack([p[k] for p in posteriors])
        for k in posteriors[0].keys()
    }


def load_salvo_posteriors(path, seed=1):
    """ load salvo posteriors generated on submit """
    with open(path, 'rb') as f:
        posteriors = pickle.load(f)
    posteriors = resample_and_reshape_posteriors(posteriors, seed=seed)
    posteriors.pop('cos_tilt_1_entropy', None)
    posteriors.pop('net_snr', None)
    posteriors['log_prior'] = np.log(posteriors.pop('prior'))
    return posteriors


def load_posteriors(path, deltas=False, seed=1):
    """ load noah posteriors """
    if deltas:
        posteriors = h5ify.load(path)
        if 'mass_1_source' in posteriors:
            posteriors['mass_1'] = posteriors.pop('mass_1_source')
        posteriors = {
            k : posteriors[k].reshape(-1, 1)
            for k in [
                'mass_1',
                'mass_ratio',
                'redshift',
                'a_1',
                'a_2',
                'cos_tilt_1',
                'cos_tilt_2',
                'log_prior'
            ]
        }
    else:
        data = h5ify.load(path)
        posteriors = list(data.values())
        for p in posteriors:
            p.pop('attrs')
        posteriors = resample_and_reshape_posteriors(posteriors, seed=seed)
        posteriors['log_prior'] = np.log(posteriors.pop('prior'))

    return posteriors
