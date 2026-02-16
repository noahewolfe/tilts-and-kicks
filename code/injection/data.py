import numpy as np


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
