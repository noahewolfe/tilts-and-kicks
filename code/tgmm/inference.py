import os
from argparse import ArgumentParser

"""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NPOC"] = "1"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".5"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["NPROC"] = "1"
os.environ["intra_op_parallelism_threads"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
"""

import h5py

import jax

from bilby.core.prior import ConditionalPriorDict
from bilby.core.prior import Uniform as Uniform_bilby
from numpyro.distributions import Uniform as Uniform_numpyro

#from gravpop import save_dict_h5
#from gravpop import load_dict_h5

from gravpop import mixture
from gravpop import FixedParameters

from gravpop import Uniform2DAnalytic
from gravpop import TruncatedGaussian2DAnalytic

from gravpop import PowerLawRedshift
from gravpop import SelectionFunction
from gravpop import MarginalizedHybridLikelihood

from gravpop.sampler.sampler import DirchletPrior, DiracDelta
from gravpop.sampler.sampler import Sampler

from models import SmoothedBrokenPowerLawTwoPeaks
from util import write_config

parser = ArgumentParser()
parser.add_argument('--outdir')
parser.add_argument('--seed', type=int, default=1701)
parser.add_argument('--thinning', type=int, default=1)
parser.add_argument('--num-samples', type=int, default=1_000)
parser.add_argument('--num-warmup', type=int, default=10_000)
parser.add_argument('--max-tree-depth', type=int, default=5)
parser.add_argument('--target-accept-prob', type=float, default=0.65)


### For saving and loading
### should be in newest gravpop version
_NONE_ATTR = "__is_none__"

def save_dict_h5(filename, data):
    def _save_group(h5group, d):
        for k, v in d.items():
            if v is None:
                g = h5group.create_group(k)
                g.attrs[_NONE_ATTR] = True
            elif isinstance(v, dict):
                _save_group(h5group.create_group(k), v)
            elif isinstance(v, pd.DataFrame):
                g = h5group.create_group(k)
                g.create_dataset("columns", data=np.array(v.columns, dtype="S"))
                g.create_dataset("values", data=v.to_numpy())
            elif isinstance(v, (list, tuple)):
                arr = np.array(v)
                if arr.dtype.kind in {"U", "O"}:  # strings (or objects that are strings)
                    arr = arr.astype("S")
                h5group.create_dataset(k, data=arr)
            elif isinstance(v, str):
                h5group.create_dataset(k, data=np.bytes_(v))  # NumPy 2.0+
            else:
                h5group.create_dataset(k, data=np.array(v))
    with h5py.File(filename, "w") as f:
        _save_group(f, data)

def load_dict_h5(filename):
    def _load_group(h5group):
        out = {}
        for k, v in h5group.items():
            if isinstance(v, h5py.Group):
                # None sentinel?
                if v.attrs.get(_NONE_ATTR, False):
                    out[k] = None
                # DataFrame?
                elif "columns" in v and "values" in v:
                    cols = [c.decode() for c in v["columns"][()]]
                    out[k] = pd.DataFrame(v["values"][()], columns=cols)
                else:
                    out[k] = _load_group(v)
            else:
                arr = v[()]
                if arr.dtype.kind == "S":
                    # Return string arrays as Python lists of str; scalars as str
                    out[k] = arr.decode() if arr.ndim == 0 else [x.decode() for x in arr.flatten()]
                else:
                    out[k] = arr.tolist() if arr.ndim == 0 else arr
        return out
    with h5py.File(filename, "r") as f:
        return _load_group(f)


def get_mass_model():
    return SmoothedBrokenPowerLawTwoPeaks()


def get_chi_model():
    chi_model = TruncatedGaussian2DAnalytic(
        a = [0,0],
        b = [1,1],
        var_names=['chi_1', 'chi_2'],
        hyper_var_names=[
            'mu_chi', 'sigma_chi', 'mu_chi', 'sigma_chi', 'rho_chi'
        ]
    )
    return FixedParameters(chi_model, {'rho_chi' : 1e-4})


def get_tilt_model():
    model_tilt1__ = TruncatedGaussian2DAnalytic(
        a=[-1, -1],
        b=[1, 1],
        var_names=['cos_tilt_1', 'cos_tilt_2'],
        hyper_var_names=[
            'mu_spin', 'sigma_spin', 'mu_spin', 'sigma_spin', 'rho_cos_tilt'
        ]
    )

    # TODO: why not default 1e-6?
    # NOTE: AH: Dont do zero because derivatives can be nan
    model_tilt1 = FixedParameters(model_tilt1__, {'rho_cos_tilt' : 1e-4})

    model_tilt_uniform = Uniform2DAnalytic(a=[-1,-1], b=[1,1],
                        var_names=['cos_tilt_1', 'cos_tilt_2'],
                        hyper_var_names=[])

    return mixture(
        [model_tilt1, model_tilt_uniform],
        ['xi_spin', 'one_minus_xi_spin']
    )


def get_model():
    mass_model = get_mass_model()
    redshift_model = PowerLawRedshift(
        var_names=['redshift'],
        hyper_var_names=['lamb'],
        z_max=3
    )
    chi_model = get_chi_model()
    tilt_model = get_tilt_model()
    return [mass_model, redshift_model, tilt_model, chi_model]


def get_selection():
    selection_data = load_dict_h5('../../data/tgmm/selection_data.h5')
    analysis_time = selection_data.pop('analysis_time')
    total_generated = selection_data.pop('total_generated')
    total_detected = selection_data.pop('total_detected')
    return SelectionFunction(
        selection_data,
        analysis_time=analysis_time,
        total_generated=total_generated,
        total_detected=total_detected
    )


def get_event_data():
    return load_dict_h5('../../data/tgmm/event_data.hdf5')


def get_priors(path, outdir=None):
    bilby_priors = ConditionalPriorDict(path)

    if outdir is not None:
        bilby_priors.to_file(outdir, 'init')

    priors = dict()
    for k, v in bilby_priors.items():
        if isinstance(v, Uniform_bilby):
            priors[k] = Uniform_numpyro(
                v.minimum,
                v.maximum
            )

    priors['lam_fractions'] = DirchletPrior(
        var_names=['lam_0', 'lam_1', 'lam_2']
    )

    return priors


def parse_args():
    args = parser.parse_args()
    write_config(args, outdir=args.outdir)
    return (
        args.outdir,
        args.seed,
        args.num_samples,
        args.num_warmup,
        args.thinning,
        args.max_tree_depth,
        args.target_accept_prob
    )


if __name__ == '__main__':
    (
        outdir,
        seed,
        num_samples,
        num_warmup,
        thinning,
        max_tree_depth,
        target_accept_prob
    ) = parse_args()

    event_data = get_event_data()
    selection_func = get_selection()

    print('loaded data')

    priors = get_priors('./lvk.prior')

    print('setup priors')

    # TODO: to be extra conservative about it, I've got two copies
    # of the population model here. TBD if this is required.
    HL = MarginalizedHybridLikelihood(
        event_data=event_data,
        selection_data=selection_func,
        models=get_model(),
        models_selection=get_model(),
        fix_kernels_selection={},
        fix_kernels_events={}
    )

    print('setup likelihood')
    print('kick off sampler...')

    samp = Sampler(
        priors=priors,
        latex_symbols={k:k for k in priors.keys()},
        likelihood=HL,
        seed=seed,
        num_samples=num_samples,
        num_warmup=num_warmup,
        thinning=thinning,
        max_tree_depth=max_tree_depth,
        target_accept_prob=target_accept_prob,
    )
    samp.sample()
    samp.samples.to_csv(f'{outdir}/run.csv')

    print('done.')