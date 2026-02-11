from bilby.core.prior import ConditionalPriorDict

from gravpop import save_dict_h5
from gravpop import load_dict_h5

from gravpop import mixture
from gravpop import FixedParameters
from gravpop import Uniform2DAnalytic
from gravpop import TruncatedGaussian2DAnalytic

from gravpop import PowerLawRedshift
from gravpop import SmoothedTwoComponentPrimaryMassRatio

from gravpop import SelectionFunction
from gravpop import MarginalizedHybridLikelihood


def get_mass_model():
    return SmoothedTwoComponentPrimaryMassRatio(
        mmin_fixed=2,
        mmax_fixed=300,
        gaussian_mass_maximum=100,
        var_names=['mass_1_source', 'mass_ratio'],
        hyper_var_names=[
            'alpha', 'beta', 'lam', 'mpp', 'sigpp', 'delta_m', 'mmin', 'mmax'
        ]
    )


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
        a=[-1,-1],
        b=[1,1],
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
    selection_data = load_dict_h5(
        '/home/asad.hussain/O4b_test/o4a_data_products/selection_data.h5'
    )
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
    return load_dict_h5(
        '/home/asad.hussain/O4b_test/o4a_data_products/event_data.hdf5'
    )


def get_priors(path):
    priors = ConditionalPriorDict(path)
    for k, v in priors.items():
    


if __name__ == '__main__':
    event_data = get_event_data()
    selection_func = get_selection()

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

