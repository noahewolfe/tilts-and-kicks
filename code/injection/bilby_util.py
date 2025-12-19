from copy import deepcopy

from bilby.core.likelihood import Likelihood
from bilby.core.prior import ConditionalPriorDict
from bilby.core.prior import Uniform
from bilby.core.prior import DirichletElement


def convert_bilby_uniform_prior(prior, backend='jax'):
    if backend == 'jax':
        import jax.numpy as xp
    elif backend == 'numpy':
        import numpy as xp

    prior_bounds = {
        k : [b.minimum, b.maximum]
        for k, b in ConditionalPriorDict(prior).items()
    }

    param_keys = list(prior_bounds.keys())
    bounds = xp.array(list(prior_bounds.values()))

    def log_prior(parameters):
        return -xp.log(xp.prod(xp.diff(bounds)))

    return param_keys, bounds, log_prior 


def convert_bilby_prior(prior, backend='jax'):
    if backend == 'jax':
        import jax.numpy as xp
    elif backend == 'numpy':
        import numpy as xp

    prior = ConditionalPriorDict(deepcopy(prior))

    label = None
    n = None

    for k, v in prior.items():
        if isinstance(v, DirichletElement):
            if label is None:
                label = v.label
                n = v.n_dimensions
            elif label != v.label:
                raise NotImplementedError('multiple dirichlet priors yet')
        elif not isinstance(v, Uniform):
            raise NotImplementedError(f'Unsupported prior {type(v)} for {k}')

    if label is None:
        def fold(x):
            return x

        param_keys, bounds, log_prior = convert_bilby_uniform_prior(
            prior, backend=backend
        )
    else:
        for i in range(n):
            if f'{label}{i}' in prior.keys():
                prior.pop(f'{label}{i}')
        for i in range(n - 1):
            prior[f'{label}{i}_unnorm'] = Uniform(minimum=0, maximum=1)

        param_keys, bounds, log_unif_prior = convert_bilby_uniform_prior(
            prior, backend='jax'
        )

        def fold(parameters):
            """ map unnormalized to normalized domain """
            # NOTE: this will be slow to compile for large n
            # NOTE: not sure if this works for other than alpha = 1
            # I think this works because our training setup---implicitly, when
            # alpha = 1---means that we initially train the flow to match a
            # uniform prior on the unnormalized dirichlet parameters

            dir_pars = xp.concatenate([
                xp.array([0.0]),
                xp.array([
                    parameters[f'{label}{i}_unnorm']
                    for i in range(n - 1)
                ]),
                xp.array([1.0])
            ])
            dir_pars = xp.sort(dir_pars)
            dir_pars = xp.diff(dir_pars)

            for i in range(n):
                parameters[f'{label}{i}'] = dir_pars[i]

            return parameters

        def log_prior(parameters):
            log_prob = log_unif_prior(parameters)
            log_prob += dirichlet_log_prob(
                label, n, parameters, backend=backend
            )
            return log_prob

    return param_keys, bounds, log_prior, fold


def dirichlet_log_prob(label, n, parameters, backend='jax'):
    if backend == 'jax':
        import jax.numpy as xp
        from jax.scipy.special import logsumexp, gammaln
    elif backend == 'numpy':
        import numpy as xp
        from scipy.special import logsumexp, gammaln

    alphas = xp.ones(n)
    x = xp.array([parameters[f'{label}{i}'] for i in range(n)])

    log_prob = xp.log(xp.prod(xp.power(x, alphas - 1)))
    log_norm = logsumexp(gammaln(alphas)) - gammaln(xp.sum(alphas))

    return log_prob - log_norm


class LikelihoodWrapper(Likelihood):
    def __init__(
        self, posteriors, injections, density, rate=False, taper=lambda _: 0,
        tobs=None
    ):
        from jax import jit

        super(LikelihoodWrapper, self).__init__(dict())

        if rate:
            if tobs is None:
                raise ValueError('you need to provide a tobs')

            def fn(parameters):
                from likelihood import rate_ln_likelihood_and_variance

                return rate_ln_likelihood_and_variance(
                    tobs, posteriors, injections, density, parameters
                )

            def extras(parameters):
                from likelihood import rate_likelihood_extras

                return rate_likelihood_extras(
                    tobs, posteriors, injections, density, parameters
                )
        else:
            def fn(parameters):
                from likelihood import shape_ln_likelihood_and_variance

                return shape_ln_likelihood_and_variance(
                    posteriors, injections, density, parameters
                )

            def extras(parameters):
                from likelihood import shape_likelihood_extras

                return shape_likelihood_extras(
                    posteriors, injections, density, parameters
                )

        self.ln_lkl_and_variance = fn

        def log_likelihood_ratio_func(parameters):
            ln_lkl, variance, _, _, _ = self.ln_lkl_and_variance(parameters)
            return ln_lkl + taper(variance)

        self.log_likelihood_ratio_func = jit(log_likelihood_ratio_func)
        self.generate_extra_statistics = jit(extras)

    def log_likelihood_ratio(self):
        return self.log_likelihood_ratio_func(self.parameters)

    def noise_log_likelihood(self):
        from jax.numpy import nan
        return nan

    def log_likelihood(self):
        return self.noise_log_likelihood() + self.log_likelihood_ratio()


def get_network(
    network, noise_curves_dir='../../sensitivity_curves',
    minimum_frequency=None, maximum_frequency=None
):
    from bilby.gw.detector import InterferometerList
    from bilby.gw.detector import Interferometer, PowerSpectralDensity

    if isinstance(network, str):
        network = network.upper()

        if network == 'CE40':
            ifos = InterferometerList(['CE40'])
        lif network == 'CE20':
            ifos = InterferometerList(['CE20'])
        elif network == 'CE2040':
            ifos = InterferometerList(['CE20', 'CE40'])
        elif network == 'CE2040ET':
            ifos = InterferometerList(['CE20', 'CE40', 'ET'])
        elif network == 'H1L1O3':
            H1 = Interferometer(
                power_spectral_density=PowerSpectralDensity(
                    asd_file=f'{noise_curves_dir}/aligo_O3actual_H1.txt'
                ),
                name='H1',
                minimum_frequency=20,
                maximum_frequency=2048,
                length=4,
                latitude=46 + 27. / 60 + 18.528 / 3600,
                longitude=-(119 + 24. / 60 + 27.5657 / 3600),
                elevation=142.554,
                xarm_azimuth=125.9994,
                yarm_azimuth=215.9994,
                xarm_tilt=-6.195e-4,
                yarm_tilt=1.25e-5
            )

            L1 = Interferometer(
                power_spectral_density=PowerSpectralDensity(
                    asd_file=f'{noise_curves_dir}/aligo_O3actual_L1.txt'
                ),
                name='H1',
                minimum_frequency=20,
                maximum_frequency=2048,
                length=4,
                latitude=30 + 33. / 60 + 46.4196 / 3600,
                longitude=-(90 + 46. / 60 + 27.2654 / 3600),
                elevation=-6.574,
                xarm_azimuth=197.7165,
                yarm_azimuth=287.7165,
                xarm_tilt=-3.121e-4,
                yarm_tilt=-6.107e-4
            )
            ifos = InterferometerList([H1, L1])
        elif network == 'H1L1O4':
            ifos = InterferometerList(['H1', 'L1'])
        elif network == 'H1L1V1O4_SALVO':
            if minimum_frequency is None or maximum_frequency is None:
                raise ValueError(
                    'You need to set a min./max. frequency manually to '
                    "replicate Salvo's VT file"
                )
            ifos = InterferometerList(['H1', 'L1', 'V1'])
            for ifo in ifos:
                ifo.minimum_frequency = minimum_frequency
                ifo.maximum_frequency = maximum_frequency
                if ifo.name == 'V1':
                    ifo.power_spectral_density = PowerSpectralDensity(
                        asd_file='../../sensitivity_curves/avirgo_O4high_NEW.txt'
                    )
                else:
                    ifo.power_spectral_density = PowerSpectralDensity(
                        asd_file='../../sensitivity_curves/aligo_O4low.txt'
                    )
        else:
            raise ValueError(
                f'Network {network} is not a valid network configuration!'
            )
    else:
        ifos = InterferometerList(network)

    return ifos
