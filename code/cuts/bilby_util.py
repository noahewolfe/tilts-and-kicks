from copy import deepcopy

import numpy as np

from bilby.core.likelihood import Likelihood
from bilby.core.prior import ConditionalPriorDict
from bilby.core.prior import Uniform
from bilby.core.prior import DirichletElement
from bilby.core.prior import Prior


def _log_expm1_over_x(x):
    """Stable log((exp(x) - 1) / x)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)

    small = np.abs(x) < 1e-7
    neg = x < -1e-7
    pos_mid = np.logical_and(x >= 1e-7, x <= 50.0)
    pos_large = x > 50.0

    # log((e^x - 1) / x) = x/2 + O(x^2) near zero
    out[small] = x[small] / 2 + x[small] ** 2 / 24
    # x < 0: use log1p(-exp(x)) - log(-x) to avoid cancellation near 0-
    out[neg] = np.log1p(-np.exp(x[neg])) - np.log(-x[neg])
    # moderate positive x: expm1 is safe
    out[pos_mid] = np.log(np.expm1(x[pos_mid])) - np.log(x[pos_mid])
    # large positive x: log(e^x - 1) = x + log1p(-e^-x), avoids overflow
    out[pos_large] = (
        x[pos_large]
        + np.log1p(-np.exp(-x[pos_large]))
        - np.log(x[pos_large])
    )
    return out


def _logdiffexp(log_a, log_b):
    """Stable log(exp(log_a) - exp(log_b)) for log_a >= log_b."""
    log_b = np.minimum(log_b, log_a)
    return log_a + np.log1p(-np.exp(log_b - log_a))


class LogInterped(Prior):
    """1D interpolated prior with log-density interpolation and log-space CDF."""
    def __init__(
        self, xx, log_yy, minimum=None, maximum=None, name=None, latex_label=None,
        unit=None, boundary=None
    ):
        self.xx = np.asarray(xx, dtype=np.float64)
        self.log_yy = np.asarray(log_yy, dtype=np.float64)

        if self.xx.ndim != 1 or self.log_yy.ndim != 1:
            raise ValueError('xx and log_yy must be one-dimensional')
        if len(self.xx) < 2 or len(self.xx) != len(self.log_yy):
            raise ValueError('xx and log_yy must have same length >= 2')
        if not np.all(np.diff(self.xx) > 0):
            raise ValueError('xx must be strictly increasing')
        if not np.all(np.isfinite(self.log_yy)):
            raise ValueError('log_yy must be finite')

        if minimum is None:
            minimum = float(self.xx[0])
        if maximum is None:
            maximum = float(self.xx[-1])
        if (
            not np.isclose(minimum, float(self.xx[0]))
            or not np.isclose(maximum, float(self.xx[-1]))
        ):
            raise ValueError(
                'LogInterped currently requires minimum/maximum to match xx bounds'
            )

        super().__init__(
            name=name,
            latex_label=latex_label,
            unit=unit,
            minimum=minimum,
            maximum=maximum,
            boundary=boundary
        )

        self._dx = np.diff(self.xx)
        self._dlogp = np.diff(self.log_yy)

        self._log_interval_mass = (
            np.log(self._dx)
            + self.log_yy[:-1]
            + _log_expm1_over_x(self._dlogp)
        )
        self._log_norm = np.logaddexp.reduce(self._log_interval_mass)

        self.log_yy = self.log_yy - self._log_norm
        self._log_interval_mass = self._log_interval_mass - self._log_norm

        self._log_cdf_knots = np.full_like(self.xx, -np.inf)
        acc = -np.inf
        for i in range(1, len(self.xx)):
            acc = np.logaddexp(acc, self._log_interval_mass[i - 1])
            self._log_cdf_knots[i] = acc
        self._log_cdf_knots[-1] = 0.0

    def _idx_and_t(self, val):
        idx = np.searchsorted(self.xx, val, side='right') - 1
        idx = np.clip(idx, 0, len(self.xx) - 2)
        t = (val - self.xx[idx]) / self._dx[idx]
        return idx, np.clip(t, 0.0, 1.0)

    def ln_prob(self, val):
        scalar = np.isscalar(val)
        arr = np.asarray(val, dtype=np.float64)
        out = np.full(arr.shape, -np.inf, dtype=np.float64)

        mask = np.logical_and(arr >= self.minimum, arr <= self.maximum)
        if np.any(mask):
            x = arr[mask]
            idx, t = self._idx_and_t(x)
            lp = self.log_yy[idx] + self._dlogp[idx] * t
            lp = np.where(x == self.maximum, self.log_yy[-1], lp)
            out[mask] = lp

        return float(out) if scalar else out

    def prob(self, val):
        return np.exp(self.ln_prob(val))

    def cdf(self, val):
        scalar = np.isscalar(val)
        arr = np.asarray(val, dtype=np.float64)
        out = np.zeros(arr.shape, dtype=np.float64)
        out[arr >= self.maximum] = 1.0

        mask = np.logical_and(arr > self.minimum, arr < self.maximum)
        if np.any(mask):
            x = arr[mask]
            idx, t = self._idx_and_t(x)
            k = self._dlogp[idx]
            log_left = self._log_cdf_knots[idx]

            log_local = np.full_like(x, -np.inf, dtype=np.float64)
            pos = t > 0
            if np.any(pos):
                kp = k[pos] * t[pos]
                log_local[pos] = (
                    np.log(self._dx[idx][pos])
                    + self.log_yy[idx][pos]
                    + np.log(t[pos])
                    + _log_expm1_over_x(kp)
                )

            log_cdf_x = np.where(
                np.isfinite(log_left),
                np.logaddexp(log_left, log_local),
                log_local
            )
            out[mask] = np.exp(log_cdf_x)

        return float(out) if scalar else out

    def rescale(self, val):
        scalar = np.isscalar(val)
        arr = np.asarray(val, dtype=np.float64)
        out = np.empty(arr.shape, dtype=np.float64)

        out[arr <= 0] = self.minimum
        out[arr >= 1] = self.maximum

        mask = np.logical_and(arr > 0, arr < 1)
        if np.any(mask):
            u = arr[mask]
            log_u = np.log(u)

            idx = np.searchsorted(self._log_cdf_knots, log_u, side='right') - 1
            idx = np.clip(idx, 0, len(self.xx) - 2)

            log_left = self._log_cdf_knots[idx]
            log_w = np.where(
                np.isfinite(log_left),
                _logdiffexp(log_u, log_left),
                log_u
            )

            log_r = log_w - self._log_interval_mass[idx]
            r = np.exp(np.clip(log_r, -np.inf, 0.0))

            k = self._dlogp[idx]
            t = np.empty_like(r)
            flat = np.abs(k) < 1e-10
            large = k > 50.0
            other = np.logical_not(np.logical_or(flat, large))

            # k -> 0 limit
            t[flat] = r[flat]
            # log(1 + r (e^k - 1)) = k + log(r + (1-r)e^-k), stable for k >> 1
            t[large] = (
                k[large]
                + np.log(r[large] + (1.0 - r[large]) * np.exp(-k[large]))
            ) / k[large]
            # safe for moderate/negative k
            t[other] = np.log1p(r[other] * np.expm1(k[other])) / k[other]
            out[mask] = self.xx[idx] + self._dx[idx] * t

        return float(out) if scalar else out


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
        elif network == 'CE20':
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
