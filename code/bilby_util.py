from bilby.core.likelihood import Likelihood
from likelihood import rate_ln_likelihood_and_variance
from likelihood import shape_ln_likelihood_and_variance
from likelihood import rate_likelihood_extras
from likelihood import shape_likelihood_extras


def convert_bilby_uniform_prior(prior, backend='jax'):
    from bilby.core.prior import ConditionalPriorDict

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


class LikelihoodWrapper(Likelihood):
    def __init__(
        self, posteriors, injections, density, rate=False, taper=lambda: 0,
        tobs=None
    ):
        from jax import jit

        super(LikelihoodWrapper, self).__init__(dict())

        if rate:
            if tobs is None:
                raise ValueError('you need to provide a tobs')

            def fn(parameters):
                return rate_ln_likelihood_and_variance(
                    tobs, posteriors, injections, density, parameters
                )

            def extras(parameters):
                return rate_likelihood_extras(
                    tobs, posteriors, injections, density, parameters
                )
        else:
            def fn(parameters):
                return shape_ln_likelihood_and_variance(
                    posteriors, injections, density, parameters
                )

            def extras(parameters):
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
