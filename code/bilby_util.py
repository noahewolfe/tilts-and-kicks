from copy import deepcopy

from bilby.core.likelihood import Likelihood
from bilby.core.prior import ConditionalPriorDict
from bilby.core.prior import Uniform
from bilby.core.prior import DirichletElement

from likelihood import rate_ln_likelihood_and_variance
from likelihood import shape_ln_likelihood_and_variance
from likelihood import rate_likelihood_extras
from likelihood import shape_likelihood_extras


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
