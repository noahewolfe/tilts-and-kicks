from bilby.core.prior import ConditionalPriorDict


def convert_bilby_uniform_prior(prior, backend='jax'):
    """ pull apart a bilby PriorDict object """
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

    def log_prior(_):
        return -xp.log(xp.prod(xp.diff(bounds)))

    return param_keys, bounds, log_prior


def dirichlet_log_prob(label, n, parameters, backend='jax'):
    """ log of dirichlet prior with n elements
        This is written to, possibly in future, support alpha != 1 if
        I ever need that ... but for now we fix all alphas to 1
        See: https://en.wikipedia.org/wiki/Dirichlet_distribution
    """
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


def convert_bilby_prior(prior, backend='jax'):
    """ convert a bilby prior which may have Uniform or Dirichlet priors """
    from copy import deepcopy
    from bilby.core.prior import Uniform
    from bilby.core.prior import DirichletElement

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
