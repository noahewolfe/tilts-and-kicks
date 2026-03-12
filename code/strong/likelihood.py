import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as lse


def mean_and_variance(weights, n):
    # mean and variance of the mean
    mean = jnp.sum(weights) / n
    variance = jnp.sum(weights**2) / n**2 - mean**2 / n
    return mean, variance


def safe_mean_and_variance(log_weights, n):
    log_n = jnp.log(n)
    log_mean = lse(log_weights) - log_n
    log_neff = 2 * lse(log_weights) - lse(2 * log_weights)
    log_variance = -log_neff + jax.nn.log1mexp(log_neff - log_n)
    return log_mean, jnp.exp(log_variance)


def ln_mean_and_variance(weights, n):
    # lazy ln(mean) and variance of ln(mean)
    mean, variance = mean_and_variance(weights, n)
    return jnp.log(mean), variance / mean**2


def safe_ln_mean_and_variance(log_weights, n):
    """ numerically safe ln(mean) and variance of ln(mean) """
    log_n = jnp.log(n)
    log_mean = lse(log_weights) - log_n

    log_neff = 2 * lse(log_weights) - lse(2 * log_weights)

    # TODO: can we do anything here (tricks with log1p or expm1)?
    #log_variance = -log_neff + jnp.log(-jnp.expm1(log_neff - log_n))
    #log_variance_of_log = log_variance - 2 * log_mean
    variance_of_log = 1 / jnp.exp(log_neff) - 1 / n

    return log_mean, variance_of_log


def selection(injections, density, parameters):
    vt_weights = density(injections, parameters) / injections['prior']
    pdet, pdet_variance = mean_and_variance(
        vt_weights, injections['total_generated']
    )
    return pdet, pdet_variance


def event_ln_likelihoods_and_selection(
    posteriors, injections, density, parameters
):
    pe_weights = density(posteriors, parameters) / posteriors['prior']

    _, npe = pe_weights.shape
    ln_lkls, pe_variances = jax.vmap(
        lambda weights: ln_mean_and_variance(weights, npe)
    )(pe_weights)

    pdet, pdet_variance = selection(injections, density, parameters)

    return ln_lkls, pdet, pe_variances, pdet_variance


def rate_ln_likelihood_and_variance(
    posteriors, injections, density, parameters
):
    (
        ln_lkls, pdet, pe_variances, pdet_variance
    ) = event_ln_likelihoods_and_selection(
        posteriors,
        injections,
        density,
        parameters
    )

    tobs = injections['time']

    nobs = len(ln_lkls)
    ln_rate = parameters['ln_rate']
    rate = jnp.exp(ln_rate)

    nexp = rate * tobs * pdet

    ln_lkl = jnp.sum(ln_lkls) + nobs * ln_rate - nexp

    pe_variance = jnp.sum(pe_variances)
    variance = pe_variance + (rate * tobs)**2 * pdet_variance

    return ln_lkl, variance, nexp, pdet, pe_variance, pdet_variance


def shape_ln_likelihood_and_variance(
    posteriors, injections, density, parameters
):
    vt_log_weights = density(injections, parameters) - injections['log_prior']

    if isinstance(posteriors, dict):
        pe_log_weights = (
            density(posteriors, parameters) - posteriors['log_prior']
        )

        nobs, npe = pe_log_weights.shape
        ln_lkls, pe_variances = jax.vmap(
            lambda lw: safe_ln_mean_and_variance(lw, npe)
        )(pe_log_weights)
        ln_lkls_sum = jnp.sum(ln_lkls)
        pe_variance = jnp.sum(pe_variances)
    else:
        nobs = len(posteriors)

        def accumulate(acc, posterior):
            pe_log_weights = (
                density(posterior, parameters) - posterior['log_prior']
            )
            ln_lkl, variance = safe_ln_mean_and_variance(
                pe_log_weights, pe_log_weights.shape[0]
            )
            return acc[0] + ln_lkl, acc[1] + variance

        ln_lkls_sum, pe_variance = jax.tree.reduce(
            accumulate,
            posteriors,
            initializer=(
                jnp.zeros((), dtype=vt_log_weights.dtype),
                jnp.zeros((), dtype=vt_log_weights.dtype)
            ),
            is_leaf=lambda x: isinstance(x, dict),
        )

    ninj = injections['total_generated']
    ln_pdet, ln_pdet_variance = safe_ln_mean_and_variance(vt_log_weights, ninj)

    ln_lkl = ln_lkls_sum - ln_pdet * nobs
    variance = pe_variance + ln_pdet_variance * nobs**2
    return ln_lkl, variance, pe_variance, ln_pdet_variance, ln_pdet


def unravel(param_keys, x):
    return dict(zip(param_keys, x))


def get_log_likelihood(model, posteriors, injections, rate=True):

    if rate:
        def log_likelihood(parameters):
            return rate_ln_likelihood_and_variance(
                posteriors, injections, model, parameters
            )
    else:
        def log_likelihood(parameters):
            return shape_ln_likelihood_and_variance(
                posteriors, injections, model, parameters
            )

    return log_likelihood


def taper(maximum_variance, v):
    return -100 * (maximum_variance - v)**2 * (v >= maximum_variance)


def get_bilby_likelihood(
    model,
    posteriors,
    injections,
    taper=lambda _: 0,
    rate=True,
):
    """
    return an instance of a bilby.core.likelihood.Likelihood
    using my own jit-compiled likelihoods.

    Yeah yeah, we should define a generic class which we can instantiate
    given different input arguments. However, bilby is slow to import
    and I don't want to import it every time I call `likelihood.py`.
    """

    from bilby.core.likelihood import Likelihood

    log_likelihood = get_log_likelihood(
        model, posteriors, injections, rate=rate
    )

    class LikelihoodWrapper(Likelihood):
        def __init__(self):
            super(LikelihoodWrapper, self).__init__(dict())

            def fn(parameters):
                out = log_likelihood(parameters)
                return out[:2]

            self.ln_lkl_and_variance = fn

            def log_likelihood_ratio_func(parameters):
                lnl, var = self.ln_lkl_and_variance(parameters)
                lnl += taper(var)
                return jnp.nan_to_num(lnl).astype(float)

            self.log_likelihood_ratio_func = jax.jit(log_likelihood_ratio_func)

            def extras(parameters):
                if rate:
                    (
                        lnl, var, nexp, pdet, pe_variance, pdet_variance
                    ) = log_likelihood(parameters)

                    return dict(
                        ln_lkl=lnl,
                        nexp=nexp,
                        pdet=pdet,
                        pdet_variance=pdet_variance,
                        pe_variance=pe_variance,
                        variance=var
                    )
                else:
                    (
                        lnl, var, pe_variance, ln_pdet_variance, ln_pdet
                    ) = log_likelihood(parameters)
                    return dict(
                        ln_lkl=lnl,
                        ln_pdet=ln_pdet,
                        ln_pdet_variance=ln_pdet_variance,
                        pe_variance=pe_variance,
                        variance=var
                    )

            self.generate_extra_statistics = jax.jit(extras)

        def log_likelihood_ratio(self, parameters=None):
            return self.log_likelihood_ratio_func(parameters)

        def noise_log_likelihood(self):
            return jnp.nan

        def log_likelihood(self, parameters=None):
            return self.log_likelihood_ratio(parameters=parameters)

    return LikelihoodWrapper()



def likelihood_extras(key, parameters, likelihood):
    """
    adapted from some code from matt;
    for use with gwpop style runs
    """
    
    extras = likelihood.generate_extra_statistics(parameters)

    vt = likelihood.selection_function.surveyed_hypervolume(parameters)
    nexp = jax.random.gamma(key, likelihood.n_posteriors)

    # K in eq. 12 of 2602.20277
    # rate of mergers (obsv'd and unobsv'd) per unit time
    analysis_time = likelihood.selection_function.analysis_time
    rate = nexp / extras['selection'] / analysis_time
    
    # K times integral over z of (1 / 1+z * dVc/dz)**(-1) * p(z)
    # and we basically always take p(z) \propto (1 + z)**(lamb_z)
    volumetric_rate = nexp / extras['selection'] / vt

    max_variance = jnp.array(
        [extras[f'var_{i}'] for i in range(likelihood.n_posteriors)]
    ).max()
    min_neff = 1 / (
        max_variance + 1 / likelihood.samples_per_posterior
    )

    selection_neff = 1 / (
        extras['selection_variance'] / likelihood.n_posteriors**2
        + 1 / likelihood.samples_per_posterior
    )

    return dict(
        rate=rate,
        volumetric_rate=volumetric_rate,
        min_neff=min_neff,
        selection_neff=selection_neff,
        **extras
    )