import jax
import jax.numpy as jnp
from util import scan


def mean_and_variance(weights, n):
    # mean and variance of the mean
    mean = jnp.sum(weights) / n
    variance = jnp.sum(weights**2) / n**2 - mean**2 / n
    return mean, variance


def ln_mean_and_variance(weights, n):
    # lazy ln(mean) and variance of ln(mean)
    mean, variance = mean_and_variance(weights, n)
    return jnp.log(mean), variance / mean**2


def event_ln_likelihoods_and_selection(
    posteriors, injections, density, parameters
):
    pe_weights = density(posteriors, parameters) / posteriors['prior']
    vt_weights = density(injections, parameters) / injections['prior']

    _, npe = pe_weights.shape
    ln_lkls, pe_variances = jax.vmap(
        lambda weights: ln_mean_and_variance(weights, npe)
    )(pe_weights)

    pdet, pdet_variance = mean_and_variance(
        vt_weights, injections['total_generated']
    )

    return ln_lkls, pdet, pe_variances, pdet_variance


def shape_ln_likelihood_and_variance(
    posteriors, injections, density, parameters
):
    (
        ln_lkls, pdet, pe_variances, pdet_variance
    ) = event_ln_likelihoods_and_selection(
        posteriors, injections, density, parameters
    )

    pe_variance = jnp.sum(pe_variances)

    nobs = len(ln_lkls)

    ln_pdet = jnp.log(pdet)
    ln_pdet_variance = pdet_variance / pdet**2

    ln_lkl = jnp.sum(ln_lkls) - ln_pdet * nobs
    variance = pe_variance + ln_pdet_variance * nobs**2
    return ln_lkl, variance, pe_variance, ln_pdet_variance, ln_pdet


def taper(maximum_variance, v):
    return -100 * (maximum_variance - v)**2 * (v >= maximum_variance)


def get_bilby_likelihood(
    model,
    posteriors,
    injections,
    taper=lambda _: 0,
    rate=False,
):
    """
    return an instance of a bilby.core.likelihood.Likelihood
    using my own jit-compiled likelihoods.

    Yeah yeah, we should define a generic class which we can instantiate
    given different input arguments. However, bilby is slow to import
    and I don't want to import it every time I call `likelihood.py`.
    """

    from bilby.core.likelihood import Likelihood

    def log_likelihood(parameters):
        return shape_ln_likelihood_and_variance(posteriors, injections, model, parameters)

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