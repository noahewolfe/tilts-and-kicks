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


def shape_likelihood_extras(posteriors, injections, density, parameters):
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

    return dict(
        ln_lkl=ln_lkl,
        ln_pdet=ln_pdet,
        ln_pdet_variance=ln_pdet_variance,
        pe_variance=pe_variance,
        variance=variance
    )


def rate_ln_likelihood_and_variance(
    tobs, posteriors, injections, density, parameters
):
    # TODO: assumes density is till the shape density, and that rate is
    # passed in as part of the parameters.
    # May not work if we use something like pixelpop.
    (
        ln_lkls, pdet, pe_variances, pdet_variance
    ) = event_ln_likelihoods_and_selection(
        posteriors, injections, density, parameters
    )

    nobs = len(ln_lkls)
    ln_rate = parameters['ln_rate']
    rate = jnp.exp(ln_rate)

    ln_lkl = jnp.sum(ln_lkls) + nobs * ln_rate - rate * tobs * pdet
    variance = jnp.sum(pe_variances) + (rate * tobs)**2 * pdet_variance

    """
    rate_density = lambda x, p: p['rate'] * density(x, p)
    pe_weights = rate_density(posteriors, parameters) / posteriors['prior']
    vt_weights = rate_density(injections, parameters) / injections['prior']

    nobs, npe = pe_weights.shape
    lkls = jnp.sum(pe_weights, axis=-1) / npe
    pe_vars = jnp.sum(pe_weights**2, axis=-1) / npe**2 - lkls**2 / npe

    ln_lkls = jnp.log(lkls)
    ln_pe_vars = pe_vars / lkls**2

    ninj = injections['total_generated']
    nexp = tobs * jnp.sum(vt_weights) / ninj
    nexp_var = tobs**2 * jnp.sum(vt_weights**2) / ninj**2 - nexp**2 / ninj

    ln_lkl = jnp.sum(ln_lkls) - nexp

    ln_pe_var = jnp.sum(ln_pe_vars)
    variance = ln_pe_var + nexp_var
    """

    return ln_lkl, variance


def rate_likelihood_extras(tobs, posteriors, injections, density, parameters):
    (
        ln_lkls, pdet, pe_variances, pdet_variance
    ) = event_ln_likelihoods_and_selection(
        posteriors, injections, density, parameters
    )

    nobs = len(ln_lkls)
    ln_rate = parameters['ln_rate']
    rate = jnp.exp(ln_rate)

    ln_lkl = jnp.sum(ln_lkls) + nobs * ln_rate - rate * tobs * pdet
    pe_variance = jnp.sum(pe_variances)
    variance = pe_variance + (rate * tobs)**2 * pdet_variance

    return dict(
        ln_lkl=ln_lkl,
        pdet=pdet,
        pdet_variance=pdet_variance,
        pe_variance=pe_variance,
        variance=variance
    )


def taper(maximum_variance, v):
    return -100 * (v - maximum_variance)**2 * (v >= maximum_variance)


def compute_extras(log_likelihood, samples):
    def fn(x):
        (
            ln_lkl,
            variance,
            pe_variance,
            ln_pdet_variance,
            ln_pdet
        ) = log_likelihood(x)
        return dict(
            log_likelihood=ln_lkl,
            variance=variance,
            pe_variance=pe_variance,
            ln_pdet_variance=ln_pdet_variance,
            ln_pdet=ln_pdet
        )
    return scan(fn, desc='extras')(samples)


def compute_prior_fraction(
    key, prior, log_likelihood, maximum_uncertainty, n=10_000
):
    keys = jax.random.split(key, n)
    samples = jax.vmap(prior.sample)(keys)
    extras = compute_extras(log_likelihood, samples)

    w = extras['variance'] < maximum_uncertainty
    frac = w.mean()
    error = ((jnp.mean(w**2) - frac**2) / n)**0.5

    return frac, error
