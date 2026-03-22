def scan(fn, desc=None):
    """ A simple jax.lax.scan wrapper with progress bar. """
    import jax
    import jax.numpy as jnp
    from jax_tqdm import scan_tqdm

    def step(_, d):
        _, x = d
        return None, fn(x)

    def tracked(xs):
        n = len(xs)
        _, ys = jax.lax.scan(
            scan_tqdm(n, desc=desc)(step),
            None,
            (jnp.arange(n), xs)
        )
        return ys

    return tracked