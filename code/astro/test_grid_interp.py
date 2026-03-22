"""Compare grid-interpolated mass_dynamical against direct KDE evaluation."""
import time

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from models import mass_dynamical, mass_dynamical_kde

parameters = dict(
    mprog_min=3,
    mprog_max=100,
    mprog_break=50,
    alpha_prog_1=2.1,
    alpha_prog_2=1.1,
    mbhmax=50.0,
    mturnover=35.0,
    beta_1g1g=1.5,
    beta_1g2g=1.5,
)

# synthetic mass data: ~70 events x 1000 posterior samples (realistic size)
key = jax.random.key(42)
k1, k2 = jax.random.split(key)
mass_1 = jax.random.uniform(k1, (70, 1000), minval=5, maxval=80)
mass_2 = jax.random.uniform(k2, (70, 1000), minval=3, maxval=50)
mass_1, mass_2 = jnp.maximum(mass_1, mass_2), jnp.minimum(mass_1, mass_2)

print(f"Data shape: {mass_1.shape}  ({mass_1.size} points)")

# both methods build KDEs from the same fixed seeds, so we can compare
# on a single shared KDE build to isolate interpolation error
print("\n--- Warmup (JIT compile) ---")
t0 = time.time()
p_grid = mass_dynamical(mass_1, mass_2, parameters)
for a in p_grid:
    a.block_until_ready()
print(f"Grid (first call):  {time.time() - t0:.2f}s")

t0 = time.time()
p_kde = mass_dynamical_kde(mass_1, mass_2, parameters)
for a in p_kde:
    a.block_until_ready()
print(f"KDE  (first call):  {time.time() - t0:.2f}s")

print("\n--- Timed runs (post-JIT) ---")
n_repeats = 5

t0 = time.time()
for _ in range(n_repeats):
    p_grid = mass_dynamical(mass_1, mass_2, parameters)
    for a in p_grid:
        a.block_until_ready()
dt_grid = (time.time() - t0) / n_repeats
print(f"Grid: {dt_grid:.4f}s per call")

t0 = time.time()
for _ in range(n_repeats):
    p_kde = mass_dynamical_kde(mass_1, mass_2, parameters)
    for a in p_kde:
        a.block_until_ready()
dt_kde = (time.time() - t0) / n_repeats
print(f"KDE:  {dt_kde:.4f}s per call")
print(f"Speedup: {dt_kde / dt_grid:.1f}x")

print("\n--- Accuracy comparison ---")
for name, g, k in [("1g1g", p_grid[0], p_kde[0]), ("1g2g", p_grid[1], p_kde[1])]:
    n_nan_g = jnp.sum(~jnp.isfinite(g))
    n_nan_k = jnp.sum(~jnp.isfinite(k))
    print(f"{name}:")
    print(f"  NaN/Inf count: grid={n_nan_g}, kde={n_nan_k}")
    print(f"  grid range: [{jnp.nanmin(g):.2e}, {jnp.nanmax(g):.2e}]")
    print(f"  kde  range: [{jnp.nanmin(k):.2e}, {jnp.nanmax(k):.2e}]")
    finite = jnp.isfinite(g) & jnp.isfinite(k)
    if finite.sum() > 0:
        g_f, k_f = g[finite], k[finite]
        abs_err = jnp.abs(g_f - k_f)
        peak = jnp.max(jnp.abs(k_f))
        mask = jnp.abs(k_f) > 1e-6 * peak
        rel_err = jnp.where(mask, abs_err / jnp.abs(k_f), 0.0)
        print(f"  finite values: {finite.sum()}/{g.size}")
        print(f"  max abs error:  {jnp.max(abs_err):.2e}")
        print(f"  max rel error (non-negligible):  {jnp.max(rel_err):.2e}")
        print(f"  mean abs error: {jnp.mean(abs_err):.2e}")
    else:
        print(f"  no finite values to compare!")
