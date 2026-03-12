"""Inspect high-mass smoothing in BrokenPowerlawPlusTwoPeaks_PrimaryMass_LowHighSmooth."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import jax.numpy as jnp

from models import BrokenPowerlawPlusTwoPeaks_PrimaryMass_LowHighSmooth
from pixelpop.models.gwpop_models import m_smoother

plt.style.use('../main.mplstyle')

# Fixed parameters
alpha_1 = 3.0
alpha_2 = 3.0
mlow_1 = 4.0
delta_m_1 = 5.0
break_mass = 30.0
mpp_1 = 10.0
sigpp_1 = 0.8
mpp_2 = 35.0
sigpp_2 = 5.0
lam_fractions = (0.5, 0.25, 0.25)
gaussian_mass_maximum = 100.0

# Varied parameters
mmax_values = [45, 100, 300]
delta_max_values = np.linspace(0, 10, 11)

# Evaluation grid (matches identifiable-gwpop.py PPD settings)
m1_grid = jnp.linspace(3.0, 300.0, 1000)

# Colormap
cmap = cm.viridis
norm = plt.Normalize(vmin=0, vmax=10)

fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

for ax, mmax in zip(axes, mmax_values):
    for delta_max in delta_max_values:
        log_p = BrokenPowerlawPlusTwoPeaks_PrimaryMass_LowHighSmooth(
            m1_grid, alpha_1, alpha_2, mlow_1, break_mass, delta_m_1,
            lam_fractions, mpp_1, sigpp_1, mpp_2, sigpp_2,
            delta_max, mmax=mmax, gaussian_mass_maximum=gaussian_mass_maximum,
        )
        p = np.array(jnp.exp(log_p))
        ax.plot(m1_grid, p, color=cmap(norm(delta_max)), lw=1.2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(3, 300)
    ax.set_ylim(1e-6, 1e0)
    ax.set_xlabel(r'$m_1$ [$\mathrm{M}_\odot$]')
    ax.set_title(rf'$m_{{\max}} = {mmax}\ \mathrm{{M}}_\odot$')

axes[0].set_ylabel(r'$p(m_1)$ [$\mathrm{M}_\odot^{-1}$]')

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=axes, label=r'$\delta_{\max}$ [$\mathrm{M}_\odot$]')

fig.savefig('high_mass_smoothing.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('Saved high_mass_smoothing.png')

# --- m_smoother alone ---
# The model uses m_smoother(-m1, -mmax, delta_max) for the high-mass taper.
fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

for ax, mmax in zip(axes2, mmax_values):
    for delta_max in delta_max_values:
        log_s = m_smoother(-m1_grid, -mmax, delta_max)
        s = np.array(jnp.exp(log_s))
        ax.plot(m1_grid, s, color=cmap(norm(delta_max)), lw=1.2)

    ax.set_xscale('log')
    ax.set_xlim(3, 300)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r'$m_1$ [$\mathrm{M}_\odot$]')
    ax.set_title(rf'$m_{{\max}} = {mmax}\ \mathrm{{M}}_\odot$')

axes2[0].set_ylabel(r'$\exp\left[\mathrm{m\_smoother}(-m_1, -m_{\max}, \delta_{\max})\right]$')

sm2 = cm.ScalarMappable(cmap=cmap, norm=norm)
fig2.colorbar(sm2, ax=axes2, label=r'$\delta_{\max}$ [$\mathrm{M}_\odot$]')

fig2.savefig('high_mass_smoother.png', bbox_inches='tight', dpi=150)
plt.close(fig2)
print('Saved high_mass_smoother.png')
