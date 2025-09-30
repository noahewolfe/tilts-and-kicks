from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import Normalize


def make_segmented_colorbar_legend(
    fig,
    ax,
    colors,
    loc,
    labels=[r'$10^{' + str(i) + r'}$' for i in [-5, -3, 0]],
    mirror_labels=None,
    mirror_label_rotation=0,
    width='30%',
    height='5%',
    ylabel='O3',
    top_xlabel=None,
    top_xlabel_coords=None,
    bottom_xlabel=r'$\rm FAR_*$ [$\mathrm{yr}^{-1}$]',
    bottom_xlabel_coords=None,
    x0=0,
    y0=-0.1,
    xticklabelpad=5,
    ylabel_pad=20,
    return_ax=False,
    orientation='horizontal'
):

    n = len(labels)

    cnorm = Normalize(vmin=0, vmax=1)
    cmap = ListedColormap(colors)
    sm = ScalarMappable(cnorm, cmap)

    cax = inset_axes(
        ax,
        width=width,
        height=height,
        loc=loc,
        bbox_to_anchor=(x0, y0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0.5
    )
    fig.colorbar(sm, cax=cax, orientation=orientation)

    major_ticks = [
        i / n / 2
        for i in range(0, 2 * n + 1)
        if i % 2 != 0
    ]

    if orientation == 'horizontal':
        cax.set_xticks([], minor=False)
        cax.set_xticks(major_ticks, minor=True)
        cax.set_xticklabels(labels, minor=True)
    else:
        cax.set_yticks([], minor=False)
        cax.set_yticks(major_ticks, minor=True)
        cax.set_yticklabels(labels, minor=True)

    if bottom_xlabel is not None:
        cax.set_xlabel(bottom_xlabel)

        if bottom_xlabel_coords is not None:
            cax.xaxis.set_label_coords(*bottom_xlabel_coords)

    if top_xlabel is not None:
        cax_twin.set_xlabel(top_xlabel)

        if top_xlabel_coords is not None:
            print(top_xlabel_coords)
            cax_twin.xaxis.set_label_coords(*top_xlabel_coords)

    if ylabel is not None:
        cax.set_ylabel(ylabel, rotation=0, labelpad=ylabel_pad, va='center')

    cax.tick_params(axis='x', which='both', length=0, pad=xticklabelpad)

    if 'left' in loc:
        cax.yaxis.set_label_position('right')

    if return_ax:
        return cax


def add_ppd(
    x,
    ppd,
    ax,
    color,
    label=None,
    linestyle=['--', ':'],
    fill=False,
    quantiles=[(0.05, 0.95), (0.005, 0.995)],
    alpha=None,
    median=True,
    xp=None,
    use_hdi=False,
    zorder=None,
    weights=None,
    hatch=None
):
    if xp is None:
        import jax.numpy as xp

    if use_hdi:
        from arviz import hdi

    if len(quantiles) == 1:
        if not isinstance(fill, list):
            fill = [fill]
        if not isinstance(alpha, list) or alpha is None:
            alpha = [alpha]
        if not isinstance(linestyle, list):
            linestyle = [linestyle]
    elif len(quantiles) > 1:
        if not isinstance(fill, list):
            fill = [fill] * len(quantiles)
        if not isinstance(alpha, list) or alpha is None:
            alpha = [alpha] * len(quantiles)
        if not isinstance(linestyle, list):
            linestyle = [linestyle] * len(quantiles)
    else:
        raise ValueError('no quantiles given')

    if median:
        if weights is not None:
            median = xp.quantile(
                ppd, 0.5, weights=weights, method='inverted_cdf', axis=0
            )
        else:
            median = xp.median(ppd, axis=0)
        ax.plot(x, median, label=label, color=color)

    for ((a, b), fi, li, al) in zip(quantiles, fill, linestyle, alpha):
        if use_hdi:
            if weights is not None:
                raise NotImplementedError('hdi with weights not implemented')
            qa, qb = hdi(ppd, hdi_prob=b - a).T
        else:
            if weights is not None:
                qa = xp.quantile(ppd, a, weights=weights, method='inverted_cdf', axis=0)
                qb = xp.quantile(ppd, b, weights=weights, method='inverted_cdf', axis=0)
            else:
                qa = xp.quantile(ppd, a, axis=0)
                qb = xp.quantile(ppd, b, axis=0)

        if fi:
            ax.fill_between(
                x, qa, qb, alpha=al, color=color, label=label, linewidth=0,
                zorder=zorder, hatch=hatch
            )
        else:
            ax.plot(x, qa, linestyle=li, color=color, zorder=zorder, alpha=al)
            ax.plot(x, qb, linestyle=li, color=color, zorder=zorder, alpha=al)
