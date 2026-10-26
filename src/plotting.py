from matplotlib.ticker import MaxNLocator, FixedLocator, FuncFormatter

def physical_axes(ax, pixel_size, unit='mm', scale=1, origin=-0.5, nbins=5, fontsize=7):
    '''Ticks in physical units on a pixel image.
    pixel_size in mm; scale=1 -> mm, scale=1000 -> µm. origin = coordinate (px) taken as 0.
    Call AFTER setting the axis limits (zoom).'''
    k = pixel_size * scale
    for axis, lims in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
        lo, hi = sorted(lims)
        phys = MaxNLocator(nbins).tick_values((lo - origin) * k, (hi - origin) * k)
        ticks_px = phys / k + origin
        axis.set_major_locator(FixedLocator(ticks_px[(ticks_px >= lo) & (ticks_px <= hi)]))
        axis.set_major_formatter(FuncFormatter(lambda v, pos: f'{round((v - origin) * k, 6):g}'))
    ax.set_xlabel(f'x [{unit}]', fontsize=fontsize)
    ax.set_ylabel(f'y [{unit}]', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)

def zoom(ax, cx, cy, half):
    '''Square zoom centred on (cx, cy) in pixel coordinates (y axis inverted, as in images).'''
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy + half, cy - half)
