import cmcrameri.cm as cmc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from pyffit.data import read_traces
from matplotlib.patches import Ellipse


stem       = '/Users/evavra/Projects/Taiwan/ALOS2/A139/F4/'
intf_paths = [
              stem + 'intf/20220807_20220918', 
              stem + 'iono_phase/intf_h/20220807_20220918',
              stem + 'iono_phase/intf_l/20220807_20220918',
              stem + 'iono_phase/intf_o/20220807_20220918']
    
data_type = 'phasefilt'
labels    = ['intf', 'intf_h', 'intf_l', 'intf_o']
mask      = False  # Apply landmask to plots
corr_min  = 0.05   # Apply coherence cutoff
grid_dims = (1, 4) #
region    = []
cmap      = cmc.roma
figsize   = (14, 8.2)


def intf_panels(intf_paths, data_type, labels, mask=False, corr_min=None, grid_dims=None, figsize=(14, 8.2), cmap=cmc.roma):
    """
    Compare the same interferometric product for different pairs.

    INPUT:
    intf_paths - paths to interferogram directories
    data_type  - product to plot (e.g., phasefilt, unwrap)
    
    OUTPUT:
    fig  - figure object
    grid - subplot grid
    """

    # Load interferograms
    intfs = []
    for intf_path in intf_paths:
        intfs.append(Interferogram(intf_path))

    # Create colorbar
    var  = np.array([[np.nanmin(intf.data[data_type]), np.nanmax(intf.data[data_type])] for intf in intfs])
    vmin = var.min()
    vmax = var.max()
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=var.min(), vmax=var.max()))

    # Make plot
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=grid_dims,  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     cbar_mode="single",
                     )

    # Loop over interferograms/subplots
    for i, ax in enumerate(grid):
        z      = intfs[i].data[data_type]
        extent = intfs[i].extent

        if corr_min != None:
            z[intfs[i].data['corr'] < corr_min] = np.nan

        if mask:
            z[intfs[i].data['landmask'] != 1] = np.nan # NOTE: modify for case of ll coordinates

        im = ax.imshow(z, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none', zorder=0)
        ax.invert_yaxis()
        ax.set_title(labels[i])
        ax.set_aspect(0.25)

        if len(region) == 4:
            ax.set_xlim(region[:2])
            ax.set_ylim(region[2:])

    grid.cbar_axes[0].colorbar(sm, label='Radians')

    plt.tight_layout()
    plt.show()
    return


def plot_map(data, faults, region):
    """
    Make horizointal velocity field map with faults.
    """

    fig = plt.figure(figsize=(14, 8.2))
    projection = ccrs.PlateCarree()
    ax  = fig.add_subplot(1, 1, 1, projection=projection)

    for fault in faults:
        ax.plot(fault[:, 0], fault[:, 1], linewidth=1, c='k')

    ax.quiver(data['LON'], data['LAT'], data['VE'], data['VN'], scale=20, scale_units='inches')


    ax.add_feature(cfeature.OCEAN.with_scale('10m'))
    ax.add_feature(cfeature.LAND.with_scale('10m'), color='gainsboro')
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.LAKES.with_scale('10m'))

    ax.set_xlim(region[:2])
    ax.set_ylim(region[2:])

    xinc = 0.5
    yinc = 0.5

    xticks = np.arange(region[0], region[1] + xinc, xinc)
    yticks = np.arange(region[2], region[3] + yinc, yinc)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    plt.show()
    
    return


def plot_horiz_field(fields, region=[], crs=ccrs.PlateCarree(), features=['land', 'ocean', 'coastline', 'lakes'], c=None, scale=40, l=20, tick_inc=1, faults={}, 
                     fault_x='Longitude', fault_y='Latitude', fault_crs=ccrs.PlateCarree(), fault_width=2, fault_colors=[], T=[], swath={}, T_swath=[], 
                     text_dict={}, cbar_dict={}, title='', cmap='viridis', fig=None, ax=None, show=False, out_name='', dpi=500, figsize=(1.5*6.4, 1.5*4.8), width=3e-2, 
                     legend_kwargs={}, legend=False, quiv_key_pos=[0.9, 1.05], headwidth=3):
    """
    Plot horizontal velocity field.
    """
    
    if fig == None:
        fig = plt.figure(figsize=figsize)
    if ax == None:
        ax = fig.add_subplot(1, 1, 1, projection=crs)

    # Add map features
    if 'land' in features:
        ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='gainsboro', edgecolor='black')
    if 'ocean' in features:
        ax.add_feature(cfeature.OCEAN.with_scale('10m'))
    if 'coastline' in features:
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    if 'lakes' in features:
        ax.add_feature(cfeature.LAKES.with_scale('10m'))

    # Get fault colors
    if (len(faults) > 0) & (len(fault_colors) == len(faults)):
        if type(fault_colors[0]) == str:
            c = fault_colors
        else:
            cmap_name  = cmap # Colormap to use
            cbar_label = 'Slip rate (mm/yr)'
            var        = fault_colors
            n_seg      = 5
            ticks      = np.linspace(var.min(), var.max(), n_seg + 1)
            alpha      = 1
            
            # Create colorbar
            cval  = (var - var.min())/(var.max() - var.min()) # Normalized color values
            cmap  = colors.LinearSegmentedColormap.from_list(cmap_name, plt.get_cmap(cmap_name, 265)(np.linspace(0, 1, 265)), n_seg)
            sm    = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=var.min(), vmax=var.max()))
            c     = cmap(cval)
            cbar  = plt.colorbar(sm, label=cbar_label)
            cbar.set_ticks(ticks)
    else:
        c = ['gray' for i in range(len(faults))]

    # Plot faults
    if len(faults) > 0:
        if type(faults) == dict:
            for i, name in enumerate(faults):
                fault = faults[name]
                ax.plot(fault[fault_x], fault[fault_y], c=c[i], linewidth=fault_width, transform=fault_crs, zorder=10, label=fault['Name'] + f'')
            if legend:
                ax.legend(**legend_kwargs)
        else:
            for i, fault in enumerate(faults):
                ax.plot(fault[fault_x], fault[fault_y], c=c[i], linewidth=fault_width, transform=fault_crs, zorder=10)

    # Plot swath bbox
    if len(swath) > 0:
        ax.plot(swath['lon'], swath['lat'], c='k', transform=crs, zorder=20)

    # Plot velocity fields
    for i, field in enumerate(fields):
        quiv = ax.quiver(field['x'], field['y'], field['dx'], field['dy'], color=field['color'], width=width, headwidth=headwidth, 
                         units='xy', angles='xy', scale=scale, transform=crs, label=field['label'], zorder=30 - i)      # gnss velocity field
    
        # Plot error ellipses if specified
        if ('sigma_x' in field.keys()) & ('sigma_y' in field.keys()):
            if (len(field['sigma_x']) == len(field['sigma_y'])) & (len(field['sigma_x']) == len(field['x'])):
                print('yeet')
                ax = add_error_ellipses(ax, quiv, field['sigma_x'].values, field['sigma_y'].values, crs, scale, color=field['color'], zorder=30 - i)


    # Add velocity key
    qkey = ax.quiverkey(quiv, quiv_key_pos[0], quiv_key_pos[1], l, f'{l} mm/yr', labelpos='E', coordinates='axes', transform=crs)

    # Add text
    if len(text_dict) > 0:
        ax.text(text_dict['x'], text_dict['y'], text_dict['str'], family='monospace', fontsize=12, transform=ax.transAxes)

    # Fix region and ticks
    if len(region) == 4:
        xticks = np.arange(np.ceil(region[0]), np.floor(region[1]) + tick_inc, tick_inc)
        yticks = np.arange(np.ceil(region[2]), np.floor(region[3]) + tick_inc, tick_inc)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xlim(region[:2])
        ax.set_ylim(region[2:])

    # Add figure title
    if len(title) > 0:
        ax.set_title(title)

    # Add colorbar 
    if len(cbar_dict) > 0:
        cbar = fig.colorbar(cbar_dict['sm'], label=cbar_dict['label'])
        cbar.set_ticks(cbar_dict['ticks'])

    if len(out_name) > 0:
        fig.savefig(out_name + '.png', dpi=dpi)
    if show:
        plt.show()

    return fig, ax


def plot_errors(errors, subset=[], bins=30, fig_name=''):
    """
    Plot error terms associated with fault model.
    """
    labels = [r'$R$ (mm/yr)', r'$R_{AMP}$ (mm/yr)']
    scales = [1000, 1, 1000, 1]
    fig, axes = plt.subplots(1, len(labels), figsize=(6, 3))

    for ax, key, label, scale in zip(axes, ['r', 'r_amp'], labels, scales):

        d   = scale*errors[key]
        dlim = [np.min(d), np.max(d)]

        ax.hist(d, bins=bins, range=dlim, color='k')

        if len(subset) > 0:
            ax.hist(d[subset], bins=bins, range=dlim, color='C3')
        ax.set_xlabel(label)
        ax.set_xlim(dlim)

    fig.tight_layout()
    
    if len(fig_name) > 0:
        fig.savefig(f'{fig_name}.png', dpi=500)
    # plt.show()
    
    return fig, axes


def add_error_ellipses(ax, quiv, sigma_x, sigma_y, crs, scale, alpha=1, color='C0', zorder=0):
    """
    Add error ellipses to a quiver plot.

    NOTE: Quiver plot units and angles must be 'xy' to work!
    
    INPUT:
    ax      - figure axes object to plot on
    quiv    - plt.quiver object
    sigma_x - x-direction uncertainties
    sigma_y - y-direction uncertainties
    
    OUTPUT:
    ax      - updated axes object
    """

    # Get arrow tips coordinates
    arrow_x = quiv.X + quiv.U/scale
    arrow_y = quiv.Y + quiv.V/scale

    # Plot ellipses
    for i in range(arrow_x.size):

        ell = Ellipse(xy=(arrow_x[i], arrow_y[i]), 
                      width=2*sigma_x[i]/scale, height=2*sigma_y[i]/scale, 
                      alpha=alpha, edgecolor=color, 
                      transform=crs, fill=False, zorder=zorder)

        ax.add_patch(ell)

    return ax

