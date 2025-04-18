import corner
import matplotlib
import numpy as np
import pandas as pd
import cmcrameri.cm as cmc
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from pyffit.data import read_traces
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.patches import Polygon, Rectangle, Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.transforms import Affine2D


stem       = '/Users/evavra/Projects/Taiwan/ALOS2/A139/F4/'
intf_paths = [
              stem + 'intf/20220807_20220918', 
              stem + 'iono_phase/intf_h/20220807_20220918',
          
              stem + 'iono_phase/intf_o/20220807_20220918']
    
data_type = 'phasefilt'
labels    = ['intf', 'intf_h', 'intf_l', 'intf_o']
mask      = False  # Apply landmask to plots
corr_min  = 0.05   # Apply coherence cutoff
grid_dims = (1, 4) #
region    = []
cmap      = cmc.roma
figsize   = (14, 8.2)


def plot_rotated_fault(fault, x, y, data, mask=[], vlim=[0, 1], cmap='viridis', title='', label='', sites=[], xlim=[-20, 30], ylim=[-10, 10], site_color='C0', s=1, marker='.',  file_name='', show=True, dpi=300):
    """
    Plot data in fault-oriented coordinate system.
    """

    fig, ax = plt.subplots(figsize=(14, 8.2), constrained_layout=True)
    transform = Affine2D().rotate_deg_around(0, 0, fault.avg_strike + 90) + ax.transData

    ax.plot(fault.trace[:, 0] - fault.origin_r[0], fault.trace[:, 1] - fault.origin_r[1], c='k', zorder=0, transform=transform)
    im = ax.scatter(x - fault.origin_r[0], y  - fault.origin_r[1], c=data, vmin=vlim[0], vmax=vlim[1], marker=marker, s=s, cmap=cmap, transform=transform)

    if len(sites) > 0:
        ax.scatter(sites['x'] - fault.origin_r[0], sites['y']  - fault.origin_r[1], facecolor=site_color, edgecolor='k', s=50, transform=transform, zorder=100)

        for i, site in sites.iterrows():
            ax.annotate(site['name'], [site['x'] - fault.origin_r[0], site['y']  - fault.origin_r[1]], xycoords=transform, zorder=1000)

    ax.set_title(title)
    ax.set_aspect(1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_facecolor('gainsboro')
    fig.colorbar(im, label=label, shrink=0.5)

    if len(file_name) > 0:
        fig.savefig(file_name, dpi=dpi)
        plt.close()
        
    if show:
        plt.show()

    return fig, ax


def animation():
    # Define the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("Animated Sine Wave")
    ax.set_xlabel("x")
    ax.set_ylabel("sin(x)")

    # Create a line object that will be updated during the animation
    line, = ax.plot([], [], lw=2)

    # Define the initialization function
    def init():
        line.set_data([], [])
        return line,

    # Define the update function for each frame
    def update(frame):
        x = np.linspace(0, 2 * np.pi, 1000)
        y = np.sin(x + frame * 0.1)  # Add frame-dependent phase shift
        line.set_data(x, y)
        return line,

    # Create the animation
    frames = 100  # Number of frames
    interval = 50  # Delay between frames in ms
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

    # Save the animation as an MP4
    output_file = "sine_wave_animation.mp4"
    writer = FFMpegWriter(fps=20, metadata=dict(artist="Matplotlib"), bitrate=1800)
    ani.save(output_file, writer=writer)

    print(f"Animation saved as {output_file}")

    return


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


def plot_field(data, faults, region):
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


def plot_horiz_field(fields, region=[], crs=ccrs.PlateCarree(), features=['land', 'ocean', 'coastline', 'lakes'], c=None, 
                     scale=40, scale_length=20, tick_inc=1, faults={}, fault_x='Longitude', fault_y='Latitude', fault_crs=ccrs.PlateCarree(), 
                     fault_width=2, fault_colors=[], T=[], swath={}, T_swath=[], text_dict={}, cbar_dict={}, title='', cmap='viridis', 
                     fig=None, ax=None, show=False, out_name='', dpi=500, figsize=(1.5*6.4, 1.5*4.8), width=3e-2, legend_kwargs={}, 
                     legend=False, quiv_key_pos=[0.9, 1.05], headwidth=3):
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
            cmap  = matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, plt.get_cmap(cmap_name, 265)(np.linspace(0, 1, 265)), n_seg)
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
                ax.plot(fault[fault_x], fault[fault_y], c=fault['color'], linewidth=fault_width, transform=fault_crs, zorder=10, label=fault['Name'] + f'')
            if legend:
                ax.legend(**legend_kwargs)
        else:
            for i, fault in enumerate(faults):
                ax.plot(fault[fault_x], fault[fault_y], c=fault['color'], linewidth=fault_width, transform=fault_crs, zorder=10)

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
                ax = add_error_ellipses(ax, quiv, field['sigma_x'].values, field['sigma_y'].values, crs, scale, color=field['color'], zorder=30 - i)


    # Add velocity key
    qkey = ax.quiverkey(quiv, quiv_key_pos[0], quiv_key_pos[1], scale_length, f'{scale_length} mm/yr', labelpos='E', coordinates='axes', transform=crs)

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


def plot_grid_map(data, extent, region=[], fig_ax=(), projection=ccrs.PlateCarree(), vlim=[], x_tick_inc=0, y_tick_inc=0, 
                  cmap='coolwarm', cbar_label='', title='',
                  dpi=500, show=False, file_name='', cbar=False):
    """
    Make map of gridded data.
    """

    if len(fig_ax) == 0:
        fig = plt.figure(figsize=(14, 8.2))
        ax  = fig.add_subplot(1, 1, 1, projection=projection)
    else:
        fig, ax = fig_ax

    if len(vlim) == 0:
        vlim = [np.nanmin(data), np.nanmax(data)]

    if len(title) > 0:
        ax.set_title(title)

    # Plot features
    ax.add_feature(cfeature.LAND.with_scale('10m'), color='white')
    ax.add_feature(cfeature.LAKES.with_scale('10m'))

    # Plot data
    im = ax.imshow(data, extent=extent, transform=projection, cmap=cmap, interpolation='none', vmin=vlim[0], vmax=vlim[1])
        
    # Fault stuff
    # ax.plot(faults[:, 0], faults[:, 1],               c='gray',    linewidth=1, transform=projection,)
    # ax.plot(qfaults[:, 0], qfaults[:, 1],             c='gray', lw=1, transform=projection,)
    # ax.plot(trace['Longitude'], trace['Latitude'],    c='k',    lw=2, transform=projection, zorder=10)
    # ax.scatter(insar_data['Longitude'], insar_data['Latitude'], c='k',    s=10, transform=projection, zorder=10)

    # Axes settings
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    if len(region) > 0:
        ax.set_xlim(region[:2])
        ax.set_ylim(region[2:])

        if x_tick_inc > 0:        
            ax.set_xticks(np.arange(region[0], region[1] + x_tick_inc, x_tick_inc))

        if y_tick_inc > 0:
            ax.set_yticks(np.arange(region[2], region[3] + y_tick_inc, y_tick_inc))


    if len(cbar_label) > 0:    
        fig.colorbar(im, ax=ax, label=cbar_label, orientation='horizontal', pad=0.05)

    if len(file_name) > 0:
        fig.savefig(file_name, dpi=dpi)
    
    if show:
        plt.show()


    return


def plot_fault_3d(mesh, triangles, c=[], fig_ax=[], edges=False, cmap_name='viridis', cbar_label='Slip (m)', 
                  labelpad=20, azim=45, elev=10, n_seg=100, n_tick=11, alpha=1, vlim_slip=[], title='', 
                  file_name='', show=True, dpi=500, invert_zaxis=False, edge_kwargs=dict(edgecolor='k', linewidth=0.25),
                  figsize=(14, 8.2), cbar_kwargs=dict(location='bottom', pad=-0.1, shrink=0.5)):
    """
    Make 3D plot of finite fault model.

    INPUT:
    mesh (M, 3)      - array of coordinates (x,y,z) of triangular mesh verticies
    triangles (N, 3) - array containing vertex indices of each triangle in the mesh. Each row corresponds to one triangle
    """

    # Make 3D plot

    if len(fig_ax) != 2:
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(projection='3d')
    else:
        fig = fig_ax[0]
        ax = fig_ax[1]

    fig.subplots_adjust(top=1.2, bottom=-.1)
    
    # Plot faces
    if len(c) > 0:
        # Create colorbar for slip patches
        if len(vlim_slip) == 0:
            vmin = np.min(c)
            vmax = np.max(c)
        else:
            vmin = vlim_slip[0]
            vmax = vlim_slip[1]

        cvar  = c
        ticks = np.linspace(vmin, vmax, n_tick)
        cval  = (cvar - vmin)/(vmax - vmin) # Normalized color values

        cmap  = matplotlib.colors.LinearSegmentedColormap.from_list(cmap_name, plt.get_cmap(cmap_name, 265)(np.linspace(0, 1, 265)), n_seg)
        sm    = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        c     = cmap(cval)

        # Add colorbar
        cbar  = fig.colorbar(sm, ax=plt.gca(), label=cbar_label, **cbar_kwargs,)
        cbar.set_ticks(ticks)

        for tri, c0 in zip(triangles, c):
            coords = mesh[tri]

            # Plot triangles
            poly  = Poly3DCollection([list(zip(coords[:, 0], coords[:, 1], coords[:, 2]))], 
                                     color=c0, alpha=alpha, linewidth=0.1)
            ax.add_collection3d(poly)
        
    # Plot edges
    if (len(c) == 0) | (edges == True):
        for tri in triangles:
            coords = np.vstack((mesh[tri], mesh[tri][0, :]))
            # Plot edges
            edge  = Line3DCollection([list(zip(coords[:, 0], coords[:, 1], coords[:, 2]))], **edge_kwargs)
            ax.add_collection3d(edge)

        # # Plot vertex IDs
        # for vtx, label in zip(mesh[tri], tri):
        #     ax.text(vtx[0], vtx[1], vtx[2], str(label), fontsize=6)

    # Set axes/aspect ratio    
    ranges = np.ptp(mesh, axis=0)
    ax.set_box_aspect(ranges) 
    ax.set_xlim(mesh[:, 0].min(), mesh[:, 0].max())
    ax.set_ylim(mesh[:, 1].min(), mesh[:, 1].max())
    ax.set_zlim(mesh[:, 2].min(), mesh[:, 2].max())

    zticks = ax.get_zticks()

    ax.set_zticklabels([f'{int(-tick)}' for tick in zticks])
    
    if invert_zaxis:
        ax.invert_zaxis()
    ax.set_xlabel('East (km)',  labelpad=labelpad)
    ax.set_ylabel('North (km)', labelpad=labelpad)
    ax.set_zlabel('Depth (km)', labelpad=labelpad/4)
    # ax.view_init(azim=45, elev=90)
    ax.view_init(azim=azim, elev=elev)
    fig.suptitle(title)
    fig.tight_layout()
    if len(file_name) > 0:
        plt.savefig(file_name, dpi=dpi)
        plt.close()

    if show:
        plt.show()
        plt.close()
    
    return fig, ax


def plot_fault_panels(panels, fault_panels, mesh, triangles, figsize=(14, 8.2), orientation='horizontal', cmap_disp='coolwarm', x_ax='east', fault_height=0.5, 
                      fault_lim='mesh', title='', fault_label='', markersize=10, trace=False, vlim_disp=[], vlim_slip=[], xlim=[], ylim=[], n_tick=11, n_seg=10, mu=0, eta=0, 
                      shrink=0.01, fontsize=8, file_name='', show=False, dpi=300):
    """
    Plot three displacement panels above side-view of fault model.
    """

    # if x_ax == 'east':
    #     xlabel = 'East (km)'
    #     x_idx  = 0
    # else:
    #     xlabel = 'North (km)'
    #     x_idx  = 1

    # Set up figure and axes
    fig   = plt.figure(figsize=figsize)
    fig.suptitle(title)
    x_idx = 0

    if orientation == 'horizontal':
        gs    = fig.add_gridspec(len(fault_panels) + 1, 4, width_ratios=(1, 1, 1, 0.05), height_ratios=[1] + [fault_height for i in range(len(fault_panels))])

        data_axes  = [fig.add_subplot(gs[0, i]) for i in range(3)]
        fault_axes = [fig.add_subplot(gs[i + 1, :-1]) for i in range(len(fault_panels))]
        caxes      = [fig.add_subplot(gs[i, -1]) for i in range(len(fault_panels) + 1)]

        data_axes[0].set_ylabel('North (km)')

        for ax in data_axes:
            ax.set_xlabel('East (km)')

    elif orientation == 'vertical':
        gs    = fig.add_gridspec(3 + len(fault_panels), 2, width_ratios=(1, 0.025), height_ratios=[1 for i in range(len(panels))] + [1 for i in range(len(fault_panels))],
                                 hspace=0.2, wspace=0.1)

        data_axes  = [fig.add_subplot(gs[i, 0]) for i in range(3)]
        fault_axes = [fig.add_subplot(gs[3 + i, 0]) for i in range(len(fault_panels))]
        caxes      = [fig.add_subplot(gs[i, 1]) for i in range(3 + len(fault_panels))]

        # labels
        for ax in data_axes:
            ax.set_ylabel('Y (km)')
            ax.set_xticklabels([])

        fault_axes[-1].set_xlabel('X (km)')

        for ax in fault_axes:
            ax.set_ylabel('Depth (km)')

    all_x = []
    all_y = []

    if len(vlim_disp) == 0:
        vmin_disp = [panel['data'].min() for panel in panels]
        vmax_disp = [panel['data'].max() for panel in panels]

    elif len(vlim_disp) == 2:
        vmin_disp = [vlim_disp[0] for panel in panels]
        vmax_disp = [vlim_disp[1] for panel in panels]

    else:
        vmin_disp = [vlim[0] for vlim in vlim_disp]
        vmax_disp = [vlim[1] for vlim in vlim_disp]

    # ---------------------------------------- Displacement panels ----------------------------------------
    for i in range(3):
        panel = panels[i]
        data  = panel['data']
        label = panel['label']

        if len(data.shape) == 2:
            # print(data.shape)
            extent = panel['extent']
            all_x.extend([extent[0], extent[1]])
            all_y.extend([extent[3], extent[2]])

            im = data_axes[i].imshow(data, extent=extent, vmin=vmin_disp[i], vmax=vmax_disp[i], cmap=cmap_disp, interpolation='none')

        else:
            x = panel['x']
            y = panel['y']

            all_x.extend([x.min(), x.max()])
            all_y.extend([y.min(), y.max()])

            im = data_axes[i].scatter(x, y, c=data, vmin=vmin_disp[i], vmax=vmax_disp[i], cmap=cmap_disp, marker='.', s=markersize)
            data_axes[i].set_aspect(1)

        if orientation == 'horizontal':
            k = 0
        else:
            k = i

        fig.colorbar(im, cax=caxes[k], label='Displacement (mm)', shrink=shrink)

    # Get axes limits
    if len(xlim) == 0:
        xlim = [np.nanmin(all_x), np.nanmax(all_x)]

    if len(ylim) == 0:
        ylim = [np.nanmin(all_y), np.nanmax(all_y)]

    # Plot fault trace
    for i in range(3):
        data_axes[i].plot(mesh[:, 0][mesh[:, 2] == 0], mesh[:, 1][mesh[:, 2] == 0], linewidth=1, c='k')

        data_axes[i].set_title(panels[i]['label'], fontsize=fontsize)
        data_axes[i].set_xlim(xlim)
        data_axes[i].set_ylim(ylim)

    # ---------------------------------------- Fault panels ----------------------------------------
    for i in range(len(fault_panels)):

        panel = fault_panels[i]

        # Set up colorbar for fault slip
        alpha = 1
        edges = True
        cvar  = panel['slip']
        vmin  =  panel['vlim'][0]
        vmax  =  panel['vlim'][1]

        ticks = np.linspace(vmin, vmax, n_tick)
        cval  = (cvar - vmin)/(vmax - vmin) # Normalized color values
        cmap  = matplotlib.colors.LinearSegmentedColormap.from_list(panel['cmap'], plt.get_cmap(panel['cmap'], 265)(np.linspace(0, 1, 265)), n_seg)
        sm    = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        c     = cmap(cval)

        # Plot fault meshes and slip distributions
        for tri, c0 in zip(triangles, c):
            pts = mesh[tri]

            # Plot triangles
            face = Polygon(list(zip(pts[:, x_idx], pts[:, 2])), color=c0, alpha=alpha, linewidth=0.1)
            fault_axes[i].add_patch(face)

            edges = Polygon(list(zip(pts[:, x_idx], pts[:, 2])), edgecolor='k', facecolor='none', alpha=1, linewidth=0.5)
            fault_axes[i].add_patch(edges)
                
        # Axis settings
        if fault_lim == 'mesh':
            fault_axes[i].set_xlim(mesh[:, x_idx].min(), mesh[:, x_idx].max())
            fault_axes[i].set_ylim(mesh[:, 2].min(), mesh[:, 2].max())

        elif fault_lim == 'map':
            fault_axes[i].set_xlim(xlim)
            fault_axes[i].set_ylim(ylim[0] - ylim[1], 0)

        fault_axes[i].set_aspect(1)
        fault_axes[i].set_title(panel['title'], fontsize=fontsize)

        fig.colorbar(sm, cax=caxes[k + i + 1], label=panel['label'], shrink=shrink)

    # Finish up
    fig.tight_layout()

    if len(file_name) > 0:
        fig.savefig(file_name, dpi=dpi)
        
    if show:
        plt.show()

    plt.close()

    return fig


def plot_quadtree(data, extent, samp_coords, samp_data, 
                  trace=[], original_data=[], cell_extents=[], 
                  cmap_disp='coolwarm', vlim_disp=[], figsize=(14, 8.2), 
                  markersize=30, file_name='', show=False, dpi=300):
    """
    Compare gridded and quadtree downsampled displacements
    """

    if len(vlim_disp) == 0:
        vlim_disp = 0.7*np.nanmax(np.abs(data))

    fig   = plt.figure(figsize=figsize)
    gs    = fig.add_gridspec(1, 3, width_ratios=(1, 1, 0.05), height_ratios=(1,))
    ax0   = fig.add_subplot(gs[0, 0])
    ax1   = fig.add_subplot(gs[0, 1])
    cax   = fig.add_subplot(gs[0, 2])
    axes  = [ax0, ax1]


    # Plot gridded data
    im = axes[0].imshow(data, extent=extent, vmin=vlim_disp[0], vmax=vlim_disp[1], cmap=cmap_disp, interpolation='none')

    if len(original_data) > 0:
        axes[1].scatter(original_data[0].flatten(), original_data[1].flatten(), c='k', marker='.', s=1)

    # Plot sampled data
    im = axes[1].scatter(samp_coords[0], samp_coords[1], c=samp_data, vmin=vlim_disp[0], vmax=vlim_disp[1], cmap=cmap_disp, marker='.', s=markersize)

    # Plot fault trace if specified
    if len(trace) > 1:
        # trace = fault_mesh[:, :2][fault_mesh[:, 2] == 0]
        axes[0].plot(trace[:, 0], trace[:, 1], linewidth=2, c='k')
        axes[1].plot(trace[:, 0], trace[:, 1], linewidth=2, c='k')

    if len(cell_extents) > 0:
        for cell_extent in cell_extents:
            for ax in axes:
                # Create a rectangle patch
                cell = Rectangle((cell_extent[0], cell_extent[2]), cell_extent[1] - cell_extent[0], cell_extent[3] - cell_extent[2], fill=False, edgecolor='k', linewidth=0.25)

                # Add the rectangle patch to the axes
                ax.add_patch(cell)

    # Axes settings
    axes[0].set_ylabel('North (km)')
    axes[0].set_xlabel('East (km)')
    axes[0].set_xlim(extent[0], extent[1])
    axes[0].set_ylim(min(extent[2:]), max(extent[2:]))

    axes[1].set_xlabel('East (km)')
    axes[1].set_ylabel('')
    axes[1].set_yticks([])
    axes[1].set_xlim(extent[0], extent[1])
    axes[1].set_ylim(min(extent[2:]), max(extent[2:]))

    ax1.set_aspect('equal')
    ax0.set_aspect('equal')

    fig.colorbar(im, cax=cax, label='LOS Displacement (mm)', shrink=0.3)

    if len(file_name) > 0:
        fig.savefig(file_name, dpi=dpi)

    if show:
        plt.show()

    plt.close()

    return fig, axes


def plot_chains(samples, samp_prob, discard, labels, units, scales, out_dir, dpi=500):
    """
    Plot Markov chains
    """
    n_dim = len(labels)

    fig, axes = plt.subplots(n_dim + 1, figsize=(6.5, 4), sharex=True)
    
    for i in range(n_dim):
        ax = axes[i]
        ax.plot(samples[discard:, :, i] * scales[i], "k", alpha=0.3, linewidth=0.5)
        # ax.plot(samples[:, :, i] * scales[i], "k", alpha=0.3, linewidth=0.5)
        # ax.plot(samples[:discard, :, i] * scales[i], color='tomato', linewidth=0.5)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    # Plot log-probability
    ax = axes[n_dim]
    ax.plot(samp_prob[discard:], "k", alpha=0.3, linewidth=0.5) # log(p(d|m))
    # ax.plot(samp_prob, "k", alpha=0.3, linewidth=0.5)
    # ax.plot(samp_prob[:discard], color='tomato', linewidth=0.5)
    ax.set_xlim(0, len(samples))
    ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.set_ylabel(r'log(p(m|d))') # log(p(d|m))
    axes[-1].set_xlabel("Step");
    fig.tight_layout()
    fig.savefig(f'{out_dir}/chains.png', dpi=dpi)
    plt.close()
    
    return


def plot_triangle(samples, priors, labels, units, scales, out_dir, limits=[],figsize=(10, 10), dpi=500, **kwargs):
    # Make corner plot
    font = {'size': 6}

    matplotlib.rc('font', **font)

    if len(limits) == 0:
        limits = [[prior * scales[i] for prior in priors[key]] for i, key in enumerate(priors.keys())]
        #print('limit test',limits)

    fig = plt.figure(figsize=figsize, tight_layout={'h_pad':0.1, 'w_pad': 0.1})
    fig = corner.corner(samples * scales, 
                        quantiles=[0.16, 0.5, 0.84], 
                        range=limits,
                        labels=[f'{label} ({unit})' for label, unit in zip(labels, units)], 
                        label_kwargs={'fontsize': 8},
                        show_titles=True,
                        title_kwargs={'fontsize': 8},
                        fig=fig, 
                        labelpad=0.1
                        )

    # fig.tight_layout(pad=1.5)
    fig.savefig(f'{out_dir}/triangle.png', dpi=dpi)
    plt.close()
    return


def plot_grid(x, y, grid, extent=[],xlabel='X', ylabel='Y',  title='', cmap='coolwarm', vlim=[], cbar=False, clabel='Displacement (mm)', background_color='w',
              figsize=(7, 6), fig_ax=[], file_name='', show=False, dpi=300):
    """
    Plot gridded dataset.  
    """

    if len(fig_ax) == 2:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(figsize=figsize)
        
    if len(vlim) != 2:
        vlim = [np.nanmin(grid), np.nanmax(grid)]

    if len(extent) != 4:
        extent = [np.min(x), np.max(x), np.max(y), np.min(y)]

    im = ax.imshow(grid, extent=extent, cmap=cmap, vmin=vlim[0], vmax=vlim[1], interpolation='none')

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[3], extent[2])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect(1)
    ax.set_facecolor(background_color)

    if len(title) > 0:
        ax.set_title(title)

    if cbar:
        plt.colorbar(im, label=clabel, pad=0.1, shrink=0.5)


    if len(file_name) > 0:
        plt.savefig(file_name, dpi=dpi)
    
    if show:
        plt.show()
            
    return fig, ax


def plot_grids_row(grids, extent, cmap=cmc.vik, region=[], titles=[], labels=[], vlims=[], figsize=(14, 8.2), xlabel='Longitude', ylabel='Latitude', cax_kwargs=dict(size="5%", pad=0.5), show=False, file_name='', dpi=300):
    
    """
    Plot row of grids
    """
    n_grid = len(grids)

    # Get color limits
    if len(vlims) != n_grid:
        vlims = [[np.min([np.nanmin(grid) for grid in grids]), np.max([np.nanmax(grid) for grid in grids])] for i in range(len(grids))]

    if len(labels) != n_grid:
        labels = ['' for i in range(n_grid)]

    # Plot
    fig, axes = plt.subplots(1, n_grid, figsize=figsize)

    for i, ax in enumerate(axes):
        # Plot grid
        im = ax.imshow(grids[i], cmap=cmap, extent=extent, interpolation='none', vmin=vlims[i][0], vmax=vlims[i][1])

        # Colorbar
        divider = make_axes_locatable(ax)
        cax     = divider.append_axes("bottom", **cax_kwargs)
        cbar    = fig.colorbar(im, cax=cax, orientation="horizontal", label=labels[i])

        # Axes settings
        axes[i].invert_yaxis()
        axes[i].set_xlabel(xlabel)
        axes[i].set_aspect(1)

        if len(titles) == n_grid:
                axes[i].set_title(titles[i])

        if len(region) == 4:
            axes[i].set_xlim(region[:2])
            axes[i].set_ylim(region[2:])

    # More axes settings
    axes[0].set_ylabel(ylabel)

    for ax in axes[1:]:
        ax.set_yticks([])

    fig.tight_layout()

    if show:
        plt.show()

    if len(file_name) > 0:
        plt.savefig(file_name, dpi=dpi)

    return fig, axes


def plot_grid_search(cost, mu, eta, label='', xlabel='', ylabel='', contours=[], cmap=cmc.roma, mc='k',
                     x_pref=0.6951927962, y_pref=1.4384498883, x_reverse=False, y_reverse=False, 
                     logx=False, logy=False, log=False, file_name='', show=False, dpi=300, vlim=[],):
    """
    Plot heatmap and marginal distributionss of grid search.
    """

    if log:
        cost_heatmap = np.log(cost)
    else:
        cost_heatmap = cost

    if len(vlim) != 2:
        vlim = [np.min(cost_heatmap), np.max(cost_heatmap)]

    # Plot
    fig = plt.figure(figsize=(10, 8))

    space = 0.5
    grid = fig.add_gridspec(3, 5,  width_ratios=(3, space/2, 1, space, 1), height_ratios=(3, space/2, 1),
    #                       left=0, right=1, bottom=0, top=1,
                          # left=grid_bbox[0], right=grid_bbox[1], bottom=grid_bbox[2], top=grid_bbox[3] ,
                          wspace=0, hspace=0)

    # Create the axes
    ax2 = fig.add_subplot(grid[2, 0])
    ax0 = fig.add_subplot(grid[0, 0], sharex=ax2)
    ax1 = fig.add_subplot(grid[0, 2], sharey=ax0)
    ax3 = fig.add_subplot(grid[0, 4], sharey=ax0)
    cax = fig.add_subplot(grid[2, 3])

    # Plot heatmap
    # im = ax0.scatter(mu, eta, c=cost_heatmap, cmap=cmap, marker='s', s=1200, vmin=vlim[0], vmax=vlim[1])
    im = ax0.contourf(mu, eta, cost_heatmap[::-1, :], 10, vmin=vlim[0], vmax=vlim[1], cmap=cmap)
    fig.colorbar(im, cax=cax, label=label, shrink=0.1)

    if len(contours) == len(cost_heatmap):
        # cs = ax0.contour(mu, eta, contours, colors='C0', linstyles='-', linewidths=0.5, zorder=10000)
        # ax0.clabel(cs)    
        ax3.plot(contours[::-1, 0], eta[:, 0], c='k')

    n_eta       = len(np.unique(eta))
    n_mu        = len(np.unique(eta))
    eta_rms_sum = np.zeros(n_eta)
    mu_rms_sum  = np.zeros(n_mu)

    # Plot x-value curves
    for mu0 in np.unique(mu):  

        mu_select = mu.flatten()[mu.flatten() == mu0]
        rms_select = cost.flatten()[mu.flatten() == mu0]

        rms_sort    = np.array([rms0 for _, rms0 in sorted(zip(mu_select, rms_select), reverse=y_reverse)])
        eta_sort    = np.sort(np.unique(eta))
        mu_rms_sum += rms_sort

        ax1.plot(rms_sort, eta_sort, c='gainsboro')

    # Plot y-value curves
    for eta0 in np.unique(eta):

        eta_select = eta.flatten()[eta.flatten() == eta0]
        rms_select = cost.flatten()[eta.flatten() == eta0]

        rms_sort     = np.array([rms0 for _, rms0 in sorted(zip(eta_select, rms_select), reverse=x_reverse)])
        mu_sort      = np.sort(np.unique(mu))
        eta_rms_sum += rms_sort

        ax2.plot(mu_sort, rms_sort, c='gainsboro')

    # Plot mean marginals
    ax1.plot(mu_rms_sum/n_eta, eta_sort, c='k')
    ax2.plot(mu_sort, eta_rms_sum/n_mu, c='k')

    # Plot preferred values
    if len(contours) > 0:
        ax3.scatter(contours[::-1, 0][np.argmin(np.abs(eta_sort - y_pref))], y_pref, c=mc, marker='o', zorder=100)
    ax1.scatter((mu_rms_sum/n_eta)[np.argmin(np.abs(eta_sort - y_pref))], y_pref, c=mc, marker='o', zorder=100)
    ax2.scatter(x_pref, (eta_rms_sum/n_mu)[np.argmin(np.abs(mu_sort - x_pref))],  c=mc, marker='o', zorder=100)
    ax0.scatter(x_pref, y_pref, c=mc, marker='o', zorder=200)

    # Scale x-parameter
    if logx:
        ax0.set_xscale('log')
        ax2.set_xscale('log')

    # Scale y-parameter
    if logy:
        ax0.set_yscale('log')
        ax1.set_yscale('log')

    # Scale z-parameter
    if log:
        ax1.set_xscale('log')
        ax2.set_yscale('log')

    # Panel settings
    ax0.xaxis.set_label_position("top")
    ax0.xaxis.tick_top()
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)

    ax1.set_xlabel(label)
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.set_ylabel(ylabel)

    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(label)

    ax3.xaxis.tick_top()
    ax3.yaxis.tick_right()
    ax3.xaxis.set_label_position("top")
    ax3.yaxis.set_label_position("right")
    ax3.set_ylabel('Data res. cutoff')
    ax3.set_xlabel('Data points')
    ax3.set_xscale('log')
    ax3.set_yscale('log')



    # ax0.set_aspect('equal')
    # ax1.set_aspect('equal')
    # ax2.set_aspect('equal')


    ax0.set_facecolor('gainsboro')

    if len(file_name) > 0:
        fig.savefig(file_name, dpi=dpi)
    
    if show:
        plt.show()

    return 

