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


def plot_fault_3d(mesh, triangles, c=[], edges=False, cmap_name='viridis', cbar_label='Slip (m)', 
                  labelpad=20, azim=45, elev=10, n_seg=100, n_tick=11, alpha=1, vlim_slip=[],
                  filename='', show=True, dpi=500, invert_zaxis=False, edge_kwargs=dict(edgecolor='k', linewidth=0.25),
                  figsize=(14, 8.2), cbar_kwargs=dict(location='bottom', pad=-0.1, shrink=0.5)):
    """
    Make 3D plot of finite fault model.

    INPUT:
    mesh (M, 3)      - array of coordinates (x,y,z) of triangular mesh verticies
    triangles (N, 3) - array containing vertex indices of each triangle in the mesh. Each row corresponds to one triangle
    """

    # Make 3D plot
    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(projection='3d')
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
        cbar  = plt.colorbar(sm, label=cbar_label, **cbar_kwargs,)
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
    fig.tight_layout()

    if len(filename) > 0:
        plt.savefig(filename, dpi=dpi)

    if show:
        plt.show()
    plt.close()
    
    return fig, ax


def plot_fault_panels(panels, mesh, triangles, slip, figsize=(14, 8.2), cmap_disp='coolwarm', cmap_slip='viridis', x_ax='east',
                      trace=False, vlim_disp=[], vlim_slip=[], xlim=[], ylim=[], markersize=10, n_tick=11, n_seg=10, mu=0, eta=0, file_name='', show=False, dpi=300):
    """
    Plot three displacement panels above side-view of fault model.
    """

    grid_dims = (1, 3)

    # Set up figure and axes
    fig   = plt.figure(figsize=figsize)
    gs    = fig.add_gridspec(2, 4, width_ratios=(1, 1, 1, 0.05), height_ratios=(1, 1))
    ax0   = fig.add_subplot(gs[0, 0])
    ax1   = fig.add_subplot(gs[0, 1])
    ax2   = fig.add_subplot(gs[0, 2])
    ax3   = fig.add_subplot(gs[1, :-1])
    cax0  = fig.add_subplot(gs[0, 3])
    cax1  = fig.add_subplot(gs[1, -1])
    axes  = [ax0, ax1, ax2, ax3]
    caxes = [cax0, cax1]

    # Set up colorbar for fault slip
    alpha = 1
    edges = True
    cvar  = slip

    if x_ax == 'east':
        xlabel = 'East (km)'
        x_idx  = 0
    else:
        xlabel = 'North (km)'
        x_idx  = 1

    if len(vlim_slip) == 0:
        vmin = slip.min()
        vmax = slip.max()
    else:
        vmin = vlim_slip[0]
        vmax = vlim_slip[1]

    ticks = np.linspace(vmin, vmax, n_tick)
    cval  = (cvar - cvar.min())/(cvar.max() - cvar.min()) # Normalized color values
    cmap  = matplotlib.colors.LinearSegmentedColormap.from_list(cmap_slip, plt.get_cmap(cmap_slip, 265)(np.linspace(0, 1, 265)), n_seg)
    sm    = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    c     = cmap(cval)

    all_x = []
    all_y = []

    if len(vlim_disp) == 0:
        all_data  = np.concatenate((panels[0]['data'].flatten(), panels[1]['data'].flatten(), panels[2]['data'].flatten()))
        vlim_disp = 0.7*np.nanmax(np.abs(all_data))

    # Plot displacement panels
    for i in range(3):
        panel = panels[i]
        data  = panel['data']
        label = panel['label']

        if len(data.shape) == 2:
            extent = panel['extent']
            all_x.extend([extent[0], extent[1]])
            all_y.extend([extent[3], extent[2]])

            im = axes[i].imshow(data, extent=extent, vmin=vlim_disp[0], vmax=vlim_disp[1], cmap=cmap_disp, interpolation='none')

        else:
            x = panel['x']
            y = panel['y']

            all_x.extend([x.min(), x.max()])
            all_y.extend([y.min(), y.max()])

            im = axes[i].scatter(x, y, c=data, vmin=vlim_disp[0], vmax=vlim_disp[1], cmap=cmap_disp, marker='.', s=markersize)
            axes[i].set_aspect('equal')


    # Get axes limits
    if len(xlim) == 0:
        xlim = [np.nanmin(all_x), np.nanmax(all_x)]

    if len(ylim) == 0:
        ylim = [np.nanmin(all_y), np.nanmax(all_y)]

    # Plot fault
    for i in range(3):

        # for z in np.unique(mesh[:, 2]):
            # axes[i].plot(mesh[:, 0][mesh[:, 2] == z], mesh[:, 1][mesh[:, 2] == z], linewidth=2)
        axes[i].plot(mesh[:, 0][mesh[:, 2] == 0], mesh[:, 1][mesh[:, 2] == 0], linewidth=1, c='k')

        axes[i].set_title(panels[i]['label'])
        axes[i].set_xlim(xlim)
        axes[i].set_ylim(ylim)

    # Plot fault mesh and slip distribution
    for tri, c0 in zip(triangles, c):
        pts = mesh[tri]

        # Plot triangles
        face = Polygon(list(zip(pts[:, x_idx], pts[:, 2])), color=c0, alpha=alpha, linewidth=0.1)
        ax3.add_patch(face)

        edges = Polygon(list(zip(pts[:, x_idx], pts[:, 2])), edgecolor='k', facecolor='none', alpha=1, linewidth=0.5)
        ax3.add_patch(edges)
            
    # Axis settings
    ax0.set_ylabel('North (km)')
    ax0.set_xlabel('East (km)')
    ax1.set_xlabel('East (km)')
    ax2.set_xlabel('East (km)')
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel('Depth (km)')
    ax3.set_xlim(mesh[:, x_idx].min(), mesh[:, x_idx].max())
    ax3.set_ylim(mesh[:, 2].min(), mesh[:, 2].max())
    ax3.set_aspect('equal')
    fig.colorbar(im, cax=cax0, label='Displacement (mm)', shrink=0.05)
    fig.colorbar(sm, cax=cax1, label='Slip (mm)')
    fig.tight_layout()

    if len(file_name) > 0:
        fig.savefig(file_name, dpi=dpi)
        
    if show:
        plt.show()

    plt.close()

    return fig, axes


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

