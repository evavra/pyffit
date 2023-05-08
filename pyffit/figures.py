import numpy as np
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

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