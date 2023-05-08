import numpy as np
import matplotlib.pyplot as plt
from grid_utilities import read_grd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

cmap  = 'coolwarm'
vlim  = 20
count = 0

def main():
    # Data file
    data_file   = 'tests/maduo/S1_LOS_D106.grd'

    # Quadtree parameters
    rms_min      = 1   # RMS threshold (data units)
    nan_frac_max = 0.4 # Fraction of NaN values allowed per cell
    width_min    = 2   # Minimum cell width (# of pixels)
    width_max    = 300 # Maximum cell width (# of pixels)
    
    # ---------- LOAD DATA ----------
    x, y, data  = read_grd(data_file)
    X, Y   = np.meshgrid(x, y)
    extent = [x.min(), x.max(), y.max(), y.min()]

    # ---------- QUADTREE ALGORITHM ----------
    print('Performing quadtree downsampling...')
    x_samp, y_samp, z_samp, pixel_count, nan_frac = quadtree(X, Y, data, 0, rms_min=rms_min, nan_frac_max=nan_frac_max, width_min=width_min, width_max=width_max,)
    
    x_samp      = np.array(x_samp)[~np.isnan(x_samp)] 
    y_samp      = np.array(y_samp)[~np.isnan(y_samp)]
    z_samp      = np.array(z_samp)[~np.isnan(z_samp)]
    pixel_count = np.array(pixel_count)[~np.isnan(pixel_count)]
    nan_frac    = np.array(nan_frac)[~np.isnan(nan_frac)]

    print('Number of initial data points:', data[~np.isnan(data)].size)
    print('Number of downsampled data points:', len(z_samp))
    print('Percent reduction: {:.2f}%'.format(100*(data[~np.isnan(data)].size - len(z_samp))/data[~np.isnan(data)].size))
    print()

    # ---------- PLOT ----------
    fig = plt.figure(figsize=(14, 8.2))
    projection = ccrs.PlateCarree()
    ax  = fig.add_subplot(1, 1, 1, projection=projection)
    ax.add_feature(cfeature.LAKES.with_scale('10m'))
    ax.scatter(x_samp, y_samp, c=z_samp, cmap=cmap, transform=projection, zorder=100)
    plt.show()
    
    return


def quadtree(x, y, data, level, rms_min=0.1, nan_frac_max=0.9, width_min=3, width_max=1000,
             x_samp=[], y_samp=[], z_samp=[], cell_dims=[], nan_frac=[]):
    """
    Perform recursive quadtree downsampling.
    
    INPUT:
    x, y      - (M, N) x and y coordinates of dataset
    z         - (M, N) gridded dataset to downsample
    rms_min   - minimum cell root-mean-square threshhold
    width_min - minimum number of data points to be included in cell
    width_min - maximum number of data points to be included in cell

    OUTPUT:
    x_samp, y_samp - downsampled coordinates
    z_samp         - downsamled data values
    std_samp       - downsampled data standard deviations
    """

    # Compute initial data statistics
    data_mean     = np.nanmean(data)            
    data_rms      = np.nanstd(data - data_mean) 
    i_nan_data    = np.isnan(data)
    n_nan_data    = np.count_nonzero(i_nan_data)
    nan_frac_data = n_nan_data/data.size
    data_dim      = np.array(data.shape)
    cell_dim      = data_dim//2

    # Get index slices for cells
    slices = [(slice(cell_dim[0]),              slice(cell_dim[1])),               # Top left
              (slice(cell_dim[0]),              slice(cell_dim[1], data_dim[1])), # Top right
              (slice(cell_dim[0], data_dim[0]), slice(cell_dim[1], data_dim[1])), # Bottom right
              (slice(cell_dim[0], data_dim[0]), slice(cell_dim[1]))]               # Bottom left

    # If (1) RMS is too high and cell size is greater than the minimum or (2) if cell size is greater than the maximum, continue to sample
    if ((data_rms >= rms_min) & all(cell_dim > width_min)) | all(cell_dim > width_max):
        for ij in slices:
            x_samp, y_samp, z_samp, cell_dims, nan_frac = quadtree(x[ij], y[ij], data[ij], level + 1, 
                               rms_min=rms_min, nan_frac_max=nan_frac_max, width_min=width_min, width_max=width_max)

        return x_samp, y_samp, z_samp, cell_dims, nan_frac

    # Otherwise, output downsampled data
    else:
        # Output downsampled data or NaN if maximum NaN fraction is exceeded
        if (nan_frac_data > nan_frac_max):
            results = ([np.nan],
                       [np.nan],
                       [np.nan],
                       [cell_dim],
                       [nan_frac_data])
        else:
            results = ([np.mean(x[~i_nan_data])],
                       [np.mean(y[~i_nan_data])],
                       [data_mean],
                       [cell_dim],
                       [nan_frac_data])

        for output, result in zip([x_samp, y_samp, z_samp, cell_dims, nan_frac], results):
            output.extend(result)

        return x_samp, y_samp, z_samp, cell_dims, nan_frac


if __name__ == '__main__':
    main()