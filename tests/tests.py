import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyffit.quadtree import quadtree
from pyffit.data import read_grd
# import cutde


def main():
    test_quadtree()

    return

def test_quadtree():
    # Data file
    data_file   = 'maduo/S1_LOS_D106.grd'

    # Quadtree parameters
    rms_min      = 1   # RMS threshold (data units)
    nan_frac_max = 0.4 # Fraction of NaN values allowed per cell
    width_min    = 2   # Minimum cell width (# of pixels)
    width_max    = 300 # Maximum cell width (# of pixels)
    
    # Plot parameters
    cmap = 'coolwarm'
    
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


if __name__ == '__main__':
    main()