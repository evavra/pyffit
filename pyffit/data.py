import os
import numpy as np
import xarray as xr


def read_grd(file, flatten=False):
    """
    Read NetCDF grid file (x, y, z) to Numpy arrays.
    Specify flatten=True to get flattened arrays and original grid dimensions
    """
    
    with xr.open_dataset(file) as grd:
        try:
            x = grd.lon.values
            y = grd.lat.values
            z = grd.z.values

        except AttributeError:
            x = grd.x.values
            y = grd.y.values
            z = grd.z.values

    if flatten:
        dims = z.shape
        X, Y = np.meshgrid(x, y)
        X    = X.flatten()
        Y    = Y.flatten()
        Z    = z.flatten()

        return X, Y, Z, dims

    else: 
        return x, y, z


def write_grd(x, y, z, file_name, T=True):

    """
    Write gridded data to GMT-compatible NetCDF file. 
    Requires GMT installation to modify grid registration (-T option).
    """

    data = xr.Dataset({'z': (['y', 'x'], z)}, coords={'y':(['y'], y), 'x':(['x'], x)})

    # Fix headers
    x_inc = np.min(np.diff(data.x.values))
    y_inc = np.min(np.diff(data.y.values))

    x_min_orig = np.min(data.x.values)
    x_max_orig = np.max(data.x.values)
    y_min_orig = np.min(data.y.values)
    y_max_orig = np.max(data.y.values)

    data.x.attrs['actual_range'] = np.array([x_min_orig, x_max_orig])
    data.y.attrs['actual_range'] = np.array([y_min_orig, y_max_orig])

    data.to_netcdf(path=file_name, mode='w')

    if T:
        x_min = np.min(data.x.values) - x_inc/2 
        x_max = np.max(data.x.values) + x_inc/2
        y_min = np.min(data.y.values) - y_inc/2
        y_max = np.max(data.y.values) + y_inc/2

        bounds = ' -R' + str(x_min) + '/' + str(x_max) + '/' + str(y_min) + '/' + str(y_max)
        # print(bounds)
        cmd = 'gmt grdedit ' + file_name + ' ' + bounds + ' -T' 
        print(cmd)
        os.system(cmd)

    return
