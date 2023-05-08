import os
import numpy as np
import xarray as xr
import pandas as pd
import numpy as np

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


def make_insar_table(data, look_table, write=True, omit_nans=True, file_name='insar.dat'):
    """
    longitude, latitude, and the LOS look vector
    """

    columns = ['Longitude', 'Latitude', 'Elevation', 'Look_E', 'Look_N', 'Look_Z', 'LOS']
    look    = np.loadtxt(look_table)
    
    print(data.shape)
    print(look.shape)

    table = np.hstack((look, data))

    df = pd.DataFrame(table, columns=columns)
    
    if omit_nans:
        df = df[~np.isnan(df['LOS'])]

    if write:
        df.to_csv(file_name, sep=' ', header=True, index=False, na_rep='NaN')
    return df


def load_gmt_fault_file(file, region=[-180, 180, -90, 90], outname=''):
    """
    Load faults from GMT ASCII file

    INPUT:
    file - fault file

    OUTPUT:
    faults - list containing arrays of fault coordinates
    """

    faults = []
    data   = pd.read_csv(file, header=None, delim_whitespace=True)

    breaks = np.where(data.iloc[:, 0].str.contains(r'>', na=True))[0]

    for i in range(len(breaks) - 1):
        fault = pd.DataFrame({'Longitude': data.iloc[breaks[i] + 1:breaks[i + 1], 0],
                              'Latitude': data.iloc[breaks[i] + 1:breaks[i + 1], 1]}).astype(float)

        if any((region[0] <= fault['Longitude']) & (fault['Longitude'] <= region[1]) & (region[2] <= fault['Latitude']) & (fault['Latitude'] <= region[3])):
            faults.append(fault)

    if len(outname) > 0:
        with open(outname, 'w') as f:
            for fault in faults:
                f.write('> -L"" \n')    
                for i in range(len(fault)):
                    f.write(f'{fault.iloc[i, 0]} {fault.iloc[i, 1]} \n')


    return faults


def read_traces(file, mode, EPSG='32611'):
    
    faults = {}

    with open(file, 'r') as f:
        # Load SCEC Community Fault Model file
        if mode == 'SCEC':
            for line in f:
                if line[0] == '#':
                    continue
                if line[0] == '>':
                    name = line.split()[-1][1:-1]
                    faults[name] = ({'name': name, 
                                      'lon': [], 
                                      'lat': []})
                else:
                    coords = [float(item) for item in line.split()]
                    faults[name]['lon'].append(coords[0])
                    faults[name]['lat'].append(coords[1])

        # Load digitized fault traces from QGIS .csv
        elif mode == 'QGIS':
            df = pd.read_csv(file, sep=',')
            for name in df['Name'].unique():
                df_fault = df[df['Name'] == name].sort_values('X')
                faults[name] = {'Name': name, 
                                'lon':  df_fault['X'].values, 
                                'lat':  df_fault['Y'].values,}
    

    # Sort so that DataFrame is in descending latitude 
    for name in faults:
        fault = faults[name]
        if fault['lat'][0] < fault['lat'][-1]:
            fault['lon'] = fault['lon'][::-1]
            fault['lat'] = fault['lat'][::-1]

    # Add UTM coordinates
    if EPSG:
        for name in faults.keys():
            faults[name]['UTMx'], faults[name]['UTMy'] = proj_ll2utm(faults[name]['lon'], faults[name]['lat'], EPSG)

    return faults