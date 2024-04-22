import os
import numpy as np
import xarray as xr
import pandas as pd
import numpy as np
from pyffit.utilities import proj_ll2utm, get_local_xy_coords


def read_grd(file, coord_type='range', check_lon=False, flatten=False):
    """
    Read NetCDF grid file (x, y, z) to numpy arrays.
    
    INPUT:
    file      - path to NetCDF file
    xy_type   - 'range' for default x/y coordinate ranges, 
                'grid' for full coordinates for each pixel in grid, or 
                'flatten' to return flattened gridded coordinates and data and original grid dimensions 
    check_lon - if True, subtract 360 from x if all(abs(x) <= 360)

    OUTPUT:
    x, y  - spatial coordinates corresponding to specified
    z     - grid values
    (dims - original grid dimensions if coord_type == 'flatten')

    """
    
    with xr.open_dataset(file) as grd:
        try:
            x = grd.lon.values
            y = grd.lat.values
            z = grd.z.values

        except AttributeError:
            try:
                x = grd.longitude.values
                y = grd.latitude.values
                z = grd.z.values

            except AttributeError:
                x = grd.x.values
                y = grd.y.values
                z = grd.z.values

    if check_lon & (all(np.abs(x) <= 360)):
        x -= 360

    if coord_type == 'grid':
        x, y = np.meshgrid(x, y)

    if (coord_type == 'flatten') | flatten:
        dims = z.shape
        X, Y = np.meshgrid(x, y)
        X    = X.flatten()
        Y    = Y.flatten()
        Z    = z.flatten()

        return X, Y, Z, dims

    else: 
        return x, y, z


def write_grd(x, y, z, file_name, T=True, V=False):

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

        if V:
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


def read_traces(file, mode, EPSG='32611', ref_point_utm=[], region=[-360, 360, -90, 90]):
    """
    Read in table of fault trace segments
    Modes:
    SCEC
    QGIS
    GMT
    """

    with open(file, 'r') as f:
        # Load SCEC Community Fault Model file
        if mode == 'SCEC':
            faults = {}
            for line in f:
                if line[0] == '#':
                    continue
                if line[0] == '>':
                    name = line
                    faults[name] = ({'name': name, 
                                      'lon': [], 
                                      'lat': []})
                else:
                    coords = [float(item) for item in line.split()]
                    faults[name]['lon'].append(coords[0])
                    faults[name]['lat'].append(coords[1])

        # Load digitized fault traces from QGIS .csv
        elif mode == 'QGIS':
            faults = {}
            df = pd.read_csv(file, sep=',')

            for name in df['Name'].unique():
                # df_fault = df[df['Name'] == name].sort_values('X')
                df_fault = df[df['Name'] == name]

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
            for name in faults.keys():
                faults[name]['UTMx'], faults[name]['UTMy'] = proj_ll2utm(faults[name]['lon'], faults[name]['lat'], EPSG)

                if len(ref_point_utm) == 2:
                    faults[name]['UTMx'] -= ref_point_utm[0]
                    faults[name]['UTMy'] -= ref_point_utm[1]
                    faults[name]['UTMx'] /= 1000
                    faults[name]['UTMy'] /= 1000

            # Sort by alphabetical order
            return dict(sorted(faults.items()))

        elif mode =='GMT':
            faults = []
            i = 0
            i_seg = []

            for line in f:
                if line[0] == '#':
                    continue
                elif '>' in line:
                    faults.append([np.nan, np.nan])
                else:
                    tmp = line.split()
                    lon, lat = float(tmp[0]), float(tmp[1])
                    faults.append([lon, lat])

            faults = np.array(faults)
            faults = faults[np.isnan(faults[:, 0]) | ((faults[:, 0] >= region[0]) & (faults[:, 0] <= region[1]) & (faults[:, 1] >= region[2]) & (faults[:, 1] <= region[3]))]

            # Get UTM coordinates
            UTMx, UTMy = proj_ll2utm(faults[:, 0], faults[:, 1], EPSG)

            if len(ref_point_utm) == 2:
                UTMx -= ref_point_utm[0]
                UTMy -= ref_point_utm[1]
                UTMx /= 1000
                UTMy /= 1000

            faults = np.hstack((faults, UTMx.reshape(-1, 1), UTMy.reshape(-1, 1)))
            return faults


def read_gnss_table(file, columns=[], EPSG='EPSG'):
    """
    Read ASCII table of gnss velocities with columns:
    ['lon', 'lat', 've_mean', 'vn_mean', 'vz_mean', 've_std', 'vn_std',  'vz_std']
    """
    df = pd.read_csv(file, delim_whitespace=True, header=0)

    if len(columns) > 0:
        df.columns = columns
    elif 'Continuous' in file:
        df.columns = ['station', 'lat', 'lon', 'vn_mean', 've_mean', 'vn_std', 've_std', 'cor']

    elif 'Survey' in file:
        df.columns = ['station', 'lat', 'lon', 'vn_mean', 've_mean', 'vz_mean', 'vn_std', 've_std', 'vz_std']

    else:
        df.columns = ['lon', 'lat', 've_mean', 'vn_mean', 've_std', 'vn_std']
    if EPSG:
        df['UTMx'], df['UTMy'] = proj_ll2utm(df['lon'].values, df['lat'].values, EPSG)

    return df


def read_Slab2_faults(file):
    """
    Read Slab 2.0 fault models
    """
    rows = []

    with open(file, 'r') as f:
        for line in f:
            if line[0] != '>':
                rows.append([float(val) for val in line.split()])

    mesh = np.array(rows)
    # print(mesh)
    return mesh
