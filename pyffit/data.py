import os
import numpy as np
import xarray as xr
import pandas as pd
import numpy as np
from pyffit.utilities import proj_ll2utm, get_local_xy_coords
from pyffit.quadtree import quadtree_unstructured, apply_unstructured_quadtree


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


def load_look_vectors(look_dir):
    """
    Load east, north, and up SAR look vectors from NetCDF files.

    INPUT:
    look_dir - directory which contains look_e.grd, look_n.grd, and look_u.grd

    OUTPUT:
    look (m, 3) - 3D array containing east, north, and up SAR look vectors
    """
    
    _, _, look_e = read_grd(f'{look_dir}/look_e.grd')
    _, _, look_n = read_grd(f'{look_dir}/look_n.grd')
    _, _, look_u = read_grd(f'{look_dir}/look_u.grd')

    return np.vstack((look_e.flatten(), look_n.flatten(), look_u.flatten())).T


def prepare_insar_datasets(insar_files, look_dirs, weights, ref_point, EPSG='32611', rms_min=0.1, nan_frac_max=0.9, width_min=0.1, width_max=2):
    """
    Ingest InSAR data for finite fault modeling

    Workflow for each dataset
        1. Read interferogram
        2. Read look vectors
        3. Project to local Cartesian coordinate system
        4. Downsample using quadtree algorithm

    INPUT:

    OUTPUT:
    datasets - dictonary containing the following attributes:
        lon (m, n)         - longitude coordinates for interferogram (deg)
        lat (m, n)         - latitude coordinates for interferogram (deg)
        x (m, n)           - y coordinates in local Cartesian coordinate system (km)
        y (m, n)           - y coordinates in local Cartesian coordinate system (km)
        extent (4,)        - extent of interferogram in Cartesian coordinates (km)
        data (m, n)        - interferogram pixel values (see file for original units)
        look (m*n, 3)      - E/N/Z components of look vectors for each pixel
        x_samp (k,)        - quadtree downsampled x-coordinates (km)
        y_samp (k,)        - quadtree downsampled y-coordinates (km)
        data_samp (k,)     - downsamled data values (see file for original units)
        data_samp_std (k,) - downsampled data standard deviations (see file for original units)
        data_index k,)     - global inidices of data contained within each quadtree cell
        data_dims (k,)     - dimensions of each cell in x/y units (km)
        data_extents (k,)  - extent of each cell in x/y coordinates (km)
        nan_frac (k,)      - fraction of nan values in each cell
        look_samp (k, 3)   - downsampled look vectors
        i_nans (k,)        - NaN locations

    """

    datasets = {}


    for i in range(len(insar_files)):
        # Get dataset name
        dataset = insar_files[i][:-4]

        # Get dataset weight
        weight  = weights[i]

        # Read data
        lon_rng, lat_rng, data = read_grd(insar_files[i])
        look                   = load_look_vectors(look_dirs[i])

        # Get full coordinates
        lon, lat = np.meshgrid(lon_rng, lat_rng)

        # Get reference point and convert coordinates to km
        x, y   = get_local_xy_coords(lon, lat, ref_point, EPSG=EPSG) 
        extent = [np.min(x), np.max(x), np.max(y), np.min(y)]

        # Downsample using quadtree algorithm
        data_extent = [x.min(), x.max(), y.min(), y.max()]
        data_index  = np.arange(0, data.size)
        x_samp, y_samp, data_samp, data_samp_std, data_tree, cell_dims, cell_extents, nan_frac = quadtree_unstructured(x.flatten(), y.flatten(), data.flatten(), data_index, data_extent, 
                                                                                                                        rms_min=rms_min, nan_frac_max=nan_frac_max, width_min=width_min, width_max=width_max,
                                                                                                                        x_samp=[], y_samp=[], data_samp=[], data_samp_std=[], data_tree=[], data_dims=[], data_extents=[], nan_frac=[])
        i_nans = np.isnan(data_samp)
        _, _, look_e_samp, _, _ = apply_unstructured_quadtree(x.flatten(), y.flatten(), look[:, 0], data_tree, nan_frac_max)
        _, _, look_n_samp, _, _ = apply_unstructured_quadtree(x.flatten(), y.flatten(), look[:, 1], data_tree, nan_frac_max)
        _, _, look_u_samp, _, _ = apply_unstructured_quadtree(x.flatten(), y.flatten(), look[:, 2], data_tree, nan_frac_max)
        look_samp = np.vstack((look_e_samp, look_n_samp, look_u_samp)).T
        i_nans = np.isnan(data_samp)

        print(f'Quadtree points: {len(data_samp)}')

        # Add datasets to dictionary
        datasets[dataset]                  = {}
        # datasets[dataset]['lon_rng']       = lon_rng
        # datasets[dataset]['lat_rng']       = lat_rng
        datasets[dataset]['lon']           = lon
        datasets[dataset]['lat']           = lat 
        datasets[dataset]['x']             = x
        datasets[dataset]['y']             = y
        datasets[dataset]['extent']        = extent
        datasets[dataset]['data']          = data
        datasets[dataset]['look']          = look
        datasets[dataset]['x_samp']        = x_samp
        datasets[dataset]['y_samp']        = y_samp
        datasets[dataset]['data_samp']     = data_samp
        datasets[dataset]['data_samp_std'] = data_samp_std
        datasets[dataset]['data_tree']     = data_tree
        datasets[dataset]['cell_dims']     = cell_dims
        datasets[dataset]['cell_extents']  = cell_extents
        datasets[dataset]['nan_frac']      = nan_frac
        datasets[dataset]['look_samp']     = look_samp
        datasets[dataset]['i_nans']        = i_nans
        datasets[dataset]['weight']        = weight

    return datasets

 