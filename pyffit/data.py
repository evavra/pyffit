import os
import glob
import datetime as dt
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


def load_gmt_fault_file(file, region=[-180, 180, -90, 90], EPSG='', ref_point=[], outname=''):
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

    if (len(ref_point) == 2) & (len(EPSG) == 5):
        ref_point_utm  = proj_ll2utm(ref_point[0], ref_point[1], EPSG)

    for i in range(len(breaks) - 1):
        fault = pd.DataFrame({'Longitude': data.iloc[breaks[i] + 1:breaks[i + 1], 0],
                              'Latitude': data.iloc[breaks[i] + 1:breaks[i + 1], 1]}).astype(float)

        if len(EPSG) == 5:
            fault['UTMx'], fault['UTMy'] = proj_ll2utm(fault['Longitude'], fault['Latitude'], EPSG)

            if len(ref_point) == 2:
                # Reference UTM coords and convert to km based on reference point
                fault['UTMx']    -= ref_point_utm[0]
                fault['UTMy']    -= ref_point_utm[1]
                fault['UTMx']    /= 1000
                fault['UTMy']    /= 1000

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


def get_data_paths(data_dir, file_format='*.grd'):
    """
    Get list of files in data_dir to ingest based on wildcard file_format
    """
    return np.sort(glob.glob(f'{data_dir}/{file_format}'))


def read_look_vectors(look_dir, filenames=['look_e.grd', 'look_n.grd', 'look_u.grd'], flatten=False):
    """
    Read east/north/up look vectors to array.
    """    
    _, _, look_e = read_grd(f'{look_dir}/{filenames[0]}')
    _, _, look_n = read_grd(f'{look_dir}/{filenames[1]}')
    _, _, look_u = read_grd(f'{look_dir}/{filenames[2]}')

    if flatten:
        return np.vstack((look_e.flatten(), look_n.flatten(), look_u.flatten())).T   
    else:
        return np.stack((look_e, look_n, look_u), axis=-1)


def modify_hdf5(file, key, data):
    """
    Update or create a dataset in an HDF5 file.
    """
    
    if key in file:
        del file[key]
    
    file.create_dataset(key, data=data)

    return file


def get_index_from_files(files, index_range, use_dates=True, use_datetime=True):
    """
    Extract scene dates from list of files given range of indices in filenames
    """

    # Get dates
    indices = [file.split('/')[-1][index_range[0]:index_range[1]] for file in files]

    # Return as Datetime objects if specified, otherwise just strings
    if use_dates:
        if use_datetime:
            return [dt.datetime.strptime(idx, '%Y%m%d') for idx in indices]
        else:
            return [f'{idx[:4]}-{idx[4:6]}-{idx[6:]}' for idx in indices]
    else:
        return indices


def read_insar_dataset(data_dir, file_format, xkey='lon', ykey='lat', check_lon=False, date_index_range=[], use_dates=True, use_datetime=True):
    """
    Read set of NetCDF files from a directory based on specified wildcard
    Example: /Users/evavra/Projects/SSAF/Data/InSAR/S1/timeseries/asc + ts_*_unfilt_ll.grd
    """

    # Load files
    files   = get_data_paths(data_dir, file_format=file_format)
    dataset = xr.open_mfdataset(files, combine='nested', concat_dim='date')

    # Check format of longitude coordinates
    if check_lon & (all(np.abs(0 <= dataset[xkey]) <= 360)):
        dataset = dataset.assign_coords(lon=(xkey, dataset[xkey].data - 360))

    # Add date dimension if specified
    if len(date_index_range) == 2:
        dates   = get_index_from_files(files, date_index_range, use_dates=use_dates, use_datetime=use_datetime)
        dataset = dataset.assign_coords(date=('date', dates))

    return dataset


def load_insar_dataset(data_dir, data_file_format, name, ref_point, data_factor=1, xkey='lon', coord_type='geographic', date_index_range=[0, 8], check_lon=False, reference_time_series=True,
                       velo_model_file='', velo_model_factor=-0.1, incremental=False, use_dates=True, use_datetime=True,
                       look_dir='', look_filenames=['look_e.grd', 'look_n.grd', 'look_u.grd',], mask_file=''):
    
    # Load InSAR data
    dataset = read_insar_dataset(data_dir, data_file_format, xkey=xkey, date_index_range=date_index_range, check_lon=check_lon, use_dates=use_dates, use_datetime=use_datetime)

    print(np.nanmean(np.abs(dataset['z'])), np.nanstd(np.abs(dataset['z'])))
    dataset['z'] = data_factor * dataset['z']
    print(np.nanmean(np.abs(dataset['z'])), np.nanstd(np.abs(dataset['z'])))


    if len(mask_file) > 0:
        # Apply mask consisting of ones and NaNs to dataset 
        _, _, mask = read_grd(mask_file)
        dim = dataset['z'].isel(date=0).shape

        if mask.shape == dim:
            print(f'Applying data mask from {mask_file}')
            dataset['z'] = dataset['z'] * mask
        else:
            print(f'Error! Mask size ({mask.shape}) does not match data size ({dim})')
    
    if coord_type == 'geographic':
        # Get full gridded coordinates
        lon, lat = np.meshgrid(dataset.coords['lon'], dataset.coords['lat'])

        # Get reference point and convert coordinates to km
        x, y = get_local_xy_coords(lon, lat, ref_point) 

        # Add Carttesian coordinates
        dataset.coords['x'] = (('lat', 'lon'), x)
        dataset.coords['y'] = (('lat', 'lon'), y)
    
    elif (dataset.coords['x'].size != dataset['z'].loc[dict(date=dataset.date[0])].size) & (dataset.coords['x'].size != dataset['z'].loc[dict(date=dataset.date[0])].size):
        print('Updating coordinates')
        dataset = dataset.rename({'x': 'x_rng','y': 'y_rng'})
        x, y    = np.meshgrid(dataset.coords['x_rng'], dataset.coords['y_rng'])
        dataset.coords['x'] = (('y_rng', 'x_rng'), x)   
        dataset.coords['y'] = (('y_rng', 'x_rng'), y)
        
    # Reference time series to start date
    if reference_time_series:
        dataset['z'] = dataset['z'] - dataset['z'].isel(date=0)
    
    # Convert from cummulative to incremental slip
    if incremental:
        print('Converting data to incremental displacements')

        for i in reversed(range(1, len(dataset.date) - 1)):

            # Load the slice of data corresponding to the selected date into memory
            z_slice = dataset['z'].isel(date=i).load()

            # # Modify the data (example: add a constant value to all elements)
            z_slice_modified = z_slice - dataset['z'].isel(date=i - 1)

            # Assign the modified data back to the Dataset
            dataset['z'].loc[dict(date=dataset.date[i])] = z_slice_modified

    # Add look vectors
    if len(look_dir) > 0:
        # Load look vectors
        look = read_look_vectors(look_dir, flatten=False, filenames=look_filenames)
    else:
        look = np.ones((dataset['z'].shape[2], dataset['z'].shape[1], 3))

    # Add look vectors
    dataset['look_e'] = (('y', 'x'), look[:, :, 0])
    dataset['look_n'] = (('y', 'x'), look[:, :, 1])
    dataset['look_u'] = (('y', 'x'), look[:, :, 2])

    # Remove interseismic velocity model
    if len(velo_model_file) > 0:
        # Load velocity model
        _, _, velo_model = read_grd(velo_model_file)
        velo_model *= velo_model_factor # Flip sign and convert to cm

        # # Convert to mm and remove interseismic deformation
        dataset['z'] = dataset['z'] - velo_model

    # Add name
    dataset.attrs["name"] = name
    return dataset