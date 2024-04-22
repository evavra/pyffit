import numpy as np
from pyffit.data import read_grd
from pyffit.utilities import get_local_xy_coords
from pyffit.quadtree import quadtree_unstructured, apply_unstructured_quadtree


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


def prepare_datasets(insar_files, look_dirs, weights, ref_point, EPSG='32611', rms_min=0.1, nan_frac_max=0.9, width_min=0.1, width_max=2):
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

 