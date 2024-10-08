import numpy as np
from pyffit.data import read_grd
from pyffit.utilities import get_local_xy_coords
from pyffit.quadtree import quadtree_unstructured, apply_unstructured_quadtree, quadtree_unstructured_new


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
        dataset = insar_files[i].split('/')[-1][:-4]

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
                                                                                                                        rms_min=rms_min, nan_frac_max=nan_frac_max, width_min=width_min, width_max=width_max,x_samp=[], y_samp=[], data_samp=[], data_samp_std=[], data_tree=[], data_dims=[], data_extents=[], nan_frac=[])
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

def prepare_datasets_smooth(insar_files, smooth_files, look_dirs, weights, ref_point, EPSG='32611', rms_min=0.1, mean_low=0.2,mean_up=0.5,nan_frac_max=0.9, width_min=0.1, width_max=2):
    """
    Ingest InSAR data for finite fault modeling. In addition to that, sample the original data based on smoothed data.
	#Xiaoyu Zou, 5/16/2024
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
        # Get smooth and original dataset name
        dataset = insar_files[i].split('/')[-1][:-4]
        smooth = smooth_files[i].split('/')[-1][:-4]

        # Get dataset weight
        weight  = weights[i]

        # Read data
        lon_rng, lat_rng, data = read_grd(insar_files[i])
        lon_rng_smooth,lat_rng_smooth, data_smooth = read_grd(smooth_files[i])
        look                   = load_look_vectors(look_dirs[i])

        # Get full coordinates
        lon, lat = np.meshgrid(lon_rng, lat_rng)
        lon_smooth, lat_smooth = np.meshgrid(lon_rng_smooth,lat_rng_smooth)

        # Get reference point and convert coordinates to km
        x, y   = get_local_xy_coords(lon, lat, ref_point, EPSG=EPSG) 
        extent = [np.min(x), np.max(x), np.max(y), np.min(y)]
        
        x_smooth, y_smooth   = get_local_xy_coords(lon_smooth, lat_smooth, ref_point, EPSG=EPSG) 
        extent_smooth = [np.min(x_smooth), np.max(x_smooth), np.max(y_smooth), np.min(y_smooth)]
        

        # Downsample the smoothed data using quadtree algorithm
        data_extent = [x_smooth.min(), x_smooth.max(), y_smooth.min(), y_smooth.max()]
        data_index  = np.arange(0, data_smooth.size)
        x_samp, y_samp, data_samp, data_samp_std, data_tree, cell_dims, cell_extents, nan_frac = quadtree_unstructured_new(x_smooth.flatten(), y_smooth.flatten(), data_smooth.flatten(), data_index, data_extent,rms_min=rms_min, nan_frac_max=nan_frac_max, mean_low=mean_low,mean_up=mean_up,width_min=width_min, width_max=width_max,x_samp=[], y_samp=[], data_samp=[], data_samp_std=[], data_tree=[], data_dims=[], data_extents=[], nan_frac=[])
                                                                                                                        
        i_nans = np.isnan(data_samp)   
        x_samp, y_samp, data_samp, data_samp_std,nan_frac = apply_unstructured_quadtree(x.flatten(), y.flatten(), data.flatten(), data_tree, nan_frac_max)
        _, _, look_e_samp, _, _ = apply_unstructured_quadtree(x.flatten(), y.flatten(), look[:, 0], data_tree, nan_frac_max)
        _, _, look_n_samp, _, _ = apply_unstructured_quadtree(x.flatten(), y.flatten(), look[:, 1], data_tree, nan_frac_max)
        _, _, look_u_samp, _, _ = apply_unstructured_quadtree(x.flatten(), y.flatten(), look[:, 2], data_tree, nan_frac_max)
        look_samp = np.vstack((look_e_samp, look_n_samp, look_u_samp)).T
        i_nans = np.isnan(data_samp)
       
        # subsample the large valued data
        #indicies1 = np.where(np.abs(data_samp) >= mean_low)[0] #high value data to be subsampled
        #indicies2 = np.where(np.abs(data_samp) <  mean_low)[0]

        
        #print('subsampled indicies length:',len(subsampled_indicies))
        #print('original indicies length:',len(indicies1))
        
        #large_y_samp=y_samp[indicies1]
        #sorted_indicies = np.argsort(large_y_samp)
        #large_y_samp=large_y_samp[sorted_indicies]
        #large_data=data_samp[sorted_indicies]
        #large_x_samp=data_samp[sorted_indicies]
        #large_look_samp=look_samp[sorted_indicies]
        #step_size=2
        #subsampled_indicies = sorted_indicies[::step_size]
        #subsampled_indicies = np.random.choice(len(large_y_samp), size=num_samples, p=weights, replace=False)
        #print('subsampled indicies length:',len(subsampled_indicies))
        #large_data=data_samp[subsampled_indicies]
        #large_x_samp=x_samp[subsampled_indicies]
        #large_y_samp=y_samp[subsampled_indicies]
        #large_look_samp=look_samp[subsampled_indicies]


        #small_data=data_samp[indicies2]
        #small_x_samp=x_samp[indicies2]
        #small_y_samp=y_samp[indicies2]
        #small_look_samp=look_samp[indicies2]
        
        #x_samp=np.concatenate((large_x_samp,small_x_samp))
        #y_samp=np.concatenate((large_y_samp,small_y_samp))
        #data_samp=np.concatenate((large_data,small_data))
        #look_samp=np.concatenate((large_look_samp,small_look_samp))




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


def load_tri_samples(tri_dir):
    """
    Load x_samp, y_samp, data_samp, look_samp, and data_std from the triangular sampled data
    """
    x_samp = np.loadtxt(f'{tri_dir}/X.txt')
    y_samp = np.loadtxt(f'{tri_dir}/Y.txt')
    data_samp = np.loadtxt(f'{tri_dir}/data.txt')
    look_samp = np.loadtxt(f'{tri_dir}/look.txt',delimiter=',')
    data_samp_std = np.loadtxt(f'{tri_dir}/data_std.txt')	
    return x_samp, y_samp, data_samp, look_samp, data_samp_std

def prepare_datasets_tri(insar_files, look_dirs, tri_dirs, weights, ref_point, EPSG='32611'):
    """
    Ingest InSAR data for finite fault modeling, using triangular R-based sampling

    Workflow for each dataset
        1. Read interferogram
        2. Read look vectors
        3. Project to local Cartesian coordinate system
        4. Downsample using triangular R-based algorithm

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
        dataset = insar_files[i].split('/')[-1][:-4]

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
        #x_samp, y_samp, data_samp, data_samp_std, data_tree, cell_dims, cell_extents, nan_frac = quadtree_unstructured(x.flatten(), y.flatten(), data.flatten(), data_index, data_extent, 
        #                                                                                                                rms_min=rms_min, nan_frac_max=nan_frac_max, width_min=width_min, width_max=width_max,
        #                                                                                                                x_samp=[], y_samp=[], data_samp=[], data_samp_std=[], data_tree=[], data_dims=[], data_extents=[], nan_frac=[])
        
        
        #i_nans = np.isnan(data_samp)
        #_, _, look_e_samp, _, _ = apply_unstructured_quadtree(x.flatten(), y.flatten(), look[:, 0], data_tree, nan_frac_max)
        #_, _, look_n_samp, _, _ = apply_unstructured_quadtree(x.flatten(), y.flatten(), look[:, 1], data_tree, nan_frac_max)
        #_, _, look_u_samp, _, _ = apply_unstructured_quadtree(x.flatten(), y.flatten(), look[:, 2], data_tree, nan_frac_max)
       #look_samp = np.vstack((look_e_samp, look_n_samp, look_u_samp)).T
       
       
        x_samp, y_samp, data_samp, look_samp, data_samp_std = load_tri_samples(tri_dirs[i])
        i_nans = np.isnan(data_samp)

        #print(f'Quadtree points: {len(data_samp)}')
        print('Triangular Samples Loaded! No Quadtree is used.')

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
        #datasets[dataset]['data_tree']     = data_tree
        #datasets[dataset]['cell_dims']     = cell_dims
        #datasets[dataset]['cell_extents']  = cell_extents
        #datasets[dataset]['nan_frac']      = nan_frac
        datasets[dataset]['look_samp']     = look_samp
        datasets[dataset]['i_nans']        = i_nans
        datasets[dataset]['weight']        = weight

    return datasets


def prepare_datasets_tri_misfit(insar_files, look_dirs, tri_dirs, misfit_dirs, weights, ref_point, EPSG='32611',rms_min=0.1, nan_frac_max=0.9, width_min=0.1, width_max=2):
    """
    Sample datasets based on misfits, and add them back to the already sampled dataset

    Workflow for each dataset
        1. Read interferogram
        2. Read look vectors
        3. Project to local Cartesian coordinate system
        4. Downsample using triangular R-based algorithm

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
        dataset = insar_files[i].split('/')[-1][:-4]

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
        

        # Downsample using quadtree algorithm, based on misfit
        data_extent = [x.min(), x.max(), y.min(), y.max()]
        data_index  = np.arange(0, data.size)
        misfit = np.loadtxt(f'{misfit_dirs[i]}/misfit.txt')
        
    
        x_samp, y_samp, data_samp, data_samp_std, data_tree, cell_dims, cell_extents, nan_frac = quadtree_unstructured(x.flatten(), y.flatten(), misfit, data_index, data_extent, 
                                                                                                                        rms_min=rms_min, nan_frac_max=nan_frac_max, width_min=width_min, width_max=width_max,x_samp=[], y_samp=[], data_samp=[], data_samp_std=[], data_tree=[], data_dims=[], data_extents=[], nan_frac=[])
        
        
        x_samp_misfit, y_samp_misfit, data_samp_misfit, data_samp_std_misfit, nan_frac = apply_unstructured_quadtree(x.flatten(), y.flatten(), data.flatten(), data_tree, nan_frac_max)
        _, _, look_e_samp, _, _ = apply_unstructured_quadtree(x.flatten(), y.flatten(), look[:, 0], data_tree, nan_frac_max)
        _, _, look_n_samp, _, _ = apply_unstructured_quadtree(x.flatten(), y.flatten(), look[:, 1], data_tree, nan_frac_max)
        _, _, look_u_samp, _, _ = apply_unstructured_quadtree(x.flatten(), y.flatten(), look[:, 2], data_tree, nan_frac_max)
        look_samp_misfit = np.vstack((look_e_samp, look_n_samp, look_u_samp)).T
        

       
       
        x_samp_ori, y_samp_ori, data_samp_ori, look_samp_ori, data_samp_std_ori = load_tri_samples(tri_dirs[i])
        
        
        # Find elements in x_samp_misfit that are not in x_samp_ori (currently, it only works when all elements in x are unique)
        mask = np.isin(x_samp_misfit, x_samp_ori, invert=True)

		# Filter arrays based on these indices
        filtered_x_samp_misfit = x_samp_misfit[mask]
        filtered_y_samp_misfit = y_samp_misfit[mask]
        filtered_data_samp_misfit = data_samp_misfit[mask]	
        filtered_data_samp_std_misfit = data_samp_std_misfit[mask]
        filtered_look_samp_misfit = look_samp_misfit[mask,:]

		# Combine the original arrays with the filtered ones 
        x_samp = np.concatenate((x_samp_ori, filtered_x_samp_misfit))
        y_samp = np.concatenate((y_samp_ori, filtered_y_samp_misfit))	
        data_samp = np.concatenate((data_samp_ori, filtered_data_samp_misfit))
        data_samp_std = np.concatenate((data_samp_std_ori, filtered_data_samp_std_misfit))
        look_samp = np.concatenate((look_samp_ori, filtered_look_samp_misfit))
		        
        i_nans = np.isnan(data_samp)
        

        print(f'Points after adding misfit samples: {len(data_samp)}')
        #print('Triangular Samples Loaded! No Quadtree is used. Misfit-based sampling is added')

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
        #datasets[dataset]['data_tree']     = data_tree
        #datasets[dataset]['cell_dims']     = cell_dims
        #datasets[dataset]['cell_extents']  = cell_extents
        #datasets[dataset]['nan_frac']      = nan_frac
        datasets[dataset]['look_samp']     = look_samp
        datasets[dataset]['i_nans']        = i_nans
        datasets[dataset]['weight']        = weight

    return datasets

def prepare_datasets_misfit(insar_files, look_dirs, misfit_dirs, weights, ref_point, EPSG='32611', rms_min=0.1, nan_frac_max=0.9, width_min=0.1, width_max=2):
    """
    Direct sample the misfit without adding to the existing sampled data

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
        dataset = insar_files[i].split('/')[-1][:-4]

        # Get dataset weight
        weight  = weights[i]

        # Read data
        lon_rng, lat_rng, data = read_grd(insar_files[i])
        look                   = load_look_vectors(look_dirs[i])

        # Get full coordinates
        lon, lat = np.meshgrid(lon_rng, lat_rng)
        
        # Get the misfit data
        misfit = np.loadtxt(f'{misfit_dirs[i]}/misfit.txt')

        # Get reference point and convert coordinates to km
        x, y   = get_local_xy_coords(lon, lat, ref_point, EPSG=EPSG) 
        extent = [np.min(x), np.max(x), np.max(y), np.min(y)]

        # Downsample using quadtree algorithm
        data_extent = [x.min(), x.max(), y.min(), y.max()]
        data_index  = np.arange(0, data.size)
        x_samp, y_samp, data_samp, data_samp_std, data_tree, cell_dims, cell_extents, nan_frac = quadtree_unstructured(x.flatten(), y.flatten(), misfit, data_index, data_extent, 
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
