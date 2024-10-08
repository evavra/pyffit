import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyffit.data import read_grd


def main():
    # # Data file
    # data_file   = 'tests/maduo/S1_LOS_D106.grd'

    # # Quadtree parameters
    # rms_min      = 1   # RMS threshold (data units)
    # nan_frac_max = 0.4 # Fraction of NaN values allowed per cell
    # width_min    = 2   # Minimum cell width (# of pixels)
    # width_max    = 300 # Maximum cell width (# of pixels)
    
    # # ---------- LOAD DATA ----------
    # x, y, data  = read_grd(data_file)
    # X, Y   = np.meshgrid(x, y)
    # extent = [x.min(), x.max(), y.max(), y.min()]

    # # ---------- QUADTREE ALGORITHM ----------
    # print('Performing quadtree downsampling...')
    # x_samp, y_samp, data_samp, pixel_count, nan_frac = quadtree(X, Y, data, 0, rms_min=rms_min, nan_frac_max=nan_frac_max, width_min=width_min, width_max=width_max,)
    
    # x_samp      = np.array(x_samp)[~np.isnan(x_samp)] 
    # y_samp      = np.array(y_samp)[~np.isnan(y_samp)]
    # data_samp      = np.array(data_samp)[~np.isnan(data_samp)]
    # pixel_count = np.array(pixel_count)[~np.isnan(pixel_count)]
    # nan_frac    = np.array(nan_frac)[~np.isnan(nan_frac)]

    # print('Number of initial data points:', data[~np.isnan(data)].size)
    # print('Number of downsampled data points:', len(data_samp))
    # print('Percent reduction: {:.2f}%'.format(100*(data[~np.isnan(data)].size - len(data_samp))/data[~np.isnan(data)].size))
    # print()

    # # ---------- PLOT ----------
    # fig = plt.figure(figsize=(14, 8.2))
    # projection = ccrs.PlateCarree()
    # ax  = fig.add_subplot(1, 1, 1, projection=projection)
    # ax.add_feature(cfeature.LAKES.with_scale('10m'))
    # ax.scatter(x_samp, y_samp, c=data_samp, cmap=cmap, transform=projection, zorder=100)
    # plt.show()
    
    return


def quadtree_unstructured(x, y, data, data_index, data_extent, fault=[], level=0, rms_min=0.1, nan_frac_max=0.9, width_min=1, width_max=50,
             x_samp=[], y_samp=[], data_samp=[], data_samp_std=[], data_tree=[], data_dims=[], data_extents=[], nan_frac=[]):
    """
    Perform recursive quadtree downsampling on unstructured data (n,) to obtain a down-sampled set of points (k,)
    based off of data gradients.

    -------------------
    |////////|////////|
    |/      /|       /|
    |/   0  /|    1  /|
    |/      /|       /|
    |/      /|////////|
    |--------|--------|
    |////////|/      /|
    |/       |/      /|
    |/   3   |/   2  /|
    |/       |/      /|
    |////////|////////|
    -------------------
    / - inclusive cell boundaries. each cell has one exclusive boundary


    INPUT:
        x, y (n,)        - x and y coordinates of dataset
        data (n,)        - gridded dataset to downsample
        index (n,)       - global indices of each original data point 
        cell_extent (4,) - spatial extent of cell

        Downsampling parameters:
        rms_min      - minimum cell root-mean-square threshhold
        nan_frac_max - maximum fraction of nan-values to permit within cell
        width_min    - minimum cell width (data units)
        width_max    - maximum cell width (data units)
        
        Recursive arguments:
        level - quadtree level. Iniial call should be level=0 and will be recursively increased throughout sampling
        All other keyword arguments are initializations of the output objects (see below).

    OUTPUT:
        x_samp, y_samp (k,) - downsampled coordinates
        data_samp (k,)      - downsamled data values
        data_samp_std (k,)  - downsampled data standard deviations
        data_index k,)      - global inidices of data contained within each quadtree cell.
        data_dims (k,)      - dimensions of each cell in x/y units.
        data_extents (k,)   - extent of each cell in x/y coordinates.
        nan_frac (k,)       - fraction of nan values in each cell.
    """

    # Get data and cell dimensions
    data_dim = np.array([data_extent[1] - data_extent[0], data_extent[3] - data_extent[2]]) # dimensions of current data cell
    cell_dim = data_dim/2                                               # dimesions of new sub-cells

    # Check to see if cell overlaps with fault

    # Compute initial data statistics
    data_mean     = np.nanmean(data)                  
    data_rms      = np.nanstd(data - data_mean) 

    # If (1) RMS is too high and cell size is greater than the minimum or 
    #    (2) if cell size is greater than the maximum, continue to sample
    if ((data_rms > rms_min) & all(cell_dim > width_min)) | all(cell_dim > width_max):
        # Get index slices for cells            
        cell_extents = [# x lower bound                         x upper bound                         y lower bound                         y upper bound
                        [data_extent[0]              , data_extent[0] + cell_dim[0], data_extent[2] + cell_dim[1], data_extent[3]              ], # [0] Top left
                        [data_extent[0] + cell_dim[0], data_extent[1]              , data_extent[2] + cell_dim[1], data_extent[3]              ], # [1] Top right
                        [data_extent[0] + cell_dim[0], data_extent[1]              , data_extent[2]              , data_extent[2] + cell_dim[1]], # [2] Bottom right
                        [data_extent[0]              , data_extent[0] + cell_dim[0], data_extent[2]              , data_extent[2] + cell_dim[1]], # [3] Bottom left    
                       ]    

        cell_slices = [# x lower bound                         x upper bound                         y lower bound                         y upper bound
                        (x >= data_extent[0])               & (x <= data_extent[0] + cell_dim[0]) & (y >  data_extent[2] + cell_dim[1]) & (y <= data_extent[3]),               # [0] Top left
                        (x >  data_extent[0] + cell_dim[0]) & (x <= data_extent[1])               & (y >= data_extent[2] + cell_dim[1]) & (y <= data_extent[3]),               # [1] Top right
                        (x >= data_extent[0] + cell_dim[0]) & (x <= data_extent[1])               & (y >= data_extent[2])               & (y <  data_extent[2] + cell_dim[1]), # [2] Bottom right
                        (x >= data_extent[0])               & (x <  data_extent[0] + cell_dim[0]) & (y >= data_extent[2])               & (y <= data_extent[2] + cell_dim[1]), # [3] Bottom left    
                       ]    

        for cell_extent, cell_slice in zip(cell_extents, cell_slices):

            # Get information for each new cell
            cell_x      = x[cell_slice]
            cell_y      = y[cell_slice]
            cell_data   = data[cell_slice]
            cell_index  = data_index[cell_slice]

            # Initiate recursive sampling
            x_samp, y_samp, data_samp, data_samp_std, data_tree, data_dims, data_extents, nan_frac = quadtree_unstructured(cell_x, cell_y, cell_data, cell_index, cell_extent,
                                rms_min=rms_min, nan_frac_max=nan_frac_max, width_min=width_min, width_max=width_max, level=level + 1,
                                x_samp=x_samp, y_samp=y_samp, data_samp=data_samp, data_samp_std=data_samp_std, data_tree=data_tree, data_dims=data_dims, data_extents=data_extents, nan_frac=nan_frac)

        if level == 0:
            # Once quadtree is complete and has returned to the top-level, return as np arrays
            x_samp        = np.array(x_samp)
            y_samp        = np.array(y_samp)
            data_samp     = np.array(data_samp)
            data_samp_std = np.array(data_samp_std)

        return x_samp, y_samp, data_samp, data_samp_std, data_tree, data_dims, data_extents, nan_frac

    # Otherwise, output downsampled data
    else:
        i_nan_data    = np.isnan(data)
        n_nan_data    = np.count_nonzero(i_nan_data)
        nan_frac_data = n_nan_data/(data.size)
        
        # if len(data) > 0:
        #     nan_frac_data = n_nan_data/(data.size)
        # else:
        #     nan_frac_data = np.nan

        # Output downsampled data or NaN if maximum NaN fraction is exceeded or if not enough pixels exist
        if (nan_frac_data > nan_frac_max) | (np.nanstd(data) == 0):
            results = (
                       [np.nan],       # x_samp
                       [np.nan],       # y_samp
                       [np.nan],       # data_samp
                       [np.nan],       # data_samp_std
                       [data_index],   # data_tree
                       [data_dim],     # data_dims
                       [data_extent],  # data_extents
                       [nan_frac_data] # nan_frac
                       )
        else:
            results = ([np.mean(x[~i_nan_data])], # x_samp 
                       [np.mean(y[~i_nan_data])], # y_samp 
                       [data_mean],               # data_samp 
                       [np.nanstd(data)],         # data_samp_std 
                       [data_index],              # data_tree 
                       [data_dim],                # data_dims 
                       [data_extent],             # data_extents 
                       [nan_frac_data]            # nan_frac 
                       )

        for output, result in zip([x_samp, y_samp, data_samp, data_samp_std, data_tree, data_dims, data_extents, nan_frac], results):
            output.extend(result)

        return x_samp, y_samp, data_samp, data_samp_std, data_tree, data_dims, data_extents, nan_frac


def quadtree(x, y, data, row_index, column_index, level=0, rms_min=0.1, nan_frac_max=0.9, width_min=3, width_max=1000,
             x_samp=[], y_samp=[], data_samp=[], data_samp_std=[], row_tree=[], column_tree=[], data_dims=[], nan_frac=[]):
    """
    Perform recursive quadtree downsampling on a grid.
    
    INPUT:
    x, y (m, n)         - x and y coordinates of dataset
    data (m, n)         - gridded dataset to downsample
    row_index (m, n)    -
    column_index (m, n) -

    level     - quadtree level. Iniial call should be level=0 and will be recursively increased
                     throughout sampling
    rms_min   - minimum cell root-mean-square threshhold
    width_min - minimum number of data points to be included in cell
    width_max - maximum number of data points to be included in cell

    OUTPUT:
    x_samp, y_samp - downsampled coordinates
    data_samp         - downsamled data values
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


    # If (1) RMS is too high and cell size is greater than the minimum or 
    #    (2) if cell size is greater than the maximum, continue to sample
    if ((data_rms >= rms_min) & all(cell_dim > width_min)) | all(cell_dim > width_max):
        # Get index slices for cells
        #               row slices                       column slices
        cell_slices = [(slice(cell_dim[0]),              slice(cell_dim[1])),              # Top left
                       (slice(cell_dim[0]),              slice(cell_dim[1], data_dim[1])), # Top right
                       (slice(cell_dim[0], data_dim[0]), slice(cell_dim[1], data_dim[1])), # Bottom right
                       (slice(cell_dim[0], data_dim[0]), slice(cell_dim[1]))]              # Bottom left

        for cell_slice in cell_slices:
            x_samp, y_samp, data_samp, data_samp_std, row_tree, column_tree, data_dims, nan_frac = quadtree(x[cell_slice], y[cell_slice], data[cell_slice], row_index[cell_slice], column_index[cell_slice],  
                                level=level + 1,rms_min=rms_min, nan_frac_max=nan_frac_max, width_min=width_min, width_max=width_max,
                                x_samp=x_samp, y_samp=y_samp, data_samp=data_samp, data_samp_std=data_samp_std, row_tree=row_tree, column_tree=column_tree, data_dims=data_dims, nan_frac=nan_frac)

        return x_samp, y_samp, data_samp, data_samp_std, row_tree, column_tree, data_dims, nan_frac

    # Otherwise, output downsampled data
    else:
        # Output downsampled data or NaN if maximum NaN fraction is exceeded
        if (nan_frac_data > nan_frac_max):
            results = ([np.nan],
                       [np.nan],
                       [np.nan],
                       [np.nan],
                       [row_index],
                       [column_index],
                       [cell_dim],
                       [nan_frac_data])
        else:
            results = ([np.mean(x[~i_nan_data])],
                       [np.mean(y[~i_nan_data])],
                       [data_mean],
                       [np.nanstd(data)],
                       [row_index],
                       [column_index],
                       [cell_dim],
                       [nan_frac_data])

        for output, result in zip([x_samp, y_samp, data_samp, data_samp_std, row_tree, column_tree, data_dims, nan_frac], results):
            output.extend(result)
        return x_samp, y_samp, data_samp, data_samp_std, row_tree, column_tree, data_dims, nan_frac


def appy_quadtree(x, y, data, row_tree, column_tree):
    """
    Downsample a gridded dataset using pre-constructed quadtree instance.

    INPUT:
    x, y (m, n) - gridded x/y coordinates
    data (m, n) - gridded data values 
    row_tree, column_tree (k,) - row and column grid indices corrsponding to each downsampled point.

    OUTPUT:
    x_samp, y_samp, data_samp, data_samp_std (k,) - quadtree downsampled coordinates, data, and bin STDs.
    """

    x_samp     = np.empty(len(row_tree))
    y_samp     = np.empty(len(row_tree))
    data_samp  = np.empty(len(row_tree))
    data_samp_std  = np.empty(len(row_tree))

    for i, (row_idx, col_idx) in enumerate(zip(row_tree, column_tree)):
        i_nan_data   = np.isnan(data[row_idx, col_idx])
        x_samp[i]    = np.mean(x[row_idx, col_idx][~i_nan_data])
        y_samp[i]    = np.mean(y[row_idx, col_idx][~i_nan_data])       
        data_samp[i] = np.nanmean(data[row_idx, col_idx])
        data_samp_std[i]  = np.nanstd(data[row_idx, col_idx])

    return x_samp, y_samp, data_samp, data_samp_std


def apply_unstructured_quadtree(x, y, data, data_tree, nan_frac_max):
    """
    Downsample a gridded dataset using pre-constructed quadtree instance.

    INPUT:
    x, y (m, n)    - gridded x/y coordinates
    data (m, n)    - gridded data values 
    data_tree (k,) - indices corrsponding to each downsampled point.
    nan_frac_max   - threshhold for maximum percentage of allowed NaN values within cell.
    OUTPUT:
    x_samp, y_samp, data_samp, data_samp_std (k,) - quadtree downsampled coordinates, data, and bin STDs.
    """

    x_samp        = np.empty(len(data_tree))
    y_samp        = np.empty(len(data_tree))
    data_samp     = np.empty(len(data_tree))
    data_samp_std = np.empty(len(data_tree))
    nan_frac      = np.empty(len(data_tree))

    # Loop over quadtree cells
    for i, idx in enumerate(data_tree):
        # Get nan information
        i_nan_data    = np.isnan(data[idx])
        n_nan_data    = np.count_nonzero(i_nan_data)
        if len(data[idx] > 0):
            nan_frac_data = n_nan_data/(data[idx].size)
        else:
            nan_frac_data = np.nan
        
        # Output downsampled data or NaN if maximum NaN fraction is exceeded, or the std = 0
        if (nan_frac_data > nan_frac_max or np.nanstd(data[idx]) == 0):
            x_samp[i]        = np.nan        # x_samp 
            y_samp[i]        = np.nan        # y_samp       
            data_samp[i]     = np.nan        # data_samp 
            data_samp_std[i] = np.nan        # data_samp 
            nan_frac[i]      = nan_frac_data # nan_frac 
        else:
            x_samp[i]        = np.mean(x[idx][~i_nan_data]) # x_samp 
            y_samp[i]        = np.mean(y[idx][~i_nan_data]) # y_samp       
            data_samp[i]     = np.nanmean(data[idx])        # data_samp 
            data_samp_std[i] = np.nanstd(data[idx])         # data_samp 
            nan_frac[i]      = nan_frac_data                # nan_frac 
           
    return x_samp, y_samp, data_samp, data_samp_std, nan_frac



def quadtree_unstructured_new(x, y, data, data_index, data_extent, fault=[], level=0, mean_low=0.2, mean_up=0.5,rms_min=0.1, nan_frac_max=0.9, width_min=1, width_max=50,
             x_samp=[], y_samp=[], data_samp=[], data_samp_std=[], data_tree=[], data_dims=[], data_extents=[], nan_frac=[]):
    """
    Perform recursive quadtree downsampling on unstructured data (n,) to obtain a down-sampled set of points (k,)
    based off of data gradients. Also, if the area has a high mean value, sample it more

    -------------------
    |////////|////////|
    |/      /|       /|
    |/   0  /|    1  /|
    |/      /|       /|
    |/      /|////////|
    |--------|--------|
    |////////|/      /|
    |/       |/      /|
    |/   3   |/   2  /|
    |/       |/      /|
    |////////|////////|
    -------------------
    / - inclusive cell boundaries. each cell has one exclusive boundary


    INPUT:
        x, y (n,)        - x and y coordinates of dataset
        data (n,)        - gridded dataset to downsample
        index (n,)       - global indices of each original data point 
        cell_extent (4,) - spatial extent of cell

        Downsampling parameters:
        rms_min      - minimum cell root-mean-square threshhold
        nan_frac_max - maximum fraction of nan-values to permit within cell
        width_min    - minimum cell width (data units)
        width_max    - maximum cell width (data units)
        mean_low     - minimum mean value threshold for sampling
        mean_up      - maximum mean value threshold for sampling
        
        Recursive arguments:
        level - quadtree level. Iniial call should be level=0 and will be recursively increased throughout sampling
        All other keyword arguments are initializations of the output objects (see below).

    OUTPUT:
        x_samp, y_samp (k,) - downsampled coordinates
        data_samp (k,)      - downsamled data values
        data_samp_std (k,)  - downsampled data standard deviations
        data_index k,)      - global inidices of data contained within each quadtree cell.
        data_dims (k,)      - dimensions of each cell in x/y units.
        data_extents (k,)   - extent of each cell in x/y coordinates.
        nan_frac (k,)       - fraction of nan values in each cell.
    """

    # Get data and cell dimensions
    data_dim = np.array([data_extent[1] - data_extent[0], data_extent[3] - data_extent[2]]) # dimensions of current data cell
    cell_dim = data_dim/2                                               # dimesions of new sub-cells

    # Check to see if cell overlaps with fault

    # Compute initial data statistics
    data_mean     = np.nanmean(data)
    #data_max      = np.nanmax(np.abs(data))
    data_rms      = np.nanstd(data - data_mean)
    mean_up = 0 #not using this parameter this time
    #print('data_rms is',data_rms)
    # If (1) RMS is too high and cell size is greater than the minimum or 
    #    (2) if cell size is greater than the maximum, continue to sample
    #if ((data_rms > rms_min) & all(cell_dim > width_min)) | ((data_rms <= 0.002) & (np.abs(data_mean) >= mean_low)  & all(cell_dim > width_min)) | all(cell_dim > width_max):
    if ((data_rms <= 0.003) & (np.abs(data_mean) >= mean_low)) | ((data_rms > rms_min) & all(cell_dim > width_min)) | all(cell_dim > width_max):    
        # Get index slices for cells
        #print('data mean is:',data_mean)
        cell_extents = [# x lower bound                         x upper bound                         y lower bound                         y upper bound
                        [data_extent[0]              , data_extent[0] + cell_dim[0], data_extent[2] + cell_dim[1], data_extent[3]              ], # [0] Top left
                        [data_extent[0] + cell_dim[0], data_extent[1]              , data_extent[2] + cell_dim[1], data_extent[3]              ], # [1] Top right
                        [data_extent[0] + cell_dim[0], data_extent[1]              , data_extent[2]              , data_extent[2] + cell_dim[1]], # [2] Bottom right
                        [data_extent[0]              , data_extent[0] + cell_dim[0], data_extent[2]              , data_extent[2] + cell_dim[1]], # [3] Bottom left    
                       ]    

        cell_slices = [# x lower bound                         x upper bound                         y lower bound                         y upper bound
                        (x >= data_extent[0])               & (x <= data_extent[0] + cell_dim[0]) & (y >  data_extent[2] + cell_dim[1]) & (y <= data_extent[3]),               # [0] Top left
                        (x >  data_extent[0] + cell_dim[0]) & (x <= data_extent[1])               & (y >= data_extent[2] + cell_dim[1]) & (y <= data_extent[3]),               # [1] Top right
                        (x >= data_extent[0] + cell_dim[0]) & (x <= data_extent[1])               & (y >= data_extent[2])               & (y <  data_extent[2] + cell_dim[1]), # [2] Bottom right
                        (x >= data_extent[0])               & (x <  data_extent[0] + cell_dim[0]) & (y >= data_extent[2])               & (y <= data_extent[2] + cell_dim[1]), # [3] Bottom left    
                       ]    

        
        for cell_extent, cell_slice in zip(cell_extents, cell_slices):

            # Get information for each new cell
            cell_x      = x[cell_slice]
            cell_y      = y[cell_slice]
            cell_data   = data[cell_slice]
            cell_index  = data_index[cell_slice]
        

            # Initiate recursive sampling
            x_samp, y_samp, data_samp, data_samp_std, data_tree, data_dims, data_extents, nan_frac = quadtree_unstructured_new(cell_x, cell_y, cell_data, cell_index, cell_extent,
                                rms_min=rms_min, mean_low=mean_low,mean_up=mean_up,nan_frac_max=nan_frac_max, width_min=width_min, width_max=width_max, level=level + 1,
                                x_samp=x_samp, y_samp=y_samp, data_samp=data_samp, data_samp_std=data_samp_std, data_tree=data_tree, data_dims=data_dims, data_extents=data_extents, nan_frac=nan_frac)

            

        if level == 0:
            # Once quadtree is complete and has returned to the top-level, return as np arrays
            x_samp        = np.array(x_samp)
            y_samp        = np.array(y_samp)
            data_samp     = np.array(data_samp)
            data_samp_std = np.array(data_samp_std)

        return x_samp, y_samp, data_samp, data_samp_std, data_tree, data_dims, data_extents, nan_frac

    # Otherwise, output downsampled data
    else:
        i_nan_data    = np.isnan(data)
        n_nan_data    = np.count_nonzero(i_nan_data)
        #nan_frac_data = n_nan_data/(data.size)
        
        if len(data) > 0:
             nan_frac_data = n_nan_data/(data.size)
        else:
             nan_frac_data = np.nan

        # Output downsampled data or NaN if maximum NaN fraction is exceeded or if not enough pixels exist
        if (nan_frac_data > nan_frac_max) | (np.nanstd(data) == 0):
            results = (
                       [np.nan],       # x_samp
                       [np.nan],       # y_samp
                       [np.nan],       # data_samp
                       [np.nan],       # data_samp_std
                       [data_index],   # data_tree
                       [data_dim],     # data_dims
                       [data_extent],  # data_extents
                       [nan_frac_data] # nan_frac
                       )
        else:
            results = ([np.mean(x[~i_nan_data])], # x_samp 
                       [np.mean(y[~i_nan_data])], # y_samp 
                       [data_mean],               # data_samp 
                       [np.nanstd(data)],         # data_samp_std 
                       [data_index],              # data_tree 
                       [data_dim],                # data_dims 
                       [data_extent],             # data_extents 
                       [nan_frac_data]            # nan_frac 
                       )

        for output, result in zip([x_samp, y_samp, data_samp, data_samp_std, data_tree, data_dims, data_extents, nan_frac], results):
            output.extend(result)

        return x_samp, y_samp, data_samp, data_samp_std, data_tree, data_dims, data_extents, nan_frac




if __name__ == '__main__':
    main()
