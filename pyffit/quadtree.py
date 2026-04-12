import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyffit.data import read_grd


def main():
    return


def quadtree_unstructured(x, y, data, data_index, data_extent, fault=[], level=0, rms_min=0.1, nan_frac_max=0.9, width_min=1, width_max=50,
             x_samp=[], y_samp=[], data_samp=[], data_samp_std=[], data_tree=[], data_dims=[], data_extents=[], nan_frac=[]):
    """
    Perform recursive quadtree downsampling on unstructured data (n,) to obtain a down-sampled set of points (k,)
    based off of data gradients.
    """

    # FIX: early exit for empty or all-NaN cells.
    # Without this guard, np.nanmean/np.nanstd called on an empty array emit:
    #   RuntimeWarning: Mean of empty slice
    #   RuntimeWarning: Degrees of freedom <= 0 for slice
    #   RuntimeWarning: invalid value encountered in scalar divide
    if data.size == 0 or np.all(np.isnan(data)):
        return x_samp, y_samp, data_samp, data_samp_std, data_tree, data_dims, data_extents, nan_frac

    # Get data and cell dimensions
    data_dim = np.array([data_extent[1] - data_extent[0], data_extent[3] - data_extent[2]])
    cell_dim = data_dim/2

    # Compute initial data statistics
    data_mean = np.nanmean(data)                  
    data_rms  = np.nanstd(data - data_mean) 

    if ((data_rms > rms_min) & all(cell_dim > width_min)) | all(cell_dim > width_max):
        cell_extents = [
                        [data_extent[0]              , data_extent[0] + cell_dim[0], data_extent[2] + cell_dim[1], data_extent[3]              ], # [0] Top left
                        [data_extent[0] + cell_dim[0], data_extent[1]              , data_extent[2] + cell_dim[1], data_extent[3]              ], # [1] Top right
                        [data_extent[0] + cell_dim[0], data_extent[1]              , data_extent[2]              , data_extent[2] + cell_dim[1]], # [2] Bottom right
                        [data_extent[0]              , data_extent[0] + cell_dim[0], data_extent[2]              , data_extent[2] + cell_dim[1]], # [3] Bottom left    
                       ]    

        cell_slices = [
                        (x >= data_extent[0])               & (x <= data_extent[0] + cell_dim[0]) & (y >  data_extent[2] + cell_dim[1]) & (y <= data_extent[3]),               # [0] Top left
                        (x >  data_extent[0] + cell_dim[0]) & (x <= data_extent[1])               & (y >= data_extent[2] + cell_dim[1]) & (y <= data_extent[3]),               # [1] Top right
                        (x >= data_extent[0] + cell_dim[0]) & (x <= data_extent[1])               & (y >= data_extent[2])               & (y <  data_extent[2] + cell_dim[1]), # [2] Bottom right
                        (x >= data_extent[0])               & (x <  data_extent[0] + cell_dim[0]) & (y >= data_extent[2])               & (y <= data_extent[2] + cell_dim[1]), # [3] Bottom left    
                       ]    

        for cell_extent, cell_slice in zip(cell_extents, cell_slices):
            cell_x     = x[cell_slice]
            cell_y     = y[cell_slice]
            cell_data  = data[cell_slice]
            cell_index = data_index[cell_slice]

            x_samp, y_samp, data_samp, data_samp_std, data_tree, data_dims, data_extents, nan_frac = quadtree_unstructured(cell_x, cell_y, cell_data, cell_index, cell_extent,
                                rms_min=rms_min, nan_frac_max=nan_frac_max, width_min=width_min, width_max=width_max, level=level + 1,
                                x_samp=x_samp, y_samp=y_samp, data_samp=data_samp, data_samp_std=data_samp_std, data_tree=data_tree, data_dims=data_dims, data_extents=data_extents, nan_frac=nan_frac)

        if level == 0:
            x_samp        = np.array(x_samp)
            y_samp        = np.array(y_samp)
            data_samp     = np.array(data_samp)
            data_samp_std = np.array(data_samp_std)

        return x_samp, y_samp, data_samp, data_samp_std, data_tree, data_dims, data_extents, nan_frac

    else:
        i_nan_data    = np.isnan(data)
        n_nan_data    = np.count_nonzero(i_nan_data)
        nan_frac_data = n_nan_data/(data.size)

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
            results = ([np.mean(x[~i_nan_data])],
                       [np.mean(y[~i_nan_data])],
                       [data_mean],
                       [np.nanstd(data)],
                       [data_index],
                       [data_dim],
                       [data_extent],
                       [nan_frac_data])

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

    level     - quadtree level. Initial call should be level=0
    rms_min   - minimum cell root-mean-square threshold
    width_min - minimum number of data points to be included in cell
    width_max - maximum number of data points to be included in cell

    OUTPUT:
    x_samp, y_samp - downsampled coordinates
    data_samp      - downsampled data values
    std_samp       - downsampled data standard deviations
    """

    # FIX: early exit for empty or all-NaN cells.
    # The recursive subdivision can produce zero-size or fully-masked sub-arrays;
    # calling np.nanmean / np.nanstd on them generates RuntimeWarnings that
    # propagate NaN/inf all the way into the MCMC log-probability, ultimately
    # causing the UnboundLocalError on samp_prob in mcmc2.py.
    if data.size == 0 or np.all(np.isnan(data)):
        return x_samp, y_samp, data_samp, data_samp_std, row_tree, column_tree, data_dims, nan_frac

    # Compute initial data statistics
    data_mean     = np.nanmean(data)                  
    data_rms      = np.nanstd(data - data_mean) 
    i_nan_data    = np.isnan(data)
    n_nan_data    = np.count_nonzero(i_nan_data)
    nan_frac_data = n_nan_data/data.size
    data_dim      = np.array(data.shape)
    cell_dim      = data_dim//2

    if ((data_rms >= rms_min) & all(cell_dim > width_min)) | all(cell_dim > width_max):
        cell_slices = [(slice(cell_dim[0]),              slice(cell_dim[1])),              # Top left
                       (slice(cell_dim[0]),              slice(cell_dim[1], data_dim[1])), # Top right
                       (slice(cell_dim[0], data_dim[0]), slice(cell_dim[1], data_dim[1])), # Bottom right
                       (slice(cell_dim[0], data_dim[0]), slice(cell_dim[1]))]              # Bottom left

        for cell_slice in cell_slices:
            x_samp, y_samp, data_samp, data_samp_std, row_tree, column_tree, data_dims, nan_frac = quadtree(x[cell_slice], y[cell_slice], data[cell_slice], row_index[cell_slice], column_index[cell_slice],  
                                level=level + 1, rms_min=rms_min, nan_frac_max=nan_frac_max, width_min=width_min, width_max=width_max,
                                x_samp=x_samp, y_samp=y_samp, data_samp=data_samp, data_samp_std=data_samp_std, row_tree=row_tree, column_tree=column_tree, data_dims=data_dims, nan_frac=nan_frac)

        return x_samp, y_samp, data_samp, data_samp_std, row_tree, column_tree, data_dims, nan_frac

    else:
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
    """

    x_samp        = np.empty(len(row_tree))
    y_samp        = np.empty(len(row_tree))
    data_samp     = np.empty(len(row_tree))
    data_samp_std = np.empty(len(row_tree))

    for i, (row_idx, col_idx) in enumerate(zip(row_tree, column_tree)):
        i_nan_data       = np.isnan(data[row_idx, col_idx])
        x_samp[i]        = np.mean(x[row_idx, col_idx][~i_nan_data])
        y_samp[i]        = np.mean(y[row_idx, col_idx][~i_nan_data])       
        data_samp[i]     = np.nanmean(data[row_idx, col_idx])
        data_samp_std[i] = np.nanstd(data[row_idx, col_idx])

    return x_samp, y_samp, data_samp, data_samp_std


def apply_unstructured_quadtree(x, y, data, data_tree, nan_frac_max):
    """
    Downsample a gridded dataset using pre-constructed quadtree instance.
    """

    x_samp        = np.empty(len(data_tree))
    y_samp        = np.empty(len(data_tree))
    data_samp     = np.empty(len(data_tree))
    data_samp_std = np.empty(len(data_tree))
    nan_frac      = np.empty(len(data_tree))

    for i, idx in enumerate(data_tree):
        i_nan_data    = np.isnan(data[idx])
        n_nan_data    = np.count_nonzero(i_nan_data)
        if len(data[idx] > 0):
            nan_frac_data = n_nan_data/(data[idx].size)
        else:
            nan_frac_data = np.nan
        
        if (nan_frac_data > nan_frac_max or np.nanstd(data[idx]) == 0):
            x_samp[i]        = np.nan
            y_samp[i]        = np.nan
            data_samp[i]     = np.nan
            data_samp_std[i] = np.nan
            nan_frac[i]      = nan_frac_data
        else:
            x_samp[i]        = np.mean(x[idx][~i_nan_data])
            y_samp[i]        = np.mean(y[idx][~i_nan_data])
            data_samp[i]     = np.nanmean(data[idx])
            data_samp_std[i] = np.nanstd(data[idx])
            nan_frac[i]      = nan_frac_data
           
    return x_samp, y_samp, data_samp, data_samp_std, nan_frac


def quadtree_unstructured_new(x, y, data, data_index, data_extent, fault=[], level=0, mean_low=0.2, rms_min=0.1, rms_min2=0.01, nan_frac_max=0.9, width_min=1, width_max=50,
             x_samp=[], y_samp=[], data_samp=[], data_samp_std=[], data_tree=[], data_dims=[], data_extents=[], nan_frac=[]):
    """
    Perform recursive quadtree downsampling on unstructured data (n,) to obtain a down-sampled set of points (k,)
    based off of data gradients. Also, if the area has a high mean value, sample it more.
    """

    # FIX: early exit for empty or all-NaN cells.
    if data.size == 0 or np.all(np.isnan(data)):
        return x_samp, y_samp, data_samp, data_samp_std, data_tree, data_dims, data_extents, nan_frac

    # Get data and cell dimensions
    data_dim = np.array([data_extent[1] - data_extent[0], data_extent[3] - data_extent[2]])
    cell_dim = data_dim/2

    # Compute initial data statistics
    data_mean = np.nanmean(data)
    data_rms  = np.nanstd(data - data_mean)

    # for lobes, the default width_min is 3 times of the width_min input
    if ((data_rms <= rms_min2) & (np.abs(data_mean) >= mean_low) & all(cell_dim > width_min*3)) | ((data_rms > rms_min) & all(cell_dim > width_min)) | all(cell_dim > width_max):    
        cell_extents = [
                        [data_extent[0]              , data_extent[0] + cell_dim[0], data_extent[2] + cell_dim[1], data_extent[3]              ], # [0] Top left
                        [data_extent[0] + cell_dim[0], data_extent[1]              , data_extent[2] + cell_dim[1], data_extent[3]              ], # [1] Top right
                        [data_extent[0] + cell_dim[0], data_extent[1]              , data_extent[2]              , data_extent[2] + cell_dim[1]], # [2] Bottom right
                        [data_extent[0]              , data_extent[0] + cell_dim[0], data_extent[2]              , data_extent[2] + cell_dim[1]], # [3] Bottom left    
                       ]    

        cell_slices = [
                        (x >= data_extent[0])               & (x <= data_extent[0] + cell_dim[0]) & (y >  data_extent[2] + cell_dim[1]) & (y <= data_extent[3]),               # [0] Top left
                        (x >  data_extent[0] + cell_dim[0]) & (x <= data_extent[1])               & (y >= data_extent[2] + cell_dim[1]) & (y <= data_extent[3]),               # [1] Top right
                        (x >= data_extent[0] + cell_dim[0]) & (x <= data_extent[1])               & (y >= data_extent[2])               & (y <  data_extent[2] + cell_dim[1]), # [2] Bottom right
                        (x >= data_extent[0])               & (x <  data_extent[0] + cell_dim[0]) & (y >= data_extent[2])               & (y <= data_extent[2] + cell_dim[1]), # [3] Bottom left    
                       ]    

        for cell_extent, cell_slice in zip(cell_extents, cell_slices):
            cell_x     = x[cell_slice]
            cell_y     = y[cell_slice]
            cell_data  = data[cell_slice]
            cell_index = data_index[cell_slice]

            x_samp, y_samp, data_samp, data_samp_std, data_tree, data_dims, data_extents, nan_frac = quadtree_unstructured_new(cell_x, cell_y, cell_data, cell_index, cell_extent,
                                rms_min=rms_min, mean_low=mean_low, rms_min2=rms_min2, nan_frac_max=nan_frac_max, width_min=width_min, width_max=width_max, level=level + 1,
                                x_samp=x_samp, y_samp=y_samp, data_samp=data_samp, data_samp_std=data_samp_std, data_tree=data_tree, data_dims=data_dims, data_extents=data_extents, nan_frac=nan_frac)

        if level == 0:
            x_samp        = np.array(x_samp)
            y_samp        = np.array(y_samp)
            data_samp     = np.array(data_samp)
            data_samp_std = np.array(data_samp_std)

        return x_samp, y_samp, data_samp, data_samp_std, data_tree, data_dims, data_extents, nan_frac

    else:
        i_nan_data    = np.isnan(data)
        n_nan_data    = np.count_nonzero(i_nan_data)
        
        if len(data) > 0:
            nan_frac_data = n_nan_data/(data.size)
        else:
            nan_frac_data = np.nan

        if (nan_frac_data > nan_frac_max) | (np.nanstd(data) == 0):
            results = (
                       [np.nan],
                       [np.nan],
                       [np.nan],
                       [np.nan],
                       [data_index],
                       [data_dim],
                       [data_extent],
                       [nan_frac_data]
                       )
        else:
            results = ([np.mean(x[~i_nan_data])],
                       [np.mean(y[~i_nan_data])],
                       [data_mean],
                       [np.nanstd(data)],
                       [data_index],
                       [data_dim],
                       [data_extent],
                       [nan_frac_data])

        for output, result in zip([x_samp, y_samp, data_samp, data_samp_std, data_tree, data_dims, data_extents, nan_frac], results):
            output.extend(result)

        return x_samp, y_samp, data_samp, data_samp_std, data_tree, data_dims, data_extents, nan_frac


if __name__ == '__main__':
    main()
