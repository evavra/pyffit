import os
import multiprocessing
import time
import numpy as np
import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
from pyffit.data import read_grd
from pyffit.utilities import check_dir_tree
from pyffit.finite_fault import get_fault_greens_functions, proj_greens_functions
import pickle
import sys
from numba import jit
import cmcrameri.cm as cmc


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


class ResQuadTree:
    """
    Quadtree object obtained through resolution-based downsampling.

    ATTRIBUTES:
    x, y (k,) - downsampled coordinates
    data (k,) - downsamled data values
    std (k,)  - downsampled data standard deviations
    """

    def __init__(self, x, y, data, look, data_index, fault, data_extent=[], mu=1, eta=1, width_min=0.1, width_max=5, smoothing=True, edge_slip=True,
                    iter=0, max_iter=100, resolution_threshold=0.02, max_intersect_width=0.5, min_fault_dist=0, poisson_ratio=0.25, verbose=False, 
                    disp_components=[0, 1, 2], slip_components=[0, 1, 2], rotation=np.nan, plot=True, run_dir='.'):

        # Save parameter values
        self.mu                   = fault.mu
        self.eta                  = fault.eta
        self.width_min            = width_min
        self.width_max            = width_max
        self.max_iter             = max_iter
        self.resolution_threshold = resolution_threshold
        self.max_intersect_width  = max_intersect_width
        self.poisson_ratio        = poisson_ratio
        self.disp_components      = disp_components
        self.slip_components      = slip_components
        self.rotation             = rotation
        self.min_fault_dist       = min_fault_dist
        self.run_dir              = run_dir
        self.smoothing            = smoothing
        self.edge_slip            = edge_slip

        if len(data_extent) != 4:
            self.extent = [x.min(), x.max(), y.min(), y.max()]

        # Get cells
        cells = resolution_sampling(x, y, data, look, data_index, fault, data_extent=self.extent, width_min=width_min, width_max=width_max, smoothing=smoothing, edge_slip=edge_slip,
                                         iter=iter, max_iter=max_iter, resolution_threshold=resolution_threshold, max_intersect_width=max_intersect_width, 
                                         disp_components=disp_components, slip_components=slip_components, rotation=rotation, plot=plot, min_fault_dist=min_fault_dist, run_dir=run_dir)
        self.compute_values(cells)

        n_points = self.x.size

        # Plot
        if plot:
            vlim = np.mean(np.abs(self.data)) + 2*np.std(np.abs(self.data))

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(x[~np.isnan(data)], y[~np.isnan(data)], marker='.', s=1, c='lightgray')
            ax.plot(fault.trace[:, 0], fault.trace[:, 1], c='k', linewidth=1)
            im = ax.scatter(self.x, self.y, marker='.', s=10, c=self.data, vmin=-vlim, vmax=vlim, cmap=cmc.vik)
            # im = ax.scatter(self.x[cell_buffer], y_origin[cell_buffer], c='gold', marker='o')
            # im = ax.scatter(self.x[cell_intersections], y_origin[cell_intersections], c='C3', marker='o')
            ax.set_aspect(1)
            ax.set_title(f'Iteration {iter}: {n_points} points')
            
            plt.colorbar(im)
            plt.savefig(f'{run_dir}/quadtree.png', dpi=300)
            plt.close()
        
    def compute_values(self, cells):

        self.cells = cells

        # Discard NaN values
        i_nans = np.where(np.isnan([cell.value for cell in self.cells]))

        for i in sorted(i_nans, reverse=True):
            self.cells = np.delete(self.cells, (i), axis=0)

        self.x          = np.empty(len(self.cells))
        self.y          = np.empty(len(self.cells))
        self.data       = np.empty(len(self.cells))  
        self.std        = np.empty(len(self.cells))  
        self.look       = np.empty((len(self.cells), 3)) 
        self.count      = np.empty(len(self.cells))  
        self.real_count = np.empty(len(self.cells))  
        self.nan_count  = np.empty(len(self.cells))  
        self.nan_frac   = np.empty(len(self.cells))  

        for i in range(len(self.cells)):
            cell = self.cells[i]

            self.x[i]          = cell.origin.x
            self.y[i]          = cell.origin.y
            self.data[i]       = cell.value  
            self.std[i]        = cell.std
            self.count[i]      = cell.data.values.size
            self.real_count[i] = np.sum(~np.isnan(cell.data.values))
            self.nan_count[i]  = self.count[i] - self.real_count[i]
            self.nan_frac[i]   = self.nan_count[i]/self.count[i]
            self.look[i, :]    = cell.look

        # print(np.mean(self.data), np.std(self.data))


    def write(self, filename):
        """
        Save ResQuadTree to disk.
        """
        print('\n' + f'Saving ResQuadTree to {filename}')

        with open(filename, 'wb') as f:
            pickle.dump(self, f)


    def histogram(self, data='cell_count', fig_kwargs={}, hist_kwargs={}, filename='', show=True):
        """
        Make histogram of scalar cell values [cell_count, real_count, nan_count, nan_frac]
        """
        data_dict = dict(cell_count=self.count, real_count=self.real_count, nan_count=self.nan_count, nan_frac=self.nan_frac)
        
        # Plot
        fig, ax = plt.subplots(**fig_kwargs)
        ax.hist(data_dict[data], **hist_kwargs)
        ax.set_xlabel(data)
        ax.set_ylabel('Count')
        ax.set_title(rf'Average = {np.mean(data_dict[data]):.1f} $\pm$ {np.std(data_dict[data]):.1f}')
        if len(filename) > 4:
            plt.savefig(filename)

        if show:
            plt.show()

        return fig, ax


    def check_parameters(self, params):
        """
        Check if  ResQuadTree has specified parameter values
        """

        for key in params.keys():
            tree_value = getattr(self, key)

            if tree_value == params[key]:
                continue
            else:
                print(f'Conflict with {key}: {tree_value} {params[key]}')
                return False

        return True


    def display_parameters(self, params):
        """
        Print specified ResQuadTree parameter values
        """

        lengths = [len(key) for key in params.keys()]

        max_length = max(lengths)

        print('\nResQuadTree parameters:')
        for key in params.keys():
            tree_value = getattr(self, key)

            pad = max_length - len(key) + 2

            print(f'    {key}:{" ".join(["" for i in range(pad)])}{tree_value}')

        return


class CellData:
    """
    Data and coordinates contained within a Cell object
    """

    def __init__(self, x, y, data, index):
        self.x      = x
        self.y      = y
        self.values = data
        self.index  = index


class CellOrigin:
    """
    Origin of quadtree Cell
    """

    def __init__(self, origin):
        self.x = origin[0]
        self.y = origin[1]


class Cell:
    """
    Class to contain attributes of a Quadtree or Resolution downsampling cell.
    
    ATTRIBUTES:
    x, y (k,)     - downsampled coordinates
    data (k,)     - downsamled data values
    std (k,)      - downsampled data standard deviations
    dims (k,)     - dimensions of each cell in x/y units.
    extents (k,)  - extent of each cell in x/y coordinates.
    nan_frac (k,) - fraction of nan values in each cell.
    """

    def __init__(self, x, y, data, look, index, origin, extent, cell_slice, trace):
        self.x          = x  
        self.y          = y  
        self.data       = CellData(x, y, data, index)  
        self.origin     = CellOrigin(origin)  
        self.extent     = extent
        self.cell_slice = cell_slice
        self.width      = np.max([extent[1] - extent[0], extent[3] - extent[2]])
        self.look       = np.mean(look, axis=0)

        # Check if cell intersects with fault
        i_intersect = np.where((trace[:, 0] >= self.extent[0]) & (trace[:, 0] <= self.extent[1]) & (trace[:, 1] >= self.extent[2]) & (trace[:, 1] <= self.extent[3]))[0]
        
        # Compute minimum distance to fault
        # dist = np.empty((2, 2))

        # for i in range(2):
        #     for j in range(2):
        #         dist[i, j] = np.min(np.sqrt((trace[:, 0] - self.extent[i])**2 + (trace[:, 1] - self.extent[j + 2])**2))

        # self.min_fault_dist = np.min(dist)
        self.min_fault_dist = get_min_fault_dist(trace, self.extent)

        # Get nan count
        if len(i_intersect) < 2:
            # No intersection, proceed as usual
            self.value    = np.nanmean(self.data.values)  
            self.std      = np.nanstd(self.data.values)  
            self.nan_frac = np.sum(np.isnan(self.data.values)/self.data.values.size)
            self.intersects = False

        else:
            # Select trace segment within cell
            trace_seg = trace[i_intersect]
            
            # Check which side the data are on
            side = np.zeros_like(self.data.values)

            for i in range(len(self.data.values)):
                side[i] = point_side_of_line(trace_seg, (self.data.x[i], self.data.y[i]))

            if sum(side == -1) > sum(side == 1):
                # Use west
                sign = -1
                label = 'West'
            else:
                # Use east
                sign = 1
                label = 'East'

            # Compute values using majority side
            self.value      = np.nanmean(self.data.values[side == sign])  
            self.std        = np.nanstd(self.data.values[side == sign])  
            self.nan_frac   = np.sum(np.isnan(self.data.values[side == sign])/self.data.values[side == sign].size)
            self.intersects = True

            # # Plot
            # fig, ax = plt.subplots(figsize=(14, 8.2))
            # fig.suptitle(label)
            # ax.scatter(self.data.x, self.data.y, c=side, cmap='coolwarm')
            # # ax.scatter(self.data.x, self.data.y, c=self.data.values, cmap='coolwarm')
            # # ax.plot(trace[:, 0], trace[:, 1], c='lightgray')
            # ax.plot(trace_seg[:, 0], trace_seg[:, 1], c='k')
            # plt.show()
            # sys.exit(1)

@jit(nopython=True)
def get_min_fault_dist(trace, extent):
    dist = np.empty((2, 2))

    for i in range(2):
        for j in range(2):
            dist[i, j] = np.min(np.sqrt((trace[:, 0] - extent[i])**2 + (trace[:, 1] - extent[j + 2])**2))
    return np.min(dist)


# ------------------ Application methods ------------------
def get_downsampled_time_series(datasets, inversion_inputs, fault, n_dim, dataset_name='data', file_name='downsampled_data.pkl'):
    """
    Generate matrix containing downsampled time series data based off of an existing quadtree structure
    """

    # Load downsampled time series matrix if already exists
    if os.path.exists(file_name):
        print('\n##### Loading time series #####')
        with open(file_name, 'rb') as f:
            d = pickle.load(f)

    # Apply existing quadtree to time-series
    else:
        print('\n##### Downsampling time series #####')
        d    = np.zeros((datasets[dataset_name].date.size, inversion_inputs[dataset_name].tree.data.size + n_dim), dtype=float)
        look = np.array([datasets[dataset_name]['look_e'].compute().data.flatten(), 
                         datasets[dataset_name]['look_n'].compute().data.flatten(), 
                         datasets[dataset_name]['look_u'].compute().data.flatten()]).T
        
        start = time.time()
        n_date = len(datasets[dataset_name].date)

        for k in range(0, n_date):
            start_date = time.time()
            
            date = datasets[dataset_name].date.values[k]

            # Select data for date
            data = datasets[dataset_name]['z'].isel(date=k).compute().data

            # Redo cells
            update_start = time.time()
            
            cells = update_cells(inversion_inputs[dataset_name].tree.cells, datasets[dataset_name].coords['x'].compute().data.flatten(), datasets[dataset_name].coords['y'].compute().data.flatten(), 
                                 data.flatten(), np.arange(0, datasets[dataset_name].coords['x'].size), look, fault.trace_interp)
            update_end = time.time() - update_start

            # Update quadtree
            inversion_inputs[dataset_name].tree.compute_values(cells)
            d[k, :inversion_inputs[dataset_name].tree.data.size] = inversion_inputs[dataset_name].tree.data
            end_date = time.time() - start_date
            print(f'{date} ({k + 1}/{n_date}) completed in {end_date:.1s}')

        print('\n' + f'Saving time series matrix d to {file_name}')

        with open(file_name, 'wb') as f:
            pickle.dump(d, f)

        end = time.time() - start
        print(f'Data computation time: {end:.1f}')
    return d


# ------------------ Resolution quadtree methods ------------------
def resolution_sampling(x, y, data, look, data_index, fault, min_fault_dist=0, data_extent=[], width_min=0.1, width_max=5, iter=0, max_iter=100, resolution_threshold=0.02, 
                        max_intersect_width=0.5, smoothing=True, edge_slip=True, disp_components=[0, 1, 2], slip_components=[0, 1, 2], rotation=np.nan, plot=True, run_dir='.',
                    #  real_min=3, mu=1, eta=1, poisson_ratio=0.25, verbose=False,
                     ):
    """
    Perform resolution-based downsampling, based on the procedure described by Lohman & Simons (2005).
    
    INPUT:
    Data
        x             - x-coordinates for original data
        y             - y-coordinates for original data
        data          - original data values 
        look          - look vectors for original data coordiantes
        data_index    - indicies for original data
        (data_extent) - bounds for sampling area  (default: [xmin, xmax, ymin, ymax])
    
    Model
        R               - smoothing regularization matrix
        E               - zero-edge-slip regularization matrix
        mesh            - fault mesh verticies 
        triangles       - fault mesh elements    
        trace           - high-resolution fault trace
        mu              - smoothing regularization value  (default = 1.0 -- should be same as inversion!)
        eta             - zero-edge-slip regularization value (default = 1.0 -- should be same as inversion!)
        poisson_ratio   - Poisson ratio for fault model (default = 0.25 -- should be same as inversion!)
        slip_components - components of fault slip to use (defaul = [0, 1, 2] for strike-slip, dip-slip, opening)
        verbose         - print updates on Green's function generation (default = False)


    Sampling
        max_iter             - max. allowed sampling iterations (default = 100)
        width_min            - min. allowed cell width (default = 0.1 km) 
        width_max            - max. allowed cell width (default = 5.0 km) 
        resolution_threshold - max. allowed resolution value for cells (default = 0.02) 
        max_intersect_width  - min. allowed size for cells that intersect the fault (default = 0.5 km) 
        min_fault_dist       - min. buffer distance to fault to enforce max_intersect_width

    OUTPUT
        cells - list of Cell objects in quadtree instance.
    """

    if len(data_extent) != 4:
        data_extent = np.array([np.min(x), np.max(x), np.min(y), np.max(y)])
    
    # Get data dimensions
    data_dim = np.array([data_extent[1] - data_extent[0], data_extent[3] - data_extent[2]]) # dimensions of current data cell

    # Get number of samples
    n_x = int(data_dim[0]/width_max) + 1
    n_y = int(data_dim[1]/width_max) + 1

    # Get initial cell grid
    x_edge, y_edge, x_origin, y_origin = get_grid_domain(data_extent, n_x, n_y)

    # Get cell info
    cells              = get_cells_grid(x, y, data, data_index, look, x_edge, y_edge, x_origin, y_origin, fault.trace_interp)
    cell_widths        = np.array([cell.width for cell in cells])
    cell_looks         = np.array([cell.look for cell in cells])
    cell_intersections = np.array([cell.intersects for cell in cells])
    cell_buffer        = np.array([cell.min_fault_dist for cell in cells]) <= min_fault_dist

    # Get resolution matrix
    N = fault.resolution_matrix(x_origin, y_origin, mode='data', smoothing=smoothing, edge_slip=edge_slip, disp_components=disp_components, slip_components=slip_components, rotation=rotation, squeeze=True)

    # Perform iterations until all cells are below resolution threshold or have reached min. cell width
    n_min_width        = 0
    n_near_field_width = sum(((cell_intersections | cell_buffer) & (cell_widths > max_intersect_width)))
    n_resolved         = sum(np.diag(N) < resolution_threshold) + n_min_width - n_near_field_width

    print('')
    print(f'Iteration {0}...')
    print(f'Number of samples:        {n_x*n_y}')
    print(f'Number of resolved cells: {n_resolved}')
    print(f'Cell widths:              {" ".join([str(w) for w in np.unique(np.round(cell_widths, 3))])}')
    print(f'N:                        {np.min(np.diag(N)):.2e} - {np.max(np.diag(N)):.2e} \n')

    while ((sum(np.diag(N) < resolution_threshold) + n_min_width - n_near_field_width) != len(cells)) & (iter != max_iter):
        
        iter += 1
        print(f'Iteration {iter}...')

        # Get unresolved cells        
        # Unresolved if resolution above threshold or if cell intersects with fault or is within buffer distance and is larger than specified max. width               
        i_unresolved = np.where((np.diag(N) > resolution_threshold) | ((cell_intersections | cell_buffer) & (cell_widths > max_intersect_width)))[0]

        for i in sorted(i_unresolved, reverse=True):
            cell_extent = cells[i].extent

            if np.max((cell_extent[1] - cell_extent[0], cell_extent[3] - cell_extent[2]))/2 > width_min:

                # Get initial cell grid
                x_edge_new, y_edge_new, x_origin_new, y_origin_new = get_grid_domain(cell_extent, 2, 2)

                # Get cell info
                cells_new = get_cells_grid(x, y, data, data_index, look, x_edge_new, y_edge_new, x_origin_new, y_origin_new, fault.trace_interp)

                # Update lists and arrays
                cells.extend(cells_new)

                # Update with new cells
                x_origin = np.concatenate((x_origin, x_origin_new))
                y_origin = np.concatenate((y_origin, y_origin_new))

                # Remove defunct cells in reverse order to avoid index shifting
                cells.pop(i)
                x_origin = np.delete(x_origin, (i), axis=0)
                y_origin = np.delete(y_origin, (i), axis=0)

            else:
                n_min_width += 1

        # Update widths
        cell_widths        = np.array([cell.width for cell in cells])
        cell_looks         = np.array([cell.look for cell in cells])
        cell_intersections = np.array([cell.intersects for cell in cells])
        cell_buffer        = np.array([cell.min_fault_dist for cell in cells]) <= min_fault_dist

        # Get resolution matrix
        N = fault.resolution_matrix(x_origin, y_origin, mode='data', smoothing=smoothing, edge_slip=edge_slip, disp_components=disp_components, slip_components=slip_components, rotation=rotation, squeeze=True)

        # Update counts``
        n_near_field_width = sum(((cell_intersections | cell_buffer) & (cell_widths > max_intersect_width)))
        n_resolved         = sum(np.diag(N) < resolution_threshold) + n_min_width - n_near_field_width

        # Print results
        print(f'Number of samples:        {len(x_origin)}')
        print(f'Number of resolved cells: {n_resolved}')
        print(f'Cell widths:              {" ".join([str(w) for w in np.unique(np.round(cell_widths, 3))])}')
        print(f'N:                        {np.min(np.diag(N)):.2e} - {np.max(np.diag(N)):.2e} \n')

        # Make plots
        # if plot:
        #     if iter == 1:
        #         check_dir_tree(run_dir + '/Sampling', clear=True)

        #     # Plot histogram and matrix
        #     fig, axes = plt.subplots(1, 2, figsize=(6, 4))
        #     fig.suptitle(f'Min. = {np.min(N):.2f}, Max. = {np.max(N):.2f}')
        #     im0 = axes[0].hist(np.diag(N), bins=50)
        #     im1 = axes[1].imshow(N, cmap='viridis', vmin=0, vmax=np.mean(N) + 3*np.std(N), interpolation='none')
        #     plt.colorbar(im1, label='Resolution')
        #     plt.savefig(f'{run_dir}/Sampling/Cell_Stats_Iteration-{iter}.png') 
        #     plt.close()

        #     # Plot point distribution
        #     n_points = x_origin.size

        #     fig, ax = plt.subplots(figsize=(6, 6))
        #     ax.scatter(x[~np.isnan(data)], y[~np.isnan(data)], marker='.', s=1, c='gainsboro')
        #     ax.plot(fault.trace[:, 0], fault.trace[:, 1], c='k', linewidth=1)
        #     im = ax.scatter(x_origin, y_origin, c=np.diag(N), marker='o', cmap='viridis', vmin=0, vmax=resolution_threshold)
        #     im = ax.scatter(x_origin[cell_buffer], y_origin[cell_buffer], c='gold', marker='o')
        #     im = ax.scatter(x_origin[cell_intersections], y_origin[cell_intersections], c='C3', marker='o')
        #     ax.set_aspect(1)
        #     ax.set_title(f'Iteration {iter}: {n_points} points')
        #     plt.colorbar(im, label='Resolution')
        #     plt.savefig(f'{run_dir}/Sampling/Cell_Map_Iteration-{iter}.png')
        #     plt.close()

    return cells


def get_grid_domain(extent, n_x, n_y):
    """
    Given a spatial region, get coordinates for uniform grid of specified dimensions.

    INPUT
    extent - spatial extent of grid [xmin, xmax, ymin, ymax]
    n_x - number of elements in x-direction
    n_y - number of elements in y-direction
    OUTPUT
    x_edge, y_edge     - vertex coordinates of grid
    x_origin, y_origin - origin coordinates of grid
    """

    # Get coordinate ranges
    x_rng_edge   = np.linspace(extent[0], extent[1], n_x + 1)
    y_rng_edge   = np.linspace(extent[2], extent[3], n_y + 1)
    x_rng_origin = x_rng_edge[:-1] + np.diff(x_rng_edge)/2
    y_rng_origin = y_rng_edge[:-1] + np.diff(y_rng_edge)/2

    # Get full grid
    x_edge, y_edge     = np.meshgrid(x_rng_edge, y_rng_edge)
    x_origin, y_origin = np.meshgrid(x_rng_origin, y_rng_origin)

    return x_edge, y_edge, x_origin.flatten(), y_origin.flatten()


def get_cells_grid(x, y, data, data_index, look, x_edge, y_edge, x_origin, y_origin, trace):
    """
    Get a uniform grid of cells
    """
    cells = []
    cell_looks = []        
    
    for i in range(x_edge.shape[0] - 1):
        for j in range(x_edge.shape[1] - 1):
            cell_extent = np.array([x_edge[i, j], x_edge[i, j + 1], y_edge[i, j], y_edge[i + 1, j]])

            # Get index slices for cells            
            cell_slice = np.where((x >= cell_extent[0]) & (x <= cell_extent[1]) & (y >= cell_extent[2]) & (y <= cell_extent[3]))[0]

            # Get information for each new cell
            cell_x     = x[cell_slice]
            cell_y     = y[cell_slice]
            cell_data  = data[cell_slice]
            cell_index = data_index[cell_slice]
            cell_look  = look[cell_slice, :]

            # Add to list
            k = np.ravel_multi_index((i, j), (x_edge.shape[0] - 1, x_edge.shape[1] - 1))
            cells.append(Cell(cell_x, cell_y, cell_data, cell_look, cell_index, (x_origin[k], y_origin[k]), cell_extent, cell_slice, trace))
    
    return cells


# def update_cell(params):
#     """
#     Get cell (origin, extent) for given data with coordinates (x, y)
#     """
#     origin, extent, x, y, data, data_index, look, trace = params

#     # Get index slices for cells            
#     cell_slice = np.where((x >= extent[0]) & (x <= extent[1]) & (y >= extent[2]) & (y <= extent[3]))[0]

#     # Get information for each new cell
#     cell_x     = x[cell_slice]
#     cell_y     = y[cell_slice]
#     cell_data  = data[cell_slice]
#     cell_index = data_index[cell_slice]
#     cell_look  = look[cell_slice, :]
    
#     return Cell(cell_x, cell_y, cell_data, cell_look, cell_index, (origin.x, origin.y), extent, cell_slice, trace)


def update_cells(cells, x, y, data, data_index, look, trace):
    """
    Populate existing cells with new data.
    """

    new_cells = []
    params    = []

    for i in range(len(cells)):
        cell = cells[i]

        # Get index slices for cells          
        cell_slice = np.where((x >= cell.extent[0]) & (x <= cell.extent[1]) & (y >= cell.extent[2]) & (y <= cell.extent[3]))[0]

        # Get information for each new cell
        cell_x     = x[cell_slice]
        cell_y     = y[cell_slice]
        cell_data  = data[cell_slice]
        cell_index = data_index[cell_slice]
        cell_look  = look[cell_slice, :]
        
        # # x, y, data, std, index, origin, extent
        new_cells.append(Cell(cell_x, cell_y, cell_data, cell_look, cell_index, (cell.origin.x, cell.origin.y), cell.extent, cell_slice, trace))
        # params.append([cell.origin, cell.extent, x, y, data, data_index, look, trace])

    # os.environ["OMP_NUM_THREADS"] = "1"
    # start       = time.time()
    # n_processes = multiprocessing.cpu_count()
    # pool        = multiprocessing.Pool(processes=n_processes)
    # new_cells     = pool.map(update_cell, params)
    # pool.close()
    # pool.join()

    return new_cells


def check_fault_intersection(cell, trace, buffer=0):
    """
    Check if cell intersects with fault trace.
    """
    # Check for intersections
    intersect = np.sum((trace[:, 0] >= cell.extent[0]) & (trace[:, 0] <= cell.extent[1]) & (trace[:, 1] >= cell.extent[2]) & (trace[:, 1] <= cell.extent[3]))
    
    if intersect > 0:

        # Check for vertices within buffer distance
        dist = np.empty((2, 2))

        for i in range(2):
            for j in range(2):
                dist[i, j] = np.sqrt((trace[:, 0] - cell_extent[i])**2 + (trace[:, 1] - cell_extent[j + 2])**2)

        if np.min(dist) >= buffer:
            return True
    else:
        return False


def point_side_of_segment(p1, p2, point):
    """
    Determine the side of a point relative to a line segment defined by two points.

    Parameters:
    p1, p2 : tuple of floats
        Coordinates of the points defining the line segment (x1, y1) and (x2, y2).
    point : tuple of floats
        Coordinates of the point to check (x, y).

    Returns:
    float
        Positive if point is on the left side, negative if on the right side, and zero if on the segment.
    """

    x1, y1 = p1
    x2, y2 = p2
    x, y   = point

    # Cross product to determine the side
    cross_product = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

    return cross_product


@jit(nopython=True)
def point_side_of_line(curve, point):
    """
    Determine the side of a point relative to a curved line defined by multiple points.

    Parameters:
    curve : array-like, shape (n, 2)
        The coordinates of the points defining the curved line (x_i, y_i).
    point : tuple of floats
        Coordinates of the point to check (x, y).

    Returns:
    float
        Positive if point is on the left side of all segments, negative if on the right side,
        and zero if on the curve.
    """

    i_nearest    = np.argmin(np.sqrt((point[0] - curve[:-1, 0])**2 + (point[1] - curve[:-1, 1])**2))
    p1           = curve[i_nearest]
    p2           = curve[i_nearest + 1]

    # overall_sign = np.sign(point_side_of_segment(p1, p2, point))
    x1, y1 = p1
    x2, y2 = p2
    x, y   = point

    # Cross product to determine the side
    cross_product = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    overall_sign = np.sign(cross_product)

    return overall_sign


# ------------------ Iterative quadtree methods ------------------
def quadtree_unstructured(x, y, data, data_index, data_extent, fault=[], level=0, rms_min=0.1, nan_frac_max=0.9, width_min=1, width_max=50, return_object=False,
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
            x_samp[i]        = np.mean(x[idx][~i_nan_data]) # x_samp 
            y_samp[i]        = np.mean(y[idx][~i_nan_data]) # y_samp       
            data_samp[i]     = np.nanmean(data[idx])        # data_samp 
            data_samp_std[i] = np.nanstd(data[idx])         # data_samp 
            nan_frac[i]      = nan_frac_data                # nan_frac 
           
    return Quadtree(x_samp, y_samp, data_samp, data_samp_std, quadtree.tree, quadtree.dims, quadtree.extents, nan_frac)



if __name__ == '__main__':
    main()
