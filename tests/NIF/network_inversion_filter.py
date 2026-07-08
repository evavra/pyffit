#!/usr/bin/env python
import os
import gc
import sys
import glob
import copy
import h5py
import time
import shutil
import pyffit
import datetime
import linecache
import tracemalloc
import numpy as np
import pandas as pd
import importlib.util
import multiprocessing
import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.transforms import Affine2D
from scipy.interpolate import griddata 
from matplotlib import colors
from types import ModuleType
import scipy
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable


def main():
    """
    Run Network Inversion Filter.
    """

    sys.stdout.flush() # remove buffering of stdout

    start = time.time()
    param_file = sys.argv[1]

    # Load parameters
    mode, params = load_parameters(param_file)
    
    if 'grid_search' in mode:
        grid_search(**params)

    if 'testing' in mode:
        testing(**params)

    if 'downsample' in mode:
        check_downsampling(**params)

    if 'filter' in mode:
        filter(**params)

    if 'covariances' in mode:
        covariances(**params)

    if 'static' in mode:
        run_static(**params)    
        
    if 'NIF' in mode:
        run_nif(**params)
        
    if 'analyze_model' in mode:
        analyze_model(**params)

    if 'analyze_disp' in mode:
        analyze_disp(**params)

    end = time.time() - start
    print(f'Total run time: {end/60:.2f}')

    return


# ------------------ Task coordination ------------------
def load_parameters(file_name):
    """
    Load parameter file for use with one of the driver methods.

    INPUT:
    file_name - Path to the Python parameter script containing parameter definitions (i.e. params.py)

    OUTPUT:
    params - A dictionary containing parameter names as keys and their values.
    """

    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Parameter script '{file_name}' not found.")

    # Make module for parameter file
    module_name = os.path.splitext(os.path.basename(file_name))[0]
    spec        = importlib.util.spec_from_file_location(module_name, file_name)
    module      = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    # Extract variables from file_name namespace
    spec.loader.exec_module(module)

    # Get run mode
    mode   = [value for key, value in vars(module).items() if key == 'mode'][0]

    # Exclude mode, built-in variables, and modules imported to parameter script
    params = {key: value for key, value in vars(module).items() if not (key == 'mode') | (key.startswith("__") | (type(value) == ModuleType))}

    return mode, params


# ------------------ Drivers ------------------
def grid_search(mesh_file, triangle_file, file_format, downsampled_dir, out_dir, data_dir, 
                date_index_range=[-22, -14], xkey='x', coord_type='xy', dataset_name='data',
                check_lon=False, reference_time_series=True, use_dates=False, use_datetime=False, dt=1, data_factor=1,
                model_file='', mask_file='', estimate_covariance=False, mask_dists=[3], n_samp=2*10**7, r_inc=0.2, r_max=80, mask_dir='.', cov_dir='.', 
                m0=[0.9, 1.5, 5], c_max=2, sv_max=2, ramp_type='none', ref_point=[-116, 33.5], avg_strike=315.8, trace_inc=0.01, poisson_ratio=0.25, 
                shear_modulus=6*10**9, disp_components=[1], slip_components=[0], resolution_threshold=2.3e-1, 
                width_min=0.1, width_max=10, max_intersect_width=100, min_fault_dist=1, max_iter=10, 
                smoothing_samp=False, edge_slip_samp=False, omega=[1e1], sigma=1e1, kappa=[2e1], mu=2e1, 
                eta=2e1, rho=1e0, steady_slip=False, constrain=False, v_sigma=1e-9, W_sigma=1,  W_dot_sigma=1, ramp_sigma=100, v_lim=(0, 3), W_lim=(0, 30), W_dot_lim=(0, 50), ramp_lim=(-100, 100),
                xlim=[-35.77475071,26.75029172], ylim=[-26.75029172, 55.08597388], vlim_slip=[0, 20], 
                vlim_disp=[[-10,10], [-10,10], [-1,1]], cmap_slip=cmc.lajolla, cmap_disp=cmc.vik, figsize=(10, 10), dpi=75, markersize=40, 
                param_file='params.py'
                ):
    """
    Run network inversion filter.
    """

    start_total = time.time()
    
    iter = 0

    pyffit.utilities.check_dir_tree(downsampled_dir)
    pyffit.utilities.check_dir_tree(downsampled_dir + '/' + dataset_name)

    # -------------------------- Prepare original data --------------------------    
    # Load fault model and regularization matrices
    fault = pyffit.finite_fault.TriFault(mesh_file, triangle_file, slip_components=slip_components, 
                                        poisson_ratio=poisson_ratio, shear_modulus=shear_modulus, 
                                        verbose=False, trace_inc=trace_inc, mu=mu, eta=eta,
                                        avg_strike=avg_strike, ref_point=ref_point)
    n_patch = len(fault.triangles)

    # Load data
    dataset = pyffit.data.load_insar_dataset(data_dir, file_format, dataset_name, ref_point, data_factor=data_factor, xkey=xkey, 
                                            coord_type=coord_type, date_index_range=date_index_range, 
                                            check_lon=check_lon, reference_time_series=reference_time_series, 
                                            incremental=False, use_dates=use_dates, use_datetime=use_datetime, 
                                            mask_file=mask_file, remove_mean=False)
    datasets = {dataset_name: dataset}
    n_obs    = len(dataset.date)
    x        = dataset.coords['x'].compute().data.flatten()
    y        = dataset.coords['y'].compute().data.flatten()

    # Load original slip model
    if len(model_file) > 0:
        try:
            with h5py.File(model_file, 'r') as file:
                slip_model = file['slip_model'][()]
        except KeyError:
            print('Error: speficied model_file could not be located')
            sys.exit(1)
    else:
        slip_model = np.full([n_patch, 3, n_obs], np.nan)

    # ------------------ Prepare inversion inputs ------------------
    # Get parameter values to check
    quadtree_dir = get_downsampled_data_directory(downsampled_dir + '/' + dataset_name, param_file)
    
    
    # Get dictionary of quadtree parameters
    quadtree_params = dict(
                            resolution_threshold=resolution_threshold,
                            width_min=width_min,
                            width_max=width_max,
                            max_intersect_width=max_intersect_width,
                            min_fault_dist=min_fault_dist,
                            max_iter=max_iter, 
                            poisson_ratio=poisson_ratio,
                            smoothing=smoothing_samp,
                            edge_slip=edge_slip_samp,
                            disp_components=disp_components,
                            slip_components=slip_components,
                            quadtree_dir=quadtree_dir,
                            )

    # Prepare inversion inputs
    inputs = pyffit.inversion.get_inversion_inputs(fault, datasets, quadtree_params, date=-1, quadtree_dir=quadtree_dir, verbose=False)
    tree   = inputs[dataset_name].tree
    n_data = inputs[dataset_name].tree.x.size
    dt    /= 365.25

    # Get ramp matrix, if specified
    if ramp_type in ['linear', 'quadratic']:
        ramp_matrix = pyffit.corrections.get_ramp_matrix(tree.x, tree.y, ramp_type=ramp_type)
    else:
        ramp_matrix = []

    # Determine number of physical parameters
    if steady_slip:
        n_fault_dim = 3 * n_patch
    else:
        n_fault_dim = 2 * n_patch

    # Determine number of ramp coefficients
    dim_ramp = 0
    if len(ramp_matrix) > 0:
        dim_ramp += ramp_matrix.shape[1]

    n_dim = n_fault_dim + dim_ramp

    # -------------------- Prepare data --------------------
    samp_file = f'cutoff={resolution_threshold:.1e}_wmin={width_min:.2f}_wmax={width_max:.2f}_max_int_w={max_intersect_width:.1f}_min_fdist={min_fault_dist:.2f}_max_it={max_iter:0f}'
    d, std    = pyffit.quadtree.get_downsampled_time_series(datasets, inputs, fault, n_fault_dim, dataset_name=dataset_name, file_name=f'{downsampled_dir}/{dataset_name}/{samp_file}.h5')

    print(f'Number of fault elements: {n_patch}')
    print(f'Number of data points:    {n_data}')
    print(f'Number of observations:   {n_obs}')
    print(f'Number of dimensions:     {n_dim}')

    # -------------------- Greens Functions --------------------
    G = -fault.greens_functions(inputs[dataset_name].tree.x, inputs[dataset_name].tree.y, disp_components=disp_components, slip_components=slip_components, rotation=avg_strike, squeeze=True)
        
    for omega0 in omega:
        for kappa0 in kappa:
            iter += 1
            print()
            print()
            print()
            print(f'#################### Working on omega = {(omega0/sigma)**2:.1e}, kappa = {(kappa0/sigma)**2:.1e} ({iter}/{len(omega) * len(kappa)}) ####################')

            # ------------------ Prepare run directories and files ------------------
            date    = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')

            
            run_dir = f'{out_dir}/omega_{omega0:.1e}__kappa_{kappa0:.1e}__sigma_{sigma:.1e}'

            pyffit.utilities.check_dir_tree(run_dir)
            pyffit.utilities.check_dir_tree(run_dir + '/Scripts/' + date)

            result_dir = f'{run_dir}/Results'
            pyffit.utilities.check_dir_tree(result_dir)
            pyffit.utilities.check_dir_tree(result_dir + '/Results')
            pyffit.utilities.check_dir_tree(result_dir + '/Slip')
            pyffit.utilities.check_dir_tree(result_dir + '/Matrices')
            pyffit.utilities.check_dir_tree(result_dir + '/Resolution')

            # Copy parameter file and nif.py version to run directory
            shutil.copy(__file__, f'{run_dir}/Scripts/{date}/nif_{date}.py')
            shutil.copy(param_file, f'{run_dir}/Scripts/{date}/params_{date}.py')
            

            print(f'Run directory: {run_dir}')

            # Don't redo existing runs
            if os.path.exists(f'{run_dir}/results_smoothing.h5') & os.path.exists(f'{run_dir}/results_smoothing.h5'):
                print(f'{run_dir} already completed.')
            else:
                # -------------------- Data covariance --------------------
                # Get covariance model
                sv_file  = f'{cov_dir}/semivariogram_params.h5' # parameters for covariance model for each observation
                cov_file = f'{quadtree_dir}/covariance.h5'      # modeled covariance matrices for each observation
                R_file   = f'{run_dir}/R.h5'                    # complete data covariance matrix for each observation 

                if estimate_covariance:
                    print(f'Using covariance estimates at {cov_file}')

                else:
                    print(f'Using dialgonal covariance matrices')
                    C = np.eye(inputs[dataset_name].tree.x.size)
                    with h5py.File(cov_file, 'w') as out_file: 
                        out_file.create_dataset(f'covariance', data=C) 

                # Set up model bounds and initial uncertainties
                if steady_slip:
                    state_lims   = [v_lim, W_lim, W_dot_lim]
                    state_sigmas = [v_sigma, W_sigma, W_dot_sigma]
                else:
                    state_lims   = [W_lim, W_dot_lim]
                    state_sigmas = [W_sigma, W_dot_sigma]
                
                if ramp_type in ['linear', 'quadratic']:
                    state_lims.append(ramp_lim)
                    state_sigmas.append(ramp_sigma)

                # Get cell distances
                dist_file = f'{quadtree_dir}/dist.h5'

                if os.path.exists(dist_file) is not True:

                    print('Computing cell distances...')
                    
                    # Compute distances
                    dist = np.empty((n_data, n_data))

                    for i in range(n_data):
                        for j in range(n_data):
                            dist[i, j] = pyffit.covariance.dist(x[i], x[j], y[i], y[j])
                    
                    # Write to disk
                    with h5py.File(dist_file, 'w') as file:                
                        file.create_dataset('dist', data=dist)

                # Get bounds on model parameters
                state_lim = pyffit.nif.get_state_constraints(fault, state_lims, ramp_matrix=ramp_matrix)

                # Perform forward Kalman filtering
                x_model_forward, x_model = pyffit.nif.network_inversion_filter(fault, G, d, std, dt, omega0, sigma, kappa0, state_sigmas, 
                                                                            cov_file=sv_file, steady_slip=steady_slip, ramp_matrix=ramp_matrix, rho=rho, constrain=constrain, 
                                                                            state_lim=state_lim, result_dir=run_dir, cost_function='state', dist_file=dist_file)

                # Compute integrated slip
                if steady_slip:
                    s_model_forward = integrate_slip(x_model_forward, dt)
                    s_model         = integrate_slip(x_model, dt)
                else:
                    s_model_forward = x_model_forward[:n_patch]
                    s_model         = x_model[:n_patch]
                    
                s_true          = slip_model[:, 0, :].T

                print(f'Forward avg. slip  {np.mean(s_model_forward):.1f} +\- {np.std(s_model_forward):.1f} | range = {s_model_forward.min():.1f} {s_model_forward.max():.1f} | v = {np.mean(x_model_forward[:, :n_patch]):.1e} +/- {np.mean(x_model_forward[:, :n_patch]):.1e} mm/yr')
                print(f'Smoothed avg. slip {np.mean(s_model):.1f} +\- {np.std(s_model):.1f} | range = {s_model.min():.1f} {s_model.max():.1f} | v = {np.mean(x_model[:, :n_patch]):.1e} +/- {np.mean(x_model[:, :n_patch]):.1e} mm/yr')
                
                if np.isnan(slip_model).all():
                    print(f'True avg. slip     {np.mean(s_true):.1f} +\- {np.std(s_true):.1f} | range = {s_true.min():.1f} {s_true.max():.1f}')

                end_total = time.time() - start_total
                print(f'Total inversion time: {end_total/60:.1f} min')

                os.remove(f'{run_dir}/results_forward.h5')
    return


def testing(mesh_file, triangle_file, file_format, downsampled_dir, out_dir, data_dir, 
                  date_index_range=[-22, -14], xkey='x', coord_type='xy', dataset_name='data',
                  check_lon=False, reference_time_series=True, use_dates=False, use_datetime=False, dt=1, data_factor=1,
                  model_file='', mask_file='', estimate_covariance=False, mask_dists=[3], n_samp=2*10**7, r_inc=0.2, r_max=80, mask_dir='.', cov_dir='.', 
                  m0=[0.9, 1.5, 5], c_max=2, sv_max=2, ramp_type='none', ref_point=[-116, 33.5], avg_strike=315.8, trace_inc=0.01, poisson_ratio=0.25, 
                  shear_modulus=6*10**9, disp_components=[1], slip_components=[0], resolution_threshold=2.3e-1, 
                  width_min=0.1, width_max=10, max_intersect_width=100, min_fault_dist=1, max_iter=10, 
                  smoothing_samp=False, edge_slip_samp=False, omega=1e1, sigma=1e1, kappa=2e1, mu=2e1, 
                  eta=2e1, rho=1e0, steady_slip=False, constrain=False, v_sigma=1e-9, W_sigma=1,  W_dot_sigma=1, ramp_sigma=100, v_lim=(0, 3), W_lim=(0, 30), W_dot_lim=(0, 50), ramp_lim=(-100, 100),
                  xlim=[-35.77475071,26.75029172], ylim=[-26.75029172, 55.08597388], vlim_slip=[0, 20], 
                  vlim_disp=[[-10,10], [-10,10], [-1,1]], cmap_slip=cmc.lajolla, cmap_disp=cmc.vik, figsize=(10, 10), dpi=75, markersize=40, 
                  param_file='params.py'
                  ):
    """
    Analyze network inversion filter results.
    """
    start      = time.time()
    cmds       = []
    run_dir    = f'{out_dir}/omega_{omega:.1e}__kappa_{kappa:.1e}__sigma_{sigma:.1e}'
    result_dir = f'{run_dir}/Results'

    # Get directory for downsampled data
    pyffit.utilities.check_dir_tree(downsampled_dir + '/' + dataset_name)
    pyffit.utilities.check_dir_tree(result_dir + '/Animations')
        
    # Get parameter values to check
    quadtree_dir = get_downsampled_data_directory(downsampled_dir + '/' + dataset_name, param_file)

    # -------------------------- Prepare original data --------------------------    
    # Load fault model and regularization matrices
    fault = pyffit.finite_fault.TriFault(mesh_file, triangle_file, slip_components=slip_components, 
                                         poisson_ratio=poisson_ratio, shear_modulus=shear_modulus, 
                                         verbose=False, trace_inc=trace_inc, mu=mu, eta=eta,
                                         avg_strike=avg_strike, ref_point=ref_point)
    n_patch = len(fault.triangles)

    if steady_slip:
        n_dim   = 3 * n_patch
    else:
        n_dim   = 2 * n_patch

    # Load data
    dataset = pyffit.data.load_insar_dataset(data_dir, file_format, dataset_name, ref_point, data_factor=data_factor, xkey=xkey, 
                                             coord_type=coord_type, date_index_range=date_index_range, 
                                             check_lon=check_lon, reference_time_series=reference_time_series, 
                                             incremental=False, use_dates=use_dates, use_datetime=use_datetime, 
                                             mask_file=mask_file, remove_mean=False)
    datasets = {dataset_name: dataset}
    n_obs    = len(dataset.date)
    x        = dataset.coords['x'].compute().data.flatten()
    y        = dataset.coords['y'].compute().data.flatten()

    # Load original slip model
    if len(model_file) > 0:
        try:
            with h5py.File(model_file, 'r') as file:
                slip_model = file['slip_model'][()]
        except KeyError:
            print('Error: speficied model_file could not be located')
            sys.exit(1)
    else:
        slip_model = np.full([n_patch, 3, n_obs], np.nan)

    # ------------------ Prepare inversion inputs ------------------
    # Get dictionary of quadtree parameters
    quadtree_params = dict(
                            resolution_threshold=resolution_threshold,
                            width_min=width_min,
                            width_max=width_max,
                            max_intersect_width=max_intersect_width,
                            min_fault_dist=min_fault_dist,
                            max_iter=max_iter, 
                            poisson_ratio=poisson_ratio,
                            smoothing=smoothing_samp,
                            edge_slip=edge_slip_samp,
                            disp_components=disp_components,
                            slip_components=slip_components,
                            quadtree_dir=quadtree_dir,
                            )

    # Prepare inversion inputs
    inputs = pyffit.inversion.get_inversion_inputs(fault, datasets, quadtree_params, date=-1, quadtree_dir=quadtree_dir, verbose=False)
    tree   = inputs[dataset_name].tree
    n_data = inputs[dataset_name].tree.x.size
    dt    /= 365.25

    # Get ramp matrix, if specified
    if ramp_type in ['linear', 'quadratic']:
        ramp_matrix = pyffit.corrections.get_ramp_matrix(tree.x, tree.y, ramp_type=ramp_type)
    else:
        ramp_matrix = []

    # -------------------------- Prepare NIF objects --------------------------
    print(f'Number of fault elements: {n_patch}')
    print(f'Number of data points:    {n_data}')

    # Prepare data
    samp_file = f'cutoff={resolution_threshold:.1e}_wmin={width_min:.2f}_wmax={width_max:.2f}_max_int_w={max_intersect_width:.1f}_min_fdist={min_fault_dist:.2f}_max_it={max_iter:0f}'
    d, std = pyffit.quadtree.get_downsampled_time_series(datasets, inputs, fault, n_dim, remove_mean=True, dataset_name=dataset_name, file_name=f'{downsampled_dir}/{dataset_name}/{samp_file}.h5')


    for k in range(len(d)):
        # data = dataset['z'].isel(date=k)
        data = d[k, :]

        print(k, np.nanmean(data))
    return


def run_static(mesh_file, triangle_file, file_format, downsampled_dir, out_dir, data_dir, 
            date_index_range=[-22, -14], xkey='x', coord_type='xy', dataset_name='data',
            check_lon=False, reference_time_series=True, use_dates=False, use_datetime=False, dt=1, data_factor=1,
            model_file='', mask_file='', estimate_covariance=False, mask_dists=[3], n_samp=2*10**7, r_inc=0.2, r_max=80, mask_dir='.', cov_dir='.', 
            m0=[0.9, 1.5, 5], c_max=2, sv_max=2, ramp_type='none', ref_point=[-116, 33.5], avg_strike=315.8, trace_inc=0.01, poisson_ratio=0.25, 
            shear_modulus=6*10**9, disp_components=[1], slip_components=[0], resolution_threshold=2.3e-1, 
            width_min=0.1, width_max=10, max_intersect_width=100, min_fault_dist=1, max_iter=10, 
            smoothing_samp=False, edge_slip_samp=False, omega=1e1, sigma=1e1, kappa=2e1, mu=2e1, 
            eta=2e1, rho=1e0, steady_slip=False, constrain=False, v_sigma=1e-9, W_sigma=1,  W_dot_sigma=1, ramp_sigma=100, v_lim=(0, 3), W_lim=(0, 30), W_dot_lim=(0, 50), ramp_lim=(-100, 100),
            xlim=[-35.77475071,26.75029172], ylim=[-26.75029172, 55.08597388], vlim_slip=[0, 20], 
            vlim_disp=[[-10,10], [-10,10], [-1,1]], cmap_slip=cmc.lajolla, cmap_disp=cmc.vik, figsize=(10, 10), dpi=75, markersize=40, 
            param_file='params.py'
            ):
    """
    Run network inversion filter.
    """

    start_total = time.time()
    
    # ------------------ Prepare run directories and files ------------------
    date    = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    run_dir = f'{out_dir}/omega_{omega:.1e}__kappa_{kappa:.1e}__sigma_{sigma:.1e}'

    pyffit.utilities.check_dir_tree(run_dir)
    pyffit.utilities.check_dir_tree(run_dir + '/Scripts/' + date)

    result_dir = f'{run_dir}/Results'
    pyffit.utilities.check_dir_tree(result_dir)
    pyffit.utilities.check_dir_tree(result_dir + '/Results')
    pyffit.utilities.check_dir_tree(result_dir + '/Slip')
    pyffit.utilities.check_dir_tree(result_dir + '/Matrices')
    pyffit.utilities.check_dir_tree(result_dir + '/Resolution')
    pyffit.utilities.check_dir_tree(downsampled_dir)
    pyffit.utilities.check_dir_tree(downsampled_dir + '/' + dataset_name)

    # Copy parameter file and nif.py version to run directory
    shutil.copy(__file__, f'{run_dir}/Scripts/{date}/nif_{date}.py')
    shutil.copy(param_file, f'{run_dir}/Scripts/{date}/params_{date}.py')
    
    print(f'Run directory: {run_dir}')

    # -------------------------- Prepare original data --------------------------    
    # Load fault model and regularization matrices
    fault = pyffit.finite_fault.TriFault(mesh_file, triangle_file, slip_components=slip_components, 
                                         poisson_ratio=poisson_ratio, shear_modulus=shear_modulus, 
                                         verbose=False, trace_inc=trace_inc, mu=mu, eta=eta,
                                         avg_strike=avg_strike, ref_point=ref_point)
    n_patch = len(fault.triangles)

    # Load data
    dataset = pyffit.data.load_insar_dataset(data_dir, file_format, dataset_name, ref_point, data_factor=data_factor, xkey=xkey, 
                                             coord_type=coord_type, date_index_range=date_index_range, 
                                             check_lon=check_lon, reference_time_series=reference_time_series, 
                                             incremental=False, use_dates=use_dates, use_datetime=use_datetime, 
                                             mask_file=mask_file, remove_mean=False)
    datasets = {dataset_name: dataset}
    n_obs    = len(dataset.date)
    x        = dataset.coords['x'].compute().data.flatten()
    y        = dataset.coords['y'].compute().data.flatten()

    # Load original slip model
    if len(model_file) > 0:
        try:
            with h5py.File(model_file, 'r') as file:
                slip_model = file['slip_model'][()]
        except KeyError:
            print('Error: speficied model_file could not be located')
            sys.exit(1)
    else:
        slip_model = np.full([n_patch, 3, n_obs], np.nan)

    # ------------------ Prepare inversion inputs ------------------
    # Get parameter values to check
    quadtree_dir = get_downsampled_data_directory(downsampled_dir + '/' + dataset_name, param_file)

    # Get dictionary of quadtree parameters
    quadtree_params = dict(
                            resolution_threshold=resolution_threshold,
                            width_min=width_min,
                            width_max=width_max,
                            max_intersect_width=max_intersect_width,
                            min_fault_dist=min_fault_dist,
                            max_iter=max_iter, 
                            poisson_ratio=poisson_ratio,
                            smoothing=smoothing_samp,
                            edge_slip=edge_slip_samp,
                            disp_components=disp_components,
                            slip_components=slip_components,
                            quadtree_dir=quadtree_dir,
                            )

    # Prepare inversion inputs
    inputs = pyffit.inversion.get_inversion_inputs(fault, datasets, quadtree_params, date=-1, quadtree_dir=quadtree_dir, verbose=False)
    tree   = inputs[dataset_name].tree
    n_data = inputs[dataset_name].tree.x.size
    dt    /= 365.25

    # Get ramp matrix, if specified
    if ramp_type in ['linear', 'quadratic']:
        ramp_matrix = pyffit.corrections.get_ramp_matrix(tree.x, tree.y, ramp_type=ramp_type)
    else:
        ramp_matrix = []

    # Determine number of physical parameters
    if steady_slip:
        n_fault_dim = 3 * n_patch
    else:
        n_fault_dim = 2 * n_patch

    # Determine number of ramp coefficients
    dim_ramp = 0
    if len(ramp_matrix) > 0:
        dim_ramp += ramp_matrix.shape[1]

    n_dim = n_fault_dim + dim_ramp

    # -------------------- Prepare data --------------------
    samp_file = f'cutoff={resolution_threshold:.1e}_wmin={width_min:.2f}_wmax={width_max:.2f}_max_int_w={max_intersect_width:.1f}_min_fdist={min_fault_dist:.2f}_max_it={max_iter:0f}'
    d, std    = pyffit.quadtree.get_downsampled_time_series(datasets, inputs, fault, n_fault_dim, dataset_name=dataset_name, file_name=f'{downsampled_dir}/{dataset_name}/{samp_file}.h5')

    print(f'Number of fault elements: {n_patch}')
    print(f'Number of data points:    {n_data}')
    print(f'Number of observations:   {n_obs}')
    print(f'Number of dimensions:     {n_dim}')

    # -------------------- Invert --------------------
    

    return


def filter(mesh_file, triangle_file, file_format, downsampled_dir, out_dir, data_dir, 
            date_index_range=[-22, -14], xkey='x', coord_type='xy', dataset_name='data',
            check_lon=False, reference_time_series=True, use_dates=False, use_datetime=False, dt=1, data_factor=1,
            model_file='', mask_file='', estimate_covariance=False, mask_dists=[3], n_samp=2*10**7, r_inc=0.2, r_max=80, mask_dir='.', cov_dir='.', 
            m0=[0.9, 1.5, 5], c_max=2, sv_max=2, ramp_type='none', ref_point=[-116, 33.5], avg_strike=315.8, trace_inc=0.01, poisson_ratio=0.25, 
            shear_modulus=6*10**9, disp_components=[1], slip_components=[0], resolution_threshold=2.3e-1, 
            width_min=0.1, width_max=10, max_intersect_width=100, min_fault_dist=1, max_iter=10, 
            smoothing_samp=False, edge_slip_samp=False, omega=1e1, sigma=1e1, kappa=2e1, mu=2e1, 
            eta=2e1, rho=1e0, steady_slip=False, constrain=False, v_sigma=1e-9, W_sigma=1,  W_dot_sigma=1, ramp_sigma=100, v_lim=(0, 3), W_lim=(0, 30), W_dot_lim=(0, 50), ramp_lim=(-100, 100),
            xlim=[-35.77475071,26.75029172], ylim=[-26.75029172, 55.08597388], vlim_slip=[0, 20], 
            vlim_disp=[[-10,10], [-10,10], [-1,1]], cmap_slip=cmc.lajolla, cmap_disp=cmc.vik, figsize=(10, 10), dpi=75, markersize=40, 
            param_file='params.py'
            ):
    """
    Run network inversion filter.
    """

    start_total = time.time()
    
    # ------------------ Prepare run directories and files ------------------
    date    = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    run_dir = f'{out_dir}/omega_{omega:.1e}__kappa_{kappa:.1e}__sigma_{sigma:.1e}'

    pyffit.utilities.check_dir_tree(run_dir)
    pyffit.utilities.check_dir_tree(run_dir + '/Scripts/' + date)

    result_dir = f'{run_dir}/Results'
    pyffit.utilities.check_dir_tree(result_dir)
    pyffit.utilities.check_dir_tree(result_dir + '/Results')
    pyffit.utilities.check_dir_tree(result_dir + '/Slip')
    pyffit.utilities.check_dir_tree(result_dir + '/Matrices')
    pyffit.utilities.check_dir_tree(result_dir + '/Resolution')
    pyffit.utilities.check_dir_tree(downsampled_dir)
    pyffit.utilities.check_dir_tree(downsampled_dir + '/' + dataset_name)

    # Copy parameter file and nif.py version to run directory
    shutil.copy(__file__, f'{run_dir}/Scripts/{date}/nif_{date}.py')
    shutil.copy(param_file, f'{run_dir}/Scripts/{date}/params_{date}.py')
    
    print(f'Run directory: {run_dir}')

    # -------------------------- Prepare original data --------------------------    
    # Load fault model and regularization matrices
    fault = pyffit.finite_fault.TriFault(mesh_file, triangle_file, slip_components=slip_components, 
                                         poisson_ratio=poisson_ratio, shear_modulus=shear_modulus, 
                                         verbose=False, trace_inc=trace_inc, mu=mu, eta=eta,
                                         avg_strike=avg_strike, ref_point=ref_point)
    n_patch = len(fault.triangles)

    # Load data
    dataset = pyffit.data.load_insar_dataset(data_dir, file_format, dataset_name, ref_point, data_factor=data_factor, xkey=xkey, 
                                             coord_type=coord_type, date_index_range=date_index_range, 
                                             check_lon=check_lon, reference_time_series=reference_time_series, 
                                             incremental=False, use_dates=use_dates, use_datetime=use_datetime, 
                                             mask_file=mask_file, remove_mean=False)
    datasets = {dataset_name: dataset}
    n_obs    = len(dataset.date)
    x        = dataset.coords['x'].compute().data
    y        = dataset.coords['y'].compute().data
    x_rng    = x[0, :]
    y_rng    = y[:, 0]
    dims     = dataset['z'].isel(date=-1).shape
    dx       = x[0, 1] - x[0, 0]
    dy       = y[1, 0] - y[0, 0]
    
    if coord_type == 'geographic':
        lon = dataset.coords['lon'].compute().data
        lat = dataset.coords['lat'].compute().data

    # Load original slip model
    if len(model_file) > 0:
        try:
            with h5py.File(model_file, 'r') as file:
                slip_model = file['slip_model'][()]
        except KeyError:
            print('Error: speficied model_file could not be located')
            sys.exit(1)
    else:
        slip_model = np.full([n_patch, 3, n_obs], np.nan)

    # Make full Green's functions matrix
    n_chunk     = 5     # Number of chunks for full model computation
    n_data_full = x.size

    full_gf_file     = f'{data_dir}/full_greens_functions.h5'
    filter_mask_file = f'{data_dir}/filter_masks.h5'
    file_stem        = lambda date, w: f'time_series_{date}.grd'

    # Get full resolution fault masks
    if not os.path.exists(filter_mask_file):
        # Get full resolution Green's functions
        if not os.path.exists(full_gf_file):
            with h5py.File(full_gf_file, 'w') as file:
                chunk_size  = n_data_full//n_chunk 
                remainder   = n_data_full % n_chunk

                # Loop over chunks
                for i in range(n_chunk):
                    start_chunk = time.time()
                    
                    # Get chunk indicies
                    if i == n_chunk - 2:
                        remainder   = n_data_full % n_chunk
                    else:
                        remainder = 0

                    start  = chunk_size * i
                    end    = chunk_size * (i + 1) + remainder
                    
                    print(f'Working on chunk {i + 1}/{n_chunk} [{start}:{end}]...')

                    # Get Greens functions and compute model prediction
                    G = -fault.greens_functions(x.flatten()[start:end], y.flatten()[start:end], disp_components=disp_components, slip_components=slip_components, rotation=avg_strike, squeeze=True)

                    # Save 
                    file.create_dataset(f'chunk_{i}', data=G)

                    end_chunk = time.time() - start_chunk
                    print(f'Completed in {end_chunk:.1f} s')

        # Get summed Green's functions
        sum_gf = np.zeros((n_data_full))

        with h5py.File(full_gf_file, 'r') as file:
            start  = 0
            for i in range(n_chunk):
                chunk = file[f'chunk_{i}'][()]
                end   = start + chunk.shape[0]
                sum_gf[start:end] += np.sum(chunk, axis=1)
                start = end
        sum_gf = sum_gf.reshape(dims)

        # Get sign masks
        sign   = np.sign(sum_gf)
        mask_E = np.ones_like(sign)
        mask_E[sign > 0] = np.nan
        mask_W = np.ones_like(sign)
        mask_W[sign < 0] = np.nan

        with h5py.File(filter_mask_file, 'w') as file:
            file.create_dataset('mask_E', data=mask_E)
            file.create_dataset('mask_W', data=mask_W)

    else:
        with h5py.File(filter_mask_file, 'r') as file:
            mask_E = file['mask_E'][()]
            mask_W = file['mask_W'][()]

    # Plot masks
    # pyffit.figures.plot_grid(x, y, sum_gf.reshape(dims), figsize=[8, 8], show=True, cmap=cmc.vik, vlim=[-0.75, 0.75])
    # pyffit.figures.plot_grid(x, y, sign, figsize=[8, 8], show=True, cmap=cmc.vik, vlim=[-1, 1])
    # pyffit.figures.plot_grid(x, y, sign_e, figsize=[8, 8], show=True, cmap=cmc.vik, vlim=[-1, 1])
    # pyffit.figures.plot_grid(x, y, sign_w, figsize=[8, 8], show=True, cmap=cmc.vik, vlim=[-1, 1])
    # pyffit.data.write_grd(x[0, :], y[:, 0], sign_e, f'{run_dir}/mask_E.grd')
    # pyffit.data.write_grd(x[0, :], y[:, 0], sign_w, f'{run_dir}/mask_W.grd')
    
    # Filter
    sigma    = 5.0               # standard deviation for Gaussian kernel
    truncate = 4.0               # truncate filter at this many sigmas

    w_x = (2*int(truncate*sigma + 0.5) + 1) * dx
    w_y = (2*int(truncate*sigma + 0.5) + 1) * dy
    w   = max((w_x, w_y))

    # Loop through all observations
    filt_dir = data_dir + f'/low_pass_filt/w-{w:.1f}km'
    pyffit.utilities.check_dir_tree(filt_dir)

    for k in range(n_obs):

        # print(f'Filtering observation {k + 1}/{n_obs} with sigma = {sigma:.1f} and truncate = {truncate:.1f}...' + np.sum(np.isnan(data_filt)))
        date = datasets[dataset_name].date[k].dt.strftime('%Y-%m-%d').item()
        data = dataset['z'].isel(date=k).data.compute()

        data_E, _ = nan_gaussian_filter(data * mask_E, sigma=sigma, truncate=truncate, return_width=True, plot=False)
        data_W, _ = nan_gaussian_filter(data * mask_W, sigma=sigma, truncate=truncate, return_width=True, plot=False)

        # Combine halves of observation
        data_filt = np.empty_like(data)
        data_filt[~np.isnan(data_W)] = data_W[~np.isnan(data_W)]
        data_filt[~np.isnan(data_E)] = data_E[~np.isnan(data_E)]
    
        print(f'Filtering observation {k + 1}/{n_obs} with sigma = {sigma:.1f} and truncate = {truncate:.1f}...' + f'{np.sum(np.isnan(data_filt))}')

        # Plot
        pyffit.figures.plot_grid(x, y, data_filt, figsize=[8, 8], show=False, cmap=cmc.vik, vlim=[-20, 20], cbar=True, 
                                 file_name=f'{filt_dir}/lp-filt-{w:.1f}km_{date}.png')

        # Save
        if coord_type == 'geographic':
            pyffit.data.write_grd(lon, lat, data_filt, f'{filt_dir}/' + file_stem(date, w))
        else:
            pyffit.data.write_grd(x_rng, y_rng, data_filt, f'{filt_dir}/' + file_stem(date, w), coords_key=coord_type)

    return


def covariances(mesh_file, triangle_file, file_format, downsampled_dir, out_dir, data_dir, 
            date_index_range=[-22, -14], xkey='x', coord_type='xy', dataset_name='data',
            check_lon=False, reference_time_series=True, use_dates=False, use_datetime=False, dt=1, data_factor=1,
            model_file='', mask_file='', estimate_covariance=False, mask_dists=[3], n_samp=2*10**7, r_inc=0.2, r_max=80, mask_dir='.', cov_dir='.', 
            m0=[0.9, 1.5, 5], c_max=2, sv_max=2, ramp_type='none', ref_point=[-116, 33.5], avg_strike=315.8, trace_inc=0.01, poisson_ratio=0.25, 
            shear_modulus=6*10**9, disp_components=[1], slip_components=[0], resolution_threshold=2.3e-1, 
            width_min=0.1, width_max=10, max_intersect_width=100, min_fault_dist=1, max_iter=10, 
            smoothing_samp=False, edge_slip_samp=False, omega=1e1, sigma=1e1, kappa=2e1, mu=2e1, 
            eta=2e1, rho=1e0, steady_slip=False, constrain=False, v_sigma=1e-9, W_sigma=1,  W_dot_sigma=1, ramp_sigma=100, v_lim=(0, 3), W_lim=(0, 30), W_dot_lim=(0, 50), ramp_lim=(-100, 100),
            xlim=[-35.77475071,26.75029172], ylim=[-26.75029172, 55.08597388], vlim_slip=[0, 20], 
            vlim_disp=[[-10,10], [-10,10], [-1,1]], cmap_slip=cmc.lajolla, cmap_disp=cmc.vik, figsize=(10, 10), dpi=75, markersize=40, 
            param_file='params.py'
            ):
    """
    Run network inversion filter.
    """

    start_total = time.time()
    
    # ------------------ Prepare run directories and files ------------------
    date    = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    run_dir = f'{out_dir}/omega_{omega:.1e}__kappa_{kappa:.1e}__sigma_{sigma:.1e}'

    pyffit.utilities.check_dir_tree(run_dir)
    pyffit.utilities.check_dir_tree(run_dir + '/Scripts/' + date)

    result_dir = f'{run_dir}/Results'
    pyffit.utilities.check_dir_tree(result_dir)
    pyffit.utilities.check_dir_tree(result_dir + '/Results')
    pyffit.utilities.check_dir_tree(result_dir + '/Slip')
    pyffit.utilities.check_dir_tree(result_dir + '/Matrices')
    pyffit.utilities.check_dir_tree(result_dir + '/Resolution')
    pyffit.utilities.check_dir_tree(downsampled_dir)
    pyffit.utilities.check_dir_tree(downsampled_dir + '/' + dataset_name)

    # Copy parameter file and nif.py version to run directory
    shutil.copy(__file__, f'{run_dir}/Scripts/{date}/nif_{date}.py')
    shutil.copy(param_file, f'{run_dir}/Scripts/{date}/params_{date}.py')
    
    print(f'Run directory: {run_dir}')

    # -------------------------- Prepare original data --------------------------    
    # Load fault model and regularization matrices
    fault = pyffit.finite_fault.TriFault(mesh_file, triangle_file, slip_components=slip_components, 
                                         poisson_ratio=poisson_ratio, shear_modulus=shear_modulus, 
                                         verbose=False, trace_inc=trace_inc, mu=mu, eta=eta,
                                         avg_strike=avg_strike, ref_point=ref_point)
    n_patch = len(fault.triangles)

    # Load data
    dataset = pyffit.data.load_insar_dataset(data_dir, file_format, dataset_name, ref_point, data_factor=data_factor, xkey=xkey, 
                                             coord_type=coord_type, date_index_range=date_index_range, 
                                             check_lon=check_lon, reference_time_series=reference_time_series, 
                                             incremental=False, use_dates=use_dates, use_datetime=use_datetime, 
                                             mask_file=mask_file, remove_mean=False)
    datasets = {dataset_name: dataset}
    n_obs    = len(dataset.date)
    x        = dataset.coords['x'].compute().data
    y        = dataset.coords['y'].compute().data
    dims     = dataset['z'].isel(date=-1).shape
    dx       = x[0, 1] - x[0, 0]
    dy       = y[1, 0] - y[0, 0]

    # ------------------ Prepare inversion inputs ------------------
    # Get parameter values to check
    quadtree_dir = get_downsampled_data_directory(downsampled_dir + '/' + dataset_name, param_file)

    # Get dictionary of quadtree parameters
    quadtree_params = dict(
                            resolution_threshold=resolution_threshold,
                            width_min=width_min,
                            width_max=width_max,
                            max_intersect_width=max_intersect_width,
                            min_fault_dist=min_fault_dist,
                            max_iter=max_iter, 
                            poisson_ratio=poisson_ratio,
                            smoothing=smoothing_samp,
                            edge_slip=edge_slip_samp,
                            disp_components=disp_components,
                            slip_components=slip_components,
                            quadtree_dir=quadtree_dir,
                            )

    # Prepare inversion inputs
    inputs = pyffit.inversion.get_inversion_inputs(fault, datasets, quadtree_params, date=-1, quadtree_dir=quadtree_dir, verbose=False)
    tree   = inputs[dataset_name].tree
    n_data = inputs[dataset_name].tree.x.size
    dt    /= 365.25

    # Get ramp matrix, if specified
    if ramp_type in ['linear', 'quadratic']:
        ramp_matrix = pyffit.corrections.get_ramp_matrix(tree.x, tree.y, ramp_type=ramp_type)
    else:
        ramp_matrix = []

    # Determine number of physical parameters
    if steady_slip:
        n_fault_dim = 3 * n_patch
    else:
        n_fault_dim = 2 * n_patch

    # Determine number of ramp coefficients
    dim_ramp = 0
    if len(ramp_matrix) > 0:
        dim_ramp += ramp_matrix.shape[1]

    n_dim = n_fault_dim + dim_ramp

    # -------------------- Prepare data --------------------
    samp_file = f'cutoff={resolution_threshold:.1e}_wmin={width_min:.2f}_wmax={width_max:.2f}_max_int_w={max_intersect_width:.1f}_min_fdist={min_fault_dist:.2f}_max_it={max_iter:0f}'
    d, std    = pyffit.quadtree.get_downsampled_time_series(datasets, inputs, fault, n_fault_dim, dataset_name=dataset_name, file_name=f'{downsampled_dir}/{dataset_name}/{samp_file}.h5')

    print(f'Number of fault elements: {n_patch}')
    print(f'Number of data points:    {n_data}')
    print(f'Number of observations:   {n_obs}')
    print(f'Number of dimensions:     {n_dim}')

    # # Load slip model
    # with h5py.File(f'{run_dir}/results_smoothing.h5', 'r') as file:
    #     slip_model = file['x_s'][:, :n_patch][()]
    #     disp_model = file['model'][()]
    #     resid      = file['resid'][()]
    #     rms        = file['rms'][()]

    # # -------------------- Parameters --------------------
    # k         = 204   # date index
    # n_chunk   = 9     # Number of chunks for full model computation
    # resid_max = 20    # Max. allows residual amplitude for masking
    # r_max     = 80    # Max. allowed relative distance for semivariogram estimation    
    # n_samp    = 10**6 # Number of samples for semivariogram estimation
    # C_min     = 1e-3  # Min. value to enforce zero covaraince
    # # ----------------------------------------------------

    # # -------------------- Get full displacement field --------------------
    # data = dataset['z'].isel(date=-1).data.compute()

    # full_model_file = f'{run_dir}/full_res_model.h5'

    # if os.path.exists(full_model_file):
    #     with h5py.File(f'{run_dir}/full_res_model.h5', 'r') as file:
    #         d_full = file[f'date_{k}'][()]

    # else:
    #     d_full = np.empty((n_data_full))
    #     n_data_full = x.size
    #     chunk_size  = n_data_full//n_chunk 
    #     remainder   = n_data_full % n_chunk

    #     for i in range(n_chunk - 1):
    #         print(f'Working on chunk {i + 1}...')
    #         start_chunk = time.time()
            
    #         # Get chunk indicies
    #         if i == n_chunk - 2:
    #             remainder   = n_data_full % n_chunk
    #         else:
    #             remainder = 0
    #         start  = chunk_size * i
    #         end    = chunk_size * (i + 1) + remainder

    #         # Get Greens functions and compute model prediciton
    #         G = -fault.greens_functions(x.flatten()[start:end], y.flatten()[start:end], disp_components=disp_components, slip_components=slip_components, rotation=avg_strike, squeeze=True)
    #         d_full[start:end] = G @ slip_model[k, :]

    #         end_chunk = time.time() - start_chunk
    #         print(f'Completed in {end_chunk:.1f} s')

    #     with h5py.File(f'{run_dir}/full_res_model.h5', 'w') as file:
    #         file.create_dataset(f'date_{k}', data=d_full)

    # d_grid = d_full.reshape(data.shape)

    # # -------------------- Estimate semivariogram --------------------
    # # Apply mask
    # full_resid = abs(data - d_grid)
    # i_mask       = full_resid > resid_max
    # mask = np.ones_like(data)
    # mask[i_mask] = np.nan


    # # Plot data
    # fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    # gs = fig.add_gridspec(1, 3, width_ratios=(1, 1, 0.05),
    #                         hspace=0.0, wspace=0.05)

    # ax0 = fig.add_subplot(gs[0, 0])
    # ax1 = fig.add_subplot(gs[0, 1])
    # # ax2 = fig.add_subplot(gs[0, 2])
    # cax = fig.add_subplot(gs[0, -1])
    # axes = [ax0, ax1]

    # data_masked = np.copy(data)
    # # data_masked[mask] = np.nan
    # data_masked *= mask
    # im = ax0.imshow(data,          cmap=cmc.vik, vmin=-20, vmax=20, interpolation='none')
    # im = ax1.imshow(data_masked,  cmap=cmc.vik, vmin=-20, vmax=20, interpolation='none')
    # # im = ax2.imshow(abs(data - d_grid), cmap=cmc.lajolla_r, vmin=0, vmax=50, interpolation='none')

    # for ax in axes:
    #     ax.invert_yaxis()
    # plt.colorbar(im, cax=cax, shrink=0.7)
    # plt.show()
    # return

    # Get mask distance
    mask_dist = mask_dists[0]

    # Get inter-cell distances
    dist_file = f'{quadtree_dir}/dist.h5'

    if os.path.exists(dist_file):
        with h5py.File(dist_file, 'r') as file:                
            dist = file['dist'][()]

    else:
        print('Computing cell distances...')
        # Compute distances
        dist = np.empty((n_data, n_data))

        for i in range(n_data):
            for j in range(n_data):
                dist[i, j] = pyffit.covariance.dist(tree.x[i], tree.x[j], tree.y[i], tree.y[j])
        
        # Write to disk
        with h5py.File(dist_file, 'w') as file:                
            file.create_dataset('dist', data=dist)

    # Get covariance model
    sv_file  = f'{cov_dir}/semivariogram_params.h5' # parameters for covariance model for each observation
    cov_file = f'{quadtree_dir}/covariance.h5'      # modeled covariance matrices for each observation

    pyffit.utilities.check_dir_tree(quadtree_dir + '/covariance')
    print(sv_file)
    print(cov_file)

    if not os.path.exists(sv_file):
        # pyffit.covariance.estimate(x, y, fault, dataset, mask_dists, mask=[], n_samp=n_samp, r_inc=r_inc, r_max=r_max, mask_dir=mask_dir, cov_dir=cov_dir, m0=m0, c_max=c_max, sv_max=sv_max,)
        pyffit.covariance.estimate(x.flatten(), y.flatten(), fault, dataset, mask_dists, mask=[], n_samp=n_samp, r_inc=r_inc, r_max=r_max, mask_dir=mask_dir, cov_dir=cov_dir, m0=m0, c_max=c_max, sv_max=sv_max,)

    # Compute covariance matrices
    # if not os.path.exists(cov_file):
    with h5py.File(cov_file, 'w') as out_file: 
        with h5py.File(sv_file, 'r') as file:  
            for k, date in enumerate(dataset.date):
                try: 
                    date = datasets[dataset_name].date[k].dt.strftime('%Y-%m-%d').item()
                except TypeError:
                    date = date     
                
                print(f'Working on {date}... {k+1}/{len(dataset.date)}')
                
                # Get semivariogram parameters
                key = f'mask_dist_{mask_dist}_km/{date}'
                sv_params = file[key][()]

                # Compute semivariance and covariance
                # sv_model = pyffit.covariance.exp_semivariogram(sv_params[0], sv_params[1], sv_params[2], dist)
                C = pyffit.covariance.exp_covariance(sv_params[0], sv_params[1], sv_params[2], dist)
                C -= np.nanmin(C)

                # cn = np.linalg.cond(C)
                cn = np.nan
                # c_model  = sv_params[0] + sv_params[1] - sv_model
                out_file.create_dataset(f'covariance/{k}', data=C) 
                out_file.create_dataset(f'condition_number/{k}', data=cn) 
                text = f'{date} condition number: {cn:.1e} | Range: {C.min():.2f} - {C.max():.2f}'
        
                fig, ax = plt.subplots(figsize=(14, 8.2))
                ax.set_title(text)
                im = ax.imshow(C, cmap=cmc.lajolla_r, interpolation='none')
                plt.colorbar(im, label=r'Covariance ($mm^2$)')
                plt.savefig(quadtree_dir + f'/covariance/C_{k}.png', dpi=200)
                plt.close()

    return
    # r_bins, sv_orig, _ = pyffit.covariance.semivariogram(x.flatten(), y.flatten(), data.flatten(), r_max=r_max, n_samp=n_samp, r_inc=0.1)
    # r_bins, sv_mask, _ = pyffit.covariance.semivariogram(x.reshape(data.shape)[mask].flatten(), y.reshape(data.shape)[mask].flatten(), data[mask].flatten(), r_max=r_max, n_samp=n_samp, r_inc=0.1)
    # r_inc = r_bins[1] - r_bins[0]
    # r     = r_bins[:-1] + r_inc/2

    # # Fit semivariograms
    # sv_params_orig = pyffit.covariance.fit_semivariogram(r, sv_orig, m0=[50, 100, 5], r_max=r_max)
    # sv_params_mask = pyffit.covariance.fit_semivariogram(r, sv_mask, m0=[50, 100, 5], r_max=r_max)

    # Get models
    # sv_model_orig  = pyffit.covariance.exp_semivariogram(sv_params_orig[0], sv_params_orig[1], sv_params_orig[2], r)
    # sv_model_mask  = pyffit.covariance.exp_semivariogram(sv_params_mask[0], sv_params_mask[1], sv_params_mask[2], r)
    # C_model_orig   = pyffit.covariance.exp_covariance(sv_params_orig[0], sv_params_orig[1], sv_params_orig[2], r)
    # C_model_mask   = pyffit.covariance.exp_covariance(sv_params_mask[0], sv_params_mask[1], sv_params_mask[2], r)


    # # Get mask distance
    # mask_dist = mask_dists[0]

    # # Get inter-cell distances
    # dist_file = f'{quadtree_dir}/dist.h5'

    # if os.path.exists(dist_file):
    #     with h5py.File(dist_file, 'r') as file:                
    #         dist = file['dist'][()]

    # else:
    #     print('Computing cell distances...')
    #     # Compute distances
    #     dist = np.empty((n_data, n_data))

    #     for i in range(n_data):
    #         for j in range(n_data):
    #             dist[i, j] = pyffit.covariance.dist(tree.x[i], tree.x[j], tree.y[i], tree.y[j])
        
    #     # Write to disk
    #     with h5py.File(dist_file, 'w') as file:                
    #         file.create_dataset('dist', data=dist)

    # # Get covariance model
    # # sv_file  = f'{cov_dir}/semivariogram_params.h5' # parameters for covariance model for each observation
    # # cov_file = f'{quadtree_dir}/covariance.h5'      # modeled covariance matrices for each observation
    # cov_file = f'./covariance.h5'      # modeled covariance matrices for each observation

    # # if not os.path.exists(sv_file):
    # #     pyffit.covariance.estimate(x, y, fault, dataset, mask_dists, n_samp=n_samp, r_inc=r_inc, r_max=r_max, mask_dir=mask_dir, cov_dir=cov_dir, m0=m0, c_max=c_max, sv_max=sv_max,)

    # # Compute semivariance and covariance
    # for sv_params, label in zip([sv_params_orig, sv_params_mask], ['Original', 'Masked']):
    #     C  = pyffit.covariance.exp_covariance(sv_params[0], sv_params[1], sv_params[2], dist)
    #     C_adj   = C - C.min()
    #     C_norm  = C.max() * (C - C.min())/(C.max() - C.min())
    #     C_trunc = np.copy(C_adj)
    #     C_trunc[C_trunc < C_min] = 0

    #     # Perform RCM sorting
    #     tmp_sparse = scipy.sparse.csr_matrix(C)
    #     perm = scipy.sparse.csgraph.reverse_cuthill_mckee(tmp_sparse, symmetric_mode=True)

    #     print(f'\n########## {label} ##########')
    #     print(f'C       condition number: {np.linalg.cond(C):.1e} | Range: {C.min():.2f} - {C.max():.2f}')
    #     print(f'C_adj   condition number: {np.linalg.cond(C_adj):.1e} | Range: {C_adj.min():.2f} - {C_adj.max():.2f}')
    #     print(f'C_norm  condition number: {np.linalg.cond(C_norm):.1e} | Range: {C_norm.min():.2f} - {C_norm.max():.2f}')
    #     print(f'C_trunc condition number: {np.linalg.cond(C_trunc):.1e} | Range: {C_trunc.min():.2f} - {C_trunc.max():.2f}')
    #     print(f'C_rcm   condition number: {np.linalg.cond(C[perm, :][:, perm]):.1e} | Range: {C[perm, :][:, perm].min():.2f} - {C[perm, :][:, perm].max():.2f}')
        
    # return



    # # Plot semivariograms
    # fig, axes = plt.subplots(1, 2, figsize=(14, 8.2))
    # fig.suptitle(f'{np.round(sv_params_orig, 2)} {np.round(sv_params_mask, 2)}')
    # axes[0].scatter(r, sv_orig, marker='.', color='k', alpha=0.2)
    # axes[0].scatter(r, sv_mask, marker='.', color='C3', alpha=0.2)
    # axes[0].plot(r, sv_model_orig, color='k',)
    # axes[0].plot(r, sv_model_mask, color='C3',)
    # # ax.plot(r, sv_model_orig_nug, color='k', linestyle='--')
    # # ax.plot(r, sv_model_mask_nug, color='C3', linestyle='--')

    # # Plot covariances
    # axes[1].plot(r, C_model_orig, color='k',)
    # axes[1].plot(r, C_model_mask, color='C3',)
    # # ax.scatter(r, sv_orig, marker='.', color='k', alpha=0.2)
    # # ax.scatter(r, sv_mask, marker='.', color='C3', alpha=0.2)

    # for ax in axes:
    #     ax.set_ylim(0, 250)
    #     ax.grid()
    # plt.show()

    k = 204
    date = datasets[dataset_name].date[k].dt.strftime('%Y-%m-%d').item()

    with h5py.File(sv_file, 'r') as file:  
        # Get semivariogram parameters
        key = f'mask_dist_{mask_dist}_km/{date}'
        sv_params = file[key][()]

    # with h5py.File(cov_file, 'r') as out_file: 

    #     for k, date in enumerate(dataset.date):
    #         date = datasets[dataset_name].date[k].dt.strftime('%Y-%m-%d').item()
    #         C = out_file[ f'covariance/{k}'][()]

    #         print(f'{date} condition number: {np.linalg.cond(C):.1e} | Range: {C.min():.2f} - {C.max():.2f}')

    # Perform RCM sorting
    tmp_sparse = scipy.sparse.csr_matrix(C)
    perm = scipy.sparse.csgraph.reverse_cuthill_mckee(tmp_sparse, symmetric_mode=True)

    C += sv_params[0]
    C_adj = C - sv_params[0]


    print(f'\n########## {k} ##########')
    print(f'C       condition number: {np.linalg.cond(C):.1e} | Range: {C.min():.2f} - {C.max():.2f}')
    print(f'C_adj   condition number: {np.linalg.cond(C_adj):.1e} | Range: {C_adj.min():.2f} - {C_adj.max():.2f}')
    # print(f'C_norm  condition number: {np.linalg.cond(C_norm):.1e} | Range: {C_norm.min():.2f} - {C_norm.max():.2f}')
    # print(f'C_trunc condition number: {np.linalg.cond(C_trunc):.1e} | Range: {C_trunc.min():.2f} - {C_trunc.max():.2f}')
    print(f'C_rcm   condition number: {np.linalg.cond(C[perm, :][:, perm]):.1e} | Range: {C[perm, :][:, perm].min():.2f} - {C[perm, :][:, perm].max():.2f}')

    # Plot
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=(1, 1, 0.05),
                                 hspace=0.0, wspace=0.05)

    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    cax  = fig.add_subplot(gs[0, -1])
     
    im = axes[0].imshow(C, cmap=cmc.lajolla_r, interpolation='none', vmin=C.min(), vmax=C.max())
    im = axes[1].imshow(C[perm, :][:, perm], cmap=cmc.lajolla_r, interpolation='none', vmin=C.min(), vmax=C.max())


    plt.colorbar(im, cax=cax)
    plt.show()

    sys.exit()


    # # Plot histogram
    # xlim = [0, 60]
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.set_title(f'Number of residuals > {resid_max} mm: {np.sum(mask):.0f}/{n_real} points ({np.sum(mask)/n_real:.2f} %)')
    # ax.hist(full_resid[i_real].flatten(), color='k', cumulative=True, range=xlim, bins=20, histtype='step', density=True)
    # ax.set_xlim(xlim)
    # plt.show()

    # # Plot data
    # fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    # gs = fig.add_gridspec(1, 3, width_ratios=(1, 1, 0.05),
    #                         hspace=0.0, wspace=0.05)

    # ax0 = fig.add_subplot(gs[0, 0])
    # ax1 = fig.add_subplot(gs[0, 1])
    # # ax2 = fig.add_subplot(gs[0, 2])
    # cax = fig.add_subplot(gs[0, -1])
    # axes = [ax0, ax1]

    # data_masked = np.copy(data)
    # data_masked[mask] = np.nan

    # im = ax0.imshow(data,          cmap=cmc.vik, vmin=-20, vmax=20, interpolation='none')
    # im = ax1.imshow(data_masked,  cmap=cmc.vik, vmin=-20, vmax=20, interpolation='none')
    # # im = ax2.imshow(abs(data - d_grid), cmap=cmc.lajolla_r, vmin=0, vmax=50, interpolation='none')

    # for ax in axes:
    #     ax.invert_yaxis()
    # plt.colorbar(im, cax=cax, shrink=0.7)
    # plt.show()


    # cov_file = f'{quadtree_dir}/covariance.h5'      # modeled covariance matrices for each observation
    # dist_file = f'{quadtree_dir}/dist.h5'

    # k = 204
    # trunc = 100
    # dist_max = 20
    # with h5py.File(cov_file, 'r') as covariance:
    #     with h5py.File(dist_file, 'r') as file:                
    #         dist = file['dist'][()]
            
    #         C = covariance[f'covariance/{k}'][()]
    #         C_trunc = np.zeros_like(C)
    #         n_data = C.shape[0]

    #         # C_trunc[dist <= dist_max] = C[dist <= dist_max]
    #         C_trunc = C.max() * (C - C.min())/(C.max() - C.min())

    #         std_norm = std[k, :]
    #         std_norm[std_norm < 1] = 1

    #         C_std = np.eye(n_data) * std_norm ** 2
            
    #         print(f'C       condition number: {np.linalg.cond(C)}')
    #         print(f'C_std   condition number: {np.linalg.cond(C_std)}')
    #         print(f'C_trunc condition number: {np.linalg.cond(C_trunc)}')

    # # Create figure and GridSpec layout
    # fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    # gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.05], hspace=0.1)

    # # Top row: image plots
    # ax0  = fig.add_subplot(gs[0, 0])
    # ax1  = fig.add_subplot(gs[0, 1])
    # ax2  = fig.add_subplot(gs[0, 2])
    # cax0 = fig.add_subplot(gs[1, 1])
    # cax1 = fig.add_subplot(gs[1, 2])

    # # Display the images with shared color limits
    # vmin = 0
    # vmax = np.max(C)

    # im0 = ax0.imshow(C,       vmin=vmin, vmax=vmax,     interpolation='none', cmap=cmc.lajolla)
    # im1 = ax1.imshow(C_trunc, vmin=vmin, vmax=vmax,     interpolation='none', cmap=cmc.lajolla)
    # im2 = ax2.imshow(C_std,   vmin=vmin,                interpolation='none', cmap=cmc.lajolla)
    # # im2 = ax2.imshow(dist,   vmin=vmin, vmax=dist_max,  interpolation='none', cmap=cmc.imola)

    # ax0.set_title(r'$C$ condition number: ' + f'{np.linalg.cond(C):.0f}')
    # ax1.set_title(r'$C_{trunc}$ condition number: ' + f'{np.linalg.cond(C_trunc):.0f}')
    # # ax2.set_title(r'Distance')
    # ax2.set_title(r'$C_{std}$ condition number: ' + f'{np.linalg.cond(C_std):.0f}')

    # # Bottom row: shared colorbar spanning both columns
    # fig.colorbar(im1, label=r'Covaraince(mm$^2$)', cax=cax0, orientation='horizontal', shrink=0.3)
    # fig.colorbar(im2, label=r'Distance (km)', cax=cax1, orientation='horizontal', shrink=0.3)

    # plt.show()

    # sys.exit(1)
    return


def run_nif(mesh_file, triangle_file, file_format, downsampled_dir, out_dir, data_dir, 
            date_index_range=[-22, -14], xkey='x', coord_type='xy', dataset_name='data',
            check_lon=False, reference_time_series=True, use_dates=False, use_datetime=False, dt=1, data_factor=1,
            model_file='', mask_file='', estimate_covariance=False, mask_dists=[3], n_samp=2*10**7, r_inc=0.2, r_max=80, mask_dir='.', cov_dir='.', 
            m0=[0.9, 1.5, 5], c_max=2, sv_max=2, ramp_type='none', ref_point=[-116, 33.5], avg_strike=315.8, trace_inc=0.01, poisson_ratio=0.25, 
            shear_modulus=6*10**9, disp_components=[1], slip_components=[0], resolution_threshold=2.3e-1, 
            width_min=0.1, width_max=10, max_intersect_width=100, min_fault_dist=1, max_iter=10, 
            smoothing_samp=False, edge_slip_samp=False, omega=1e1, sigma=1e1, kappa=2e1, mu=2e1, 
            eta=2e1, rho=1e0, steady_slip=False, constrain=False, v_sigma=1e-9, W_sigma=1,  W_dot_sigma=1, ramp_sigma=100, v_lim=(0, 3), W_lim=(0, 30), W_dot_lim=(0, 50), ramp_lim=(-100, 100),
            xlim=[-35.77475071,26.75029172], ylim=[-26.75029172, 55.08597388], vlim_slip=[0, 20], 
            vlim_disp=[[-10,10], [-10,10], [-1,1]], cmap_slip=cmc.lajolla, cmap_disp=cmc.vik, figsize=(10, 10), dpi=75, markersize=40, 
            param_file='params.py'
            ):
    """
    Run network inversion filter.
    """

    start_total = time.time()
    
    # ------------------ Prepare run directories and files ------------------
    date    = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    run_dir = f'{out_dir}/omega_{omega:.1e}__kappa_{kappa:.1e}__sigma_{sigma:.1e}'

    pyffit.utilities.check_dir_tree(run_dir)
    pyffit.utilities.check_dir_tree(run_dir + '/Scripts/' + date)

    result_dir = f'{run_dir}/Results'
    pyffit.utilities.check_dir_tree(result_dir)
    pyffit.utilities.check_dir_tree(result_dir + '/Results')
    pyffit.utilities.check_dir_tree(result_dir + '/Slip')
    pyffit.utilities.check_dir_tree(result_dir + '/Matrices')
    pyffit.utilities.check_dir_tree(result_dir + '/Resolution')
    pyffit.utilities.check_dir_tree(downsampled_dir)
    pyffit.utilities.check_dir_tree(downsampled_dir + '/' + dataset_name)

    # Copy parameter file and nif.py version to run directory
    shutil.copy(__file__, f'{run_dir}/Scripts/{date}/nif_{date}.py')
    shutil.copy(param_file, f'{run_dir}/Scripts/{date}/params_{date}.py')
    
    print(f'Run directory: {run_dir}')

    # -------------------------- Prepare original data --------------------------    
    # Load fault model and regularization matrices
    fault = pyffit.finite_fault.TriFault(mesh_file, triangle_file, slip_components=slip_components, 
                                         poisson_ratio=poisson_ratio, shear_modulus=shear_modulus, 
                                         verbose=False, trace_inc=trace_inc, mu=mu, eta=eta,
                                         avg_strike=avg_strike, ref_point=ref_point)
    n_patch = len(fault.triangles)

    # Load data
    dataset = pyffit.data.load_insar_dataset(data_dir, file_format, dataset_name, ref_point, data_factor=data_factor, xkey=xkey, 
                                             coord_type=coord_type, date_index_range=date_index_range, 
                                             check_lon=check_lon, reference_time_series=reference_time_series, 
                                             incremental=False, use_dates=use_dates, use_datetime=use_datetime, 
                                             mask_file=mask_file, remove_mean=False)
    datasets = {dataset_name: dataset}
    n_obs    = len(dataset.date)
    x        = dataset.coords['x'].compute().data.flatten()
    y        = dataset.coords['y'].compute().data.flatten()

    # Load original slip model
    if len(model_file) > 0:
        try:
            with h5py.File(model_file, 'r') as file:
                slip_model = file['slip_model'][()]
        except KeyError:
            print('Error: speficied model_file could not be located')
            sys.exit(1)
    else:
        slip_model = np.full([n_patch, 3, n_obs], np.nan)

    # ------------------ Prepare inversion inputs ------------------
    # Get parameter values to check
    quadtree_dir = get_downsampled_data_directory(downsampled_dir + '/' + dataset_name, param_file)
    
    
    # Get dictionary of quadtree parameters
    quadtree_params = dict(
                            resolution_threshold=resolution_threshold,
                            width_min=width_min,
                            width_max=width_max,
                            max_intersect_width=max_intersect_width,
                            min_fault_dist=min_fault_dist,
                            max_iter=max_iter, 
                            poisson_ratio=poisson_ratio,
                            smoothing=smoothing_samp,
                            edge_slip=edge_slip_samp,
                            disp_components=disp_components,
                            slip_components=slip_components,
                            quadtree_dir=quadtree_dir,
                            )

    # Prepare inversion inputs
    inputs = pyffit.inversion.get_inversion_inputs(fault, datasets, quadtree_params, date=-1, quadtree_dir=quadtree_dir, verbose=False)
    tree   = inputs[dataset_name].tree
    n_data = inputs[dataset_name].tree.x.size
    dt    /= 365.25

    # Get ramp matrix, if specified
    if ramp_type in ['linear', 'quadratic']:
        ramp_matrix = pyffit.corrections.get_ramp_matrix(tree.x, tree.y, ramp_type=ramp_type)
    else:
        ramp_matrix = []

    # Determine number of physical parameters
    if steady_slip:
        n_fault_dim = 3 * n_patch
    else:
        n_fault_dim = 2 * n_patch

    # Determine number of ramp coefficients
    dim_ramp = 0
    if len(ramp_matrix) > 0:
        dim_ramp += ramp_matrix.shape[1]

    n_dim = n_fault_dim + dim_ramp

    # -------------------- Prepare data --------------------
    samp_file = f'cutoff={resolution_threshold:.1e}_wmin={width_min:.2f}_wmax={width_max:.2f}_max_int_w={max_intersect_width:.1f}_min_fdist={min_fault_dist:.2f}_max_it={max_iter:0f}'
    d, std    = pyffit.quadtree.get_downsampled_time_series(datasets, inputs, fault, n_fault_dim, dataset_name=dataset_name, file_name=f'{downsampled_dir}/{dataset_name}/{samp_file}.h5')

    if ramp_type == 'saved':
        print('Removing ramps fit from previous run residuals')

        with h5py.File(f'{run_dir}/Ramps/corrected.h5') as ramps:
            d[:, :n_data] -= ramps['ramp']

            run_dir = f'{run_dir}/Deramped'
            pyffit.utilities.check_dir_tree(run_dir)

    print(f'Number of fault elements: {n_patch}')
    print(f'Number of data points:    {n_data}')
    print(f'Number of observations:   {n_obs}')
    print(f'Number of dimensions:     {n_dim}')

    # -------------------- Greens Functions --------------------
    G = -fault.greens_functions(inputs[dataset_name].tree.x, inputs[dataset_name].tree.y, disp_components=disp_components, slip_components=slip_components, rotation=avg_strike, squeeze=True)
    
    # -------------------- Data covariance --------------------
    # Get covariance model
    cov_file  = f'{cov_dir}/semivariogram_params.h5' # parameters for covariance model for each observation
    # cov_file = f'{quadtree_dir}/covariance.h5'      # modeled covariance matrices for each observation
    # R_file   = f'{run_dir}/R.h5'                    # complete data covariance matrix for each observation 

    if estimate_covariance:
        print(f'Using covariance estimates at {cov_file}')
        # # Get mask distance
        # mask_dist = mask_dists[0]

        # # Get inter-cell distances
        # dist_file = f'{quadtree_dir}/dist.h5'

        # if os.path.exists(dist_file):
        #     with h5py.File(dist_file, 'r') as file:                
        #         dist = file['dist'][()]

        # else:
        #     print('Computing cell distances...')
        #     # Compute distances
        #     dist = np.empty((n_data, n_data))

        #     for i in range(n_data):
        #         for j in range(n_data):
        #             dist[i, j] = pyffit.covariance.dist(tree.x[i], tree.x[j], tree.y[i], tree.y[j])
            
        #     # Write to disk
        #     with h5py.File(dist_file, 'w') as file:                
        #         file.create_dataset('dist', data=dist)



        # if not os.path.exists(sv_file):
        #     pyffit.covariance.estimate(x, y, fault, dataset, mask_dists, n_samp=n_samp, r_inc=r_inc, r_max=r_max, mask_dir=mask_dir, cov_dir=cov_dir, m0=m0, c_max=c_max, sv_max=sv_max,)

        # # Compute covariance matrices
        # if not os.path.exists(cov_file):
        #     with h5py.File(cov_file, 'w') as out_file: 
        #         with h5py.File(sv_file, 'r') as file:  
        #             for k, date in enumerate(dataset.date):
        #                 date = datasets[dataset_name].date[k].dt.strftime('%Y-%m-%d').item()
        #                 print(f'Working on {date}...')
                        
        #                 # Get semivariogram parameters
        #                 key = f'mask_dist_{mask_dist}_km/{date}'
        #                 sv_params = file[key][()]

        #                 # Compute semivariance and covariance
        #                 sv_model = pyffit.covariance.exp_semivariogram(sv_params[0], sv_params[1], sv_params[2], dist)
        #                 c_model  = sv_params[0] + sv_params[1] - sv_model
        #                 out_file.create_dataset(f'covariance/{k}', data=c_model) 

    else:
        print(f'Using dialgonal covariance matrices')
        C = np.eye(inputs[dataset_name].tree.x.size)
        with h5py.File(cov_file, 'w') as out_file: 
            out_file.create_dataset(f'covariance', data=C) 

    # Set up model bounds and initial uncertainties
    if steady_slip:
        state_lims   = [v_lim, W_lim, W_dot_lim]
        state_sigmas = [v_sigma, W_sigma, W_dot_sigma]
    else:
        state_lims   = [W_lim, W_dot_lim]
        state_sigmas = [W_sigma, W_dot_sigma]
    
    if ramp_type in ['linear', 'quadratic']:
        state_lims.append(ramp_lim)
        state_sigmas.append(ramp_sigma)

    # Get cell distances
    dist_file = f'{quadtree_dir}/dist.h5'

    if os.path.exists(dist_file) is not True:

        print('Computing cell distances...')
        
        # Compute distances
        dist = np.empty((n_data, n_data))

        for i in range(n_data):
            for j in range(n_data):
                dist[i, j] = pyffit.covariance.dist(x[i], x[j], y[i], y[j])
        
        # Write to disk
        with h5py.File(dist_file, 'w') as file:                
            file.create_dataset('dist', data=dist)

    # Get bounds on model parameters
    state_lim = pyffit.nif.get_state_constraints(fault, state_lims, ramp_matrix=ramp_matrix)

    # Perform forward Kalman filtering
    x_model_forward, x_model = pyffit.nif.network_inversion_filter(fault, G, d, std, dt, omega, sigma, kappa, state_sigmas, 
                                                                   cov_file=cov_file, steady_slip=steady_slip, ramp_matrix=ramp_matrix, rho=rho, constrain=constrain, 
                                                                   state_lim=state_lim, result_dir=run_dir, cost_function='state', dist_file=dist_file)

    # Compute integrated slip
    if steady_slip:
        s_model_forward = integrate_slip(x_model_forward, dt)
        s_model         = integrate_slip(x_model, dt)
    else:
        s_model_forward = x_model_forward[:n_patch]
        s_model         = x_model[:n_patch]
        
    s_true          = slip_model[:, 0, :].T

    print(f'Forward avg. slip  {np.mean(s_model_forward):.1f} +\- {np.std(s_model_forward):.1f} | range = {s_model_forward.min():.1f} {s_model_forward.max():.1f} | v = {np.mean(x_model_forward[:, :n_patch]):.1e} +/- {np.mean(x_model_forward[:, :n_patch]):.1e} mm/yr')
    print(f'Smoothed avg. slip {np.mean(s_model):.1f} +\- {np.std(s_model):.1f} | range = {s_model.min():.1f} {s_model.max():.1f} | v = {np.mean(x_model[:, :n_patch]):.1e} +/- {np.mean(x_model[:, :n_patch]):.1e} mm/yr')
    
    if np.isnan(slip_model).all():
        print(f'True avg. slip     {np.mean(s_true):.1f} +\- {np.std(s_true):.1f} | range = {s_true.min():.1f} {s_true.max():.1f}')

    end_total = time.time() - start_total
    print(f'Total inversion time: {end_total/60:.1f} min')


    os.remove(f'{run_dir}/results_forward.h5')

    return


def analyze_disp(mesh_file, triangle_file, data_dir, file_format, run_dir, samp_file, site_file, creepmeter_file):
    """
    Analyze network inversion filter results.
    """
    start = time.time()

    result_dir = f'{run_dir}/Results'
    site_dir   = f'{result_dir}/Sites'
    pyffit.utilities.check_dir_tree(site_dir)
    pyffit.utilities.check_dir_tree(f'{result_dir}/Data')
    pyffit.utilities.check_dir_tree(f'{result_dir}/Displacements')

    # Get most recent parameter file from run directory
    files = sorted(glob.glob(f'{run_dir}/Scripts/*/*param*'))
    mode, params = load_parameters(files[-1])

    # Extract parameters
    get_params = True
    if get_params:
        slip_components = params['slip_components']
        poisson_ratio=params['poisson_ratio'] 
        shear_modulus = params['shear_modulus']
        trace_inc = params['trace_inc']
        mu = params['mu']
        eta = params['eta']
        steady_slip = params['steady_slip']
        dataset_name = params['dataset_name']
        ref_point = params['ref_point']
        data_factor = params['data_factor']
        xkey = params['xkey']
        coord_type = params['coord_type']
        date_index_range = params['date_index_range']
        check_lon = params['check_lon']
        reference_time_series = params['reference_time_series']
        use_dates = params['use_dates']
        use_datetime = params['use_datetime']
        # mask_file = params['mask_file']
        mask_file = ''
        # model_file = params['model_file']
        dataset_name = params['dataset_name']
        downsampled_dir = params['downsampled_dir']
        resolution_threshold = params['resolution_threshold']
        width_min = params['width_min']
        width_max = params['width_max']
        max_intersect_width = params['max_intersect_width']
        min_fault_dist = params['min_fault_dist']
        max_iter = params['max_iter']
        smoothing_samp = params['smoothing_samp']
        edge_slip_samp = params['edge_slip_samp']
        disp_components = params['disp_components']
        dt = params['dt']
        vlim_disp = params['vlim_disp']
        avg_strike = params['avg_strike']
        param_file = params['param_file']

    # # Add arrows from top and bottom subplots to the middle
    # add_arrow(fig, ax_top2, ax_middle)
    # add_arrow(fig, ax_bottom1, ax_middle)
    # add_arrow(fig, ax_bottom2, ax_middle)

    # -------------------------- Prepare original data -------------------------- =       
    # Load fault model and regularization matrices
    fault = pyffit.finite_fault.TriFault(mesh_file, triangle_file, slip_components=slip_components, 
                                         poisson_ratio=poisson_ratio, shear_modulus=shear_modulus, 
                                         verbose=False, trace_inc=trace_inc, mu=mu, eta=eta,
                                         avg_strike=avg_strike, ref_point=ref_point)
    
    n_patch = len(fault.triangles)
    slip_start = steady_slip * n_patch  # Get start of transient slip

    if steady_slip:
        n_dim   = 3 * n_patch
    else:
        n_dim   = 2 * n_patch

    # Load data
    dataset = pyffit.data.load_insar_dataset(data_dir, file_format, dataset_name, ref_point, data_factor=data_factor, xkey=xkey, 
                                             coord_type=coord_type, date_index_range=date_index_range, 
                                             check_lon=check_lon, reference_time_series=reference_time_series, 
                                             incremental=False, use_dates=use_dates, use_datetime=use_datetime, 
                                             mask_file=mask_file, remove_mean=False)
    datasets = {dataset_name: dataset}
    n_obs    = len(dataset.date)
    x        = dataset.coords['x'].compute().data.flatten()
    y        = dataset.coords['y'].compute().data.flatten()
    x_grid   = dataset.coords['x'].compute().data
    y_grid   = dataset.coords['y'].compute().data

    # ------------------ Prepare data and model ------------------
    # Get directory for downsampled data
    pyffit.utilities.check_dir_tree(downsampled_dir + '/' + dataset_name)
        
    # Get parameter values to check
    quadtree_dir = get_downsampled_data_directory(downsampled_dir + '/' + dataset_name, param_file)

    # Get dictionary of quadtree parameters
    quadtree_params = dict(
                            resolution_threshold=resolution_threshold,
                            width_min=width_min,
                            width_max=width_max,
                            max_intersect_width=max_intersect_width,
                            min_fault_dist=min_fault_dist,
                            max_iter=max_iter, 
                            poisson_ratio=poisson_ratio,
                            smoothing=smoothing_samp,
                            edge_slip=edge_slip_samp,
                            disp_components=disp_components,
                            slip_components=slip_components,
                            quadtree_dir=quadtree_dir,
                            )

    # Prepare inversion inputs
    inputs = pyffit.inversion.get_inversion_inputs(fault, datasets, quadtree_params, date=-1, quadtree_dir=quadtree_dir, verbose=False)
    n_data = inputs[dataset_name].tree.x.size
    tree = inputs[dataset_name].tree
    dt    /= 365.25

    # Prepare downsampled data
    samp_file = f'cutoff={resolution_threshold:.1e}_wmin={width_min:.2f}_wmax={width_max:.2f}_max_int_w={max_intersect_width:.1f}_min_fdist={min_fault_dist:.2f}_max_it={max_iter:0f}'
    d, std    = pyffit.quadtree.get_downsampled_time_series(datasets, inputs, fault, n_dim, dataset_name=dataset_name, file_name=f'{downsampled_dir}/{dataset_name}/{samp_file}.h5')
    
    # Load slip model
    with h5py.File(f'{run_dir}/results_smoothing.h5', 'r') as file:
        slip_model = file['x_s'][:, slip_start:slip_start + n_patch][()]
        disp_model = file['model'][()]
        resid      = file['resid'][()]
        rms        = file['rms'][()]

    disp = d[:, :n_data]

    # Load sites to extract time series
    sites = load_site_table(creepmeter_file, ref_point, mode='creepmeter')

    # avg_resid       = np.mean(np.abs(resid), axis=0)
    # min_fault_dists = [cell.min_fault_dist for cell in tree.cells]
    # i_near          = [cell.min_fault_dist <= min_fault_dist for cell in tree.cells]
    # i_good = i_near & (avg_resid <= avg_resid.mean() + avg_resid.std()) & (avg_resid >= avg_resid.mean() - avg_resid.std())

    # Get rotated coordinate system
    x_r, y_r = pyffit.utilities.rotate(tree.x, tree.y, np.deg2rad(avg_strike + 90))
    x_r  -= fault.origin_r[0]
    y_r  -= fault.origin_r[1]
    x_r   = -(x_r - fault.trace_r[:, 0].max())
    x_end = -(fault.trace_r[:, 0] - fault.trace_r[:, 0].max())

    # Plot time series
    for img in glob.glob(f'{result_dir}/Displacements/time_series*'):
        os.remove(img)

    # # Get 10 best time series
    # metric = avg_resid                                          # Mean residual
    # metric = np.abs(d[-1, :n_data])                             # Final displacement
    # metric = np.abs(d[-1, :n_data])/avg_resid                   # Final displacement normalized by avg. residual
    # metric     = d[-1, :n_data]/avg_resid                         # Final displacement normalized by avg. residual
    # mode       = 'max'
    # max_dist   = 0.2
    # n_samp     = 10

    # if mode == 'min':
    #     i_sort = np.argsort(metric[i_near])[:n_samp] # get min. values
    # else:
    #     i_sort = np.argsort(metric[i_near])[-n_samp:] # get max. values
    # i_samp = i_near[i_sort]

    # for i in i_samp:
    #     samp_dist = x_r[i]
    #     title     = f'd = {samp_dist:.1f} km | disp. = {d[-1, i]:.1f} mm | resid. = {avg_resid[i]:.1f} mm'
    #     file_name = f'{result_dir}/Displacements/time_series_{samp_dist:.1f}_km.png'

    #     fig, ax = plot_time_series(dataset.date, disp[:, i], disp_model[:, i], title=title, disp_kwargs=dict(c='k' , label='Data'), model_kwargs=dict(c='C3' , label='Model'), file_name=file_name)

    # Get evenly sampled time series
        # Get near-field cells
    fault_samp = np.arange(2, 45, 10)
    nodes      = np.empty((len(fault_samp), 2))
    max_dist   = 0.300
    
    # Get E/W side indices
    near         = [cell.min_fault_dist <= max_dist for cell in tree.cells]
    i_near       = np.where(near)[0]
    offset       = np.empty((n_obs, len(fault_samp)))
    offset_model = np.empty((n_obs, len(fault_samp)))
    sides = np.array([cell.side for cell in tree.cells])
    i_e = np.where((sides > 0) & near)[0]
    i_w = np.where((sides < 0) & near)[0]
    
    for i, samp_dist in enumerate(fault_samp):
        # Get nodes
        j       = np.argmin(np.abs(x_end - samp_dist))
        nodes[i, :] = [x_end[j], fault.trace_r[j, 1]]

        # Get nearby points
        r = np.sqrt((x_r - nodes[i, 0])**2 + (y_r - nodes[i, 1])**2) # distance to node

        # i_sort_e = np.argsort(r[i_e])
        i_samp_e = i_e[r[i_e] <= max_dist]

        # i_sort_w = np.argsort(r[i_w])
        i_samp_w = i_w[r[i_w] <= max_dist]

        # Compute offsets
        offset[:, i]       = disp[:, i_samp_w].mean(axis=1)       - disp[:, i_samp_e].mean(axis=1) 
        offset_model[:, i] = disp_model[:, i_samp_w].mean(axis=1) - disp_model[:, i_samp_e].mean(axis=1) 

        # Plot
        # title     = f'd = {samp_dist:.1f} km | disp. = {d[-1, i]:.1f} mm | resid. = {avg_resid[i]:.1f} mm'
        # title     = f'd = {samp_dist:.1f} km'
        # file_name = f'{result_dir}/Displacements/time_series_{samp_dist:.1f}_km.png'

        # fig, ax = plot_time_series(dataset.date, disp[:, i_samp], disp_model[:, i_samp], title=title, disp_kwargs=dict(c='k' , label='Data'), model_kwargs=dict(c='C3' , label='Model'), file_name=file_name)
        # fig, ax = pyffit.figures.plot_time_series(dataset.date, offset, offset_model, title=title, disp_kwargs=dict(c='k' , label='Data'), model_kwargs=dict(c='C3' , label='Model'), file_name=file_name)


    # Add arrows from top and bottom plots to the middle plot
    def add_arrow(fig, ax_from, ax_to, coords=(0.5, 0.5), arrow_start='bottom', color='k', y_offset=0.03, **arrow_kwargs):

        # Get bounding boxes in figure coordinates
        bbox_from = ax_from.get_position()

        # Start at bottom or top of the reference plot
        if arrow_start == 'bottom':
            x_start = (bbox_from.x0 + bbox_from.x1) / 2
            y_start = bbox_from.y0
        else:
            x_start = (bbox_from.x0 + bbox_from.x1) / 2
            y_start = bbox_from.y1
            y_offset *= -1

        # End in the middle subplot
        bbox_to = ax_to.get_position()
        x_end = bbox_to.x0 + coords[0] * (bbox_to.x1 - bbox_to.x0)
        y_end = bbox_to.y0 + (coords[1] + y_offset) * (bbox_to.y1 - bbox_to.y0)

        # Add arrow
        fig.patches.append(patches.FancyArrowPatch(
                           (x_start, y_start), (x_end, y_end),
                           transform=fig.transFigure,
                           **arrow_kwargs))

    xlim      = [-5, 50]
    ylim      = [-7, 7]
    ylim_disp = [-3, 48]
    vlines    = [datetime.datetime(2017, 9, 8, 0, 0, 0), datetime.datetime(2019, 7, 5, 0, 0, 0)]

    fig  = plt.figure(figsize=(10, 8))
    ncol = 11
    gs   = gridspec.GridSpec(3, ncol, figure=fig, height_ratios=[1, 2, 1], width_ratios=[1 + i - i for i in range(ncol)], hspace=0.1)

    # Top row: 3 subplots
    ax_top1 = fig.add_subplot(gs[0,  :3])
    ax_top2 = fig.add_subplot(gs[0, 4:7])
    ax_top3 = fig.add_subplot(gs[0, 8:11])

    # Middle row: 1 large subplot
    ax_map = fig.add_subplot(gs[1, :])

    # Bottom row: 2 centered subplots
    ax_bottom1 = fig.add_subplot(gs[2, 2:5])
    ax_bottom2 = fig.add_subplot(gs[2, 6:9])
    axes       = [ax_top3, ax_bottom2, ax_top2, ax_bottom1, ax_top1,]

    # time series panels
    for i in range(len(fault_samp)):
        pyffit.figures.plot_time_series(dataset.date, offset[:, i], offset_model[:, i], fig_ax=(fig, axes[i]), ylabel='Offset (mm)',
                                                       disp_kwargs=dict(c='k', label='Data'), model_kwargs=dict(c='C3', label='Model'), close=False)

        # Axis-specific settings
        if i % 2 == 0:
            arrow_start = 'bottom'
            axes[i].xaxis.set_label_position('top')
            axes[i].xaxis.tick_top()
        else:
            arrow_start = 'top'

        # Get dimensionless axis coordinates
        coords = (-(nodes[i, 0] - xlim[1])/(xlim[1] - xlim[0]), (1 + (nodes[i, 1] - ylim[1])/(ylim[1] - ylim[0])))

        # Plot arrows
        add_arrow(fig, axes[i], ax_map, y_offset=0.04, arrow_start=arrow_start, coords=coords, arrowstyle='-|>', facecolor='k', linewidth=1, mutation_scale=20, )

        # Other settings
        axes[i].set_ylim(ylim_disp)
        axes[i].tick_params(axis='x', labelrotation=45) # Rotates x-axis tick labels
        axes[i].vlines(vlines, ylim_disp[0], ylim_disp[1], color='gainsboro', linestyle='--')
        
    # Map
    im = ax_map.scatter(x_r, y_r, c=d[-1, :n_data], cmap=cmc.vik, vmin=-20, vmax=20, s=5)
    ax_map.plot(-(fault.trace_r[:, 0] - fault.trace_r[:, 0].max()), fault.trace_r[:, 1], c='k', linewidth=1)
    ax_map.scatter(nodes[:, 0], nodes[:, 1], c='gold', marker='o', s=50, edgecolors='k', zorder=100)

    # Settings
    ax_map.set_xlabel('X (km)')
    ax_map.set_ylabel('Y (km)')
    ax_map.set_xlim(xlim)
    ax_map.set_ylim(ylim)
    ax_map.set_aspect(1)
    ax_map.invert_xaxis()

    ax_map.set_facecolor('silver')

    cax = fig.add_axes([0.78, 0.43, 0.1, 0.012])  # [left, bottom, width, height] in figure coords
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Displacement (mm)", fontsize=9)
    
    plt.savefig(f'{result_dir}/Displacements/panel_time_series.png', dpi=300)

    return
    # Plot map
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x_r, y_r, c=d[-1, :n_data], cmap=cmc.vik, vmin=-20, vmax=20, s=5)
    # ax.scatter(x_r[i_near], y_r[i_near], c='C4', edgecolors='k')
    ax.plot(-(fault.trace_r[:, 0] - fault.trace_r[:, 0].max()), fault.trace_r[:, 1], c='k', linewidth=1)
    # ax.scatter(x_r[i_samp], y_r[i_samp], c='gold', edgecolors='k', zorder=100)
    ax.scatter(nodes[:, 0], nodes[:, 1], c='gold', edgecolors='k', zorder=100)

    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect(1)
    ax.invert_xaxis()
    ax.set_facecolor('silver')
    plt.show()
    plt.savefig(f'{result_dir}/Displacements/map_r.png', dpi=300)


    return




    fig, ax = plt.subplots(figsize=(14, 8.2))
    ax.scatter(x_r, y_r, c='gray', marker='.', s=10)
    ax.plot(x_end, fault.trace_r[:, 1], c='k', marker='.')
    # ax.scatter(x_r[i_near], y_r[i_near], c='C0', marker='o')
    # ax.plot(node[0], node[1], c='C3', marker='o')


    xlim = [-3, 48]
    ylim = [-10, 10]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_xaxis()
    ax.set_aspect(1)
    plt.show()


    # for i, cell in enumerate(tree.cells):
        # print(i, cell.min_fault_dist)

    return



    # print(d.shape)
    # print(resid.shape)

    # return
    # ------------------ Rotated plots ------------------
    # Plot absolute average error
    int_abs_resid = np.sum(np.abs(resid), axis=0)/(len(dataset.date))
    vlim       = [0, 10]
    label      = 'Average absolute misfit (mm)'
    cmap       = cmc.lajolla
    site_color = 'C0'
    s          = 10
    pyffit.figures.plot_rotated_fault(fault, tree.x, tree.y, int_abs_resid, s=s, vlim=vlim, cmap=cmap, sites=sites, site_color=site_color, label=label, close=True, file_name=f'{site_dir}/avg_misfit_abs.png')

    # Plot true average error
    int_resid  = -np.sum(resid, axis=0)/(len(dataset.date))
    vlim       = [-10, 10]
    label      = 'Average misfit (mm)'
    cmap       = cmc.vik
    site_color = 'gold'
    s          = 10

    pyffit.figures.plot_rotated_fault(fault, tree.x, tree.y, int_resid, s=s,  vlim=vlim, cmap=cmap, sites=sites, site_color=site_color, label=label, close=True, file_name=f'{site_dir}/avg_misfit.png')

    # Plot cell STD
    avg_std  = np.mean(std[:, :len(tree.x)], axis=0)
    vlim       = [0, 10]
    label      = 'Average standard deviation (mm)'
    cmap       = cmc.hawaii
    site_color = 'k'
    s          = 10
    pyffit.figures.plot_rotated_fault(fault, tree.x, tree.y, avg_std,  s=s, vlim=vlim, cmap=cmap, sites=sites, site_color=site_color, label=label, close=True, file_name=f'{site_dir}/avg_std.png')
    
    # Plot cell counts
    for count in range(1, 5):
        pixel_count          = np.zeros_like(tree.x)
        i_count              = tree.real_count == count
        pixel_count[i_count] = 1

        vlim       = [0, 1]
        label      = 'Single pixel cells locations'
        cmap       = cmc.lajolla
        site_color = 'k'
        file_name  = f'{site_dir}/pixel_count_{count}.png'

        fig, ax, im = pyffit.figures.plot_rotated_fault(fault, tree.x, tree.y, pixel_count, vlim=vlim, cmap=cmap, site_color=site_color, label=label, show=False)
        transform = Affine2D().rotate_deg_around(0, 0, fault.avg_strike + 90) + ax.transData
        ax.scatter(tree.x[i_count] - fault.origin_r[0], 
                tree.y[i_count]  - fault.origin_r[1], 
                c=pixel_count[i_count], 
                vmin=vlim[0], vmax=vlim[1], marker='.', cmap=cmap, transform=transform, zorder=100)
        ax.set_title(f'Number of {count}-pixel cells: {sum(i_count)}/{pixel_count.size}')
        fig.savefig(file_name, dpi=300)
        plt.close()
        
    # Plot STDs
    for std in range(10, 51, 10):
        std_count          = np.zeros_like(tree.x)
        i_count            = tree.std > std
        std_count[i_count] = 1

        vlim       = [0, 1]
        label      = f'Cell standard deviation >{std:.0f} mm'
        cmap       = cmc.lajolla
        site_color = 'k'
        file_name  = f'{result_dir}/Data/std_{std}.png'

        fig, ax, im = pyffit.figures.plot_rotated_fault(fault, tree.x, tree.y, std_count, vlim=vlim, cmap=cmap, site_color=site_color, label=label, show=False)
        transform = Affine2D().rotate_deg_around(0, 0, fault.avg_strike + 90) + ax.transData
        ax.scatter(tree.x[i_count] - fault.origin_r[0], 
                tree.y[i_count]  - fault.origin_r[1], 
                c=std_count[i_count], 
                vmin=vlim[0], vmax=vlim[1], marker='.', cmap=cmap, transform=transform, zorder=100)
        ax.set_title(f'Cell standard deviation > {std:.0f} mm {sum(i_count)}/{std_count.size}')
        fig.savefig(file_name, dpi=300)
        plt.close()
    
    # ------------------ Analyze site time series ------------------
    # Extract time series for each site
    site_dist = 0.2 # km
    vlines    = [datetime.datetime(2017, 9, 8, 0, 0, 0), datetime.datetime(2019, 7, 5, 0, 0, 0)]
    ylim      = [-50, 50]

    for name in sites['name']:
        # Get coordinates
        site   = sites[sites['name'] == name]
        x_site = site['x'].values[0]
        y_site = site['y'].values[0]

        # Get pixels within site_dist
        dist_pix  = np.sqrt((x_site - x_grid)**2 + (y_site - y_grid)**2)
        i_x, i_y = np.where(dist_pix <= site_dist)
        n_pixel  = len(i_x)

        # Get pixels used in nearest cell
        dist_cell = np.sqrt((site['x'].values[0] - tree.x)**2 + (site['y'].values[0] - tree.y)**2)
        i_cell    = np.argmin(dist_cell)
        i_cell_E  = np.argmin(dist_cell[tree.side == 1])
        i_cell_W  = np.argmin(dist_cell[tree.side == -1])
        cell      = tree.cells[i_cell]
        cell_E    = tree.cells[tree.side == 1][i_cell_E]
        cell_W    = tree.cells[tree.side == -1][i_cell_W]
        n_cell    = cell.data.x.size
        n_cell_E  = cell_E.data.x.size
        n_cell_W  = cell_W.data.x.size

        # Extract cell timeseries
        cell_data = np.empty((n_obs, n_cell))

        for i in range(n_cell):
            dist_pixel = np.sqrt((cell.data.x[i] - x_grid)**2 + (cell.data.y[i] - y_grid)**2)
            j_x, j_y = np.where(dist_pixel == dist_pixel.min())

            if np.isnan(dataset['z'][0, j_x, j_y]):
                cell_data[:, i] = np.nan
            else:
                cell_data[:, i] = dataset['z'][:, j_x, j_y].compute().data.flatten()
 
         # ---------------------------------- Plot offset time series ----------------------------------
        plot_offsets = True
        if plot_offsets:
            fig = plt.figure(figsize=(9, 9), constrained_layout=True)
            gs = fig.add_gridspec(2, 2, width_ratios=(1, 1), height_ratios=[1, 1],
                                    hspace=0.0, wspace=0.05)

            ax0 = fig.add_subplot(gs[0, :])
            ax1 = fig.add_subplot(gs[1, 0])
            ax2 = fig.add_subplot(gs[1, 1])
            axes = [ax0, ax1, ax2]

            offsets     = d[:, i_cell_E] - d[:, i_cell_W]
            offsets_nif = disp_model[:, i_cell_E] - disp_model[:, i_cell_W]
            axes[2].vlines(vlines, ylim[0], ylim[1], color='gainsboro', linestyle='--')
            axes[2].plot(dataset.date, offsets, c='k', label='Downsampled data')
            axes[2].plot(dataset.date, offsets_nif, c='C3', label='Model')
            axes[2].legend()
            axes[2].set_ylim(ylim)
            axes[2].set_box_aspect(1)
            axes[2].set_xlabel('Date')
            axes[2].set_ylabel('Displacement (mm)')
            axes[2].yaxis.tick_right()
            axes[2].yaxis.set_label_position('right') 
            axes[2].set_title(f'Cell time series')

            # Plot map
            # transform = Affine2D().rotate_deg_around(0, 0, fault.avg_strike + 90) + axes[1].transData
            # axes[1].plot(fault.trace[:, 0], fault.trace[:, 1], c='k', transform=transform)
            # ax.scatter(x_grid[i_x, i_y], y_grid[i_x, i_y], c='gainsboro', s=100)
            # ax.scatter(x_grid[i_x, i_y], y_grid[i_x, i_y], c=dataset['z'][-1, :, :].compute().data[dist_pix <= site_dist], cmap=cmc.vik, vmin=vlim_disp[0][0], vmax=vlim_disp[0][1], s=100)
            # axes[1].scatter(x_site, y_site, facecolor='gold', edgecolor='k', s=200)
            # axes[1].scatter(cell.data.x, cell.data.y, c=cell.data.side, cmap=cmc.vik_r, s=100)
            # axes[1].scatter(cell.origin.x, cell.origin.y, facecolor='C1', edgecolor='k', s=200)
            axes[1].plot(fault.trace[:, 0], fault.trace[:, 1], c='k', zorder=0)
            # axes[1].scatter(tree.x , tree.y , c=d[-1, :n_data], marker='o', cmap=cmc.vik, vmin=vlim_disp[0][0], vmax=vlim_disp[0][1])
            for cell in [cell_E, cell_W]:
                axes[1].scatter(cell.origin.x, cell.origin.y, facecolor='gold', edgecolor='k', s=50, zorder=100)
                axes[1].scatter(cell.data.x , cell.data.y, c='gray',  marker='s', s=100, zorder=0, )
                axes[1].scatter(cell.data.x , cell.data.y, c=cell.data.side * 0.5 * abs(cell.data.values/cell.data.values), cmap=cmc.vik_r, marker='s', s=100, zorder=0, vmin=-1, vmax=1)

            xmin = np.min([cell_E.data.x.min(), cell_W.data.x.min()]) - np.max([cell_E.width, cell_W.width])
            xmax = np.max([cell_E.data.x.max(), cell_W.data.x.max()]) + np.max([cell_E.width, cell_W.width])
            ymin = np.min([cell_E.data.y.min(), cell_W.data.y.min()]) - np.max([cell_E.width, cell_W.width])
            ymax = np.max([cell_E.data.y.max(), cell_W.data.y.max()]) + np.max([cell_E.width, cell_W.width])
            xlim_cells = [xmin, xmax]
            ylim_cells = [ymin, ymax]
   
            axes[1].set_xlim(xlim_cells)
            axes[1].set_ylim(ylim_cells)

            axes[1].set_xlabel('East (km)')
            axes[1].set_ylabel('North (km)')
            axes[1].set_box_aspect(1)
            axes[1].set_title(f'Quadtree cell pixels')

            # Plot map
            transform = Affine2D().rotate_deg_around(0, 0, fault.avg_strike + 90) + axes[0].transData
            axes[0].plot(fault.trace[:, 0] - fault.origin_r[0], fault.trace[:, 1] - fault.origin_r[1], c='k', zorder=0, transform=transform)
            axes[0].scatter(tree.x - fault.origin_r[0], tree.y  - fault.origin_r[1], c=d[-1, :n_data], marker='.', cmap=cmc.vik, vmin=vlim_disp[0][0], vmax=vlim_disp[0][1], transform=transform)
            axes[0].scatter(sites['x'] - fault.origin_r[0], sites['y']  - fault.origin_r[1], facecolor='gold', edgecolor='k', s=50, transform=transform, zorder=100)

            # Add annotation
            xy = np.array([site['x'].values[0] - fault.origin_r[0], site['y'].values[0]  - fault.origin_r[1]])
            axes[0].annotate(site['name'].values[0], xy, xytext=xy + 2,
                            arrowprops=dict(arrowstyle="->",  color='black'),
                            bbox=dict(facecolor='w', boxstyle='round,pad=0.5', alpha=0.85),
                            fontsize=12, fontweight='bold', horizontalalignment='left',
                            xycoords=transform, zorder=1000)
            
            axes[0].set_aspect(1)
            axes[0].set_xlim(-20, 30)
            axes[0].set_ylim(-10, 10)
            axes[0].set_xlabel('X (km)')
            axes[0].set_ylabel('Y (km)')
            axes[0].set_title(f'Site: {name}')

            plt.tight_layout()
            file_name = f'{site_dir}/offsets_{name}.png'
            plt.savefig(file_name, dpi=300)
            plt.close()
            print(file_name)

        # ---------------------------------- Plot cell time series ----------------------------------
        plot_cell = False
        if plot_cell:
            fig = plt.figure(figsize=(9, 9), constrained_layout=True)
            gs = fig.add_gridspec(2, 2, width_ratios=(1, 1), height_ratios=[1, 1],
                                    hspace=0.0, wspace=0.05)

            ax0 = fig.add_subplot(gs[0, :])
            ax1 = fig.add_subplot(gs[1, 0])
            ax2 = fig.add_subplot(gs[1, 1])
            axes = [ax0, ax1, ax2]

            for i in range(n_cell):

                if cell.data.side[i] == cell.side:
                    if cell.side == 1:
                        c = 'C0'
                    else:
                        c = 'C3'
                    axes[2].plot(dataset.date, cell_data[:, i], c='gainsboro')

                else:
                    c = 'gainsboro'

            axes[2].vlines(vlines, ylim[0], ylim[1], color='gainsboro', linestyle='--')
            axes[2].plot(dataset.date, d[:, i_cell], c='k', label='Downsampled data')
            axes[2].plot(dataset.date, disp_model[:, i_cell], c='C3', label='Model')
            axes[2].legend()
            axes[2].set_ylim(ylim)
            axes[2].set_box_aspect(1)
            axes[2].set_xlabel('Date')
            axes[2].set_ylabel('Displacement (mm)')
            axes[2].yaxis.tick_right()
            axes[2].yaxis.set_label_position('right') 
            axes[2].set_title(f'Cell time series')

            # Plot map
            # transform = Affine2D().rotate_deg_around(0, 0, fault.avg_strike + 90) + axes[1].transData
            # axes[1].plot(fault.trace[:, 0], fault.trace[:, 1], c='k', transform=transform)
            # ax.scatter(x_grid[i_x, i_y], y_grid[i_x, i_y], c='gainsboro', s=100)
            # ax.scatter(x_grid[i_x, i_y], y_grid[i_x, i_y], c=dataset['z'][-1, :, :].compute().data[dist_pix <= site_dist], cmap=cmc.vik, vmin=vlim_disp[0][0], vmax=vlim_disp[0][1], s=100)
            # axes[1].scatter(x_site, y_site, facecolor='gold', edgecolor='k', s=200)
            # axes[1].scatter(cell.data.x, cell.data.y, c=cell.data.side, cmap=cmc.vik_r, s=100)
            # axes[1].scatter(cell.origin.x, cell.origin.y, facecolor='C1', edgecolor='k', s=200)
            axes[1].plot(fault.trace[:, 0], fault.trace[:, 1], c='k', zorder=0)
            # axes[1].scatter(tree.x , tree.y , c=d[-1, :n_data], marker='o', cmap=cmc.vik, vmin=vlim_disp[0][0], vmax=vlim_disp[0][1])
            axes[1].scatter(cell.origin.x, cell.origin.y, facecolor='gold', edgecolor='k', s=50, zorder=100)
            axes[1].scatter(cell.data.x , cell.data.y, c='gray',  marker='s', s=100, zorder=0, )
            axes[1].scatter(cell.data.x , cell.data.y, c=cell.data.side * 0.5 * abs(cell.data.values/cell.data.values), cmap=cmc.vik_r, marker='s', s=100, zorder=0, vmin=-1, vmax=1)
            axes[1].set_xlim(cell.origin.x - cell.width, cell.origin.x + cell.width)
            axes[1].set_ylim(cell.origin.y - cell.width, cell.origin.y + cell.width)

            axes[1].set_xlabel('East (km)')
            axes[1].set_ylabel('North (km)')
            axes[1].set_box_aspect(1)
            axes[1].set_title(f'Quadtree cell pixels')

            # Plot map
            transform = Affine2D().rotate_deg_around(0, 0, fault.avg_strike + 90) + axes[0].transData
            axes[0].plot(fault.trace[:, 0] - fault.origin_r[0], fault.trace[:, 1] - fault.origin_r[1], c='k', zorder=0, transform=transform)
            axes[0].scatter(tree.x - fault.origin_r[0], tree.y  - fault.origin_r[1], c=d[-1, :n_data], marker='.', cmap=cmc.vik, vmin=vlim_disp[0][0], vmax=vlim_disp[0][1], transform=transform)
            axes[0].scatter(sites['x'] - fault.origin_r[0], sites['y']  - fault.origin_r[1], facecolor='gold', edgecolor='k', s=50, transform=transform, zorder=100)

            # Add annotation
            xy = np.array([site['x'].values[0] - fault.origin_r[0], site['y'].values[0]  - fault.origin_r[1]])
            axes[0].annotate(site['name'].values[0], xy, xytext=xy + 2,
                            arrowprops=dict(arrowstyle="->",  color='black'),
                            bbox=dict(facecolor='w', boxstyle='round,pad=0.5', alpha=0.85),
                            fontsize=12, fontweight='bold', horizontalalignment='left',
                            xycoords=transform, zorder=1000)
            
            axes[0].set_aspect(1)
            axes[0].set_xlim(-20, 30)
            axes[0].set_ylim(-10, 10)
            axes[0].set_xlabel('X (km)')
            axes[0].set_ylabel('Y (km)')
            axes[0].set_title(f'Site: {name}')

            plt.tight_layout()
            plt.savefig(f'{site_dir}/site_{name}.png', dpi=300)
            plt.close()



    # Load
    # # Load results
    # results_forward = h5py.File(f'{run_dir}/results_forward.h5', 'r')
    # x_model_forward = results_forward['x_a']

    # results_smoothing = h5py.File(f'{run_dir}/results_smoothing.h5', 'r')
    # x_model           = results_smoothing['x_s']
    
    # # Compute integrated slip
    # if steady_slip:
    #     s_model_forward = integrate_slip(results_forward['x_a'], dt)
    #     s_model         = integrate_slip(results_smoothing['x_s'], dt)
    # else:
    #     s_model_forward = x_model_forward[:, :n_patch]
    #     s_model = x_model[:, :n_patch]

    # s_true = slip_model[:, 0, :].T

    print(run_dir)
    return


def analyze_model(mesh_file, triangle_file, file_format, downsampled_dir, out_dir, data_dir, 
                  date_index_range=[-22, -14], xkey='x', coord_type='xy', dataset_name='data',
                  check_lon=False, reference_time_series=True, use_dates=False, use_datetime=False, dt=1, data_factor=1,
                  model_file='', mask_file='', estimate_covariance=False, mask_dists=[3], n_samp=2*10**7, r_inc=0.2, r_max=80, mask_dir='.', cov_dir='.', 
                  m0=[0.9, 1.5, 5], c_max=2, sv_max=2, ramp_type='none', ref_point=[-116, 33.5], avg_strike=315.8, trace_inc=0.01, poisson_ratio=0.25, 
                  shear_modulus=6*10**9, disp_components=[1], slip_components=[0], resolution_threshold=2.3e-1, 
                  width_min=0.1, width_max=10, max_intersect_width=100, min_fault_dist=1, max_iter=10, 
                  smoothing_samp=False, edge_slip_samp=False, omega=1e1, sigma=1e1, kappa=2e1, mu=2e1, 
                  eta=2e1, rho=1e0, steady_slip=False, constrain=False, v_sigma=1e-9, W_sigma=1,  W_dot_sigma=1, ramp_sigma=100, v_lim=(0, 3), W_lim=(0, 30), W_dot_lim=(0, 50), ramp_lim=(-100, 100),
                  xlim=[-35.77475071,26.75029172], ylim=[-26.75029172, 55.08597388], vlim_slip=[0, 20], 
                  vlim_disp=[[-10,10], [-10,10], [-1,1]], cmap_slip=cmc.lajolla, cmap_disp=cmc.vik, figsize=(10, 10), dpi=75, markersize=40, 
                  param_file='params.py'
                  ):
    """
    Analyze network inversion filter results.
    """
    start      = time.time()
    cmds       = []
    run_dir    = f'{out_dir}/omega_{omega:.1e}__kappa_{kappa:.1e}__sigma_{sigma:.1e}'

    if ramp_type == 'saved':
        ramp_file = f'{run_dir}/Ramps/corrected.h5'
        run_dir = f'{run_dir}/Deramped'

        print(ramp_file)

    result_dir = f'{run_dir}/Results'
    ramp_dir   = f'{run_dir}/Ramps/'

    # Get directory for downsampled data
    pyffit.utilities.check_dir_tree(downsampled_dir + '/' + dataset_name)
    pyffit.utilities.check_dir_tree(result_dir + '/Animations')
    pyffit.utilities.check_dir_tree(ramp_dir)
        
    # Get parameter values to check
    quadtree_dir = get_downsampled_data_directory(downsampled_dir + '/' + dataset_name, param_file)

    # -------------------------- Prepare original data --------------------------    
    # Load fault model and regularization matrices
    fault = pyffit.finite_fault.TriFault(mesh_file, triangle_file, slip_components=slip_components, 
                                         poisson_ratio=poisson_ratio, shear_modulus=shear_modulus, 
                                         verbose=False, trace_inc=trace_inc, mu=mu, eta=eta,
                                         avg_strike=avg_strike, ref_point=ref_point)
    n_patch = len(fault.triangles)

    if steady_slip:
        n_dim   = 3 * n_patch
    else:
        n_dim   = 2 * n_patch

    # Load data
    dataset = pyffit.data.load_insar_dataset(data_dir, file_format, dataset_name, ref_point, data_factor=data_factor, xkey=xkey, 
                                             coord_type=coord_type, date_index_range=date_index_range, 
                                             check_lon=check_lon, reference_time_series=reference_time_series, 
                                             incremental=False, use_dates=use_dates, use_datetime=use_datetime, 
                                             mask_file=mask_file, remove_mean=False)
    datasets = {dataset_name: dataset}
    n_obs    = len(dataset.date)
    x        = dataset.coords['x'].compute().data.flatten()
    y        = dataset.coords['y'].compute().data.flatten()

    # Load original slip model
    if len(model_file) > 0:
        try:
            with h5py.File(model_file, 'r') as file:
                slip_model = file['slip_model'][()]
        except KeyError:
            print('Error: speficied model_file could not be located')
            sys.exit(1)
    else:
        slip_model = np.full([n_patch, 3, n_obs], np.nan)

    # ------------------ Prepare inversion inputs ------------------
    # Get dictionary of quadtree parameters
    quadtree_params = dict(
                            resolution_threshold=resolution_threshold,
                            width_min=width_min,
                            width_max=width_max,
                            max_intersect_width=max_intersect_width,
                            min_fault_dist=min_fault_dist,
                            max_iter=max_iter, 
                            poisson_ratio=poisson_ratio,
                            smoothing=smoothing_samp,
                            edge_slip=edge_slip_samp,
                            disp_components=disp_components,
                            slip_components=slip_components,
                            quadtree_dir=quadtree_dir,
                            )

    # Prepare inversion inputs
    inputs = pyffit.inversion.get_inversion_inputs(fault, datasets, quadtree_params, date=-1, quadtree_dir=quadtree_dir, verbose=False)
    tree   = inputs[dataset_name].tree
    n_data = inputs[dataset_name].tree.x.size
    dt    /= 365.25

    # Get ramp matrix, if specified
    if ramp_type in ['linear', 'quadratic']:
        ramp_matrix = pyffit.corrections.get_ramp_matrix(tree.x, tree.y, ramp_type=ramp_type)
    else:
        ramp_matrix = []

    # -------------------------- Prepare NIF objects --------------------------
    print(f'Number of fault elements: {n_patch}')
    print(f'Number of data points:    {n_data}')

    # Prepare data
    samp_file = f'cutoff={resolution_threshold:.1e}_wmin={width_min:.2f}_wmax={width_max:.2f}_max_int_w={max_intersect_width:.1f}_min_fdist={min_fault_dist:.2f}_max_it={max_iter:0f}'
    d, std = pyffit.quadtree.get_downsampled_time_series(datasets, inputs, fault, n_dim, dataset_name=dataset_name, file_name=f'{downsampled_dir}/{dataset_name}/{samp_file}.h5')

    if ramp_type == 'saved':
        with h5py.File(ramp_file, 'r') as f:
            d[:, :n_data] -= f['ramp'][()]

    # -------------------------- Analyze results --------------------------
    # Load results
    results_forward = h5py.File(f'{run_dir}/likelihood.h5', 'r')
    x_model_forward = results_forward['x_a']

    results_smoothing = h5py.File(f'{run_dir}/results_smoothing.h5', 'r')
    x_model           = results_smoothing['x_s']

    # Fit ramps to residuals
    resid     = results_smoothing['resid']
    model     = results_smoothing['model']
    ramp      = np.empty_like(resid)
    corrected = np.copy(d)

    for k in range(n_obs):
        date = dataset.date[k].dt.strftime('%Y-%m-%d').item()

        print(f'Fitting ramp for {date} ({k + 1}/{n_obs})')

        # ramp[k, :] = pyffit.corrections.fit_ramp(tree.x, tree.y, resid[k, :], ramp_type='linear', region=[-360, 360, -90, 90], grid=True)
        ramp[k, :] = np.ones(n_data) * np.mean(resid[k, :])
        
        corrected[k, :n_data] = d[k, :n_data] - ramp[k, :]


        plot = False
        if plot:
            vlim = np.std(np.abs(d[k, :])) * 2

            # Plot
            panels  = [[d[k, :n_data], model[k, :], resid[k, :]], 
                    [ramp[k, :], resid[k, :] - ramp[k, :], corrected[k, :n_data]]]
            
            labels = [['Data', 'Model', 'Residual'], 
                    ['Ramp', 'Deramped residual', 'Deramped data']]
                    
            fig, axes = plt.subplots(len(panels), len(panels[0]), figsize=(10, 8))
            fig.suptitle(date)

            for i in range(len(axes[:, 0])):
                for j in range(len(axes[0, :])):
                    axes[i, j].set_title(labels[i][j])
                    axes[i, j].scatter(tree.x, tree.y, c=panels[i][j], marker='.', s=5, cmap=cmc.vik, vmin=-vlim, vmax=vlim)

                    axes[i, j].set_xlabel('East (km)')
                    axes[i, j].set_facecolor('darkgray')
                    axes[i, j].set_aspect(1)

                    if i == 0:
                        axes[i, j].set_ylabel('North (km)')
                    if i > 0:
                        axes[i, j].set_yticklabels([])

            plt.tight_layout()
            plt.savefig(f'{ramp_dir}/{k}.png', dpi=150)

    with h5py.File(f'{ramp_dir}/corrected.h5', 'w') as f:
        f.create_dataset('ramp', data=ramp)
        f.create_dataset('corrected', data=corrected)

    # Compute integrated slip
    if steady_slip:
        s_model_forward = integrate_slip(results_forward['x_a'], dt)
        s_model         = integrate_slip(results_smoothing['x_s'], dt)
    else:
        s_model_forward = x_model_forward[:, :n_patch]
        s_model         = x_model[:, :n_patch]

    s_true = slip_model[:, 0, :].T

    # Get along-strike projected coordinates
    mesh_r         = np.copy(fault.mesh)
    x_mesh, y_mesh = pyffit.utilities.rotate(fault.mesh[:, 0], fault.mesh[:, 1], np.deg2rad(avg_strike + 90))
    mesh_r[:, 0]   = x_mesh
    mesh_r[:, 1]   = y_mesh
    
    # Get centroid values
    x_center = np.array([mesh_r[tri][:, 0].mean() for tri in fault.triangles])
    z_center = np.array([mesh_r[tri][:, 2].mean() for tri in fault.triangles])
    x_center = -(x_center - x_center.max())

    # Interpolate
    n_x = int(x_center.max())*20
    n_z = int(abs(z_center).max())*20
    dt  = convert_timedelta((dataset['date'][1] - dataset['date'][0]).values)

    x_rng = np.linspace(x_center.min(), x_center.max(), n_x)
    z_rng = np.linspace(z_center.min(), z_center.max(), n_z)
    x_grid, z_grid = np.meshgrid(x_rng, z_rng)

    slip_history      = np.empty((s_model.shape[0], n_z, n_x))
    slip_history_filt = np.empty((s_model.shape[0], n_z, n_x))
    slip_history_true = np.empty((s_model.shape[0], n_z, n_x))

    # Interpolate slip history to regular grid
    for k in range(len(s_model)):
        slip_history[k, :, :]      = griddata((x_center, z_center), s_model[k, :], (x_grid, z_grid), method='cubic')
        slip_history_filt[k, :, :] = griddata((x_center, z_center), s_model_forward[k, :], (x_grid, z_grid), method='cubic')
        slip_history_true[k, :, :] = griddata((x_center, z_center), s_true[k, :], (x_grid, z_grid), method='cubic')

    slip_rate_history      = get_slip_rate_history(slip_history, dt=dt)
    slip_rate_history_filt = get_slip_rate_history(slip_history_filt, dt=dt)
    slip_rate_history_true = get_slip_rate_history(slip_history_true, dt=dt)

    hlines = [datetime.datetime(2017, 9, 8, 0, 0, 0), datetime.datetime(2019, 7, 5, 0, 0, 0)]

    # Toggle plot modes
    summary_plots        = True
    validation           = False
    regular_fault_panels = True
    rotated_fault_panels = True
    fault_3d             = True
    animation            = True

    # -------------------- Summary figures --------------------
    if summary_plots:
        dpi    = 300
        # -------------------- Avg. slip rate --------------------       
        vmin   = 0
        vmax   = 10
        label  = 'Slip rate (mm/yr)'
        title  = 'Depth averaged slip rate (smoothed)'
        file_name = f'{result_dir}/History_rate_smooth.png'
        plot_slip_history(x_rng, dataset['date'], np.mean(slip_rate_history, axis=1), cmap=cmap_slip, vmin=vmin, vmax=vmax, title=title, label=label, hlines=hlines, file_name=file_name)
        title  = 'Depth averaged slip rate (filtered)'
        file_name = f'{result_dir}/History_rate_filt.png'
        plot_slip_history(x_rng, dataset['date'], np.mean(slip_rate_history_filt, axis=1), cmap=cmap_slip, vmin=vmin, vmax=vmax, title=title, label=label, hlines=hlines, file_name=file_name)
        vmin   = -5
        vmax   = 5
        title  = 'Depth averaged slip rate (smoothed - filtered)'
        file_name = f'{result_dir}/History_rate_smoothed-filt.png'
        plot_slip_history(x_rng, dataset['date'], np.mean(slip_rate_history, axis=1) - np.mean(slip_rate_history_filt, axis=1), cmap=cmc.vik, vmin=vmin, vmax=vmax, title=title, label=label, hlines=hlines, file_name=file_name)

        # -------------------- Avg. slip  --------------------
        vmin   = 0
        vmax   = 15
        label  = 'Slip rate (mm/yr)'
        title  = 'Surface slip rate (smoothed)'
        file_name = f'{result_dir}/History_surface_rate_smooth.png'
        plot_slip_history(x_rng, dataset['date'], slip_rate_history[:, -1, :], cmap=cmap_slip, vmin=vmin, vmax=vmax, title=title, label=label, hlines=hlines, file_name=file_name)
        title  = 'Surface slip rate (filtered)'
        file_name = f'{result_dir}/History_surface_rate_filt.png'
        plot_slip_history(x_rng, dataset['date'], slip_rate_history_filt[:, -1, :], cmap=cmap_slip, vmin=vmin, vmax=vmax, title=title, label=label, hlines=hlines, file_name=file_name)
        vmin   = -5
        vmax   = 5
        title  = 'Surface slip rate (smoothed - filtered)'
        file_name = f'{result_dir}/History_surface_rate_smoothed-filt.png'
        plot_slip_history(x_rng, dataset['date'], slip_rate_history[:, -1, :] - slip_rate_history_filt[:, -1, :], cmap=cmc.vik, vmin=vmin, vmax=vmax, title=title, label=label, hlines=hlines, file_name=file_name)


        # -------------------- Surface slip rate --------------------       
        vmin   = 0
        vmax   = 30
        label  = 'Slip (mm)'

        title  = 'Depth averaged slip (smoothed)'
        file_name = f'{result_dir}/History_slip_smooth.png'
        plot_slip_history(x_rng, dataset['date'], np.mean(slip_history, axis=1), cmap=cmap_slip, vmin=vmin, vmax=vmax, title=title, label=label, hlines=hlines, file_name=file_name)
        title  = 'Depth averaged slip (filtered)'
        file_name = f'{result_dir}/History_slip_filt.png'
        plot_slip_history(x_rng, dataset['date'], np.mean(slip_history_filt, axis=1), cmap=cmap_slip, vmin=vmin, vmax=vmax, title=title, label=label, hlines=hlines, file_name=file_name)
        vmin   = -5
        vmax   =  5
        title  = 'Depth averaged slip (smoothed - filtered)'
        file_name = f'{result_dir}/History_slip_smoothed-filt.png'
        plot_slip_history(x_rng, dataset['date'], np.mean(slip_history, axis=1) - np.mean(slip_history_filt, axis=1), cmap=cmc.vik, vmin=vmin, vmax=vmax, title=title, label=label, hlines=hlines, file_name=file_name)

        # -------------------- Surface slip --------------------       
        vmin   = 0
        vmax   = 40
        label  = 'Slip (mm)'

        title  = 'Surface slip (smoothed)'
        file_name = f'{result_dir}/History_surface_slip_smooth.png'
        plot_slip_history(x_rng, dataset['date'], slip_history[:, -1, :], cmap=cmap_slip, vmin=vmin, vmax=vmax, title=title, label=label, hlines=hlines, file_name=file_name)
        title  = 'Surface slip (filtered)'
        file_name = f'{result_dir}/History_surface_slip_filt.png'
        plot_slip_history(x_rng, dataset['date'], slip_history_filt[:, -1, :], cmap=cmap_slip, vmin=vmin, vmax=vmax, title=title, label=label, hlines=hlines, file_name=file_name)
        vmin   = -5
        vmax   =  5
        title  = 'Surface slip (smoothed - filtered)'
        file_name = f'{result_dir}/History_surface_slip_smoothed-filt.png'
        plot_slip_history(x_rng, dataset['date'], slip_history[:, -1, :] - slip_history_filt[:, -1, :], cmap=cmc.vik, vmin=vmin, vmax=vmax, title=title, label=label, hlines=hlines, file_name=file_name)

        # -------------------- Slip s --------------------
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlabel('Time')
        ax.set_ylabel('Slip (mm)')
        ax.set_ylim(0, np.max(s_model))

        for i in range(n_patch):
            ax.plot(dataset.date, s_true[:, i], c='k', alpha=0.3, zorder=0)
        plt.savefig(f'{result_dir}/Evolution_slip_0.png', dpi=300)

        for i in range(n_patch):
            ax.plot(dataset.date, s_model_forward[:, i], c='C0', alpha=0.2, )
        plt.savefig(f'{result_dir}/Evolution_slip_1.png', dpi=300)
        
        for i in range(n_patch):
            ax.plot(dataset.date, s_model[:, i], c='C3', alpha=0.3, )
        # ax.legend()
        plt.savefig(f'{result_dir}/Evolution_slip_2.png', dpi=300)
        plt.close()

        # -------------------- Slip s by depth --------------------
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlabel('Time')
        ax.set_ylabel('Slip (mm)')
        ax.set_ylim(-0.2, np.max(s_model))

        vmin      = -3
        vmax      = 0
        n_tick    = 6
        cmap_name = cmc.roma
        n_seg     = 10

        cvar  = np.min(fault.patches[:, :, 2], axis=1)
        cval  = (cvar - vmin)/(vmax - vmin) # Normalized color values
        ticks = np.linspace(vmin, vmax, n_tick)

        cmap  = colors.LinearSegmentedColormap.from_list(cmap_name, plt.get_cmap(cmap_name, 265)(np.linspace(0, 1, 265)), n_seg)
        sm    = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        c     = cmap(cval)

        for i in range(n_patch):
            ax.plot(dataset.date, s_model[:, i], c=c[i], alpha=0.3, zorder=int(abs(cval[i])))

        # ax.legend()
        plt.colorbar(sm, ax=plt.gca(), label='Depth (km)')
        plt.savefig(f'{result_dir}/Evolution_slip_depth.png', dpi=300)
        plt.close()

        # -------------------- Residuals --------------------
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlabel('Time')
        ax.set_ylabel('Residual (mm)')
        # ax.set_xlim(0, 200)
        ax.plot(dataset.date, np.mean(np.abs(results_forward['resid']), axis=1), c='C0', alpha=1.0, zorder=0, label='Filtered')
        ax.plot(dataset.date, np.mean(np.abs(results_smoothing['resid']), axis=1), c='C3', alpha=1.0, zorder=0, label='Smoothed')
        ax.legend()
        plt.savefig(f'{result_dir}/Evolution_residuals.png', dpi=300)
        plt.close()

        # -------------------- RMS --------------------
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlabel('Time')
        ax.set_ylabel('RMS (mm)')
        # ax.set_xlim(0, 200)
        ax.plot(dataset.date, results_forward['rms'][()], c='C0', alpha=1, zorder=0, label='Filtered')
        ax.plot(dataset.date, results_smoothing['rms'][()], c='C3', alpha=1, zorder=0, label='Smoothed')
        ax.legend()
        plt.savefig(f'{result_dir}/Evolution_rms.png', dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlabel('Time')
        ax.set_ylabel('Change in RMS (mm)')
        # ax.set_xlim(0, 200)
        ax.plot(dataset.date, results_forward['rms'][()] - results_smoothing['rms'][()], c='k', alpha=1, zorder=0, label='Filtered - smoothed')
        ax.legend()
        plt.savefig(f'{result_dir}/Evolution_rms_diff.png', dpi=300)
        plt.close()

        # -------------------- Slip rate v --------------------
        if steady_slip:
            fig, ax = plt.subplots(figsize=(6, 4))

            for i in range(0, n_patch):
                ax.plot(dataset.date, results_forward['x_a'][:, i], c='C0', alpha=0.2,)
                ax.plot(dataset.date, results_smoothing['x_s'][:, i], c='C3', alpha=0.2,)

            ax.set_xlabel('Time')
            ax.set_ylabel('Slip rate (mm/yr)')
            ax.legend()
            plt.savefig(f'{result_dir}/Evolution_v.png', dpi=300)
            plt.close()

        # -------------------- Transient slip W --------------------
        fig, ax = plt.subplots(figsize=(6, 4))
        for i in range(steady_slip * n_patch, (1 + steady_slip) * n_patch):
            ax.plot(dataset.date, results_forward['x_a'][:, i], c='C0', alpha=0.2, )
            ax.plot(dataset.date, results_smoothing['x_s'][:, i], c='C3', alpha=0.2, )
            ax.plot(dataset.date, s_true[:, i - n_patch], c='k', alpha=0.2, )

            
        ax.set_xlabel('Time')
        ax.set_ylabel('Transient slip (mm)')
        plt.savefig(f'{result_dir}/Evolution_W.png', dpi=300)
        plt.close()

        # -------------------- Transient slip rate W_dot --------------------
        fig, ax = plt.subplots(figsize=(6, 4))
        
        print('Transient slip rate:')
        for i in range((steady_slip + 1) * n_patch, (steady_slip + 2) * n_patch):
            ax.plot(dataset.date, results_forward['x_a'][:, i], c='C0', alpha=0.2)
            ax.plot(dataset.date, results_smoothing['x_s'][:, i], c='C3', alpha=0.2)

        ax.set_xlabel('Time')
        ax.set_ylabel('Transient slip rate (mm/yr)')
        plt.savefig(f'{result_dir}/Evolution_W_dot.png', dpi=300)
        plt.close()
    
   
   # -------------------- Validation --------------------
    if validation:
        
        s_resid_forward = s_true - s_model_forward 
        s_resid = s_true - s_model 

        # Compute RMS with respect to true slip model
        s_rms_forward   = np.mean(s_resid_forward**2, axis=1)**0.5
        s_rms           = np.mean(s_resid**2, axis=1)**0.5
        s_rms_forward_avg = np.mean(s_resid_forward**2)**0.5
        s_rms_avg       = np.mean(s_resid**2)**0.5

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlabel('Time')
        ax.set_ylabel('Slip RMS (mm)')
        # ax.set_xlim(0, 200)
        ax.plot(dataset.date, s_rms_forward, c='C0', alpha=1, zorder=0, label=f'Filtered (avg. = {s_rms_forward_avg:.1f} mm)')
        ax.plot(dataset.date, s_rms, c='C3', alpha=1, zorder=0, label=f'Smoothed (avg. = {s_rms_avg:.1f} mm)')
        ax.legend()
        plt.savefig(f'{result_dir}/Evolution_rms_slip.png', dpi=300)
        plt.close()
        
        # -------------------- Plot history --------------------
        dpi    = 300
        cmap_resid = cmc.vik
        # hlines = [datetime.datetime(2017, 9, 8, 0, 0, 0), datetime.datetime(2019, 7, 5, 0, 0, 0)]
        # hlines = [datetime.datetime(2015, 9, 13, 0, 0, 0), datetime.datetime(2019, 7, 5, 0, 0, 0)]
        # hlines = []

        vmin   = -10
        vmax   = 10
        title  = 'Residual depth averaged slip rate'
        label  = 'Slip rate (mm/yr)'
        file_name = f'{result_dir}/History_residual_slip_rate.png'

        plot_slip_history(x_rng, dataset['date'], np.mean(slip_rate_history_true - slip_rate_history, axis=1), cmap=cmap_resid, vmin=vmin, vmax=vmax, title=title, label=label, hlines=hlines, file_name=file_name)

        vmin   = -10
        vmax   = 10
        title  = 'Residual surface slip rate'
        label  = 'Slip rate (mm/yr)'
        file_name = f'{result_dir}/History_residual_surface_slip_rate.png'

        plot_slip_history(x_rng, dataset['date'], slip_rate_history_true[:, -1, :] - slip_rate_history[:, -1, :], cmap=cmap_resid, vmin=vmin, vmax=vmax, title=title, label=label, hlines=hlines, file_name=file_name)

        vmin   = -10
        vmax   = 10
        title  = 'Residual depth averaged slip'
        label  = 'Slip (mm)'
        file_name = f'{result_dir}/History_residual_slip.png'
        plot_slip_history(x_rng, dataset['date'], np.mean(slip_history_true - slip_history, axis=1), cmap=cmap_resid, vmin=vmin, vmax=vmax, title=title, label=label, hlines=hlines, file_name=file_name)

        vmin   = -10
        vmax   = 10
        title  = 'Residual surface slip'
        label  = 'Slip (mm)'
        file_name = f'{result_dir}/History_residual_surface_slip.png'
        plot_slip_history(x_rng, dataset['date'], slip_history_true[:, -1, :] - slip_history[:, -1, :], cmap=cmap_resid, vmin=vmin, vmax=vmax, title=title, label=label, hlines=hlines, file_name=file_name)
        

   # -------------------- Fault panels --------------------
    if regular_fault_panels:
        reg_dir = f'{result_dir}/Results/regular'
        pyffit.utilities.check_dir_tree(reg_dir)
        
        # -------------------- Plot fault and model fit --------------------
        params      = []
        date        = datasets[dataset_name].date[-1]
        title       = f'Date: {date}'
        orientation = 'horizontal'
        fault_lim   = 'mesh'
        show        = False
        figsize     = (10.5, 8)
        markersize  = 5

        # for run, results, s in zip(['Smoothed'], [results_forward, results_smoothing], [s_model_forward, s_model]):
        for run, results, s in zip(['Smoothed'], [results_smoothing], [s_model]):
            for k in range(len(x_model)):
                
                if use_datetime:
                    date = datasets[dataset_name].date[k].dt.strftime('%Y-%m-%d').item()
                else:
                    date = datasets[dataset_name].date[k]

                slip_resid = s[k, :] - s_true[k, :]

                title       = f'[{run}] Date: {date}'
                file_name   = f'{reg_dir}/{run}_{date}.png'

                data_panels = [
                            dict(x=inputs[dataset_name].tree.x, y=inputs[dataset_name].tree.y, data=d[k, :n_data],          label=dataset_name),
                            dict(x=inputs[dataset_name].tree.x, y=inputs[dataset_name].tree.y, data=results['model'][k, :], label='Model'),
                            dict(x=inputs[dataset_name].tree.x, y=inputs[dataset_name].tree.y, data=results['resid'][k, :], label=f"Residuals ({np.abs(results['resid'][k, :]).mean():.2f}"+ r'$\pm$' + f"{np.abs(results['resid'][k, :]).std():.2f})"),
                            ]
                
                fault_panels = [
                            dict(slip=s[k, :], cmap=cmap_slip, vlim=vlim_slip, title='Slip model', label='Slip (mm)'),
                            # dict(slip=slip_resid, cmap=cmc.vik,   vlim=[-1, 1], title=f'Residuals ({np.abs(slip_resid).mean():.2f}'+ r'$\pm$' + f'{np.abs(slip_resid).std():.2f})',  label='Residuals (mm)'),
                            ]
                
                params.append([data_panels, fault_panels, fault, figsize, title, markersize, orientation, fault_lim, vlim_slip, cmap_disp, vlim_disp, xlim, ylim, dpi, show, file_name])

                if k == len(x_model):
                    print(np.mean(np.abs(d[k, :n_data])), np.std(np.abs(d[k, :n_data])))
        # cmds.append(f'magick -loop 0 -delay 0 {result_dir}/Results/Forward_20*png {result_dir}/Animations/animation_forward.gif')
        # cmds.append(f'magick -loop 0 -delay 0 {result_dir}/Results/Smoothed_20*png {result_dir}/Animations/animation_smoothed.gif')

        # Parallel
        os.environ["OMP_NUM_THREADS"] = "1"
        start       = time.time()
        n_processes = multiprocessing.cpu_count()
        pool        = multiprocessing.Pool(processes=n_processes - 1)
        results     = pool.map(fault_panel_wrapper, params)
        pool.close()
        pool.join()
        del pool
        del results
        gc.collect()
    

    # -------------------- Plot rotated fault and model fit --------------------
    if rotated_fault_panels:
        start_panels = time.time()
        rot_dir = f'{result_dir}/Results/rotated'
        resid_dir = f'{result_dir}/Results/residuals'
        pyffit.utilities.check_dir_tree(rot_dir)
        pyffit.utilities.check_dir_tree(resid_dir)

        print('Plotting fault panels...')
        # Rotate data so fault is horizontal
        fault_r        = copy.deepcopy(fault)
        x_r, y_r       = pyffit.utilities.rotate(inputs[dataset_name].tree.x, inputs[dataset_name].tree.y, np.deg2rad(avg_strike + 90))
        x_mesh, y_mesh = pyffit.utilities.rotate(fault.mesh[:, 0], fault.mesh[:, 1], np.deg2rad(avg_strike + 90))
        x_trace, y_trace = pyffit.utilities.rotate(fault.trace[:, 0], fault.trace[:, 1], np.deg2rad(avg_strike + 90))
        x_r           -= np.mean(x_trace)
        y_r           -= np.mean(y_trace)
        x_mesh        -= np.mean(x_trace)
        y_mesh        -= np.mean(y_trace)
        x_trace       -= np.mean(x_trace)
        y_trace       -= np.mean(y_trace)
        fault_r.mesh[:, 0] = x_mesh
        fault_r.mesh[:, 1] = y_mesh
        fault_r.trace[:, 0] = x_trace
        fault_r.trace[:, 1] = y_trace
    
        params      = []
        date        = datasets[dataset_name].date[-1]
        title       = f'Date: {date}'
        orientation = 'vertical'
        fault_lim   = 'map'
        show        = False
        xlim        = [-25, 25]
        ylim        = [-5, 5] 
        markersize  = 5
        
        # for run, results, s in zip(['Forward', 'Smoothed'], [results_forward, results_smoothing], [s_model_forward, s_model]):
        for run, results, s in zip(['Smoothed'], [results_smoothing], [s_model]):
            for k in range(len(x_model)):
                # Get ramp
                if run == 'Forward':
                    key = 'x_a'
                else:
                    key = 'x_s'

                # Get names
                if use_datetime:
                    date = datasets[dataset_name].date[k].dt.strftime('%Y-%m-%d').item()
                else:
                    date = datasets[dataset_name].date[k]
                title       = f'[{run}] Date: {date}'
                file_name   = f'{rot_dir}/Rotated_{run}_{date}.png'

                # Set up input objects
                if len(ramp_matrix) > 0:
                    ramp = ramp_matrix @ results[key][k, -3:]
                    data_panels = [
                        dict(x=x_r, y=y_r, data=d[k, :n_data],          label=dataset_name),
                        dict(x=x_r, y=y_r, data=results['model'][k, :], label=f'Model'),
                        dict(x=x_r, y=y_r, data=ramp,                   label=f'Ramp ({ramp.min():.2f} - {ramp.max():.2f})'),
                        dict(x=x_r, y=y_r, data=results['resid'][k, :], label=f"Residuals ({np.abs(results['resid'][k, :]).mean():.2f}"+ r'$\pm$' + f"{np.abs(results['resid'][k, :]).std():.2f})"),
                        ]
                    
                else:
                    data_panels = [
                        dict(x=x_r, y=y_r, data=d[k, :n_data],          label=dataset_name),
                        dict(x=x_r, y=y_r, data=results['model'][k, :], label=f'Model'),
                        dict(x=x_r, y=y_r, data=results['resid'][k, :], label=f"Residuals ({np.abs(results['resid'][k, :]).mean():.2f}"+ r'$\pm$' + f"{np.abs(results['resid'][k, :]).std():.2f})"),
                        ]
                    figsize = (8, 6)

                fault_panels = [
                            dict(slip=s[k, :], cmap=cmap_slip, vlim=vlim_slip, title='Slip model', label='Slip (mm)'),
                            # dict(slip=slip_resid, cmap=cmc.vik,   vlim=[-1, 1], title=f'Residuals ({np.abs(slip_resid).mean():.2f}'+ r'$\pm$' + f'{np.abs(slip_resid).std():.2f})',  label='Residuals (mm)'),
                            ]
                
                params.append([data_panels, fault_panels, fault_r, figsize, title, markersize, orientation, fault_lim, vlim_slip, cmap_disp, vlim_disp, xlim, ylim, dpi, show, file_name])

                if len(model_file) > 0:
                    resid = s_true[k, :] - s[k, :]
                    # Add slip model residual
                    fault_panels = [
                                dict(slip=resid, cmap=cmc.vik, vlim=[-10, 10], title=f'Slip model residual ({np.abs(resid).mean():.2f}'+ r'$\pm$' + f'{np.abs(resid).std():.2f})', label='Residual slip (mm)'),
                                # dict(slip=slip_resid, cmap=cmc.vik,   vlim=[-1, 1], title=f'Residuals ({np.abs(slip_resid).mean():.2f}'+ r'$\pm$' + f'{np.abs(slip_resid).std():.2f})',  label='Residuals (mm)'),
                                ] 
                    file_name   = f'{resid_dir}/Rotated_{run}_residual_{date}.png'
                    
                    params.append([data_panels, fault_panels, fault_r, figsize, title, markersize, orientation, fault_lim, vlim_slip, cmap_disp, vlim_disp, xlim, ylim, dpi, show, file_name])
                    
                if k == len(x_model):
                    print(np.mean(np.abs(d[k, :n_data])), np.std(np.abs(d[k, :n_data])))

        results_smoothing.close()
        # results_forward.close()

        # fault_panel_wrapper([data_panels, fault_panels, fault_r, figsize, title, markersize, orientation, fault_lim, vlim_slip, cmap_disp, vlim_disp, xlim, ylim, dpi, True, file_name])
        # sys.exit()
        # Parallel
        os.environ["OMP_NUM_THREADS"] = "1"
        n_processes = multiprocessing.cpu_count()
        pool        = multiprocessing.Pool(processes=n_processes - 1)
        results     = pool.map(fault_panel_wrapper, params)
        pool.close()
        pool.join()
        del pool
        del results
        gc.collect()

        end_panels = time.time() - start_panels
        print(f'Fault panels complete ({end_panels:.1f} s)')


    # -------------------- 3D fault --------------------
    if fault_3d:
        params      = []
        edges       = True
        cbar_label  = 'Dextral slip (mm)'
        labelpad    = 10
        azim        = 235
        elev        = 17 
        n_seg       = 10 
        n_tick      = 6
        alpha       = 1
        show        = False
        figsize     = (6, 3)
        cbar_kwargs = dict(location='right', pad=0.05, shrink=0.4)

        for k in range(len(x_model)):
            # Plot smoothed modeled slip 
            date      = datasets[dataset_name].date[k].dt.strftime('%Y-%m-%d').item()
            # title     = f'{date}: Mean = {s_model[k, :].mean():.2f} | range = {s_model[k, :].min():.2f}-{s_model[k, :].max():.2f}'
            title     = f"{datasets[dataset_name].date[0].dt.strftime('%Y-%m-%d').item()} to {datasets[dataset_name].date[k].dt.strftime('%Y-%m-%d').item()}"
            file_name = f'{result_dir}/Slip/slip_smoothing_{date}.png'
            c         = s_model[k, :]
            params.append([fault.mesh, fault.triangles, s_model[k, :], edges, cmap_slip, cbar_label, vlim_slip, labelpad, azim, elev, n_seg, n_tick, alpha, title, show, figsize, cbar_kwargs, file_name])

        for k in range(len(x_model) - 1):
            # Plot smoothed modeled slip 
            date      = datasets[dataset_name].date[k].dt.strftime('%Y-%m-%d').item()
            # title     = f'{date}: Mean = {s_model[k, :].mean():.2f} | range = {s_model[k, :].min():.2f}-{s_model[k, :].max():.2f}'
            title     = f"{datasets[dataset_name].date[0].dt.strftime('%Y-%m-%d').item()} to {datasets[dataset_name].date[k].dt.strftime('%Y-%m-%d').item()}"
            file_name = f'{result_dir}/Slip/inc_slip_smoothing_{date}.png'
            c         = s_model[k, :]
            params.append([fault.mesh, fault.triangles, s_model[k + 1, :] - s_model[k, :], edges, cmap_slip, cbar_label, [-5, 5], labelpad, azim, elev, n_seg, n_tick, alpha, title, show, figsize, cbar_kwargs, file_name])


        # Parallel
        n_processes = multiprocessing.cpu_count()
        pool        = multiprocessing.Pool(processes=n_processes - 1)
        results     = pool.map(fault_3d_wrapper, params)
        pool.close()
        pool.join()
        del pool
        del results
        gc.collect()


    if animation:
        start_animation = time.time()
        animation_file = f'{result_dir}/Animations/animation_rotated_smoothed.gif'
        if os.path.exists(animation_file):
            os.remove(animation_file)
        os.system(f'magick -loop 0 -delay 0 {result_dir}/Results/rotated/Rotated_Smoothed_20*png {animation_file}')

        if len(model_file) > 0:
            animation_file = f'{result_dir}/Animations/animation_rotated_smoothed_residual.gif'
            if os.path.exists(animation_file):
                os.remove(animation_file)
            os.system(f'magick -loop 0 -delay 0 {result_dir}/Results/residuals/Rotated_Smoothed_residual_20*png {animation_file}')
        # Make animations
        # cmds.append(f'magick -loop 0 -delay 0 {result_dir}/Results/Rotated_Forward*png {result_dir}/Animations/animation_rotated_forward.gif')
        # cmds.append(f'magick -loop 0 -delay 0 {result_dir}/Results/Rotated_Smoothed*png {result_dir}/Animations/animation_rotated_smoothed.gif')
        # cmds.append(f'magick -loop 0 -delay 0 {result_dir}/Slip/slip*png {result_dir}/Animations/animation_slip.gif')

        # for cmd in cmds:
        #     print('Working on ', cmd)

        # start_animation = time.time()
        # n_processes = multiprocessing.cpu_count()
        # pool        = multiprocessing.Pool(processes=n_processes - 1)
        # results     = pool.map(os.system, cmds)
        # pool.close()
        # pool.join()

        end_animation = time.time() - start_animation
        print(f'Total animation time: {end_animation:.1f} s')

    end = time.time() - start
    print(f'Total plotting time: {end:.1f} s')
    print(result_dir)
    return


def check_downsampling(mesh_file, triangle_file, file_format, downsampled_dir, out_dir, data_dir, 
            date_index_range=[-22, -14], xkey='x', coord_type='xy', dataset_name='data',
            check_lon=False, reference_time_series=True, use_dates=False, use_datetime=False, dt=1, data_factor=1,
            model_file='', mask_file='', estimate_covariance=False, mask_dists=[3], n_samp=2*10**7, r_inc=0.2, r_max=80, mask_dir='.', cov_dir='.', 
            m0=[0.9, 1.5, 5], c_max=2, sv_max=2, ramp_type='none', ref_point=[-116, 33.5], avg_strike=315.8, trace_inc=0.01, poisson_ratio=0.25, 
            shear_modulus=6*10**9, disp_components=[1], slip_components=[0], resolution_threshold=2.3e-1, 
            width_min=0.1, width_max=10, max_intersect_width=100, min_fault_dist=1, max_iter=10, 
            smoothing_samp=False, edge_slip_samp=False, omega=1e1, sigma=1e1, kappa=2e1, mu=2e1, 
            eta=2e1, rho=1e0, steady_slip=False, constrain=False, v_sigma=1e-9, W_sigma=1,  W_dot_sigma=1, ramp_sigma=100, v_lim=(0, 3), W_lim=(0, 30), W_dot_lim=(0, 50), ramp_lim=(-100, 100),
            xlim=[-35.77475071,26.75029172], ylim=[-26.75029172, 55.08597388], vlim_slip=[0, 20], 
            vlim_disp=[[-10,10], [-10,10], [-1,1]], cmap_slip=cmc.lajolla, cmap_disp=cmc.vik, figsize=(10, 10), dpi=75, markersize=40, 
            param_file='params.py'
                        ):
    """
    Perform downsampling.
    """

    start_total = time.time()

    # ------------------ Prepare run directories and files ------------------
    # Get directory for downsampled data
    pyffit.utilities.check_dir_tree(downsampled_dir + '/' + dataset_name)
        
    # Get parameter values to check
    quadtree_dir = get_downsampled_data_directory(downsampled_dir + '/' + dataset_name, param_file)

    # -------------------------- Prepare original data --------------------------    
    # Load fault model and regularization matrices
    fault = pyffit.finite_fault.TriFault(mesh_file, triangle_file, slip_components=slip_components, 
                                         poisson_ratio=poisson_ratio, shear_modulus=shear_modulus, 
                                         verbose=False, trace_inc=trace_inc, mu=mu, eta=eta,
                                         avg_strike=avg_strike, ref_point=ref_point)
    n_patch = len(fault.triangles)

    if steady_slip:
        n_dim   = 3 * n_patch
    else:
        n_dim   = 2 * n_patch

    # Load data
    dataset = pyffit.data.load_insar_dataset(data_dir, file_format, dataset_name, ref_point, data_factor=data_factor, xkey=xkey, 
                                             coord_type=coord_type, date_index_range=date_index_range, 
                                             check_lon=check_lon, reference_time_series=reference_time_series, 
                                             incremental=False, use_dates=use_dates, use_datetime=use_datetime, 
                                             mask_file=mask_file, remove_mean=False)
    datasets = {dataset_name: dataset}
    n_obs    = len(dataset.date)
    x        = dataset.coords['x'].compute().data.flatten()
    y        = dataset.coords['y'].compute().data.flatten()

    # Load original slip model
    if len(model_file) > 0:
        try:
            with h5py.File(model_file, 'r') as file:
                slip_model = file['slip_model'][()]
        except KeyError:
            print('Error: speficied model_file could not be located')
            sys.exit(1)
    else:
        slip_model = np.full([n_patch, 3, n_obs], np.nan)

    # ------------------ Prepare inversion inputs ------------------
    # Get dictionary of quadtree parameters
    quadtree_params = dict(
                            resolution_threshold=resolution_threshold,
                            width_min=width_min,
                            width_max=width_max,
                            max_intersect_width=max_intersect_width,
                            min_fault_dist=min_fault_dist,
                            max_iter=max_iter, 
                            poisson_ratio=poisson_ratio,
                            smoothing=smoothing_samp,
                            edge_slip=edge_slip_samp,
                            disp_components=disp_components,
                            slip_components=slip_components,
                            quadtree_dir=quadtree_dir,
                            )

    # Prepare inversion inputs
    inputs = pyffit.inversion.get_inversion_inputs(fault, datasets, quadtree_params, date=-1, quadtree_dir=quadtree_dir, verbose=False)
    tree   = inputs[dataset_name].tree
    n_data = inputs[dataset_name].tree.x.size
    dt    /= 365.25

    # -------------------------- Prepare NIF objects --------------------------
    print(f'Number of fault elements: {n_patch}')
    print(f'Number of data points:    {n_data}')

    # Prepare data
    samp_file = f'cutoff={resolution_threshold:.1e}_wmin={width_min:.2f}_wmax={width_max:.2f}_max_int_w={max_intersect_width:.1f}_min_fdist={min_fault_dist:.2f}_max_it={max_iter:0f}'
    d, std = pyffit.quadtree.get_downsampled_time_series(datasets, inputs, fault, n_dim, dataset_name=dataset_name, file_name=f'{downsampled_dir}/{dataset_name}/{samp_file}.h5')

    d_samp = d[:, :n_data]
    avg_std = np.mean(std, axis=0)
    avg_v   = np.mean(d_samp, axis=0)

    # -------------------------- Plotting --------------------------
    # Plot pixels with value opposide of expected
    label = 'Sign'
    cmap  = cmc.vik_r
    vlim  = [-2, 2]
    c     = np.sign(-np.mean(d_samp, axis=0))
    file_name = ''

    # Plot fault intesecting cells
    label = 'Intersections'
    cmap  = cmc.lajolla
    vlim  = [-2, 2]
    c     = tree.intersects
    file_name = ''
    
    # Plot standard deviation
    label     = 'Standard deviation (mm)'
    cmap      = cmc.lajolla
    vlim      = [0, 10]
    show      = False
    s         = 5
    c         = avg_std
    file_name = f'{quadtree_dir}/rotated_std.png'
    title     = f'Number of cells with ' + r'$\sigma > $' + f'{vlim[1]} mm: {np.sum(avg_std > vlim[1])}/{len(avg_std)}'
    pyffit.figures.plot_rotated_fault(fault, tree.x, tree.y, c, title=title, vlim=vlim, cmap=cmap, label=label, s=s, show=show, file_name=file_name)

    # Plot standard devation with respect to mean velocity
    label     = 'Relative standard deviation'
    cmap      = cmc.vik
    vlim      = [0, 2]
    show      = False
    s         = 5
    c         = avg_std/np.abs(avg_v)
    file_name = f'{quadtree_dir}/rotated_std_relative.png'

    pyffit.figures.plot_rotated_fault(fault, tree.x, tree.y, c, vlim=vlim, cmap=cmap, label=label, s=s, show=show, file_name=file_name)
    
    # Plot rotated downsampled data
    label     = 'Displacement (mm)'
    cmap      = cmc.vik
    vlim      = [-20, 20]
    show      = False
    s         = 20
    c         = avg_std/np.abs(avg_v)
    file_name = f'{quadtree_dir}/rotated_quadtree.png'
    pyffit.figures.plot_rotated_fault(fault, tree.x, tree.y, d_samp[-1, :], vlim=vlim, cmap=cmc.vik, label=label, s=s, title=title, show=show, file_name=file_name)

    # Plot mean STD histogram
    xlim = [0, 30]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f'Number of quadtree cells: {std.shape[1]}')
    ax.hist(avg_std, range=xlim, bins=50, align='mid', histtype='step', color='k', cumulative=False, density=False)
    ax.set_xlim(xlim)
    ax.set_xlabel('Cell standard deviation (mm/yr)')
    ax.set_ylabel('Count')
    # ax1 = ax.twinx()
    # ax1.set_ylabel('Density')
    plt.savefig(f'{quadtree_dir}/cell_std.png', dpi=300)
    plt.close()

    # Plot relative STD histogram
    xlim = [0, 10]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f'Number of quadtree cells: {std.shape[1]}')
    ax.hist(avg_std/np.abs(avg_v), range=xlim, bins=50, align='mid', histtype='step', color='k', cumulative=False, density=False)
    ax.set_xlim(xlim)
    ax.set_xlabel('Cell relative standard deviation')
    ax.set_ylabel('Count')
    # ax1 = ax.twinx()
    # ax1.set_ylabel('Density')
    plt.savefig(f'{quadtree_dir}/cell_std_relative.png', dpi=300)
    plt.close()


    # Plot cell count histogram
    xlim = [0, 10]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f'Number of quadtree cells: {std.shape[1]}')
    ax.hist(tree.real_count, range=[0, 10], bins=10, align='mid', histtype='step', color='k', cumulative=True, density=False)
    ax.set_xlim(xlim)
    ax.set_xlabel('Number of pixels')
    ax.set_ylabel('Count')
    ax1 = ax.twinx()
    ax1.set_ylabel('Density')
    plt.savefig(f'{quadtree_dir}/pixel_count.png', dpi=300)
    plt.close()

    return


def test_resolution():
    # -------------------------- Settings -------------------------------------------------------------------------------------------------------------------------
    # Files
    mesh_file           = '/Users/evavra/Projects/SSAF/Analysis/Finite_Fault_Modeling/Mesh/Geometry/mesh_points_simple.txt'
    triangle_file       = '/Users/evavra/Projects/SSAF/Analysis/Finite_Fault_Modeling/Mesh/Geometry/mesh_connectivity_simple.txt'
    model_file          = '/Users/evavra/Software/pyffit/tests/NIF/synthetic_data/slip_model.h5'
    data_dir           = '/Users/evavra/Software/pyffit/tests/NIF/synthetic_data_full/signal'
    mask_file           = '/Users/evavra/Software/pyffit/tests/NIF/synthetic_data_full/decorr_mask.grd'
    file_format         = 'signal_*.grd'
    date_index_range    = [-7, -4]
    xkey                = 'x'
    ykey                = 'y'
    out_dir             = f'/Users/evavra/Software/pyffit/tests/NIF/Benchmar_High_Res/Decorrelated'

    n_samp     = 20
    # cutoff_rng = np.linspace(0.1, 0.9, n_samp)
    cutoff_rng = np.logspace(-2, 0, n_samp)
    mu_rng     = np.logspace(-4, 2, n_samp)
    kappa_rng  = mu_rng ** -0.5
    plot_results = False

    # Geographic parameters
    EPSG        = '32611' 
    ref_point   = [-116, 33.5]
    data_region = [-116.4, -115.7, 33.25, 34.0]
    avg_strike  = 315.8
    trace_inc   = 0.01

    # Elastic parameters
    poisson_ratio = 0.25       # Poisson ratio
    shear_modulus = 6 * 10**9  # Shear modulus (Pa)

    # Resolution based resampling
    resolution_threshold = 0.2 # cutoff value for resolution matrix (lower values = more points)
    width_min            = 0.1 # Min. allowed cell size (km)
    width_max            = 10   # Max. allowed cell size (km)
    max_intersect_width  = 100  # Max. allowed size for fault-intersecting cells (km)
    min_fault_dist       = 1  # Buffer distance from fault to enforce max_intersect width
    max_iter             = 10   # Max. allowed sampling iterations

    # NIF parameters
    omega           = 1e1  # temporal smoothing hyperparameter
    sigma           = 1e1   # data covariance scaling hyperparameter (Note: for single dataset, and single kappa value for steady-state velocity, transient slip, and transient velocity, sigma becomes reduntant)
    disp_components = [1] # displacement components to use [0 for fault-perpendicular, 1 for fault-parallel, 2 for vertical]
    slip_components = [0] # slip components to use [0 for strike-slip, 1 for dip-slip, 2 for opening]
    smoothing_samp  = False
    edge_slip_samp  = False
    smoothing_inv   = True
    edge_slip_inv   = False

    slip_lim        = [0, 20]
    slip_lim        = [0, np.inf]

    # Plot parameters
    xlim        = [-35.77475071, 26.75029172]
    ylim        = [-26.75029172, 55.08597388]

    xlim_r    = [-21, 26] 
    ylim_r    = [-5, 5] 
    vlim_slip = [0,   20]
    vlim_disp = [[-10, 10],
                 [-10, 10],
                 [-1, 1]] 
    cmap_disp  = cmc.vik
    cmap_slip  = 'viridis'
    figsize    = (10, 10)
    dpi        = 100
    markersize = 40

    # kappa_rng = np.logspace(-4, 4, 9)
    # mu_rng    = kappa_rng**-2
    # mu_rng    = np.logspace(-5, 5, 11)

    
    # -------------------------- Prepare original data --------------------------    
    # Load data
    dataset = pyffit.data.load_insar_dataset(data_dir, file_format, 'data', ref_point, xkey=xkey, coord_type='xy', date_index_range=date_index_range, check_lon=False, 
                                            reference_time_series=True, incremental=False, use_dates=False, use_datetime=False, mask_file=mask_file)

    # Load fault model and regularization matrices
    fault = pyffit.finite_fault.TriFault(mesh_file, triangle_file, slip_components=slip_components, poisson_ratio=poisson_ratio, shear_modulus=shear_modulus, 
                                        verbose=True, trace_inc=trace_inc)
    
    # Set up inputs
    datasets = {'data': dataset}
    
    result_dir = f'{out_dir}/Static'
    pyffit.utilities.check_dir_tree(result_dir)
    out_file = h5py.File(f'{result_dir}/results.h5', 'w')

    scores = np.empty((cutoff_rng.size, mu_rng.size))
    n_data = np.empty((cutoff_rng.size, mu_rng.size))

    start = time.time()
    
    # Loop over data resolution cutoff ranges
    for i, cutoff in enumerate(cutoff_rng[::-1]):
        test_dir = f'{result_dir}/Resolution/cutoff_{cutoff:.1e}'.replace('e+0', 'e+').replace('e-0', 'e-')
            
        pyffit.utilities.check_dir_tree(test_dir)
        # pyffit.utilities.check_dir_tree(test_dir + f'/Results')

        # -------------------------- Prepare inversion inputs --------------------------    
        # Get dictionary of quadtree parameters
        quadtree_params = dict(
                                resolution_threshold=cutoff,
                                width_min=width_min,
                                width_max=width_max,
                                max_intersect_width=max_intersect_width,
                                min_fault_dist=min_fault_dist,
                                max_iter=max_iter, 
                                poisson_ratio=poisson_ratio,
                                smoothing=smoothing_samp,
                                edge_slip=edge_slip_samp,
                                disp_components=disp_components,
                                slip_components=slip_components,
                                run_dir=test_dir,
                                )
        
        # Prepare downsampled data for inversion
        inputs = pyffit.inversion.get_inversion_inputs(fault, datasets, quadtree_params, date=-1, LOS=False, disp_components=disp_components, slip_components=slip_components, rotation=avg_strike, squeeze=True, run_dir=test_dir, verbose=False)
        
        # Get Green's Functions
        GF           = fault.greens_functions(inputs['data'].tree.x, inputs['data'].tree.y, disp_components=disp_components, slip_components=slip_components, rotation=avg_strike, squeeze=True)
        n_data[i, :] = inputs['data'].tree.x.size
        n_patch      = GF.shape[1]
        
        for j, (mu, kappa) in enumerate(zip(mu_rng, kappa_rng)):
            mu_dir = test_dir + f'/mu_{mu:.1e}'.replace('e+0', 'e+').replace('e-0', 'e-')
            pyffit.utilities.check_dir_tree(mu_dir)

            # Load fault model and regularization matrices
            fault = pyffit.finite_fault.TriFault(mesh_file, triangle_file, slip_components=slip_components, poisson_ratio=poisson_ratio, shear_modulus=shear_modulus, 
                                                verbose=True, trace_inc=trace_inc, mu=mu)
            n_patch = len(fault.triangles)

            # Load original slip model
            model = h5py.File(model_file, 'r+')
            slip_true = model['slip_model'][()]

            # # Rotate data so fault is horizontal
            # mesh_r         = np.copy(fault.mesh)
            # x_r, y_r       = pyffit.utilities.rotate(inputs['data'].tree.x, inputs['data'].tree.y, np.deg2rad(avg_strike + 90))
            # x_mesh, y_mesh = pyffit.utilities.rotate(fault.mesh[:, 0], fault.mesh[:, 1], np.deg2rad(avg_strike + 90))
            # y_r           -= np.nanmean(y_mesh[fault.mesh[:, 2] == 0])
            # y_mesh        -= np.nanmean(y_mesh[fault.mesh[:, 2] == 0])
            # mesh_r[:, 0]   = x_mesh
            # mesh_r[:, 1]   = y_mesh
            # yticks   = np.arange(-4, 5, 2)

            recovered = np.empty((n_patch, n_patch))

            # Loop over Green's functions
            tree = inputs['data'].tree
            for k in range(n_patch):
                i_patch = pyffit.utilities.get_padded_integer_string(i, 106)

                print(f'Working on Patch {i_patch}...')

                patch = GF[:, k].reshape(-1)
                tree.data = patch

                # ---------- Run inversion ----------
                inputs     = dict(data=pyffit.inversion.InversionDataset(tree, GF))    
                Inversion  = pyffit.inversion.LinearInversion(fault, inputs, smoothing=smoothing_inv, edge_slip=edge_slip_inv, verbose=False)
                slip_model = Inversion.run(slip_lim=slip_lim)[:, 0]
                recovered[:, k] = slip_model

                slip_true    = np.zeros(slip_model.shape)
                slip_true[i] = 1
                slip_resid   = slip_model - slip_true

                for g in range(n_patch):
                    print(f'Patch {g} recovered amplitude: {recovered[g, g]:.2f}')

                score = 100 * np.sum(np.diag(recovered))/n_patch 
                scores[i, j] = score

            #     # ---------- Plot ----------
            #     # print(f'Plotting Patch {i_patch}...')
            #     title = f'Patch: {i_patch}'
            #     filestem = f'mu-{mu:.1e}_{i_patch}.png'
            #     vlim_GF    = np.max(np.abs(patch)) * 0.75
            #     # vlim_resid = np.max(np.abs(Inversion.datasets['data'].resids))
            #     vlim_resid = vlim_GF * 0.1
            #     # vlim_slip  = np.max(np.abs(slip_resid))
            #     vlim_slip  = 0.3
            #     vlim_disp  = [[-vlim_GF, vlim_GF], [-vlim_GF, vlim_GF], [-vlim_resid, vlim_resid]]

            #     data_panels = [
            #                 dict(x=x_r, y=y_r, data=Inversion.datasets['data'].tree.data, label='Input'),
            #                 dict(x=x_r, y=y_r, data=Inversion.datasets['data'].model,     label='Model'),
            #                 dict(x=x_r, y=y_r, data=Inversion.datasets['data'].resids,    label='Residual'),
            #                 ]

            #     fault_panels = [
            #                 dict(slip=slip_model, cmap='Greys', vlim=[0, np.max(slip_model)],   title='Slip model', label='Slip (mm)'),
            #                 dict(slip=slip_resid, cmap=cmc.vik, vlim=[-vlim_slip, vlim_slip], title=f'Residuals ({np.abs(slip_resid).mean():.2f}'+ r'$\pm$' + f'{np.abs(slip_resid).std():.2f})',  label='Residuals (mm)'),
            #                 ]
                
            #     if plot_results:
            #         pyffit.figures.plot_fault_panels(data_panels, fault_panels, mesh_r, fault.triangles, figsize=(9, 10), title=title, n_seg=100, cmap_disp=cmc.vik, markersize=markersize,
            #                                                 orientation='vertical', fault_lim='map', fault_height=1, vlim_slip=vlim_slip, vlim_disp=vlim_disp, xlim=xlim_r, 
            #                                                 ylim=ylim_r, dpi=dpi, show=False, file_name=f'{mu_dir}/Rotated_results_{filestem}') 
                
            # Plot 3D fault
            fig, ax = pyffit.figures.plot_fault_3d(fault.mesh, fault.triangles, c=np.diag(recovered), edges=True, cmap_name=cmc.lajolla, cbar_label='Slip (mm)', 
                            vlim_slip=[0, 1], labelpad=10, azim=235, elev=17, n_seg=10, n_tick=6, alpha=1, show=False, figsize=(8, 4), title=r'$\mu =$' + f'{mu:.1e} | ' + r'$\kappa =$' + f'{kappa:.1e} | Recovery score = {score:.1f}',
                            cbar_kwargs=dict(location='bottom', pad=-0.1, shrink=0.4), file_name=f'{mu_dir}/Recovered_mu_{mu:.1e}.png', dpi=500)

            print(f"Green's functions: {GF.shape}")
            print(f"Smoothing value:   {mu}")
            print(f"Recovery score:    {score}")

    # out_file.create_dataset(f'quadtree_params', data=quadtree_params)
    out_file.create_dataset(f'scores', data=scores)
    out_file.create_dataset(f'cutoff', data=cutoff_rng)
    out_file.create_dataset(f'mu',     data=mu_rng)
    out_file.create_dataset(f'kappa',  data=kappa_rng)
    out_file.create_dataset(f'n_data', data=n_data)
    out_file.close()

    # fig, ax = plt.subplots(figsize=(7, 6))
    # im = ax.imshow(scores, cmap=cmc.lajolla, vmin=0, vmax=100, interpolation='none', extent=[cutoff_rng.min(), cutoff_rng.max(), mu_rng.min(), mu_rng.max()])
    # ax.set_xlabel(r'Smoothing $\mu$')
    # ax.set_ylabel(r'Data resolution cutoff')
    # plt.colorbar(im, label='Recovery score')
    # plt.savefig(f'{result_dir}/grid_search.png', dpi=500)
    # plt.show()

    mu_grid, cutoff_grid = np.meshgrid(mu_rng, cutoff_rng)
    pyffit.figures.plot_grid_search(scores, mu_grid, cutoff_grid, contours=n_data, x_pref=2e-2, y_pref=2e-1, 
                                   label='Recovery score', xlabel=r'Smoothing weight $\mu$', ylabel=r'Data resolution cutoff',
                                   logx=True, logy=True, log=False, x_reverse=True, y_reverse=False, file_name=f'{result_dir}/grid_search.png',
                                   show=False,)


    end = time.time() - start
    print(f'Grid search time: {end:.1f} s for {n_samp*n_samp} samples')
    

    return



    # # out_file = h5py.File(f'{run_dir}/G.h5', 'w')
    # # out_file.create_dataset(f'G', data=G)
    # # out_file.close()
    # # out_file = h5py.File(f'{run_dir}/G.h5', 'r')
    # # G = out_file['G'][()]
    # # extent = [dataset.x.min(), dataset.x.max(), dataset.y.max(), dataset.y.min()]
    # width = 4

    # for k in range(G.shape[1]):
    #     patch = fault.patches[k]

    #     if any(patch[:, 2] == 0):
    #         print(f'Working on {k}')

    #         vlim = np.max(np.abs(G[:, k]))

    #         fig, ax = plt.subplots(figsize=(6, 4))
    #         # im = ax.imshow(G[:, k].reshape(dataset.x.shape), vmin=-vlim, vmax=vlim, interpolation='none', cmap=cmc.vik, extent=extent)
    #         im = ax.scatter(inputs['data'].tree.x, inputs['data'].tree.y, c=G[:, k], vmin=-vlim, vmax=vlim, cmap=cmc.vik, s=5)
    #         ax.plot(fault.trace[:, 0], fault.trace[:, 1], c='k', linewidth=0.5)
    #         ax.set_xlim(patch[:, 0].mean() - width, patch[:, 0].mean() + width)
    #         ax.set_ylim(patch[:, 1].mean() - width, patch[:, 1].mean() + width)

    #         # ax.invert_yaxis()
    #         ax.set_title(f'Patch {k}')

    #         plt.colorbar(im, shrink=0.5)
    #         plt.savefig(f'{run_dir}/Greens_Functions/GF_{k}.png', dpi=300)
    #         plt.close()
    # return


    # Get resolution matrix
    N  = fault.resolution_matrix(inputs['data'].tree.x, inputs['data'].tree.y, mode='model', smoothing=smoothing, edge_slip=edge_slip, disp_components=disp_components, slip_components=slip_components, rotation=avg_strike, squeeze=True)
    n = np.diag(N)

    # ------------------ Perform inversion ------------------
    Inversion  = pyffit.inversion.LinearInversion(fault, inputs, smoothing=False, edge_slip=False, verbose=True)
    slip_model = Inversion.run(slip_lim=slip_lim, lsq_kwargs=dict(method='trf', lsq_solver='exact'))
    fault.update_slip(slip_model)

    # Display model info
    print('\n' + f'Model information:')
    print(f'Slip range:    {fault.slip[:, 0].min():.1f} {fault.slip[:, 0].max():.1f}')
    print(f'Moment:        {fault.moment:.2e} dyne-cm')
    print(f'M_w:           {fault.magnitude:.1f}')
    print(f'Mean residual: {np.nanmean(np.abs(Inversion.resids)):.2f}')
    print(f'RMSE:          {np.sqrt(np.nanmean(Inversion.resids**2)):.2f}')
    
    # Save some things to disk
    out_file.create_dataset(f'{date}/slip_model', data=fault.slip)
    out_file.create_dataset(f'{date}/magnitude',  data=fault.magnitude)

    # ------------------ Plot results ------------------
    start = time.time()
    show = False
    for key in Inversion.datasets.keys():
        filestem = f'{key}_mu-{mu:.1e}_eta-{eta:.1e}_{date}.png'.replace('e+0', 'e+').replace('e-0', 'e-')
        title = f'{key}: {date}'

        # Save results to file
        out_file.create_dataset(f'{date}/{key}/x',     data=Inversion.datasets[key].tree.x)
        out_file.create_dataset(f'{date}/{key}/y',     data=Inversion.datasets[key].tree.y)
        out_file.create_dataset(f'{date}/{key}/data',  data=Inversion.datasets[key].tree.data)
        out_file.create_dataset(f'{date}/{key}/model', data=Inversion.datasets[key].model)

        # # Plot fault and model fit
        panel_labels = [rf'Downsampled data', 
                        rf'Model ($\mu$ = {mu:.1e}, $\eta$ = {eta:.1e})', 
                        rf'Residuals ({np.nanmean(np.abs(Inversion.datasets[key].resids)):.2f} $\pm$ {np.nanstd(np.abs(Inversion.datasets[key].resids)):.2f}) mm']
        data_panels = [
                    dict(x=Inversion.datasets[key].tree.x, y=Inversion.datasets[key].tree.y, data=Inversion.datasets[key].tree.data, label=panel_labels[0]),
                    dict(x=Inversion.datasets[key].tree.x, y=Inversion.datasets[key].tree.y, data=Inversion.datasets[key].model,     label=panel_labels[1]),
                    dict(x=Inversion.datasets[key].tree.x, y=Inversion.datasets[key].tree.y, data=Inversion.datasets[key].resids,    label=panel_labels[2]),
                    ]


        slip_resid = fault.slip[:, 0] - slip_true[:, 0, -1]

        fault_panels = [
                    dict(slip=fault.slip[:, 0], cmap='viridis', vlim=[0, 20],   title='Slip model', label='Slip (mm)'),
                    dict(slip=slip_resid,       cmap=cmc.vik,   vlim=[-10, 10], title=f'Residuals ({np.abs(slip_resid).mean():.2f}'+ r'$\pm$' + f'{np.abs(slip_resid).std():.2f})',  label='Residuals (mm)'),
                    ]

        pyffit.figures.plot_fault_panels(data_panels, fault_panels, fault.mesh, fault.triangles, figsize=(12, 8), title=title, 
                                            orientation='horizontal', fault_lim='mesh', vlim_slip=vlim_slip, vlim_disp=vlim_disp, 
                                            xlim=xlim, ylim=ylim, dpi=dpi, show=show, file_name=f'{run_dir}/Static/Results/Full_results_{filestem}') 

        # Rotate data so fault is horizontal
        mesh_r         = np.copy(fault.mesh)
        x_r, y_r       = pyffit.utilities.rotate(Inversion.datasets[key].tree.x, Inversion.datasets[key].tree.y, np.deg2rad(avg_strike + 90))
        x_mesh, y_mesh = pyffit.utilities.rotate(fault.mesh[:, 0], fault.mesh[:, 1], np.deg2rad(avg_strike + 90))
        y_r           -= np.nanmean(y_mesh[fault.mesh[:, 2] == 0])
        y_mesh        -= np.nanmean(y_mesh[fault.mesh[:, 2] == 0])
        mesh_r[:, 0]   = x_mesh
        mesh_r[:, 1]   = y_mesh
        yticks   = np.arange(-4, 5, 2)

        data_panels = [
                    dict(x=x_r, y=y_r, data=Inversion.datasets[key].tree.data, label=panel_labels[0]),
                    dict(x=x_r, y=y_r, data=Inversion.datasets[key].model,     label=panel_labels[1]),
                    dict(x=x_r, y=y_r, data=Inversion.datasets[key].resids,    label=panel_labels[2]),
                    ]

        pyffit.figures.plot_fault_panels(data_panels, fault_panels, mesh_r, fault.triangles, figsize=(9, 10), title=title, markersize=1,
                                                    orientation='vertical', fault_lim='map', fault_height=1, vlim_slip=vlim_slip, vlim_disp=vlim_disp, xlim=xlim_r, 
                                                    ylim=ylim_r, dpi=dpi, show=show, file_name=f'{run_dir}/Static/Results/Rotated_results_{filestem}') 


    # Plot 3D fault
    fig, ax = pyffit.figures.plot_fault_3d(fault.mesh, fault.triangles, c=fault.slip[:, 0], edges=True, cmap_name='viridis', cbar_label='Slip (mm)', 
                    vlim_slip=[], labelpad=10, azim=235, elev=17, n_seg=100, n_tick=8, alpha=1, show=show, figsize=(8, 4), title=date,
                    cbar_kwargs=dict(location='bottom', pad=-0.1, shrink=0.4), file_name=f'{run_dir}/Static/Results/Fault_3D_{date}_{filestem}', dpi=dpi)
    
    end_plot = time.time() - start 
    print('\n' + f'Plotting time: {end_plot:.2f} s')
    

    # -------------------- Plots --------------------
    resolution_lim = [-np.max(np.abs(n)), np.max(np.abs(n))]

    fig, ax = pyffit.figures.plot_fault_3d(fault.mesh, fault.triangles, c=n[:n_patch], edges=True, cmap_name=cmc.vik, cbar_label='Resolution', 
                                           vlim_slip=resolution_lim, labelpad=10, azim=235, elev=17, n_seg=10, n_tick=6, alpha=1, title=f'Mean = {n.mean():.2e} | range = {n.min():.2e}-{n.max():.2e}',
                                           show=False, figsize=(7, 5), cbar_kwargs=dict(location='right', pad=0.05, shrink=0.4))
    fig.savefig(f'{result_dir}/Resolution/fault_resolution_v.png', dpi=300)
    plt.close()

    fig, ax = pyffit.figures.plot_fault_3d(fault.mesh, fault.triangles, c=n[n_patch:2*n_patch], edges=True, cmap_name=cmc.vik, cbar_label='Resolution', 
                                           vlim_slip=resolution_lim, labelpad=10, azim=235, elev=17, n_seg=10, n_tick=6, alpha=1, title=f'Mean = {n.mean():.2e} | range = {n.min():.2e}-{n.max():.2e}',
                                           show=False, figsize=(7, 5), cbar_kwargs=dict(location='right', pad=0.05, shrink=0.4))
    fig.savefig(f'{result_dir}/Resolution/fault_resolution_W.png', dpi=300)
    plt.close()

    fig, ax = pyffit.figures.plot_fault_3d(fault.mesh, fault.triangles, c=n[2*n_patch:], edges=True, cmap_name=cmc.vik, cbar_label='Resolution', 
                                           vlim_slip=resolution_lim, labelpad=10, azim=235, elev=17, n_seg=10, n_tick=6, alpha=1, title=f'Mean = {n.mean():.2e} | range = {n.min():.2e}-{n.max():.2e}',
                                           show=False, figsize=(7, 5), cbar_kwargs=dict(location='right', pad=0.05, shrink=0.4))
    fig.savefig(f'{result_dir}/Resolution/fault_resolution_W_dot.png', dpi=300)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(N, vmin=resolution_lim[0], vmax=resolution_lim[1], cmap=cmc.vik, interpolation='none')
    ax.set_title(f'Resolution matrix: Mean = {N.mean():.4f} | off-diagonal range = {N[N != 1].min():.4f}-{N[N != 1].max():.4f}', fontsize=6)
    fig.colorbar(im)
    fig.tight_layout()
    plt.savefig(f'{result_dir}/Resolution/fault_resolution_matrix.png', dpi=300)

    print(result_dir + '/Resolution')
    return


# ------------------ Utilities ------------------
def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def make_decorr_mask():
    x, y, asc = pyffit.data.read_grd('/Users/evavra/Projects/SSAF/Data/InSAR/Sentinel_1/timeseries/interp/filt/asc_spline_20211215_interp_filt_10km.grd')
    x, y, des = pyffit.data.read_grd('/Users/evavra/Projects/SSAF/Data/InSAR/Sentinel_1/timeseries/interp/filt/des_spline_20211215_interp_filt_10km.grd')

    # Get union of NaN values
    mask = np.ones_like(asc)
    mask[np.isnan(asc) | np.isnan(des)] = np.nan

    # Save to file
    pyffit.data.write_grd(x, y, mask, f'/Users/evavra/Software/pyffit/tests/NIF/synthetic_data_full/decorr_mask.grd')
    # print(mask.shape)
    # vlim = np.nanmean(np.abs(asc))

    # fig, ax = plt.subplots(1, 2, figsize=(14, 8.2))
    # ax[0].imshow(asc * mask, cmap='coolwarm', vmin=-vlim, vmax=vlim)
    # ax[1].imshow(des * mask, cmap='coolwarm', vmin=-vlim, vmax=vlim)
    # ax[0].invert_yaxis()
    # ax[1].invert_yaxis()

    plt.show()
    return


def convert_timedelta(td, unit='Y'):
    """
    Convert timedelta to years (Y) or days (D)
    """
    dt = td/np.timedelta64(1, 'D')

    if unit == 'Y':
        dt /= 365.25
    return dt


def load_site_table(file, ref_point, EPSG='32611', unit='km', mode='site'):
    """
    Load table of site coordinates to make time series plots for.
    """
    skiprows = []
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if line[0] == '#':
                skiprows.append(i)

    if mode == 'site':
        header = 0
        xkey = 'lon'
        ykey = 'lat'
    elif mode == 'creepmeter':
        header = 0
        xkey = 'Lon'
        ykey = 'Lat'
    else:
        print("Must choose mode='site' or mode='creepmeter'")

    table = pd.read_csv(file, header=header, skiprows=skiprows, delim_whitespace=True)
    table['x'], table['y'] = pyffit.utilities.get_local_xy_coords(table[xkey], table[ykey], ref_point, EPSG=EPSG, unit=unit) 

    return table
    

def check_parameters(new_params, old_params, verbose=False):
    """
    Check if parameter file has specified values
    """

    for key in new_params.keys():
        old_value = old_params[key]

        if old_value == new_params[key]:
            continue
        else:
            if verbose:
                print(f'Conflict with {key}: {old_value} {new_params[key]}')
            return False

    return True


def get_downsampled_data_directory(base_dir, param_file):
    """
    Get directory for downsampled time series.
    """
    
    # Load current parameter file
    params = load_parameters(param_file)[1]

    # Extract parameters relevant to downsampling
    check_keys   = ['mesh_file', 'triangle_file', 'data_dir', 'file_format', 'ref_point', 'resolution_threshold', 'width_min', 'width_max', 'max_intersect_width', 'min_fault_dist', 'max_iter', 'smoothing_samp', 'edge_slip_samp',]
    check_params = dict((key, params[key]) for key in check_keys)

    # Get list of existing parameter files
    param_files = glob.glob(f'{base_dir}/*/params_*.py')

    # Check parameter files
    for file in param_files:
        # Load parameters
        old_params = load_parameters(file)[1]

        # Check against specified values
        confirm = check_parameters(check_params, old_params)
        
        if confirm:
            downsampled_data_dir = '/'.join(file.split('/')[:-1])
            print(f'Using existing dataset at {downsampled_data_dir}')
            
            return downsampled_data_dir
        
    # If no match is found, make new directory
    date = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    downsampled_data_dir = base_dir + '/' + date
    pyffit.utilities.check_dir_tree(downsampled_data_dir)
    shutil.copy(param_file, f'{downsampled_data_dir}/params_{date}.py')
    print(f'Creating new dataset at {downsampled_data_dir}')

    return downsampled_data_dir
       

def integrate_slip(x, dt):
    """
    Compute integrated slip at time k for matrix of state vectors x.
    dt (years) is the elapsed time between states.
    """

    # Size of state vector is 3x number of fault elements (v, W, and W_dot for each element)
    n_patch = x.shape[1]//3
    
    # Integrate slip
    s = np.empty((x.shape[0], n_patch))

    for k in range(len(x)):
        t = k * dt # Get time since start of time series
        s[k, :] = x[k, :n_patch] * t + x[k, n_patch:2*n_patch]

    return s


def nan_gaussian_filter(U, truncate=4.0, sigma=2.0, mode='nearest', return_width=False, plot=False,):
    """
    Gaussian filter that handles NaN values.
    Based on scipy.ndimage.gaussian_filter()

    """
    w = 2*int(truncate*sigma + 0.5) + 1

    # print(f'Width = {w}')

    # Set NaN values in original array to zero
    V = np.copy(U)
    V[np.isnan(U)] = 0

    # Get auxilary array with zeros indicating NaN positions
    W = np.ones_like(U)
    W[np.isnan(U)] = 0

    # Filter arrays
    VV = scipy.ndimage.gaussian_filter(V, sigma=sigma, truncate=truncate, mode=mode)
    WW = scipy.ndimage.gaussian_filter(W, sigma=sigma, truncate=truncate, mode=mode)

    # Get filtered U array and restore NaNs
    UU = VV/WW
    UU[np.isnan(U)] = np.nan

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 8.2))
        axes[0].imshow(U, interpolation='none', cmap=cmc.vik, vmin=-20, vmax=20)
        axes[1].imshow(UU, interpolation='none', cmap=cmc.vik, vmin=-20, vmax=20)

        for ax in axes:
            ax.invert_yaxis()
        plt.show()

    if return_width:
        return UU, w
    else:
        return UU
        

def get_slip_rate_history(slip_history, dt=1):
    slip_rate_history = np.zeros_like(slip_history)
    A = np.array([(slip_history[k, :, :] - slip_history[k - 1, :, :])/dt for k in range(1, len(slip_history))])
    A = np.array([(slip_history[k + 1, :, :] - slip_history[k - 1, :, :])/(2*dt) for k in range(1, len(slip_history) - 1)])
    slip_rate_history[1:-1, :, :] = A
    return slip_rate_history


# ------------------ Plotting ------------------
def grid_search_plot():
    grid_search_dir = '/Users/evavra/Software/pyffit/tests/NIF/Benchmark/Testing/Static'
    out_file = h5py.File(grid_search_dir + '/results.h5', 'r')

    scores     = out_file['scores'][()]
    cutoff_rng = out_file['cutoff'][()]
    mu_rng     = out_file['mu'][()]
    kappa_rng  = out_file['kappa'][()]
    n_data     = out_file['n_data'][()]

    print(n_data)
    mu_grid, cutoff_grid = np.meshgrid(mu_rng, cutoff_rng)

    pyffit.figures.plot_grid_search(scores, mu_grid, cutoff_grid, contours=n_data, x_pref=2e-2, y_pref=2e-1, 
                                   label='Recovery score', xlabel=r'Smoothing weight $\mu$', ylabel=r'Data resolution cutoff',
                                   logx=True, logy=True, log=False, x_reverse=True, y_reverse=False, file_name=f'{grid_search_dir}/grid_search.png',
                                   show=False,)
    
    return

    mu    = np.logspace(-7, 7)
    kappa = mu ** -0.5

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mu, kappa, c='k')
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.grid()
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$\kappa$')
    plt.show()
    return


def make_fault_movie():
    # -------------------------- testing --------------------------
    mesh_file       = '/Users/evavra/Projects/SSAF/Analysis/Finite_Fault_Modeling/Mesh/Geometry/mesh_points_simple.txt'
    triangle_file   = '/Users/evavra/Projects/SSAF/Analysis/Finite_Fault_Modeling/Mesh/Geometry/mesh_connectivity_simple.txt'
    model_file      = 'synthetic_data/slip_model.h5'
    file_name       = 'synthetic_slip'
    slow_factor     = 1
    
    # Model and plot parameters
    slip_components = [0] # slip components to use [0 for strike-slip, 1 for dip-slip, 2 for opening]
    poisson_ratio   = 0.25       # Poisson ratio
    shear_modulus   = 6 * 10**9  # Shear modulus (Pa)
    trace_inc       = 0.01
    mu              = 2   # smoothing regularization value
    eta             = 2   # zero-edge-slip regularization value
    slip_components = [0] # slip components to use [0 for strike-slip, 1 for dip-slip, 2 for opening]
    figsize         = (10, 5)
    dpi             = 500
    
    # vmin        = 0
    # vmax        = 30
    # cmap_name   = 'viridis'
    vmin        = 0
    vmax        = 0.3
    cmap_name   = 'inferno'


    cbar_label  = 'Slip (mm)'
    labelpad    = 10
    azim        = 45
    elev        = 15
    n_seg       = 100
    n_tick      = 7
    alpha       = 1
    show        = False
    cbar_kwargs = dict(location='bottom', pad=-0.23, shrink=0.4)

    # Load results
    model      = h5py.File(model_file, 'r+')
    slip_model = model['slip_model'][()]

    # Load fault
    fault = pyffit.finite_fault.TriFault(mesh_file, triangle_file, slip_components=slip_components, poisson_ratio=poisson_ratio, shear_modulus=shear_modulus, 
                                         verbose=True, trace_inc=trace_inc, mu=mu, eta=eta)

    # Create the figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Initial triangles
    triangles = [fault.mesh[tri] for tri in fault.triangles]
    poly3d = Poly3DCollection(triangles, edgecolors="k", linewidths=0.2, alpha=1)
    ax.add_collection3d(poly3d)

    # Colorbar
    cvar  = np.zeros(len(fault.triangles))
    cval  = (cvar - vmin)/(vmax - vmin) # Normalized color values
    ticks = np.linspace(vmin, vmax, n_tick)

    cmap  = colors.LinearSegmentedColormap.from_list(cmap_name, plt.get_cmap(cmap_name, 265)(np.linspace(0, 1, 265)), n_seg)
    sm    = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    c     = cmap(cval)

    # Set axis limits
    ranges = np.ptp(fault.mesh, axis=0)
    ax.set_box_aspect(ranges) 
    ax.set_xlim(fault.mesh[:, 0].min(), fault.mesh[:, 0].max())
    ax.set_ylim(fault.mesh[:, 1].min(), fault.mesh[:, 1].max())
    ax.set_zlim(fault.mesh[:, 2].min(), fault.mesh[:, 2].max())
    ax.set_xlabel('East (km)',  labelpad=labelpad)
    ax.set_ylabel('North (km)', labelpad=labelpad)
    ax.set_zlabel('Depth (km)', labelpad=labelpad/4)
    zticks = ax.get_zticks()
    ax.set_zticklabels([f'{int(-tick)}' for tick in zticks])
    ax.view_init(azim=azim, elev=elev)
    fig.tight_layout()
    cbar = fig.colorbar(sm, label=cbar_label, **cbar_kwargs,)
    cbar.set_ticks(ticks)
    fig.subplots_adjust(top=1.5, bottom=-0.1)

    def update_slip(i):
        """
        Update the fault patch colors
        """

        i = i//slow_factor

        date   = dates[i]
        cvar   = results[f'{date}/slip_model'][()][:, 0]
        cval   = (cvar - vmin)/(vmax - vmin) # Normalized color values
        colors = cmap(cval)

        print(f'Working on {date} ({np.nanmin(cvar):1e} - {np.nanmax(cvar):1e})')

        poly3d.set_facecolor(colors)
        poly3d.set_edgecolor("k")  # Keep edges black

        fig.suptitle(date)

    def update_incremental_slip(i):
        """
        Update the fault patch colors
        """
        print(f'Working on {i}')

        if i == 0:
            cvar = slip_model[:, 0, 0]
            title = f'0'
        else: 
            cvar = slip_model[:, 0, i] - slip_model[:, 0, i - 1]
            title = f'{i - 1} to {i}'
        cval   = (cvar - vmin)/(vmax - vmin) # Normalized color values
        colors = cmap(cval)

        poly3d.set_facecolor(colors)
        poly3d.set_edgecolor("k")  # Keep edges black

        fig.suptitle(title)

    def update_sum_slip(i):
        """
        Update the fault patch colors
        """

        print(f'Working on {i}')

        cvar = slip_model[:, 0, i]
        cval   = (cvar - vmin)/(vmax - vmin) # Normalized color values
        colors = cmap(cval)

        poly3d.set_facecolor(colors)
        poly3d.set_edgecolor("k")  # Keep edges black

        fig.suptitle(i)

    # Create the animation
    ani         = FuncAnimation(fig, update_incremental_slip, frames=len(slip_model[0, 0, :]), interval=10)
    output_file = f"{file_name}.mp4"
    # output_file = "cummulative_slip_slow.mp4"
    writer      = FFMpegWriter(fps=100, metadata=dict(artist="Matplotlib"), bitrate=3600)
    ani.save(output_file, writer=writer, dpi=dpi)

    print(f"Animation saved as {output_file}")

    return


def plot_matrix(A, cmap='coolwarm'):
    fig, ax = plt.subplots(figsize=(14, 8.2))
    ax.imshow(A, cmap=cmap, interpolation='none')
    ax.set_title(f'Range: [{A.min():.2e} {A.max():.2e}] | Mean: {A.mean():.2e}' + r'$\pm$' + f'{A.std():.2e}')
    plt.show()
    return


def plot_slip_history(x, y, c, vmin=0, vmax=10, tick_inc=5, cmap=cmc.lajolla_r, label='Slip (mm)', hlines=[], 
                        title='', figsize=(10, 6), file_name='', show=False, dpi=300):
    """
    Make heatmap of slip/slip rate history.
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(x, y, c, shading='nearest', cmap=cmap, vmin=vmin, vmax=vmax)

    if len(hlines) > 0:
        ax.hlines(hlines, x.min(), x.max(), color='gainsboro', linestyle='--')
    
    ax.set_ylim(y[-1], y[0])
    ax.set_aspect(1/100)
    ax.invert_xaxis()
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Date')
    ax.set_facecolor('gainsboro')
    ax.set_title(title)
    plt.colorbar(im, label=label, shrink=0.5, ticks=np.arange(vmin, vmax + tick_inc, tick_inc))
    plt.tight_layout()

    if len(file_name) > 0:
        plt.savefig(file_name, dpi=dpi)

    if show:
        plt.show()
    return


def fault_panel_wrapper(params):

    data_panels, fault_panels, fault, figsize, title, markersize, orientation, fault_lim, vlim_slip, cmap_disp, vlim_disp, xlim, ylim, dpi, show, file_name = params
    
    pyffit.figures.plot_fault_panels(data_panels, fault_panels, fault.mesh, fault.triangles, figsize=figsize, title=title, markersize=markersize,
                                        orientation=orientation, fault_lim=fault_lim, vlim_slip=vlim_slip, cmap_disp=cmap_disp, vlim_disp=vlim_disp, 
                                        xlim=xlim, ylim=ylim, dpi=dpi, show=show, file_name=file_name) 
    # print(file_name)
    return


def fault_3d_wrapper(params):

    mesh, triangles, c, edges, cmap_name, cbar_label, vlim_slip, labelpad, azim, elev, n_seg, n_tick, alpha, title, show, figsize, cbar_kwargs, file_name = params
    
    fig, ax = pyffit.figures.plot_fault_3d(mesh, triangles, c=c, 
                                            edges=edges, 
                                            cmap_name=cmap_name, 
                                            cbar_label=cbar_label, 
                                            vlim_slip=vlim_slip, 
                                            labelpad=labelpad, 
                                            azim=azim, 
                                            elev=elev, 
                                            n_seg=n_seg, 
                                            n_tick=n_tick, 
                                            alpha=alpha, 
                                            title=title,
                                            show=show, 
                                            figsize=figsize, 
                                            cbar_kwargs=cbar_kwargs)
    fig.tight_layout()
    fig.savefig(file_name, dpi=200)
    plt.close()
    print(file_name)
    return


if __name__ == '__main__':
    main()