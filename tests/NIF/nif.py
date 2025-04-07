#!/usr/bin/env python
import os
import gc
import sys
import h5py
import time
import shutil
import pickle
import pyffit
import warnings
import numpy as np
import pandas as pd
import importlib.util
import multiprocessing
import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import minimize 
from scipy.interpolate import griddata 
from matplotlib import colors
import datetime
from types import ModuleType
from numba import jit

from collections import Counter
import linecache
import tracemalloc

def main():
    """
    Run Network Inversion Filter.
    """
    # grid_search_plot()
    # make_decorr_mask()
    # test_resolution()
    # run_network_inversion_filter()
    # make_synthetic_data()
    # make_fault_movie()

    # sys.stdout = pyffit.utilities.Logger('/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/log.txt')
    start = time.time()
    param_file = sys.argv[1]

    # Load parameters
    mode, params = load_parameters(param_file)
    
    # Perform standard NIF run
    if 'NIF' in mode:
        run_nif(**params)
        
    if 'analyze' in mode:
        analyze_nif(**params)

    end = time.time() - start
    print(f'Total run time: {end/60:.2f}')

    return


# greens_function_dir='.',
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
def analyze_nif(mesh_file, triangle_file, file_format, downsampled_dir, out_dir, data_dir, 
            date_index_range=[-22, -14], xkey='x', coord_type='xy', dataset_name='data',
            check_lon=False, reference_time_series=True, use_dates=False, use_datetime=False, dt=1, data_factor=1,
            model_file='', mask_file='', estimate_covariance=False, mask_dists=[3], n_samp=2*10**7, r_inc=0.2, r_max=80, mask_dir='.', cov_dir='.', 
            m0=[0.9, 1.5, 5], c_max=2, sv_max=2, ref_point=[-116, 33.5], avg_strike=315.8, trace_inc=0.01, poisson_ratio=0.25, 
            shear_modulus=6*10**9, disp_components=[1], slip_components=[0], resolution_threshold=2.3e-1, 
            width_min=0.1, width_max=10, max_intersect_width=100, min_fault_dist=1, max_iter=10, 
            smoothing_samp=False, edge_slip_samp=False,  omega=1e1, sigma=1e1, kappa=2e1, mu=2e1, 
            eta=2e1, steady_slip=False, constrain=False, v_sigma=1e-9, W_sigma=1,  W_dot_sigma=1, v_lim=(0,3), W_lim=(0,30), W_dot_lim=(0,50), 
            xlim=[-35.77475071,26.75029172], ylim=[-26.75029172, 55.08597388], vlim_slip=[0, 20], 
            vlim_disp=[[-10,10], [-10,10], [-1,1]], cmap_disp=cmc.vik, figsize=(10, 10), dpi=75, markersize=40, 
            param_file='params.py',

            # look_dir, asc_velo_model_file, des_velo_model_file, ykey='y', EPSG='32611', 
            # data_region=[-116.4, -115.7, 33.25, 34.0], smoothing_inv=True, edge_slip_inv=False,
            ):
    """
    Analyze network inversion filter results.
    """
    start = time.time()

    run_dir    = f'{out_dir}/omega_{omega:.1e}__kappa_{kappa:.1e}__sigma_{sigma:.1e}'
    result_dir = f'{run_dir}/Results'

    # -------------------------- Prepare original data --------------------------    
    # Load fault model and regularization matrices
    fault = pyffit.finite_fault.TriFault(mesh_file, triangle_file, slip_components=slip_components, 
                                         poisson_ratio=poisson_ratio, shear_modulus=shear_modulus, 
                                         verbose=False, trace_inc=trace_inc, mu=mu, eta=eta)
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
                                             mask_file=mask_file)
    datasets = {dataset_name: dataset}
    n_obs    = len(dataset.date)
    x        = dataset.coords['x'].compute().data.flatten()
    y        = dataset.coords['y'].compute().data.flatten()

    # dt = convert_timedelta(dataset.date[1] - dataset.date[0], unit='D')

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
                            run_dir=run_dir,
                            )

    # Prepare inversion inputs
    inputs = pyffit.inversion.get_inversion_inputs(fault, datasets, quadtree_params, date=-1, run_dir=run_dir, verbose=False)
    n_data = inputs[dataset_name].tree.x.size
    dt    /= 365.25

    # Prepare downsampled ata
    samp_file = f'cutoff={resolution_threshold:.1e}_wmin={width_min:.2f}_wmax={width_max:.2f}_max_int_w={max_intersect_width:.1f}_min_fdist={min_fault_dist:.2f}_max_it={max_iter:0f}'
    d         = pyffit.quadtree.get_downsampled_time_series(datasets, inputs, fault, n_dim, dataset_name=dataset_name, file_name=f'{downsampled_dir}/{samp_file}.pkl')

    # Load results
    # results_forward = read_kalman_filter_results(f'{run_dir}/results_forward.h5')
    # x_model_forward = results_forward.x_a
    results_forward = h5py.File(f'{run_dir}/results_forward.h5', 'r')
    x_model_forward = results_forward['x_a']

    # results_smoothing = read_kalman_filter_results(f'{run_dir}/results_smoothing.h5')
    # x_model = results_smoothing.x_s
    results_smoothing = h5py.File(f'{run_dir}/results_smoothing.h5', 'r')
    x_model           = results_smoothing['x_s']

    # Compute integrated slip
    if steady_slip:
        s_model_forward = integrate_slip(results_forward['x_a'], dt)
        s_model         = integrate_slip(results_smoothing['x_s'], dt)
    else:
        s_model_forward = x_model_forward[:, :n_patch]
        s_model = x_model[:, :n_patch]
    s_true = slip_model[:, 0, :].T

    # Compute integrated slip
    # slip_model = integrate_slip(results_smoothing.x_s, dt)

    # Get along-strike projected coordinates
    mesh_r         = np.copy(fault.mesh)
    x_mesh, y_mesh = pyffit.utilities.rotate(fault.mesh[:, 0], fault.mesh[:, 1], np.deg2rad(avg_strike + 90))
    mesh_r[:, 0]   = x_mesh
    mesh_r[:, 1]   = y_mesh
    
    # Get centroid values
    x_center = np.array([mesh_r[tri][:, 0].mean() for tri in fault.triangles])
    z_center = np.array([mesh_r[tri][:, 2].mean() for tri in fault.triangles])
    x_center -= x_center.min()

    # Interpolate
    # slip_interp = griddata((x_center, z_center), slip_model[-1, :], (x_grid, z_grid), method='cubic')
    n_x = int(x_center.max())*20
    n_z = int(abs(z_center).max())*20
    dt  = convert_timedelta((dataset['date'][1] - dataset['date'][0]).values)

    x_rng = np.linspace(x_center.min(), x_center.max(), n_x)
    z_rng = np.linspace(z_center.min(), z_center.max(), n_z)
    x_grid, z_grid = np.meshgrid(x_rng, z_rng)

    slip_history = np.empty((s_model.shape[0], n_z, n_x))

    # Interpolate slip history to regurlar grid
    print(x_center.shape, z_center.shape, s_model.shape)
    for k in range(len(s_model)):
        slip_history[k, :, :] = griddata((x_center, z_center), s_model[k, :], (x_grid, z_grid), method='cubic')

    slip_rate_history = np.zeros_like(slip_history)
    A = np.array([(slip_history[k, :, :] - slip_history[k - 1, :, :])/dt for k in range(1, len(slip_history))])
    A = np.array([(slip_history[k + 1, :, :] - slip_history[k - 1, :, :])/(2*dt) for k in range(1, len(slip_history) - 1)])
    slip_rate_history[1:-1, :, :] = A

    # -------------------- Plot history --------------------
    dpi    = 300
    cmap   = cmc.lajolla_r
    hlines = [datetime.datetime(2017, 9, 8, 0, 0, 0), datetime.datetime(2019, 7, 5, 0, 0, 0)]
 
 
    vmin   = 0
    vmax   = 5
    title  = 'Depth averaged slip rate'
    label  = 'Slip rate (mm/yr)'
    file_name = f'{result_dir}/History_slip_rate.png'

    plot_slip_history(x_rng, dataset['date'], np.mean(slip_rate_history, axis=1), vmin=vmin, vmax=vmax, title=title, label=label, hlines=hlines, file_name=file_name)

    vmin   = 0
    vmax   = 15
    title  = 'Surface slip rate'
    label  = 'Slip rate (mm/yr)'
    file_name = f'{result_dir}/History_surface_slip_rate.png'

    plot_slip_history(x_rng, dataset['date'], slip_rate_history[:, -1, :], vmin=vmin, vmax=vmax, title=title, label=label, hlines=hlines, file_name=file_name)

    vmin   = 0
    vmax   = 10
    title  = 'Depth averaged slip'
    label  = 'Slip (mm)'
    file_name = f'{result_dir}/History_slip.png'
    plot_slip_history(x_rng, dataset['date'], np.mean(slip_history, axis=1), vmin=vmin, vmax=vmax, title=title, label=label, hlines=hlines, file_name=file_name)


    vmin   = 0
    vmax   = 40
    title  = 'Surface slip'
    label  = 'Slip (mm)'
    file_name = f'{result_dir}/History_surface_slip.png'
    plot_slip_history(x_rng, dataset['date'], slip_history[:, -1, :], vmin=vmin, vmax=vmax, title=title, label=label, hlines=hlines, file_name=file_name)


    
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
    plt.colorbar(sm, label='Depth (km)')
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

    for i in range((steady_slip + 1) * n_patch, (steady_slip + 2) * n_patch):
        ax.plot(dataset.date, results_forward['x_a'][:, i], c='C0', alpha=0.2)
        ax.plot(dataset.date, results_smoothing['x_s'][:, i], c='C3', alpha=0.2)

    ax.set_xlabel('Time')
    ax.set_ylabel('Transient slip rate (mm/yr)')
    plt.savefig(f'{result_dir}/Evolution_W_dot.png', dpi=300)
    plt.close()
    
    # # Plot covariances
    # vlim = np.max(abs(results_forward.P_a[-1, :, :]))

    # fig, ax = plt.subplots(figsize=(4, 4))
    # im = ax.imshow(results_forward.P_a[-1, :, :], vmin=-vlim, vmax=vlim, cmap='coolwarm', interpolation='none')
    # ax.set_title(r'$P_a$ (backward pass)')
    # fig.colorbar(im)
    # plt.savefig(f'{result_dir}/Matrices/P_forward.png', dpi=300)
    # # plt.show()
    # plt.close()

    # vlim = np.max(abs(results_smoothing.P_s[-1, :, :]))

    # fig, ax = plt.subplots(figsize=(4, 4))
    # im = ax.imshow(results_smoothing.P_s[0, :, :], vmin=-vlim, vmax=vlim, cmap='coolwarm', interpolation='none')
    # ax.set_title(r'$P_a$ (forward pass)')
    # fig.colorbar(im)
    # plt.savefig(f'{result_dir}/Matrices/P_smoothed.png', dpi=300)
    # plt.close()
    
    # Plot fault and model fit
    params = []

    date        = datasets[dataset_name].date[-1]
    title       = f'Date: {date}'
    orientation = 'horizontal'
    fault_lim   = 'mesh'
    show        = False

    for run, results, s in zip(['Forward', 'Smoothed'], [results_forward, results_smoothing], [s_model_forward, s_model]):
        for k in range(len(x_model)):
            
            if use_datetime:
                date = datasets[dataset_name].date[k].dt.strftime('%Y-%m-%d')
            else:
                date = datasets[dataset_name].date[k]

            slip_resid = s[k, :] - s_true[k, :]

            title       = f'[{run}] Date: {date}'
            orientation = 'horizontal'
            fault_lim   = 'mesh'
            show        = False
            file_name   = f'{result_dir}/Results/{run}_{date}.png'

            data_panels = [
                        dict(x=inputs[dataset_name].tree.x, y=inputs[dataset_name].tree.y, data=d[k, :n_data],       label=dataset_name),
                        dict(x=inputs[dataset_name].tree.x, y=inputs[dataset_name].tree.y, data=results['model'][k, :], label='Model'),
                        dict(x=inputs[dataset_name].tree.x, y=inputs[dataset_name].tree.y, data=results['resid'][k, :], label=f"Residuals ({np.abs(results['resid'][k, :]).mean():.2f}"+ r'$\pm$' + f"{np.abs(results['resid'][k, :]).std():.2f})"),
                        ]
            
            fault_panels = [
                        dict(slip=s[k, :], cmap='viridis', vlim=vlim_slip, title='Slip model', label='Slip (mm)'),
                        # dict(slip=slip_resid, cmap=cmc.vik,   vlim=[-1, 1], title=f'Residuals ({np.abs(slip_resid).mean():.2f}'+ r'$\pm$' + f'{np.abs(slip_resid).std():.2f})',  label='Residuals (mm)'),
                        ]
            
            params.append([data_panels, fault_panels, fault, figsize, title, markersize, orientation, fault_lim, vlim_slip, cmap_disp, vlim_disp, xlim, ylim, dpi, show, file_name])

            if k == len(x_model):
                print(np.mean(np.abs(d[k, :n_data])), np.std(np.abs(d[k, :n_data])))

    # Parallel
    os.environ["OMP_NUM_THREADS"] = "1"
    start       = time.time()
    n_processes = multiprocessing.cpu_count()
    pool        = multiprocessing.Pool(processes=n_processes - 1)
    results     = pool.map(fault_plot_wrapper, params)
    pool.close()
    pool.join()
    del pool
    del results
    gc.collect()

    # 3D fault
    for k in range(len(x_model)):
    
        # # Plot forward modeled slip 
        # date = datasets[dataset_name].date[k]

        # fig, ax = pyffit.figures.plot_fault_3d(fault.mesh, fault.triangles, c=s_model_forward[k, :], edges=True, cmap_name='viridis', cbar_label='Dextral slip (mm)', 
        #                                     vlim_slip=[0, 20], labelpad=10, azim=235, elev=17, n_seg=10, n_tick=6, alpha=1, title=f'Mean = {s_model_forward[k, :].mean():.2f} | range = {s_model_forward[k, :].min():.2f}-{s_model_forward[k, :].max():.2f}',
        #                                     show=False, figsize=(7, 5), cbar_kwargs=dict(location='right', pad=0.05, shrink=0.4))
        # fig.savefig(f'{result_dir}/Slip/slip_forward_{date}', dpi=100)
        # plt.close()

        # Plot smoothed modeled slip 
        # date = datasets[dataset_name].date[-(k + 1)]
        date = datasets[dataset_name].date[k].dt.strftime('%Y-%m-%d')

        fig, ax = pyffit.figures.plot_fault_3d(fault.mesh, fault.triangles, c=s_model[k, :], 
                                               edges=True, 
                                               cmap_name='viridis', 
                                               cbar_label='Dextral slip (mm)', 
                                               vlim_slip=vlim_slip, 
                                               labelpad=10, 
                                               azim=235, 
                                               elev=17, 
                                               n_seg=10, 
                                               n_tick=6, alpha=1, title=f'{date}: Mean = {s_model[k, :].mean():.2f} | range = {s_model[k, :].min():.2f}-{s_model[k, :].max():.2f}',
                                               show=False, figsize=(8, 5), cbar_kwargs=dict(location='right', pad=0.05, shrink=0.4))
        fig.tight_layout()
        fig.savefig(f'{result_dir}/Slip/slip_smoothing_{date}.png', dpi=100)
        plt.close()

        # # Plot residuals
        # fig, ax = pyffit.figures.plot_fault_3d(fault.mesh, fault.triangles, c=s_model[k, :] - s_true[k, :], vlim_slip=[-20, 20], edges=True, cmap_name='coolwarm', cbar_label='Residual (mm)', 
        #                                      labelpad=10, azim=235, elev=17, n_seg=10, n_tick=6, alpha=1, 
        #                                     show=False, figsize=(7, 5), cbar_kwargs=dict(location='right', pad=0.05, shrink=0.4))
        # fig.savefig(f'{result_dir}/residual_{date}', dpi=100)
        # plt.close()


    end = time.time() - start
    print(f'Total plotting time: {end:.1f} s')
    print(result_dir)
    return


def run_nif(mesh_file, triangle_file, file_format, downsampled_dir, out_dir, data_dir, 
            date_index_range=[-22, -14], xkey='x', coord_type='xy', dataset_name='data',
            check_lon=False, reference_time_series=True, use_dates=False, use_datetime=False, dt=1, data_factor=1,
            model_file='', mask_file='', estimate_covariance=False, mask_dists=[3], n_samp=2*10**7, r_inc=0.2, r_max=80, mask_dir='.', cov_dir='.', 
            m0=[0.9, 1.5, 5], c_max=2, sv_max=2, ref_point=[-116, 33.5], avg_strike=315.8, trace_inc=0.01, poisson_ratio=0.25, 
            shear_modulus=6*10**9, disp_components=[1], slip_components=[0], resolution_threshold=2.3e-1, 
            width_min=0.1, width_max=10, max_intersect_width=100, min_fault_dist=1, max_iter=10, 
            smoothing_samp=False, edge_slip_samp=False, omega=1e1, sigma=1e1, kappa=2e1, mu=2e1, 
            eta=2e1, steady_slip=False, constrain=False, v_sigma=1e-9, W_sigma=1,  W_dot_sigma=1, v_lim=(0,3), W_lim=(0,30), W_dot_lim=(0,50), 
            xlim=[-35.77475071,26.75029172], ylim=[-26.75029172, 55.08597388], vlim_slip=[0, 20], 
            vlim_disp=[[-10,10], [-10,10], [-1,1]], cmap_disp=cmc.vik, figsize=(10, 10), dpi=75, markersize=40, 
            param_file='params.py'
            # look_dir, asc_velo_model_file, des_velo_model_file, ykey='y', EPSG='32611', 
            # data_region=[-116.4, -115.7, 33.25, 34.0], smoothing_inv=True, edge_slip_inv=False,
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

    # Copy parameter file and nif.py version to run directory
    shutil.copy(__file__, f'{run_dir}/Scripts/{date}/nif_{date}.py')
    shutil.copy(param_file, f'{run_dir}/Scripts/{date}/params_{date}.py')
    
    print(f'Run directory: {run_dir}')

    # -------------------------- Prepare original data --------------------------    
    # Load fault model and regularization matrices
    fault = pyffit.finite_fault.TriFault(mesh_file, triangle_file, slip_components=slip_components, 
                                         poisson_ratio=poisson_ratio, shear_modulus=shear_modulus, 
                                         verbose=False, trace_inc=trace_inc, mu=mu, eta=eta)
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
                                             mask_file=mask_file)
    datasets = {dataset_name: dataset}
    n_obs    = len(dataset.date)
    x        = dataset.coords['x'].compute().data.flatten()
    y        = dataset.coords['y'].compute().data.flatten()
    # dt = convert_timedelta(dataset.date[1] - dataset.date[0], unit='D')

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
                            run_dir=run_dir,
                            )

    # Prepare inversion inputs
    inputs = pyffit.inversion.get_inversion_inputs(fault, datasets, quadtree_params, date=-1, run_dir=run_dir, verbose=False)
    n_data = inputs[dataset_name].tree.x.size
    # dt     = 12 # time step (days)
    dt    /= 365.25

    # Define covariance matrices
    # r = get_distances(inputs[dataset_name].tree.x, inputs[dataset_name].tree.y)
    # C = exponential_covariance(r, 1, 1, file_name=f'{run_dir}/exponential_covariance.png', show=False)

    if estimate_covariance:
        pyffit.covariance.estimate(x, y, fault, dataset, mask_dists, n_samp=n_samp, r_inc=r_inc, r_max=r_max, mask_dir=mask_dir, cov_dir=cov_dir, m0=m0, c_max=c_max, sv_max=sv_max,)
    else:
        C = np.eye(inputs[dataset_name].tree.x.size)

    # -------------------------- Prepare NIF objects --------------------------
    print(f'Number of fault elements: {n_patch}')
    print(f'Number of data points:    {n_data}')

    # Prepare data
    samp_file = f'cutoff={resolution_threshold:.1e}_wmin={width_min:.2f}_wmax={width_max:.2f}_max_int_w={max_intersect_width:.1f}_min_fdist={min_fault_dist:.2f}_max_it={max_iter:0f}'
    d         = pyffit.quadtree.get_downsampled_time_series(datasets, inputs, fault, n_dim, dataset_name=dataset_name, file_name=f'{downsampled_dir}/{samp_file}.pkl')

    # Get Greens Functions
    G = -fault.greens_functions(inputs[dataset_name].tree.x, inputs[dataset_name].tree.y, disp_components=disp_components, slip_components=slip_components, rotation=avg_strike, squeeze=True)
    
    # Set up model bounds and initial uncertainties
    if steady_slip:
        state_lims   = [v_lim, W_lim, W_dot_lim]
        state_sigmas = [v_sigma, W_sigma, W_dot_sigma]
    else:
        state_lims   = [W_lim, W_dot_lim]
        state_sigmas = [W_sigma, W_dot_sigma]
    
    # Get bounds on model parameters
    state_lim = get_state_constraints(fault, state_lims)

    # Perform forward Kalman filtering
    x_model_forward, x_model = network_inversion_filter(fault, G, d, C, dt, omega, sigma, kappa, state_sigmas, steady_slip=steady_slip, constrain=constrain, state_lim=state_lim, result_dir=run_dir, cost_function='state')

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

    return


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


def fault_plot_wrapper(params):

    data_panels, fault_panels, fault, figsize, title, markersize, orientation, fault_lim, vlim_slip, cmap_disp, vlim_disp, xlim, ylim, dpi, show, file_name = params
    
    pyffit.figures.plot_fault_panels(data_panels, fault_panels, fault.mesh, fault.triangles, figsize=figsize, title=title, markersize=markersize,
                                        orientation=orientation, fault_lim=fault_lim, vlim_slip=vlim_slip, cmap_disp=cmap_disp, vlim_disp=vlim_disp, 
                                        xlim=xlim, ylim=ylim, dpi=dpi, show=show, file_name=file_name) 
    print(file_name)
    return


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


# ------------------ NIF ------------------
def network_inversion_filter(fault, G, d, C, dt, omega, sigma, kappa, state_sigmas, steady_slip=False, constrain=False, state_lim=[], result_dir='.', cost_function='state'):
    """
    Invert time series of surface displacements for fault slip.
    Based on the method of Bekaert et al. (2016)

    INPUT:
    x_init (n_dim,)      - intial state vector (n_dim - # of model parameters) 
    P_init (n_dim, n_dim)- intial state covariance matrix 
    d (n_obs, n_data)    - observations (n_obs - # of time points, n_data - # of data points)

    H (n_data, n_dim)    - observation matrix: maps model output x to observations d
    R (n_data, n_data)   - data covariance matrix
    T (n_dim, n_dim)     - state transition matrix: maps state x_k to next state x_k+1
    Q (n_data, n_data)   - process noise covariance matrix: prescribes covariances (i.e. tuning parameters) related to temporal smoothing
    """

    # -------------------- Prepare NIF --------------------
    n_patch = len(fault.triangles)

    if steady_slip:
        n_dim   = 3 * n_patch
    else:
        n_dim   = 2 * n_patch

    # Form smoothing matrix S
    L = fault.smoothing_matrix

    # Form observation matrix H
    # H = make_observation_matrix(G, fault.smoothing_matrix, t=dt)

    # Form transition matrix T
    T = make_transition_matrix(n_patch, dt, steady_slip=steady_slip)

    # Form process covariance matrix Q
    Q = make_process_covariance_matrix(n_patch, dt, omega, steady_slip=steady_slip)

    # Form data covariance matrix R
    R = make_data_covariance_matrix(C, n_patch, sigma, kappa, steady_slip=steady_slip)

    # Define initial state vector
    x_init = np.zeros(n_dim)

    # Form initial state prediction covariance matrix P_init
    P_init = make_prediction_covariance_matrix(n_patch, state_sigmas)

    # -------------------- Run NIF --------------------
    # # 1) Perform forward Kalman filtering
    model, resid, rms = kalman_filter(x_init, P_init, d, dt, G, L, T, R, Q, steady_slip=steady_slip, constrain=constrain, state_lim=state_lim, cost_function=cost_function, file_name=f'{result_dir}/results_forward.h5')
    # write_kalman_filter_results(model, resid, rms, x_f=x_f, x_a=x_a, P_f=P_f, P_a=P_a, backward_smoothing=False, file_name=f'{result_dir}/results_forward.h5')
    # results_forward = read_kalman_filter_results(f'{result_dir}/results_forward.h5')

    with h5py.File(f'{result_dir}/results_forward.h5', 'r') as file:
        x_model_forward = file['x_a'][()]

    # Form transition matrix T
    # T = make_transition_matrix(n_patch, -dt, steady_slip=steady_slip)
    
    # Form process covariance matrix Q
    # Q = make_process_covariance_matrix(n_patch, -dt, omega)

    # Perform backward smooothing
    # Update input objects using forward pass results
    # d      = d[::-1, :] # Reverse time
    # x_init = results_forward.x_a[-1, :]    # Use final state estimate as initial state
    # P_init = results_forward.P_a[-1, :, :] # Use final covariance estimate as initial state covariance

    # Clear memory before proceding
    # del results_forward
    gc.collect()

    # Perform backward smoothing
    # results_smoothing = kalman_filter(x_init, P_init, d, dt, G, L, T, R, Q, constrain=constrain, state_lim=state_lim, cost_function=cost_function, backward_smoothing=True)
    # results_smoothing = backward_smoothing(results_forward.x_f, results_forward.x_a, results_forward.P_f, results_forward.P_a, d, dt, G, L, T,constrain=constrain, state_lim=state_lim, cost_function=cost_function)
    model, resid, rms, x_s, P_s = backward_smoothing(f'{result_dir}/results_forward.h5', d, dt, G, L, T, steady_slip=steady_slip, constrain=constrain, state_lim=state_lim, cost_function=cost_function)
    write_kalman_filter_results(model, resid, rms, x_s=x_s, P_s=P_s, backward_smoothing=True, file_name=f'{result_dir}/results_smoothing.h5',)
    x_model_smoothing = x_s
    gc.collect()
    # del results_smoothing
    # gc.collect()

    # results_smoothing = backward_smoothing(results_forward.x_f, results_forward.x_a, results_forward.P_f, results_forward.P_a, d, dt, G, L, T, 
                                        #    state_lim=state_lim, cost_function='state')

    return x_model_forward, x_model_smoothing


def kalman_filter(x_init, P_init, d, dt, G, L, T, R, Q, state_lim=[], steady_slip=False, constrain=False, cost_function='state', backward_smoothing=False, overwrite=True, file_name='results_forward.h5'):
    """
    INPUT:
    x_init (n_dim,)     - intial state (n_dim - # of model parameters) 
    d (n_obs, n_data)   - observations (n_obs - # of time points, n_data - # of data points)

    G - greens funcions matrix
    L - fault smoothing matrix

    H (n_data, n_dim)   - observation matrix: maps model output x to observations d
    R (n_data, n_data)  - data covariance matrix
    T (n_dim, n_dim)    - state transition matrix: maps state x_k to next state x_k+1
    Q (n_data, n_data)  - process noise covariance matrix: prescribes covariances (i.e. tuning parameters) related to temporal smoothing
    """

    # ---------- Kalman Filter ---------- 
    n_dim   = x_init.size
    if steady_slip:
        n_patch = x_init.size//3
    else:
        n_patch = x_init.size//2

    n_obs   = d.shape[0]
    n_data  = d.shape[1] - n_dim
    slip_start = steady_slip * n_patch  # Get start of transient slip
    
    x_f     = np.empty((n_obs, n_dim))        # forecasted states
    x_a     = np.empty((n_obs, n_dim))        # analyzed states
    P_f     = np.empty((n_obs, n_dim, n_dim)) # forecasted covariances
    P_a     = np.empty((n_obs, n_dim, n_dim)) # analyzed covariances
    model   = np.empty((n_obs, n_data))
    resid   = np.empty((n_obs, n_data))
    rms     = np.empty((n_obs,))

    print()

    if backward_smoothing:
        print('############### Running Kalman filter in backward mode ###############')
    else:
        print('############### Running Kalman filter ###############')

    start_total = time.time()

    for k in range(0, n_obs):
        start_step = time.time()
        print(f'\n##### Working on Step {k} #####')

        # Update observation matrix H        
        start_H = time.time()
        H = make_observation_matrix(G, L, t=dt*k, steady_slip=steady_slip)
        end_H = time.time() - start_H
        print(f'H-matrix time: {end_H:.2f} s')

        # 1) ---------- Forecast ----------
        # Make forecast 
        start_forecast = time.time()

        x_f[k, :] = T @ x_init
        
        # Compute covariance forecast
        if k == 0:
            P_f[k, :, :] = P_init

            # Print object information
            print(f'[G] Greens function matrix:     shape = {G.shape} | {G.size:.1e} elements | {100 * np.sum(G == 0)/G.size:.2f}% sparsity) | mean = {np.mean(G)} | std = {np.std(G)}')
            print(f'[H] Observation matrix:         shape = {H.shape} | {H.size:.1e} elements | {100 * np.sum(H == 0)/H.size:.2f}% sparsity) | mean = {np.mean(H)} | std = {np.std(H)}')
            print(f'[R] Data covariance matrix:     shape = {R.shape} | {R.size:.1e} elements | {100 * np.sum(R == 0)/R.size:.2f}% sparsity) | mean = {np.mean(R)} | std = {np.std(R)}')
            print(f'[T] Transition matrix:          shape = {T.shape} | {T.size:.1e} elements | {100 * np.sum(T == 0)/T.size:.2f}% sparsity) | mean = {np.mean(T)} | std = {np.std(T)}')
            print(f'[Q] Process covariance matrix:  shape = {Q.shape} | {Q.size:.1e} elements | {100 * np.sum(Q == 0)/Q.size:.2f}% sparsity) | mean = {np.mean(Q)} | std = {np.std(Q)}')

        else:
            P_f[k, :, :] = T @ P_a[k - 1, :, :] @ T.T + Q 
        
        end_forecast = time.time() - start_forecast
        print(f'Forecast time: {end_forecast:.2f} s')

        # 2) ---------- Analysis ----------
        # Get Kalman gain
        HPfH = H @ P_f[k, :, :] @ H.T 
        HPfH_R = HPfH + R

        start_pinv = time.time()
        HPfHT_inv = np.linalg.pinv(HPfH_R, hermitian=True)
        end_pinv = time.time() - start_pinv
        print(f'HPfHT_inv time: {end_pinv:.2f} s')

        # Get Kalman gain
        K = P_f[k, :, :] @ H.T @ HPfHT_inv 

        # Update state and covariance        
        start_analysis = time.time()
        x_a[k, :]      = x_f[k, :] + K @ (d[k, :] - H @ x_f[k, :])
        P_a[k, :, :]   = P_f[k, :, :] - K @ H @ P_f[k, :, :]
        end_analysis   = time.time() - start_analysis
        print(f'Analysis time: {end_forecast:.2f} s')
                                    
        # Constrain state estimate 
        start_opt = time.time()
        print(f'State range: {x_a[k, :].min():.2f} - {x_a[k, :].max():.2f}')

        if len(state_lim) == n_dim:

            bounds_flag = False
            
            # Define initial guess as unconstrained state vector with bounds enforced on parameters outside of range
            x_0 = np.zeros_like(x_a[k, :])

            for i in range(len(x_0)):
                # Enforce parameter bounds
                if x_a[k, i] < state_lim[i][0]:
                    x_0[i] = state_lim[i][0]
                    bounds_flag = True

                elif x_a[k, i] > state_lim[i][1]:
                    x_0[i] = state_lim[i][1]
                    bounds_flag = True
                
                else:
                    x_0[i] = x_a[k, i]
                
            # If value is less than the previous one, set to be previous one
            for i in range(len(x_0)):
                if k > 0:
                    if backward_smoothing:
                        if x_a[k, i] > x_a[k - 1, i]:
                            x_0[i] = x_a[k - 1, i]
                    elif x_a[k, i] < x_a[k - 1, i]:
                            x_0[i] = x_a[k - 1, i]

            # If at least one value falls outside of the supplied bounds, perform optimization
            if bounds_flag:

                print('Constraining...')
                # Define monotonic increase constraint (or decrease if in backward smoothing mode)
                if k != 0:
                    if backward_smoothing:
                        # State at k should be less than at state k - 1
                        constraints = (
                                        {'type': 'ineq', 'fun': lambda x: x_a[k - 1, slip_start:slip_start + n_patch] - x[slip_start:slip_start + n_patch]}, # W_k-1 - W_k >= 0
                                        )
                    else:
                        # # State at k should be greater than at state k - 1
                        # constraints = (
                        #                 {'type': 'ineq', 'fun': lambda x: x[slip_start:slip_start + n_patch] - x_a[k - 1, slip_start:slip_start +n_patch]}, # W_k - W_k-1 >= 0
                        #                 )


                        # Supply element-wise constraints on transient slip
                        constraints = []

                        for j in range(slip_start, slip_start + n_patch):
                            def monotonic(x):
                                return x[j] - x_a[k - 1, j]

                        constraints.append({'type': 'ineq', 'fun': lambda x: monotonic(x)}) # W[k, j] - W[k - 1, j]  >= 0

                else:
                    # There is no reference state if at k = 0
                    constraints = ()

                # Define cost function for optimization
                # if cost_function == 'joint':
                #     # cost_function = lambda x: np.linalg.norm(d[k, :] - H @ x, ord=2) + np.linalg.norm(P_a[k, :, :]**-1 * (x - x_a[k, :]), ord=2)
                #     cost_function = lambda x: np.linalg.norm(d[k, :] - H @ x, ord=2) + np.linalg.norm(x - x_a[k, :], ord=2)
                # else:

                cost_function = lambda x: np.linalg.norm(x - x_a[k, :], ord=2)
                # cost_function = lambda x: np.linalg.norm(d[k, :] - H @ x, ord=2) + np.linalg.norm(x - x_a[k, :], ord=2)

                # # Perform minimization
                if constrain:
                    results = minimize(cost_function, x_0, bounds=state_lim, method='COBYLA', constraints=constraints, tol=1e-6)
                    end_opt = time.time() - start_opt

                    x_a[k, :] = results.x

                    print(f'Constraint time: {end_opt:.2f} s')

                else:
                    end_opt = 0
                    x_a[k, :] = x_0

                print(f'Updated state range: {x_a[k, :].min():.2f} - {x_a[k, :].max():.2f}')
                
            else:
                print(f'Step {k} state is within bounds')
                end_opt = 0

        # Use current analysis initialize next forecast
        x_init = x_a[k, :] 

        # Compute error terms
        pred        = H @ x_a[k, :]
        model[k, :] = pred[:n_data]
        resid[k, :] = (d[k, :] - pred)[:n_data]
        rms[k]      = np.sqrt(np.mean(resid[k, :]**2))  

        end_step = time.time() - start_step
        print(f'Step {k} time: {end_step:.2f} s (inv: {end_pinv/end_step * 100:.1f} %, opt: {end_opt/end_step * 100:.1f} %, other: {(end_step - end_pinv - end_opt)/end_step * 100:.1f} %)')


    end_total = time.time() - start_total

    if backward_smoothing:
        print('\n##### Backward smoothing complete #####')
    else:
        print('\n##### Kalman filter complete #####')
    print(f'Elapsed time: {end_total/60:.1f} min')
    print()

    # return KalmanFilterResults(model, resid, rms, x_f=x_f, x_a=x_a, P_f=P_f, P_a=P_a, x_s=x_a, P_s=P_a, backward_smoothing=backward_smoothing)

    # Save results
    if os.path.exists(file_name):
        if overwrite:
            os.remove(file_name)
        else:
            print(f'Error! {file_name} already exists.')
            print(f'Set "overwrite=True" to replace')
            sys.exit(1)

    with h5py.File(f'{file_name}', 'w') as file:

        file.create_dataset('model', data=model)
        file.create_dataset('resid', data=resid)
        file.create_dataset('rms',   data=rms)
        file.create_dataset('backward_smoothing', data=backward_smoothing)

        if backward_smoothing:
            file.create_dataset('x_s', data=x_a)
            file.create_dataset('P_s', data=P_a)
            del P_a
            gc.collect()

        else:
            print('Saving forecast...')
            file.create_dataset('x_f', data=x_f)
            file.create_dataset('P_f', data=P_f)
            del P_f
            gc.collect()

    if not backward_smoothing:
        with h5py.File(f'{file_name}', 'a') as file:
            print('Saving analysis...')
            file.create_dataset('x_a', data=x_a)
            file.create_dataset('P_a', data=P_a)
            del P_a
            gc.collect()


    # if backward_smoothing:
    #     return model, resid, rms, x_a, P_a
    # else:
        # return model, resid, rms, x_f, x_a, P_f, P_a
    return model, resid, rms


def backward_smoothing(result_file, d, dt, G, L, T, steady_slip=False, constrain=False, state_lim=[], cost_function='state'):
    """
    Dimensions:
    n_obs  - # of time points
    n_dim  - # of model parameters
    n_data - # of data points

    INPUT:
    x_f, x_a (n_obs, n_dim)         - forecasted and analyzed state vectors
    P_f, P_a (n_obs, n_dim, n_dim)  - forecasted and analyzed covariance matrices 
    T        (n_dim, n_dim)         - state transition matrix: maps state x_k to next state x_k-1

    OUTPUT:
    """

    start_total = time.time()
    file = h5py.File(f'{result_file}', 'r')
    x_f = file['x_f'][()]
    x_a = file['x_a'][()]

    # ---------- Kalman Filter ---------- 
    # Get lengths
    n_obs   = x_f.shape[0]
    n_dim   = x_f.shape[1]
    if steady_slip:
        n_patch = n_dim//3
    else:
        n_patch = n_dim//2    
    n_data  = d.shape[1] - n_dim
    slip_start = steady_slip * n_patch # Get start of transient slip
    
    # Initialize
    x_s     = np.empty((n_obs, n_dim))        # analyzed states
    P_s     = np.empty((n_obs, n_dim, n_dim)) # analyzed covariances
    S       = np.empty((n_obs - 1, n_dim, n_dim)) # analyzed covariances
    model   = np.empty((n_obs, n_data))
    resid   = np.empty((n_obs, n_data))
    rms     = np.empty((n_obs,))

    print(f'\n##### Running backward smoothing ##### ')

    # Set last epoch to be equal to forward filtering result
    x_s[-1, :]    = x_a[-1, :]
    P_s[-1, :, :] = file['P_a'][-1, :, :]

    for k in reversed(range(0, n_obs - 1)):
        print(f'\n##### Working on Step {k} #####')
        
        # Compute smoothing matrix S
        P_f_inv = np.linalg.pinv(file['P_f'][k + 1, :, :], hermitian=True)
        S[k, :, :] = file['P_a'][k, :, :] @ T.T @ P_f_inv

        # Get smoothed states and covariances
        x_s[k, :]    =         x_a[k, :] + S[k, :, :] @ (x_s[k + 1, :]               - x_f[k + 1, :])
        P_s[k, :, :] = file['P_a'][k, :] + S[k, :, :] @ (P_s[k + 1, :, :] - file['P_f'][k + 1, :, :]) @ S[k, :, :].T
        
         # Constrain state estimate 
        start_opt = time.time()
        print(f'State range: {x_s[k, :].min():.2f} - {x_s[k, :].max():.2f}')
        
        if len(state_lim) == n_dim:
            
            H = make_observation_matrix(G, L, t=dt*k, steady_slip=steady_slip)

            bounds_flag = False
            
            # Define initial guess as unconstrained state vector with bounds enforced on parameters outside of range
            x_0 = np.zeros_like(x_s[k, :])

            for i in range(len(x_0)):
                # Enforce parameter bounds
                if x_s[k, i] < state_lim[i][0]:
                    x_0[i] = state_lim[i][0]
                    bounds_flag = True

                elif x_s[k, i] > state_lim[i][1]:
                    x_0[i] = state_lim[i][1]
                    bounds_flag = True
                
                else:
                    x_0[i] = x_s[k, i]
                
            # If value is less than the previous one, set to be previous one
            for i in range(len(x_0)):
                if x_0[i] > x_s[k + 1, i]:
                    x_0[i] = x_s[k + 1, i]
    
            # If at least one value falls outside of the supplied bounds, perform optimization
            if bounds_flag:
                print('Constraining...')
                # Define monotonic increase constraint (or decrease if in backward smoothing mode)
                if (k < n_obs - 1):
                    # State at k should be less than at state k - 1
                    # constraints = (
                    #                 {'type': 'ineq', 'fun': lambda x: x_s[k + 1, n_patch:2*n_patch] - x[n_patch:2*n_patch]}, # W_k+1 - W_k >= 0
                    #                 )
                    
                    # Supply element-wise constraints on transient slip
                    constraints = []

                    for j in range(slip_start, slip_start + n_patch):
                        def monotonic(x):
                            return x_s[k + 1, j] - x[j]

                    constraints.append({'type': 'ineq', 'fun': lambda x: monotonic(x)}) # W[k+1, j] - W[k, j] >= 0

                else:
                    # There is no reference state if at k = n_obs - 1
                    constraints = ()

                # Define cost function for optimization
                # if cost_function == 'joint':
                #     # cost_function = lambda x: np.linalg.norm(d[k, :] - H @ x, ord=2) + np.linalg.norm(P_a[k, :, :]**-1 * (x - x_a[k, :]), ord=2)
                #     cost_function = lambda x: np.linalg.norm(d[k, :] - H @ x, ord=2) + np.linalg.norm(x - x_a[k, :], ord=2)
                # else:
                cost_function = lambda x: np.linalg.norm(x - x_s[k, :], ord=2)
                # cost_function = lambda x: np.linalg.norm(d[k, :] - H @ x, ord=2) + np.linalg.norm(x - x_s[k, :], ord=2)

                # # Perform minimization
                if constrain:
                    results = minimize(cost_function, x_0, bounds=state_lim, method='COBYLA', constraints=constraints, tol=1e-6)
                    end_opt = time.time() - start_opt

                    x_s[k, :] = results.x

                    print(f'Constraint time: {end_opt:.2f} s')
                    
                else:
                    end_opt = 0
                    x_s[k, :] = x_0

                print(f'Updated state range: {x_s[k, :].min():.2f} - {x_s[k, :].max():.2f}')
                
            else:
                print(f'Step {k} state is within bounds')
                end_opt = 0

    for k in range(n_obs):
        H = make_observation_matrix(G, L, t=dt*k)

        # Compute error terms
        pred        = H @ x_s[k, :]
        model[k, :] = pred[:n_data]
        resid[k, :] = (d[k, :] - pred)[:n_data]
        rms[k]      = np.sqrt(np.mean(resid[k, :]**2))  


    file.close()
    gc.collect()
    end_total = time.time() - start_total
    print('##### Backward smoothing complete #####')
    print(f'Elapsed time: {end_total/60:.1f} min')
    print()

    return model, resid, rms, x_s, P_s


def read_kalman_filter_results(file_name):
    """
    Read Kalman filter results file and add to KalmanFilterResults object
    """

    with h5py.File(f'{file_name}', 'r') as file:

        for key in file.keys():
            print(key)
        backward_smoothing = file['backward_smoothing'][()]

        if backward_smoothing:
            # model = file['model'][()][::-1]  
            # resid = file['resid'][()][::-1]  
            # rms   = file['rms'][()][::-1]    
            # x_s = file['x_s'][()][::-1] 
            # P_s = file['P_s'][()][::-1] 
            model = file['model'][()]  
            resid = file['resid'][()]  
            rms   = file['rms'][()]    
            x_s = file['x_s'][()] 
            P_s = file['P_s'][()] 
            return KalmanFilterResults(model, resid, rms, x_s=x_s, P_s=P_s, backward_smoothing=backward_smoothing)

        else:
            model = file['model'][()] 
            resid = file['resid'][()] 
            rms   = file['rms'][()]   
            x_f = file['x_f'][()] 
            x_a = file['x_a'][()] 
            P_f = file['P_f'][()] 
            P_a = file['P_a'][()] 
            return KalmanFilterResults(model, resid, rms, x_f=x_f, x_a=x_a, P_f=P_f, P_a=P_a, backward_smoothing=backward_smoothing)


def write_kalman_filter_results(model, resid, rms, x_f=[], x_a=[], P_f=[], P_a=[], x_s=[], P_s=[], backward_smoothing=False, file_name='results.h5', overwrite=True):
    """
    Write Kalman filter results to HDF5 file.
    """

    if os.path.exists(file_name):
        if overwrite:
            os.remove(file_name)
        else:
            print(f'Error! {file_name} already exists.')
            print(f'Set "overwrite=True" to replace')
            sys.exit(1)

    with h5py.File(f'{file_name}', 'w') as file:

        file.create_dataset('model', data=model)
        file.create_dataset('resid', data=resid)
        file.create_dataset('rms',   data=rms)
        file.create_dataset('backward_smoothing', data=backward_smoothing)

        if backward_smoothing:
            file.create_dataset('x_s', data=x_s)
            file.create_dataset('P_s', data=P_s)
        else:
            print('Saving forecast...')
            file.create_dataset('x_f', data=x_f)
            file.create_dataset('P_f', data=P_f)
            del P_f
            gc.collect()

    if not backward_smoothing:
        with h5py.File(f'{file_name}', 'a') as file:
            print('Saving analysis...')
            file.create_dataset('x_a', data=x_a)
            file.create_dataset('P_a', data=P_a)

    return


def write_kalman_filter_results_old(results, file_name='results.h5', overwrite=True, backward_smoothing=False):
    """
    Write Kalman filter results to HDF5 file.
    """

    if os.path.exists(file_name):
        if overwrite:
            os.remove(file_name)
        else:
            print(f'Error! {file_name} already exists.')
            print(f'Set "overwrite=True" to replace')
            sys.exit(1)

    with h5py.File(f'{file_name}', 'w') as file:

        file.create_dataset('model', data=results.model)
        file.create_dataset('resid', data=results.resid)
        file.create_dataset('rms',   data=results.rms)
        file.create_dataset('backward_smoothing', data=results.backward_smoothing)

        if backward_smoothing:
            file.create_dataset('x_s', data=results.x_s)
            file.create_dataset('P_s', data=results.P_s)
        else:
            file.create_dataset('x_f', data=results.x_f)
            file.create_dataset('P_f', data=results.P_f)

    if not backward_smoothing:
        with h5py.File(f'{file_name}', 'a') as file:
            file.create_dataset('x_a', data=results.x_a)
            file.create_dataset('P_a', data=results.P_a)

    return


def backward_smoothing_old(x_f, x_a, P_f, P_a, d, dt, G, L, T, constrain=False, state_lim=[], cost_function='state'):
    """
    Dimensions:
    n_obs  - # of time points
    n_dim  - # of model parameters
    n_data - # of data points

    INPUT:
    x_f, x_a (n_obs, n_dim)         - forecasted and analyzed state vectors
    P_f, P_a (n_obs, n_dim, n_dim)  - forecasted and analyzed covariance matrices 
    T        (n_dim, n_dim)         - state transition matrix: maps state x_k to next state x_k-1

    OUTPUT:
    """

    start_total = time.time()

    # ---------- Kalman Filter ---------- 
    # Get lengths
    n_obs   = x_f.shape[0]
    n_dim   = x_f.shape[1]
    n_patch = n_dim//3
    n_data  = d.shape[1] - n_dim

    # Initialize
    x_s     = np.empty((n_obs, n_dim))        # analyzed states
    P_s     = np.empty((n_obs, n_dim, n_dim)) # analyzed covariances
    S       = np.empty((n_obs - 1, n_dim, n_dim)) # analyzed covariances
    model   = np.empty((n_obs, n_data))
    resid   = np.empty((n_obs, n_data))
    rms     = np.empty((n_obs,))

    # # Define cost function for optimization
    # if cost_function == 'joint':
    #     # cost_function = lambda x: np.linalg.norm(d[k, :] - H @ x, ord=2) + np.linalg.norm(P_a[k, :, :]**-1 * (x - x_a[k, :]), ord=2)
    #     cost_function = lambda x: np.linalg.norm(d[k, :] - H @ x, ord=2) + np.linalg.norm(x - x_a[k, :], ord=2)
    # else:

    print(f'\n##### Running backward smoothing ##### ')

    x_s[-1, :]    = x_a[-1, :]
    P_s[-1, :, :] = P_a[-1, :, :]

    for k in reversed(range(0, n_obs - 1)):
        # print(k)
        
        # Compute smoothing matrix S
        P_f_inv = np.linalg.pinv(P_f[k + 1, :, :], hermitian=True)
        S[k, :, :] = P_a[k, :, :] @ T.T @ P_f_inv

        # Get smoothed states and covariances
        x_s[k, :]    = x_a[k, :] + S[k, :, :] @ (x_s[k + 1, :] - x_f[k + 1, :])
        P_s[k, :, :] = P_a[k, :] + S[k, :, :] @ (P_s[k + 1, :, :] - P_f[k + 1, :, :]) @ S[k, :, :].T
        
         # Constrain state estimate 
        start_opt = time.time()
        print(f'State range: {x_a[k, :].min():.2f} - {x_a[k, :].max():.2f}')
        if len(state_lim) == n_dim:

            bounds_flag = False
            
            # Define initial guess as unconstrained state vector with bounds enforced on parameters outside of range
            x_0 = np.zeros_like(x_s[k, :])

            for i in range(len(x_0)):
                # Enforce parameter bounds
                if x_s[k, i] < state_lim[i][0]:
                    x_0[i] = state_lim[i][0]
                    bounds_flag = True

                elif x_s[k, i] > state_lim[i][1]:
                    x_0[i] = state_lim[i][1]
                    bounds_flag = True
                
                else:
                    x_0[i] = x_s[k, i]
                
            # If value is less than the previous one, set to be previous one
            for i in range(len(x_0)):
                if k < len(n_obs):
                    if backward_smoothing:
                        if x_a[k, i] > x_s[k - 1, i]:
                            x_0[i] = x_s[k - 1, i]
                    elif x_a[k, i] < x_s[k - 1, i]:
                            x_0[i] = x_s[k - 1, i]

            # If at least one value falls outside of the supplied bounds, perform optimization
            if bounds_flag:
                print('Constraining...')
                # Define monotonic increase constraint (or decrease if in backward smoothing mode)
                if k != 0:
                    if backward_smoothing:
                        # State at k should be less than at state k - 1
                        constraints = (
                                        {'type': 'ineq', 'fun': lambda x: x_s[k - 1, n_patch:2*n_patch] - x[n_patch:2*n_patch]}, # W_k-1 - W_k >= 0
                                        )
                    else:
                        # State at k should be greater than at state k - 1
                        constraints = (
                                        {'type': 'ineq', 'fun': lambda x: x[n_patch:2*n_patch] - x_s[k - 1, n_patch:2*n_patch]}, # W_k - W_k-1 >= 0
                                        )
                else:
                    # There is no reference state if at k = 0
                    constraints = ()

                # Define cost function for optimization
                # if cost_function == 'joint':
                #     # cost_function = lambda x: np.linalg.norm(d[k, :] - H @ x, ord=2) + np.linalg.norm(P_a[k, :, :]**-1 * (x - x_a[k, :]), ord=2)
                #     cost_function = lambda x: np.linalg.norm(d[k, :] - H @ x, ord=2) + np.linalg.norm(x - x_a[k, :], ord=2)
                # else:
                cost_function = lambda x: np.linalg.norm(x - x_s[k, :], ord=2)
                # cost_function = lambda x: np.linalg.norm(d[k, :] - H @ x, ord=2) + np.linalg.norm(x - x_s[k, :], ord=2)

                # # Perform minimization
                if constrain:
                    results = minimize(cost_function, x_0, bounds=state_lim, method='COBYLA', constraints=constraints, tol=1e-6)
                    end_opt = time.time() - start_opt

                    x_s[k, :] = results.x

                    print(f'Constraint time: {end_opt:.2f} s')

                else:
                    end_opt = 0
                    x_s[k, :] = x_0

                print(f'Updated state range: {x_s[k, :].min():.2f} - {x_s[k, :].max():.2f}')
                
            else:
                print(f'Step {k} state is within bounds')
                end_opt = 0


    for k in range(n_obs):
        H = make_observation_matrix(G, L, t=dt*k)

        # Compute error terms
        pred        = H @ x_s[k, :]
        model[k, :] = pred[:n_data]
        resid[k, :] = (d[k, :] - pred)[:n_data]
        rms[k]      = np.sqrt(np.mean(resid[k, :]**2))  


    end_total = time.time() - start_total
    print('##### Backward smoothing complete #####')
    print(f'Elapsed time: {end_total/60:.1f} min')
    print()

    return KalmanFilterResults(model, resid, rms, x_f=x_f, x_a=x_a, P_f=P_f, P_a=P_a, x_s=x_a, P_s=P_a, backward_smoothing=True)
    # return   


def make_observation_matrix(G, R, t=1, steady_slip=False):
    """
    Form observation from Greens function matrix G and smoothing matrix R.
    """

    zeros_G = np.zeros_like(G)
    zeros_R = np.zeros_like(R)

    if steady_slip:
        # Form rows
        data         = np.hstack((  G * t,       G, zeros_G))
        v_smooth     = np.hstack((      R, zeros_R, zeros_R))
        W_smooth     = np.hstack((zeros_R,       R, zeros_R))
        W_dot_smooth = np.hstack((zeros_R, zeros_R,       R))
        H            = np.vstack((data, v_smooth, W_smooth, W_dot_smooth))
   
    else:
        # Form rows
        data         = np.hstack((      G, zeros_G))
        W_smooth     = np.hstack((      R, zeros_R))
        W_dot_smooth = np.hstack((zeros_R,       R))
        H            = np.vstack((data, W_smooth, W_dot_smooth))

    # Plot 
    # fig, ax = plt.subplots(figsize=(14, 8.2))
    # ax.imshow(H[:10000, :], cmap='coolwarm', vmin=-1e-4, vmax=1e-4)
    # ax.set_aspect(0.01)
    # plt.show()

    return H


def make_transition_matrix(n_patch, dt, steady_slip=False):
    """
    Form transtion matrix from Greens function matrix G and smoothing matrix R.
    """

    if steady_slip:
        n_dim = 3 * n_patch
    else:
        n_dim = 2 * n_patch

    T = np.eye(n_dim)
    T[-2*n_patch:-n_patch, -n_patch:] = np.eye(n_patch) * dt # add slip rate term

    # # Form base matrices
    # I     = np.eye(n_patch)
    # zeros = np.zeros((n_patch, n_patch))
    # # Form rows
    # v     = np.hstack((    I,  zeros,  zeros))
    # W     = np.hstack((zeros,      I, I * dt))
    # W_dot = np.hstack((zeros,  zeros,      I))
    # T     = np.vstack((v, W, W_dot))

    return T


def make_prediction_covariance_matrix(n_patch, state_sigmas):
    """
    Form initial prediction covariance matrix P_0 based on uncertainties on model parameters.
    state_sigmas have length n_dim and should correspond to uncertainties on v and/or W and W_dot.
    v_sigma, W_sigma, W_dot_sigma should have units of v, W, W_dot (mm/yr and mm, typically).

    
    """
    n_param = len(state_sigmas)
    P_0     = np.eye(n_param * n_patch)
    
    for i in range(n_patch):
        for j in range(n_param):
            P_0[i + j*n_patch, i + j*n_patch] *= state_sigmas[j]

    # # Form base matrices
    # I     = np.eye(n_patch)
    # zeros = np.zeros((n_patch, n_patch))
    # # Form rows
    # v     = np.hstack((v_sigma * I,  zeros,                 zeros))
    # W     = np.hstack((zeros,        W_sigma * I,           zeros))
    # W_dot = np.hstack((zeros,        zeros,       W_dot_sigma * I))
    # P_0     = np.vstack((v, W, W_dot))

    return P_0


def make_process_covariance_matrix(n_patch, dt, omega, steady_slip=True):
    """
    Form process covariance matrix Q from number of fault elements n_patch, time-step dt, and temporal smoothing parameter omega.
    """

    # Form base matrices
    I     = np.eye(n_patch)
    # zeros = np.zeros((n_patch, n_patch))

    # # Form rows 
    # v     = np.hstack((zeros,                      zeros,                      zeros))
    # W     = np.hstack((zeros, (omega**2) * (dt**3)/3 * I, (omega**2) * (dt**2)/2 * I))
    # W_dot = np.hstack((zeros, (omega**2) * (dt**2)/2 * I,        (omega**2) * dt * I))
    # Q     = np.vstack((v, W, W_dot))

    # Form W/W_dot submatrix
    W     = np.hstack(((omega**2) * (dt**3)/3 * I, (omega**2) * (dt**2)/2 * I))
    W_dot = np.hstack(((omega**2) * (dt**2)/2 * I,        (omega**2) * dt * I))
    Q_sub = np.vstack((W, W_dot))

    # Form complete Q matrix
    if steady_slip:
        Q = np.zeros((3*n_patch, 3*n_patch))
        Q[-2*n_patch:, -2*n_patch:] = Q_sub
    
    else:
        Q = Q_sub
        
    return Q


def make_data_covariance_matrix(C, n_patch, sigma, kappa, steady_slip=False):
    """
    Form data covariance matrix R from observation covariance matrix C, data covariance weight sigma, and spatial smoothing weight kappa.
    """

    if steady_slip:
        n_dim = 3 * n_patch
    else:
        n_dim = 2 * n_patch

    n_data = len(C)

    # Form base matrices
    I       = np.eye(n_patch)

    # Form R matrix
    R = np.eye((n_data + n_dim))
    R[:n_data, :n_data]  = sigma**2 * C
    R[n_data:, n_data:] *= kappa**2

    # zeros_I = np.zeros_like(I)
    # zeros_CI = np.zeros((C.shape[0], n_patch))
    # zeros_IC = np.zeros((n_patch, C.shape[1]))
    # # Form rows
    # d     = np.hstack((sigma**2 * C,     zeros_CI,     zeros_CI,     zeros_CI))
    # v     = np.hstack((    zeros_IC, kappa**2 * I,      zeros_I,      zeros_I))
    # W     = np.hstack((    zeros_IC,      zeros_I, kappa**2 * I,      zeros_I))
    # W_dot = np.hstack((    zeros_IC,      zeros_I,      zeros_I, kappa**2 * I))
    # R     = np.vstack((d, v, W, W_dot))
    return R


def exponential_covariance(h, a, b, file_name='', show=False, units='km'):

    """
    Compute exponential covariance function as described by Wackernagel (2003).

    INPUT:
    h - distance
    a - drop-off scaling parameter
    b - amplitude scaling parameter
    """
    
    C_exp = lambda h, a, b: b * np.exp(-np.abs(h)/a)

    # Plot
    if len(file_name) > 0:
        if h.size > 1:

            h_rng = np.linspace(0, np.max(h), 100)
            C_rng = C_exp(h_rng, a, b)

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.vlines(x=a, ymin=0, ymax=b, linestyle='--')
            ax.plot(h_rng, C_rng, linewidth=2, c='k')
            ax.set_xlabel('Distance (km)')
            ax.set_ylabel('Covariance')

            fig.savefig(file_name, dpi=300)

            if show:
                plt.show()

            plt.close()

    return C_exp(h, a, b)


def get_distances(x, y):
    """
    Compute Euclidian distance between points defined by (x, y)
    """
    r = np.empty((x.size, x.size))
    for i in range(x.size):
        for j in range(x.size):
            r[i, j] = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
    return r


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
    

def get_state_constraints(fault, bounds):
    """
    Map supplied parameter bounds to the dimensions of the state vector.
    Typically bounds will consist of [v_lim, W_lim, W_dot_lim].
    Note that len(fault.patches) * len(bounds) should equal len(x) (or, n_dim).
    """
    state_lim = []
    for i in range(len(bounds)):
        for j in range(len(fault.patches)):
            state_lim.append(bounds[i])

    return state_lim


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


@jit(nopython=True)
def update_state(x_f, K, d, H):
    return x_f + K @ (d - H @ x_f)


@jit(nopython=True)
def update_covariance(P_f, K, H):
    return P_f - K @ H @ P_f


def convert_timedelta(td, unit='Y'):
    """
    Convert timedelta to years (Y) or days (D)
    """
    dt = td/np.timedelta64(1, 'D')

    if unit == 'Y':
        dt /= 365.25
    return dt


# ------------------ Plotting ------------------
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


# ------------------ Classes ------------------
class KalmanFilterResults:

    """
    Object containing results of a Kalman filtering procedure.

    ATTRIBUTES:
    x_f, x_a, x_s (n_obs, n_dim)        - forecasted, analyzed, and smoothed state vectors
    P_f, P_a, x_s (n_obs, n_dim, n_dim) - forecasted, analyzed, and smoothed covariance matrices
    resid (n_obs, n_data)          - model misfits at each time step
    rms (n_obs,)                   - RMS error after analysis at each time step
    """

    def __init__(self, model, resid, rms, x_f=[], x_a=[], P_f=[], P_a=[], x_s=[], P_s=[], backward_smoothing=False):
        """
        Set attributes.
        """
        
        self.backward_smoothing = backward_smoothing

        if backward_smoothing:
            # self.x_s   = x_s[::-1, :]
            # self.P_s   = P_s[::-1, :]
            # self.model = model[::-1, :]
            # self.resid = resid[::-1]
            # self.rms   = rms[::-1]
            self.x_s   = x_s
            self.P_s   = P_s
            self.model = model
            self.resid = resid
            self.rms   = rms
        else:
            self.x_f   = x_f
            self.x_a   = x_a
            self.P_f   = P_f
            self.P_a   = P_a

            self.model = model
            self.resid = resid
            self.rms   = rms


if __name__ == '__main__':
    main()