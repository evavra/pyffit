import os
import gc
import sys
import h5py
import time
import pickle
import pyffit
import datetime
import warnings
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import colors
from matplotlib.animation import FuncAnimation, FFMpegWriter
import pandas as pd
from scipy.optimize import minimize 
import cmcrameri.cm as cmc
warnings.filterwarnings("ignore")


def main():
    make_synthetic_data()
    return


def make_synthetic_data():
    # ------------------ Settings ------------------
    # File paths
    working_dir   = '/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/data/synthetic_3'
    mesh_file     = f'/Users/evavra/Projects/SSAF/Analysis/Finite_Fault_Modeling/Mesh/Geometry/mesh_points_529.txt'
    triangle_file = f'/Users/evavra/Projects/SSAF/Analysis/Finite_Fault_Modeling/Mesh/Geometry/mesh_connectivity_529.txt' 
    # gf_file       = '/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/synthetic_greens_functions.h5'
    gf_file       = '/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/full_greens_functions_529.h5'
    nan_mask_file = '/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/nan_mask.grd'

    # Ooperation modes
    # make = ['signal', 'aps', 'intf', 'ts', 'gif']
    # make = ['aps', 'intf', 'ts', 'gif']
    make = ['signal', 'intf', 'ts', 'gif']
    # make = ['intf', 'ts', 'gif']
    # make = ['signal']

    # Synthetic data and noise 
    max_slip    = 50  # mm
    n_aps       = 1   # number of APS to stack
    aps_amp     = 5   # APS amplitude range
    L_c         = 2   # exponential decay distance
    noise_amp   = 30  # Gaussian noise amplitude (mm)
    noise_width = 0.1 # Gaussian noise amplitude (mm)

    # Coordinates
    # n_x         = 960       # number of points in x-direction
    # n_y         = 1330      # number of points in y-direction
    # xlim        = [-35.77475071, 26.75029172]
    # ylim        = [-26.75029172, 55.08597388]
    n_scene     = 50       # Number of SAR scenes
    start_date  = '2014-04-03'     
    dt          = 12

    # Fault parameters
    ref_point       = [-116, 33.5]
    avg_strike      = 315.8
    disp_components = [1]       # displacement components to use [0 for fault-perpendicular, 1 for fault-parallel, 2 for vertical]
    trace_inc       = 0.01
    poisson_ratio   = 0.25      # Poisson ratio
    shear_modulus   = 6 * 10**9 # Shear modulus (Pa)
    slip_components = [0]       # slip components to use [0 for strike-slip, 1 for dip-slip, 2 for opening]

    # Plotting 
    # cmap       = 'coolwarm'
    # vlim       = [-10, 10]

    # ------------------ Generate synthetic dataset ------------------
    pyffit.utilities.check_dir_tree(working_dir, clear=False)  

    # Get nans and coordinates
    # x_rng = np.linspace(xlim[0], xlim[1], n_x)
    # y_rng = np.linspace(ylim[0], ylim[1], n_y)
    lon_rng, lat_rng, nan_mask = pyffit.data.read_grd(nan_mask_file)
    lon, lat = np.meshgrid(lon_rng, lat_rng)

    # Get reference point and convert coordinates to km
    x, y = pyffit.data.get_local_xy_coords(lon, lat, ref_point) 

    # Load fault model and regularization matrices
    fault = pyffit.finite_fault.TriFault(mesh_file, triangle_file, slip_components=slip_components, poisson_ratio=poisson_ratio, shear_modulus=shear_modulus, 
                                        verbose=True, trace_inc=trace_inc)



    # lon_grd, lat_grd, data = pyffit.data.read_grd('/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/data/synthetic_2/time_series/time_series_2017-07-04.grd')

    # fig, ax = plt.subplots(figsize=(14, 8.2))
    # ax.plot(fault.trace[:, 0], fault.trace[:, 1], )
    # # im = ax.scatter(x_samp, y_samp, c=np.arange(0, x_samp.size), cmap=cmc.lajolla, s=1, marker='.')
    # im = ax.scatter(x.flatten(), y.flatten(), c=data.flatten(), cmap=cmc.vik,  marker='.')
    # plt.colorbar(im)
    # plt.show()

    # return
    # -------------------- TESTING GREENS FUNCTIONS --------------------
    # # ----- check subset of grid -----
    # test_gf = np.empty_like(x)
    # test_gf[:, :] = np.nan
    # i = (x < 25) & (x > -10) & (y < 20) & (y > -15) & (nan_mask == 1)
    # # mask_samp = nan_mask[i]
    # x_samp = x[i]
    # y_samp = y[i]
    # # x_samp = x_crop[mask_samp == 1]
    # # y_samp = y_crop[mask_samp == 1]
    # print(x_samp.size)

    # # ----- check random points near fault trace -----
    # n_samp = 10000
    # x_samp = np.empty(n_samp)
    # y_samp = np.empty(n_samp)

    # for k in range(n_samp):
    #     i = np.random.randint(low=0, high=len(fault.trace))
    #     x_samp[k] = fault.trace[i, 0] + np.random.uniform(low=-1, high=1)
    #     y_samp[k] = fault.trace[i, 1] + np.random.uniform(low=-1, high=1)

    # GF = -fault.greens_functions(x_samp.flatten(), y_samp.flatten(), disp_components=disp_components, slip_components=slip_components, rotation=avg_strike, squeeze=True)
    # GF_sum = np.sum(GF, axis=1)
    # test_gf[i] = GF_sum

    # # fig, ax = plt.subplots(figsize=(14, 8.2))
    # # ax.plot(fault.trace[:, 0], fault.trace[:, 1], )
    # # # im = ax.scatter(x_samp, y_samp, c=np.arange(0, x_samp.size), cmap=cmc.lajolla, s=1, marker='.')
    # # im = ax.scatter(x.flatten(), y.flatten(), c=test_gf.flatten(), cmap=cmc.vik,  marker='.')
    # # plt.colorbar(im)
    # # plt.show()

    # pyffit.data.write_grd(lon_rng, lat_rng, test_gf, 'test.grd', coords_key='geographic', T=True)
    # pyffit.data.write_grd(lon_rng, lat_rng, test_gf, 'test2.grd',coords_key='geographic', T=False)

    # lon_grd, lat_grd, data = pyffit.data.read_grd('test.grd')
    # Lon_grid, Lat_grid = np.meshgrid(lon_grd, lat_grd)
    # x_grd, y_grd = pyffit.data.get_local_xy_coords(Lon_grid, Lat_grid, ref_point) 


    # fig, ax = plt.subplots(figsize=(14, 8.2))
    # ax.set_title('test1')
    # ax.plot(fault.trace[:, 0], fault.trace[:, 1], )
    # # im = ax.scatter(x_samp, y_samp, c=np.arange(0, x_samp.size), cmap=cmc.lajolla, s=1, marker='.')
    # im = ax.scatter(x_grd.flatten(), y_grd.flatten(), c=data.flatten(), cmap=cmc.vik,  marker='.')
    # plt.colorbar(im)
    # plt.show()

    # lon_grd, lat_grd, data = pyffit.data.read_grd('test2.grd')
    # Lon_grid, Lat_grid = np.meshgrid(lon_grd, lat_grd)
    # x_grd, y_grd = pyffit.data.get_local_xy_coords(Lon_grid, Lat_grid, ref_point) 


    # fig, ax = plt.subplots(figsize=(14, 8.2))
    # ax.set_title('test2')
    # ax.plot(fault.trace[:, 0], fault.trace[:, 1], )
    # # im = ax.scatter(x_samp, y_samp, c=np.arange(0, x_samp.size), cmap=cmc.lajolla, s=1, marker='.')
    # im = ax.scatter(x_grd.flatten(), y_grd.flatten(), c=data.flatten(), cmap=cmc.vik,  marker='.')
    # plt.colorbar(im)
    # plt.show()

    # ----- Plot summed greens functions -----
    # GF = np.empty((lon_rng.size*lat_rng.size, len(fault.triangles)))

    # with h5py.File(gf_file, 'r') as file:
    #     n_chunk = len(file.keys())
    #     start  = 0
    #     for i in range(n_chunk):
    #         chunk = file[f'chunk_{i}'][()]
    #         end   = start + chunk.shape[0]
    #         GF[start:end, :] += chunk
    #         start = end

    # GF_sum = np.sum(GF, axis=1)

    # fig, ax = plt.subplots(figsize=(14, 8.2))
    # ax.plot(fault.trace[:, 0], fault.trace[:, 1], )
    # ax.scatter(x.flatten(), y.flatten(), c=GF_sum.flatten(), cmap=cmc.vik)
    # plt.show()
    # ------------------------------------------------------------

    # -------------------- Make projection vectors --------------------
    # u_perp = np.array([np.cos(np.deg2rad(avg_strike)), np.sin(np.deg2rad(avg_strike))])
    # u_para = np.array([-np.sin(np.deg2rad(avg_strike)), np.cos(np.deg2rad(avg_strike))]) 

    # -------------------- Make signal --------------------    
    # Set signal time dependence
    # One stationary pulse
    # n_zero = 43
    n_zero = 20
    u      = np.linspace(1e-0, 1e8, n_scene - n_zero)
    s0     = np.log(u)**2
    s      = (s0 - s0.min())/(s0.max() - s0.min())
    sig    = np.concatenate((np.zeros(n_zero), s))
    d_sig  = np.diff(sig, prepend=0)

    # Two migrating pulses
    # n_zero = 20
    # u   = np.linspace(1e-2, 1e5, n_scene - n_zero)
    # s0  = np.log(u)
    # t0  = 1 - (u**2)
    # t   = (t0 - t0.min())/(t0.max() - t0.min())
    # s   = (s0 - s0.min())/(s0.max() - s0.min())
    # sig = np.concatenate((np.zeros(n_zero), s * t))
    # sig += np.concatenate((np.zeros(2*n_zero), 0.5*(s * t)[:-n_zero]))
    # sig += np.concatenate((np.zeros(3*n_zero), 0.3*(s * t)[:-2*n_zero]))
    # sig /= sig.max()

    # Plot signal evolution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(n_scene), sig, marker='.',  c='C0',)
    ax.plot(np.arange(n_scene), d_sig, marker='.', c='C3',)
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.set_xlim(0, n_scene)
    # ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(f'{working_dir}/signal_evolution.png', dpi=500)

    # # Signal 0: stationary pulse 
    signal_0 = {'x':    np.zeros(n_scene),
                'slip': max_slip * d_sig,
                'l':    0.3 * np.ones(n_scene),
                'w':    1.5 * np.ones(n_scene),}

    # # Signal 1: north propagating pulse 
    # inc_slip = max_slip/n_scene
    # signal_1 = {'x':    np.linspace(-0.5, 2, n_scene),
    #             'slip': 2*inc_slip * sig,
    #             'l':    0.1 * np.ones(n_scene),
    #             'w':    2.0 * np.ones(n_scene),}

    # # Signal 2: south propagating pulse
    # signal_2 = {'x':    np.linspace(0, -2, n_scene),
    #             'slip': inc_slip * sig,
    #             'l':    0.1 * np.ones(n_scene),
    #             'w':    2.0 * np.ones(n_scene),}

    # # Signal 3: delayed shallow creep
    # signal_3 = {'x':    0.0 * np.ones(n_scene),
    #             'slip': inc_slip * np.linspace(0, 1, n_scene),
    #             'l':    1.0 * np.ones(n_scene),
    #             'w':    1.0 * np.ones(n_scene),}

    # signals = [signal_1, signal_2, signal_3]
    # signals = [signal_1, signal_2]
    signals = [signal_0,]

    # Get dates
    start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    dates = [start_datetime + datetime.timedelta(days=k * dt) for k in range(n_scene)]

    # Make synthetic data
    generate(fault, lon_rng, lat_rng, ref_point, n_scene, dates, signals, n_aps, aps_amp, noise_amp, noise_width, 
             disp_components=disp_components, slip_components=slip_components, avg_strike=avg_strike, 
              nan_mask=nan_mask, aps_kwargs=dict(aps_amp=aps_amp, L_c=L_c), make=make, working_dir=working_dir, gf_file=gf_file)
    return


def generate(fault, lon_rng, lat_rng, ref_point, n_scene, dates, signals, n_aps, aps_amp, noise_amp, noise_width, 
             make=['signal', 'aps', 'intf', 'ts'], disp_components=[0, 1, 2],
            slip_components=[0, 1, 2], avg_strike=0, nan_mask=[], aps_kwargs={},
            clear=False, working_dir='.', gf_file='./greens_functions.h5'):
    """
    Generate synthetic dataset
    """

    start_master = time.time()

    # Prepare output directories
    prep_out_dir(working_dir, clear=clear)

    # Get full gridded geographic coordinates
    lon, lat = np.meshgrid(lon_rng, lat_rng)

    # Get reference point and convert coordinates to km
    x, y = pyffit.data.get_local_xy_coords(lon, lat, ref_point) 
    # x_data, y_data = np.meshgrid(x_rng, y_rng)
    # x_rng = x_data[0, :]
    # y_rng = y_data[:, 0]

    # Get date strings
    # dates_str = [date.strftime('%Y-%m-%d') for date in dates]

    if 'signal' in make:
        # Generate Greens functions
        # GF = pyffit.finite_fault.get_fault_greens_functions(x_data.flatten(), y_data.flatten(), np.zeros_like(x_data.flatten()), fault.mesh, fault.triangles, verbose=False)
        print('Loading Greens functions')

        # Make full Green's functions matrix
        n_chunk     = 5     # Number of chunks for full model computation
        n_data_full = x.size

        # full_gf_file = f'{run_dir}/full_greens_functions.h5'
        if not os.path.exists(gf_file):
            with h5py.File(gf_file, 'w') as file:

                chunk_size  = n_data_full//n_chunk 
                remainder   = n_data_full % n_chunk

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
        else:
            print(f'Using Greens funtions at {gf_file}')

        start_gf = time.time()
        GF = np.empty((x.size, len(fault.triangles)))
        GF[:] = np.nan

        with h5py.File(gf_file, 'r') as file:
            n_chunk = len(file.keys())
            start  = 0
            for i in range(n_chunk):
                chunk = file[f'chunk_{i}'][()]
                end   = start + chunk.shape[0]
                GF[start:end, :] = chunk
                start = end

        end_gf = time.time() - start_gf
        print(f'Done: {end_gf:.1f} s')
        print(f'Greens functions: {GF.shape} ({GF.size:.1e})')
        
        # Include only strike-slip contribution
        # GF = GF[:, :, :, 0] 

        # Project to fault parallel and fault perpendicular directions
        # _, GF = pyffit.utilities.rotate(GF[:, 0, :] ,GF[:, 1, :], np.deg2rad(avg_strike))

        # Prep input params
        signal_params = []
        slip_model = np.zeros((*fault.triangles.shape, n_scene), dtype='float64')

        for i in range(n_scene):
            # signal_inc = []

            # Concatenate signals
            for signal in signals:
                slip_model[:, :, i] += pyffit.synthetics.get_synthetic_slip(fault.mesh, fault.triangles, max_slip=signal['slip'][i], strike_scale=signal['l'][i], dip_scale=signal['w'][i], x_shift=signal['x'][i])
            
            # Sum cummulative slip
            if i > 0:
                slip_model[:, :, i] += slip_model[:, :, i - 1]

            # i_label = pyffit.utilities.get_padded_integer_string(i, n_scene)
            date = dates[i].strftime('%Y-%m-%d')
            signal_params.append((fault, GF, slip_model[:, :, i], lon, lat,
                                f'{working_dir}/signal/plots/signal_{date}.png', f'{working_dir}/signal/plots/slip_{date}.png', f'{working_dir}/signal/signal_{date}.grd'))

        with h5py.File(f'{working_dir}/slip_model.h5', 'w') as file:
            file = pyffit.data.modify_hdf5(file, 'slip_model', slip_model)

        # -------------------- Make signal --------------------
        print(f'Generating signal...')
        start = time.time()

        if (n_scene < 500) | (GF.size > 10**7):
        # if (n_scene < 5):
            # Serial
            for i in range(n_scene):
                print(f'Working on {i + 1}/{n_scene}...')
                make_signal(signal_params[i])
        else:
            # Parallel
            os.environ["OMP_NUM_THREADS"] = "1"
            start       = time.time()
            n_processes = multiprocessing.cpu_count()
            pool        = multiprocessing.Pool(processes=n_processes)
            results     = pool.map(make_signal, signal_params)
            pool.close()
            pool.join()
        
        end = time.time() - start
        del GF
        gc.collect()
        print(f'Signal computation time: {end:.1f} sec | {end/n_scene:.2f} sec/scene')

    # -------------------- Make APS --------------------
    if 'aps' in make:
        print(f'Generating APS...')

        # Make APS
        aps_params = []

        # Prep input params
        for i in range(n_scene):
            # i_label = pyffit.utilities.get_padded_integer_string(i, n_scene)
            date = dates[i].strftime('%Y-%m-%d')

            aps_params.append((lon_rng, lat_rng, n_aps, noise_amp, noise_width, aps_kwargs, f'{working_dir}/aps/plots/aps_{date}.png', f'{working_dir}/aps/aps_{date}.grd'))

        start = time.time()

        if n_scene < 20:
            # Series
            for i in range(n_scene):
                make_aps(aps_params[i])
        else:
            # Parallel 
            os.environ["OMP_NUM_THREADS"] = "1"
            start       = time.time()
            n_processes = multiprocessing.cpu_count()
            pool        = multiprocessing.Pool(processes=n_processes)
            results     = pool.map(make_aps, aps_params)
            pool.close()
            pool.join()

        end = time.time() - start
        del aps_params
        gc.collect()
        print(f'APS computation time: {end:.1f} sec | {end/n_scene:.2f} sec/scene')

    # -------------------- Make interferograms --------------------    
    if 'intf' in make:
        print(f'Preparing interferograms...')

        # Prep input parameters
        intf_params = []

        for i in range(n_scene - 1):
            # a_label = pyffit.utilities.get_padded_integer_string(i, n_scene)
            # b_label = pyffit.utilities.get_padded_integer_string(i + 1, n_scene)
            date_a = dates[i].strftime('%Y-%m-%d')
            date_b = dates[i + 1].strftime('%Y-%m-%d')

            intf_params.append((lon_rng, lat_rng, fault.trace, working_dir, date_a, date_b, nan_mask))

        start = time.time()

        if n_scene < 20:
            # Series
            for i in range(n_scene - 1):
                make_intf(intf_params[i])
        else:
            # Parallel 
            os.environ["OMP_NUM_THREADS"] = "1"
            start       = time.time()
            n_processes = multiprocessing.cpu_count()
            pool        = multiprocessing.Pool(processes=n_processes)
            results     = pool.map(make_intf, intf_params)
            pool.close()
            pool.join()

        del intf_params
        gc.collect()
        end = time.time() - start
        print(f'Interferogram computation time: {end:.1f} sec | {end/(n_scene - 1):.2f} sec/intf')

    # -------------------- Make time series --------------------    
    if 'ts' in make:
        print(f'Preparing time series...')
        stack = np.zeros_like(lon) * nan_mask

        for i in range(n_scene):
            # Get intf labels
            # a_label = pyffit.utilities.get_padded_integer_string(i - 1, n_scene)
            # b_label = pyffit.utilities.get_padded_integer_string(i, n_scene)
            date_a = dates[i - 1].strftime('%Y-%m-%d')
            date_b = dates[i].strftime('%Y-%m-%d')

            # Start at zero
            if i > 0:
                _, _, intf = pyffit.data.read_grd(f'{working_dir}/intf/intf_{date_a}-{date_b}.grd')
                stack += intf

            # Write grid to file
            pyffit.data.write_grd(lon_rng, lat_rng, stack, f'{working_dir}/time_series/time_series_{date_b}.grd', T=True, V=False)

            # Plot intf
            fig, ax = pyffit.figures.plot_grid(lon_rng, lat_rng, stack, vlim=[-15, 15], title=f'Date: {date_b} | Min: {np.nanmin(stack):.2f} Max. {np.nanmax(stack):.2f}', cbar=True)
            ax.plot(fault.trace[:, 0], fault.trace[:, 1], c='k')
            fig.savefig(f'{working_dir}/time_series/plots/time_series_{date_b}.png', dpi=300)
            
            # a_label = pyffit.utilities.get_padded_integer_string(i, n_scene)
            # b_label = pyffit.utilities.get_padded_integer_string(i + 1, n_scene)
            
            # _, _, intf = pyffit.data.read_grd(f'{working_dir}/intf/intf_{a_label}-{b_label}.grd')
            # stack += intf

            # # Write intf to file
            # pyffit.data.write_grd(x_rng, y_rng, stack, f'{working_dir}/time_series/time_series_{a_label}-{b_label}.grd', T=True, V=False)

            # # Plot intf
            # fig, ax = pyffit.figures.plot_grid(x_rng, y_rng, stack, vlim=[-15, 15])
            # ax.plot(trace[0], trace[1], c='k')
            # fig.savefig(f'{working_dir}/time_series/plots/time_series_{a_label}-{b_label}.png', dpi=300)


    if 'gif' in make:
        print(f'Making animation...')
        os.system(f'magick -loop 0 -delay 0 {working_dir}/time_series/plots/time_series_*.png {working_dir}/time_series/plots/animation_time_series.gif')

    end_master = time.time() - start_master
    print(f'Total time: {end_master/60:.1f} min')

    return


# ------------------ Wrappers for parallelization ------------------
def make_aps(params):
    """
    Wrapper for computing, plotting, and saving synthetic APS.
    """
    lon_rng, lat_rng, n_aps, noise_amp, noise_width, aps_kwargs, plot_dir, grid_dir = params

    # Make APS
    aps = np.zeros((len(lat_rng), len(lon_rng)))
    
    for i in range(n_aps):
        aps += pyffit.synthetics.make_synthetic_aps(lon_rng, lat_rng, **aps_kwargs, randomize=False)

    aps /= n_aps

    # Make speckle noise
    noise = noise_amp * np.random.normal(loc=0, scale=noise_width, size=aps.shape)
    # noise      = noise_amp * np.random.uniform(low=-1, high=1, size=data.shape)
    # aps += noise

    # Plot
    # pyffit.figures.plot_grid(x_rng, y_rng, aps, plot_dir, vlim=[-10, 10])
    vlim = np.mean(np.abs(aps)) + np.std(np.abs(aps))
    date = plot_dir.split('/')[-1][4:-4]
    fig, ax = pyffit.figures.plot_grid(lon_rng, lat_rng, aps, vlim=[-vlim, vlim], title=f'Date: {date} | Min: {np.nanmin(aps):.2f} Max. {np.nanmax(aps):.2f}', cbar=True, clabel='Displacement (mm)')
    fig.savefig(plot_dir, dpi=300)

    # Write
    pyffit.data.write_grd(lon_rng, lat_rng, aps, grid_dir, T=True, V=False)
    
    return aps


def make_signal(params):

    fault, GF, slip_model, lon, lat, signal_plot_dir, slip_plot_dir, signal_dir = params

    mesh = fault.mesh
    triangles = fault.triangles

    # print(f'Working on {k}')
    lon_rng = lon[0, :]
    lat_rng = lat[:, 0]
    
    # Compute displacements
    # disp = pyffit.finite_fault.get_fault_displacements(x_data, y_data, np.zeros_like(x_data), mesh, triangles, slip_model)
    
    # Compute displacements
    disp = GF.dot(slip_model[:, 0].flatten())
    

    # Write displacements to file
    # print(disp.reshape(x_data.shape).min(), disp.reshape(x_data.shape).max())
    pyffit.data.write_grd(lon_rng, lat_rng, disp.reshape(lon.shape), signal_dir, T=True, V=False)
    
    # Plot fault slip 
    fig, ax = pyffit.figures.plot_fault_3d(mesh, triangles, c=slip_model, edges=True, cmap_name='viridis', cbar_label='Dextral slip (mm)', 
                                           vlim_slip=[0, 20], labelpad=10, azim=235, elev=17, n_seg=10, n_tick=6, alpha=1,
                                           show=False, figsize=(7, 5), cbar_kwargs=dict(location='right', pad=0.05, shrink=0.4))
    fig.savefig(slip_plot_dir, dpi=300)
    
    # Plot surface displacements
    data = disp.reshape(lon.shape)
    fig, ax = pyffit.figures.plot_grid(lon_rng, lat_rng, data, vlim=[-20, 20], title=f'Date: {signal_plot_dir[-7:-4]} | Min: {np.nanmin(data):.2f} Max. {np.nanmax(data):.2f}', cbar=True, clabel='Displacement (mm)')
    ax.plot(mesh[mesh[:, 2] == 0][:, 0], mesh[mesh[:, 2] == 0][:, 1], c='k')
    # ax.plot(mesh[mesh[:, 2] == 0][:, 0], mesh[mesh[:, 2] == 0][:, 1], c='k')
    fig.savefig(signal_plot_dir, dpi=300)

    return slip_model, disp


def make_intf(params):
    """
    Wrapper for computing, plotting, and saving synthetic interferograms.
    """
    x_rng, y_rng, trace, working_dir, date_a, date_b, nan_mask = params

    # Read datasets
    _, _, aps_a    = pyffit.data.read_grd(f'{working_dir}/aps/aps_{date_a}.grd')
    _, _, aps_b    = pyffit.data.read_grd(f'{working_dir}/aps/aps_{date_b}.grd')
    _, _, signal_a = pyffit.data.read_grd(f'{working_dir}/signal/signal_{date_a}.grd')
    _, _, signal_b = pyffit.data.read_grd(f'{working_dir}/signal/signal_{date_b}.grd')

    # Form intf
    intf = (signal_b + aps_b - signal_a - aps_a) * nan_mask

    # Write intf to file
    pyffit.data.write_grd(x_rng, y_rng, intf, f'{working_dir}/intf/intf_{date_a}-{date_b}.grd',  T=True, V=False)

    # Plot intf
    fig, ax = pyffit.figures.plot_grid(x_rng, y_rng, intf, vlim=[-15, 15], title=f'Epoch :{date_a}-{date_b} | Min: {np.nanmin(intf):.2f} Max. {np.nanmax(intf):.2f}', cbar=True)
    ax.plot(trace[:, 0], trace[:, 1], c='k')
    fig.savefig(f'{working_dir}/intf/plots/intf_{date_a}-{date_b}.png', dpi=300)

    return intf
 

def prep_out_dir(working_dir, clear=False):
    """
    Prepare output subdirectories
    """
    pyffit.utilities.check_dir_tree(working_dir, clear=clear)  
    pyffit.utilities.check_dir_tree(working_dir + '/aps', clear=clear)  
    pyffit.utilities.check_dir_tree(working_dir + '/aps/plots', clear=clear)  
    pyffit.utilities.check_dir_tree(working_dir + '/signal', clear=clear)  
    pyffit.utilities.check_dir_tree(working_dir + '/signal/plots', clear=clear)  
    pyffit.utilities.check_dir_tree(working_dir + '/intf', clear=clear)  
    pyffit.utilities.check_dir_tree(working_dir + '/intf/plots', clear=clear)  
    pyffit.utilities.check_dir_tree(working_dir + '/time_series', clear=clear)  
    pyffit.utilities.check_dir_tree(working_dir + '/time_series/plots', clear=clear)  
    return


if __name__ == '__main__':
    main()