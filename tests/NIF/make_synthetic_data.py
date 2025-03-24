import os
import h5py
import time
import pickle
import pyffit
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
    working_dir   = 'synthetic_data_full'
    mesh_file     = f'/Users/evavra/Projects/SSAF/Analysis/Finite_Fault_Modeling/Mesh/Geometry/mesh_points_simple.txt'
    triangle_file = f'/Users/evavra/Projects/SSAF/Analysis/Finite_Fault_Modeling/Mesh/Geometry/mesh_connectivity_simple.txt' 

    # Synthetic data 
    n_scene     = 200       # Number of SAR scenes
    n_processes = 8         # number of threads for parallelization
    max_slip    = 100       # mm
    aps_amp     = 2        # mm
    n_x         = 960       # number of points in x-direction
    n_y         = 1330       # number of points in y-direction
    # xlim        = [-16, 30]
    # ylim        = [-21, 30]
    xlim        = [-35.77475071, 26.75029172]
    ylim        = [-26.75029172, 55.08597388]
    avg_strike  = 315.8

    # Plotting 
    cmap       = 'coolwarm'
    vlim       = [-10, 10]

    # Generate full coordinates
    x_rng = np.linspace(xlim[0], xlim[1], n_x)
    y_rng = np.linspace(ylim[0], ylim[1], n_y)
    x_data, y_data = np.meshgrid(x_rng, y_rng)

    # ------------------ Generate synthetic dataset ------------------
    start_master = time.time()
    pyffit.utilities.check_dir_tree(working_dir + '/aps', clear=True)  
    pyffit.utilities.check_dir_tree(working_dir + '/aps/plots', clear=True)  
    pyffit.utilities.check_dir_tree(working_dir + '/signal', clear=True)  
    pyffit.utilities.check_dir_tree(working_dir + '/signal/plots', clear=True)  
    pyffit.utilities.check_dir_tree(working_dir + '/intf', clear=True)  
    pyffit.utilities.check_dir_tree(working_dir + '/intf/plots', clear=True)  
    pyffit.utilities.check_dir_tree(working_dir + '/time_series', clear=True)  
    pyffit.utilities.check_dir_tree(working_dir + '/time_series/plots', clear=True)  
    file = h5py.File(f'{working_dir}/slip_model.h5', 'w')

    # -------------------- Make projection vectors --------------------
    u_perp = np.array([np.cos(np.deg2rad(avg_strike)), np.sin(np.deg2rad(avg_strike))])
    u_para = np.array([-np.sin(np.deg2rad(avg_strike)), np.cos(np.deg2rad(avg_strike))]) 

    # -------------------- Make signal --------------------    
    # Load fault model
    mesh      = np.loadtxt(mesh_file,     delimiter=',', dtype=float)
    triangles = np.loadtxt(triangle_file, delimiter=',', dtype=int)
    trace     = [mesh[mesh[:, 2] == 0][:, 0], mesh[mesh[:, 2] == 0][:, 1]]

    # Generate Greens functions
    GF = pyffit.finite_fault.get_fault_greens_functions(x_data.flatten(), y_data.flatten(), np.zeros_like(x_data.flatten()), mesh, triangles, verbose=False)

    # Include only strike-slip contribution
    GF = GF[:, :, :, 0] 

    # Project to fault parallel and fault perpepndicualr directions
    _, GF = pyffit.utilities.rotate(GF[:, 0, :] ,GF[:, 1, :], np.deg2rad(avg_strike))

    # Set signal time dependence
    n_zero = 20
    u   = np.linspace(1e-2, 1e5, n_scene - n_zero)
    s0  = np.log(u)
    t0  = 1 - (u**2)
    t   = (t0 - t0.min())/(t0.max() - t0.min())
    s   = (s0 - s0.min())/(s0.max() - s0.min())
    sig = np.concatenate((np.zeros(n_zero), s * t))
    sig += np.concatenate((np.zeros(2*n_zero), 0.5*(s * t)[:-n_zero]))
    sig += np.concatenate((np.zeros(3*n_zero), 0.3*(s * t)[:-2*n_zero]))
    sig /= sig.max()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(n_scene), sig, marker='.', c='C3',)
    ax.set_xlabel('Time')
    ax.set_ylabel('Signal amplitude')
    ax.set_xlim(0, n_scene)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(f'{working_dir}/signal_evolution.png', dpi=500)

    inc_slip = max_slip/n_scene

    # Signal 1: north propagating pulse 
    signal_1 = {'x':    np.linspace(-0.5, 2, n_scene),
                'slip': 2*inc_slip * sig,
                'l':    0.1 * np.ones(n_scene),
                'w':    2.0 * np.ones(n_scene),}

    # Signal 2: south propagating pulse
    signal_2 = {'x':    np.linspace(0, -2, n_scene),
                'slip': inc_slip * sig,
                'l':    0.1 * np.ones(n_scene),
                'w':    2.0 * np.ones(n_scene),}

    # Signal 3: delayed shallow creep
    signal_3 = {'x':    0.0 * np.ones(n_scene),
                'slip': inc_slip * np.linspace(0, 1, n_scene),
                'l':    1.0 * np.ones(n_scene),
                'w':    1.0 * np.ones(n_scene),}

    # signals = [signal_1, signal_2, signal_3]
    signals = [signal_1, signal_2]
    signal_params = []

    # Prep input params
    slip_model = np.zeros((*triangles.shape, n_scene), dtype='float64')

    for i in range(n_scene):
        signal_inc = []

        # Concatenate signals
        for signal in signals:
            slip_model[:, :, i] += pyffit.synthetics.get_synthetic_slip(mesh, triangles, max_slip=signal['slip'][i], strike_scale=signal['l'][i], dip_scale=signal['w'][i], x_shift=signal['x'][i])

        # Sum cummulative slip
        if i > 0:
            slip_model[:, :, i] += slip_model[:, :, i - 1]

        i_label = pyffit.utilities.get_padded_integer_string(i, n_scene)
        signal_params.append((mesh, triangles, GF, slip_model[:, :, i], x_data, y_data, f'{working_dir}/signal/plots/signal_{i_label}.png', f'{working_dir}/signal/plots/slip_{i_label}.png', f'{working_dir}/signal/signal_{i_label}.grd'))

    file = pyffit.data.modify_hdf5(file, 'slip_model', slip_model)

    # -------------------- Make signal --------------------
    # Parallel 
    print(f'Generating signal...')
    start = time.time()

    if n_scene < 5:
        # Series
        for i in range(n_scene):
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
    print(f'Signal computation time: {end:.1f} sec | {end/n_scene:.2f} sec/scene')

    
    # -------------------- Make APS --------------------
    print(f'Generating APS...')

    # Make APS
    aps_params = []

    # Prep input params
    for i in range(n_scene):
        i_label = pyffit.utilities.get_padded_integer_string(i, n_scene)
        aps_params.append((x_rng, y_rng, aps_amp, f'{working_dir}/aps/plots/aps_{i_label}.png', f'{working_dir}/aps/aps_{i_label}.grd'))\

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
    print(f'APS computation time: {end:.1f} sec | {end/n_scene:.2f} sec/scene')


    # -------------------- Make interferograms --------------------    
    print(f'Preparing interferogram...')

    # Prep input parameters
    intf_params = []

    for i in range(n_scene - 1):
        a_label = pyffit.utilities.get_padded_integer_string(i, n_scene)
        b_label = pyffit.utilities.get_padded_integer_string(i + 1, n_scene)
        intf_params.append((x_rng, y_rng, trace, working_dir, a_label, b_label))

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

    end = time.time() - start
    print(f'Interferogram computation time: {end:.1f} sec | {end/(n_scene - 1):.2f} sec/intf')

    end_master = time.time() - start_master
    print(f'Total time: {end_master:.1f}')

    # -------------------- Make time series --------------------    
    stack = np.zeros_like(x_data)

    for i in range(n_scene):
        # Get intf labels
        a_label = pyffit.utilities.get_padded_integer_string(i - 1, n_scene)
        b_label = pyffit.utilities.get_padded_integer_string(i, n_scene)

        # Start at zero
        if i > 0:
            _, _, intf = pyffit.data.read_grd(f'{working_dir}/intf/intf_{a_label}-{b_label}.grd')
            stack += intf

        # Write grid to file
        pyffit.data.write_grd(x_rng, y_rng, stack, f'{working_dir}/time_series/time_series_{b_label}.grd', T=True, V=False)

        # Plot intf
        fig, ax = pyffit.figures.plot_grid(x_rng, y_rng, stack, vlim=[-15, 15], title=f'Date: {b_label} | Min: {stack.min():.2f} Max. {stack.max():.2f}', cbar=True)
        ax.plot(trace[0], trace[1], c='k')
        fig.savefig(f'{working_dir}/time_series/plots/time_series_{b_label}.png', dpi=300)
        
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
    return



# ------------------ Wrappers for parallelization ------------------
def make_aps(params):
    """
    Wrapper for computing, plotting, and saving synthetic APS.
    """
    x_rng, y_rng, aps_amp, plot_dir, grid_dir = params

    # Make APS
    aps = pyffit.synthetics.make_synthetic_aps(x_rng, y_rng, manual_amp=aps_amp)

    # Plot
    pyffit.figures.plot_grid(x_rng, y_rng, aps, plot_dir, vlim=[-10, 10])
    
    # Write
    pyffit.data.write_grd(x_rng, y_rng, aps, grid_dir, T=True, V=False)
    
    return aps


def make_signal(params):

    mesh, triangles, GF, slip_model, x_data, y_data, signal_plot_dir, slip_plot_dir, signal_dir = params
    x_rng = x_data[0, :]
    y_rng = y_data[:, 0]

    # Compute displacements
    # disp = pyffit.finite_fault.get_fault_displacements(x_data, y_data, np.zeros_like(x_data), mesh, triangles, slip_model)
    
    # Compute displacements
    disp = GF.dot(slip_model[:, 0].flatten())
    
    # Write displacements to file
    # print(disp.reshape(x_data.shape).min(), disp.reshape(x_data.shape).max())
    pyffit.data.write_grd(x_rng, y_rng, disp.reshape(x_data.shape), signal_dir, T=True, V=False)
    
    # Plot fault slip 
    fig, ax = pyffit.figures.plot_fault_3d(mesh, triangles, c=slip_model, edges=True, cmap_name='viridis', cbar_label='Dextral slip (mm)', 
                                           vlim_slip=[0, 20], labelpad=10, azim=235, elev=17, n_seg=10, n_tick=6, alpha=1,
                                           show=False, figsize=(7, 5), cbar_kwargs=dict(location='right', pad=0.05, shrink=0.4))
    fig.savefig(slip_plot_dir, dpi=300)
    
    # Plot surface displacements
    data = disp.reshape(x_data.shape)
    fig, ax = pyffit.figures.plot_grid(x_rng, y_rng, data, vlim=[-20, 20], title=f'Date: {signal_plot_dir[-7:-4]} | Min: {data.min():.2f} Max. {data.max():.2f}', cbar=True, clabel='Displacement (mm)')
    ax.plot(mesh[mesh[:, 2] == 0][:, 0], mesh[mesh[:, 2] == 0][:, 1], c='k')
    fig.savefig(signal_plot_dir, dpi=300)

    return slip_model, disp


def make_intf(params):
    """
    Wrapper for computing, plotting, and saving synthetic interferograms.
    """
    x_rng, y_rng, trace, working_dir, a_label, b_label = params

    # Read datasets
    _, _, aps_a    = pyffit.data.read_grd(f'{working_dir}/aps/aps_{a_label}.grd')
    _, _, aps_b    = pyffit.data.read_grd(f'{working_dir}/aps/aps_{b_label}.grd')
    _, _, signal_a = pyffit.data.read_grd(f'{working_dir}/signal/signal_{a_label}.grd')
    _, _, signal_b = pyffit.data.read_grd(f'{working_dir}/signal/signal_{b_label}.grd')

    # Form intf
    intf = signal_b + aps_b - signal_a - aps_a

    # Write intf to file
    pyffit.data.write_grd(x_rng, y_rng, intf, f'{working_dir}/intf/intf_{a_label}-{b_label}.grd', T=True, V=False)

    # Plot intf
    fig, ax = pyffit.figures.plot_grid(x_rng, y_rng, intf, vlim=[-15, 15], title=f'Epoch :{a_label}-{b_label} | Min: {intf.min():.2f} Max. {intf.max():.2f}', cbar=True)
    ax.plot(trace[0], trace[1], c='k')
    fig.savefig(f'{working_dir}/intf/plots/intf_{a_label}-{b_label}.png', dpi=300)

    return intf

