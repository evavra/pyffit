import os
import sys
import glob
import time
import h5py
import emcee
import numba
import shutil
import corner
import matplotlib
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from map_utilities import proj_ll2utm, add_utm_coords, add_ll_coords
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from grid_utilities import read_grd
from multiprocessing import Pool
from matplotlib import colors
from sys import platform
from pyproj import Proj


# A few global parameters
dpi      = 300  # Resolution of  
EPSG     = '32611'   # UTM zone for study region (used for projecting data to Cartesian coordinates)
std_max  = 10    # Cutoff for bin standard devations (above are discarded)
 
# The number of fault patches and their discretization is fixed globally to decrease
# the computation time of the forward model/likelihood function.
n_patch     = 20
patch_edges = np.linspace(0, 1, n_patch + 1)

# Configure matplotlib for headless use if run on Linux
if platform != 'darwin':
    matplotlib.use('Agg')


def main():
    """
    Usages:
        python profile_inversion.py                 - perform entire inversion      
        python profile_inversion.py reload          - remake all plots from most recent run
        python profile_inversion.py summary out_dir - remake summary plots for specified run
    """

    # -------------------- Parameters --------------------
    # Machine-specific parameters
    if platform == 'darwin':
        # Mac
        file       = '../../Data/InSAR/S1/horiz_creep_20.grd' 
        fault_file = '../../Data/SSAF_trace.dat'
        n_process  = 12

    else:
        # Linux
        file       = 'horiz_creep_20.grd' 
        fault_file = 'SSAF_trace.dat'
        n_process  = 24

    # Parameters
    bounds        = [-116.11, -115.76,  # Longitude limits
                       33.38,   33.68]  # Latitude limits
    swath_inc     = 1                   # approx. sampled fault segment length for swath selection (km)
    param_inc     = 2                   # approx. sampled fault segment length for parameter estimation (km)
    l             = 10000               # fault perpendicular length (m)
    w             = 500                 # fault parallel width (m) 
    l_bin         = 30                  # Number of fault-perpendicular averaging bins
    x_min         = -l                  # min. x-distance to include (m)
    x_max         = l                   # max. x-distance to include (m)
    norm_EW       = False               # Equally weight both east and west sides of fault
    bin_min       = 5                   # Min. data count in each averaging bin
    bin_mode      = 'log'               # Swath bin spacing - log or linear
    kernel_mode   = 'Gaussian'          # Gaussian, tent, or uniform
    kernel_width  = 2                   # Gaussian filter width (km) if kernel_mode is Gaussian
    max_reg_width = 3                   # Max. distance from node to include (km)
    init_mode     = 'uniform'               # initialization of walkers - uniform or gaussian
    n_walkers     = 10                      # number of walkers in ensemble
    n_step        = 10000                   # number of steps for each walker to take
    moves         = [(emcee.moves.DEMove(), 0.8),         # Choice of walker moves 
                     (emcee.moves.DESnookerMove(), 0.2)]  # emcee default is [emcee.moves.StretchMove(), 1.0]

    # -------------------- 2D fault model --------------------    
    # # 1. Uniform prior (must be defined even if using a Gaussian prior!)
    # prior_mode      = 'uniform' # Type of prior to be used
    # top_dir         = 'Filter_50km/0_Average'
    # top_dir         = 'Filter_50km/1_Unconstrained'
    # v_mode          = 'elliptical'  # elliptical, elliptical_shift, or elliptical_shift_vert
    # fault_dip       = None
    # parallel        = 'swaths'
    # v0_lim          = [0, 10e-3]    # m/yr (positive is right-lateral)
    # D_lim           = [0, 15e3]     # m
    # dip_lim         = [0, 180]      # deg
    # priors_uniform  = {'v0':  v0_lim, 'D': D_lim , 'dip': dip_lim}
    # labels          = [r'$v_0$', 'D', r'$\theta$']
    # units           = ['mm/yr', 'km', 'deg']
    # scales          = [1e3, 1e-3, 1]


    # # 2. Gaussian prior From Run_008
    # prior_mode = 'Gaussian' # Type of prior to be used
    # top_dir    = 'Filter_50km/2_Constrained'
    # v0_val     = [ 3.12e-3,   3*0.10e-3] # m/yr (positive is right-lateral)
    # D_val      = [ 1.97e3,    3*0.13e3 ] # m
    # dip_val    = [83.14,      3*2.10   ] # deg
    # priors_gaussian = {'v0':  v0_val, 'D': D_val, 'dip': dip_val}

    # 3. Vertical fixed geometry
    prior_mode      = 'uniform' # Type of prior to be used
    top_dir         = 'Filter_50km/3_Vertical'
    v_mode          = 'fixed_dip'  
    fault_dip       = np.ones(44)*90
    parallel        = None
    v0_lim          = [0, 10e-3]    # m/yr (positive is right-lateral)
    D_lim           = [0, 15e3]     # m
    priors_uniform  = {'v0':  v0_lim, 'D': D_lim }
    labels          = [r'$v_0$', 'D']
    units           = ['mm/yr', 'km']
    scales          = [1e3, 1e-3]
    
    # -------------------- 2D fault model with shift --------------------    
    # # Uniform prior (must be defined even if using a Gaussian prior!)
    # v_mode        = 'elliptical_shift'  # elliptical, elliptical_shift, or elliptical_shift_vert
    # v0_lim          = [0, 10e-3]    # m/yr (positive is right-lateral)
    # D_lim           = [0, 15e3]     # m
    # dip_lim         = [0, 180]      # deg
    # vc_lim          = [-2e-3, 2e-3] # m
    # priors_uniform  = {'v0':  v0_lim, 'D': D_lim , 'dip': dip_lim, 'vc':  vc_lim}
    # labels          = [r'$v_0$', 'D', r'$\theta$', r'$v_c$']
    # units           = ['mm/yr', 'km', 'deg', 'mm/yr']
    # scales          = [1e3, 1e-3, 1, 1e3]

    # # Gaussian prior
    # # Using results from Run_001
    # v0_val    = [ 2.40e-3,   0.30e-3] # m/yr (positive is right-lateral)
    # D_val     = [ 3.30e3,    0.30e3 ] # m
    # dip_val   = [62.13,      4.03   ] # deg
    # vc_val    = [ 0.66e-3,   0.05e-3] # m

    # Include dip as free parameter
    # priors_gaussian = {'v0':  v0_val, 'D': D_val , 'dip': dip_val, 'vc':  vc_val}
    # labels          = [r'$v_0$', 'D', r'$\theta$', r'$v_c$']
    # units           = ['mm/yr', 'km', 'deg', 'mm/yr']
    # scales          = [1e3, 1e-3, 1, 1e3]

    # # Fixed geometry with velocity shift
    # priors_uniform  = {'v0':  v0_lim, 'D': D_lim , 'vc':  vc_lim}
    # labels          = [r'$v_0$', 'D', r'$v_c$']
    # units           = ['mm/yr', 'km', 'mm/yr']
    # scales          = [1e3, 1e-3, 1e3]

    # -------------------- Shallow and deep slip --------------------
    # priors_uniform  = {'s_0':   [0,     10e-3], 
    #                    'D_s':   [0,     10e3], 
    #                    'dip_s': [0,       180], 
    #                    's_pl':  [10e-3, 30e-3], 
    #                    'D_d':   [10e3,   15e3], 
    #                    'dip_d': [0,       180], 
    #                   }

    # labels = [r'$s_s$', r'$D_s$', r'$\theta_s$', r'$s_d$', r'$D_d$', r'$\theta_d$']
    # units = ['mm/yr', 'km', 'deg', 'mm/yr', 'km', 'deg']
    # scales = np.array([1e3, 1e-3, 1, 1e3, 1e-3, 1])

    # With shift
    # priors_uniform  = {'s_0':   [0,     10e-3], 
    #                    'D_s':   [0,      5e3], 
    #                    'dip_s': [0,       180], 
    #                    's_pl':  [10e-3,     30e-3], 
    #                    'D_d':   [10e3,   15e3], 
    #                    'dip_d': [0,       180], 
    #                    'v_ref': [-1,        1], 
    #                   }

    # labels = [r'$s_s$', r'$D_s$', r'$\theta_s$', r'$s_d$', r'$D_d$', r'$\theta_d$', r'$v_{ref}$']
    # units = ['mm/yr', 'km', 'deg', 'mm/yr', 'km', 'deg', 'mm/yr']
    # scales = np.array([1e3, 1e-3, 1, 1e3, 1e-3, 1, 1e3])

    # Plotting parameters
    vlim = 5
    cmap = 'coolwarm'

    if prior_mode == 'uniform':
        log_prob = log_prob_uniform
        priors   = priors_uniform

    elif prior_mode == 'Gaussian':
        log_prob = log_prob_gaussian
        priors   = priors_gaussian

    else:
        print('Error! Must specify prior mode!')
        return

    # Get plot ticks from parameter bounds/uniform distribution
    ticks = []
    for var, scale in zip(priors.keys(), scales):    
        ticks.append(np.round(np.array([np.min(priors[var]), np.mean(priors[var]), np.max(priors[var])]) * scale))

    # -------------------- BEGIN ANALYSIS -------------------- 
    # Configure top-level parameters
    param_names  = ['file', 'swath_inc', 'param_inc', 'l', 'w', 'l_bin', 'bin_min', 'kernel_mode', 'kernel_width', 'max_reg_width', 'top_dir', 'v_mode', 'priors', 'init_mode', 'n_walkers', 'n_step', 'n_process', 'n_patch', 'std_max', 'moves']  
    param_values = [ file,   swath_inc,   param_inc,   l,   w,   l_bin,   bin_min,   kernel_mode,   kernel_width,   max_reg_width,   top_dir,   v_mode,   priors,   init_mode,   n_walkers,   n_step ,  n_process,   n_patch,   std_max,   moves ]  

    # Load fault trace and add UTM coordinates
    fault = read_fault(fault_file)
    fault = add_utm_coords(fault, EPSG)

    # Interpolate nodes to specified increments 
    s_nodes = fit_fault_spline(fault, bounds, swath_inc, key_str='S')
    f_nodes = fit_fault_spline(fault, bounds, param_inc, key_str='F')
    s_nodes = add_utm_coords(s_nodes, EPSG)
    f_nodes = add_utm_coords(f_nodes, EPSG)

    # Load InSAR dataset
    X, Y, Z, dims = read_grd(file, flatten=True)
    # Z -= np.nanmean(Z)

    # Check mode
    if len(sys.argv) > 1:
        if sys.argv[1] == 'preview':
            mode = sys.argv[1]
            print()
            print('##########################')
            print('# Making preview #')
            print('##########################')

        if sys.argv[1] == 'reload':
            mode = sys.argv[1]
            print()
            print('##########################')
            print('# Reloading previous run #')
            print('##########################')
    else:
        mode = ''

    # Make output directory
    out_dir = prep_out_dir(param_names, param_values, mode, top_dir)

    # Pre-compute profile info to hard-code in probability function
    params   = []
    stds     = []
    X_swaths = []
    Y_swaths = []
    Z_swaths = []

    # Create HDF5 file for saving swath data
    print(out_dir)
    if os.path.isfile(f'{out_dir}/profile_data.h5'):
        os.remove(f'{out_dir}/profile_data.h5')

    h5f = h5py.File(f'{out_dir}/profile_data.h5', 'a')
    h5f.create_group('Fault') 
    h5f.create_group('Kernel') 

    # Construct fine-sampled swaths
    for (key, node) in s_nodes.iterrows():

        # Generate data profile and append to list
        X_p, Y_p, Z_p, X_r, Y_r, node_r = get_swath(X, Y, Z, [node['Longitude'], node['Latitude']], node['Strike'], l=l, w=w)

        X_swaths.append(X_r[~np.isnan(Z_p)] - node_r[0])
        Y_swaths.append(Y_r[~np.isnan(Z_p)] - node_r[1])
        Z_swaths.append(Z_p[~np.isnan(Z_p)])

        # Get bounding coordinates for swath
        bbox_df = get_swath_bbox(node, x_min, x_max, w)

        h5f.create_dataset(f'Swaths/{key}/x',              data=X_swaths[-1])
        h5f.create_dataset(f'Swaths/{key}/y',              data=Y_swaths[-1])
        h5f.create_dataset(f'Swaths/{key}/z',              data=Z_swaths[-1])
        h5f.create_dataset(f'Swaths/{key}/Longitude',      data=node['Longitude'])
        h5f.create_dataset(f'Swaths/{key}/Latitude',       data=node['Latitude'])
        h5f.create_dataset(f'Swaths/{key}/UTMx',           data=node['UTMx'])
        h5f.create_dataset(f'Swaths/{key}/UTMy',           data=node['UTMy'])
        h5f.create_dataset(f'Swaths/{key}/Strike',         data=node['Strike'])
        h5f.create_dataset(f'Swaths/{key}/bbox/Longitude', data=bbox_df['Longitude'].to_numpy())
        h5f.create_dataset(f'Swaths/{key}/bbox/Latitude',  data=bbox_df['Latitude'].to_numpy())
        h5f.create_dataset(f'Swaths/{key}/bbox/UTMx',      data=bbox_df['UTMx'].to_numpy())
        h5f.create_dataset(f'Swaths/{key}/bbox/UTMy',      data=bbox_df['UTMy'].to_numpy())

    # Update swath node DataFrame
    s_nodes['X'] = X_swaths
    s_nodes['Y'] = Y_swaths
    s_nodes['Z'] = Z_swaths

    # Get function for smoothing kernel 
    h5f.create_dataset(f'Kernel/max_reg_width', data=max_reg_width)   

    if kernel_mode == 'Gaussian':
        kernel = get_kernel(kernel_mode, max_reg_width, w=kernel_width)
        h5f.create_dataset(f'Kernel/width', data=width)   
    else:
        kernel = get_kernel(kernel_mode, max_reg_width)

    # Plot smoothing kernel   
    plot_kernel(max_reg_width, kernel, out_dir)

    # Prepare inputs for each profile
    for i, (key, node) in enumerate(f_nodes.iterrows()):
        print()
        print(f'Working on {key}...') 

        # Add fault node information to file
        for col in node.index:
            h5f.create_dataset(f'Fault/{key}/{col}', data=node[col][()])

        # Aggregate data from swaths within allowed regularization kernel
        d0      = node['dist']
        i_swath = (np.abs(s_nodes['dist'] - d0)) <= (max_reg_width + 0.1)

        X0 = s_nodes['X'][i_swath]
        Y0 = s_nodes['Y'][i_swath]
        Z0 = s_nodes['Z'][i_swath]
        D0 = s_nodes['dist'][i_swath] - d0

        # Get mean and STD of binned data  
        x_loc, d_avg, d_std, B, d_total_std, d_total_count = bin_swath(X0, Z0, D0, l, l_bin, bin_min, bin_mode, kernel, norm_EW, x_min=x_min, x_max=x_max) 

        # Save to file
        h5f.create_dataset(f'Fault/{key}/i_swath', data=i_swath) # indicies of subswaths to include in inversion
        h5f.create_dataset(f'Fault/{key}/x_bin',   data=x_loc)   # new x-data after binning
        h5f.create_dataset(f'Fault/{key}/d_avg',   data=d_avg)   # bin means
        h5f.create_dataset(f'Fault/{key}/d_std',   data=d_std)   # bin standard deviations
        h5f.create_dataset(f'Fault/{key}/B',       data=B)       # weights

        # Plot histograms of all and discarded bins
        plot_data_hist(d_total_std, std_max, d_total_count, bin_min, out_dir, key)

        fig, ax = plt.subplots(figsize=(7, 3))

        for x_raw, z_raw in zip(X0, Z0):
            ax.scatter(x_raw/1000, z_raw, s=1, c='lightgray', alpha=0.5, zorder=1)
        
        ax.grid(linewidth=0.25, color='k', zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel('Off-fault distance (km)')
        ax.set_ylabel('Velocity (mm/yr')
        ax.set_xlim(x_min/1000, x_max/1000)
        ax.set_ylim(-10, 10)
        fig.tight_layout()
        fig.savefig(f'{out_dir}/profiles/raw/{key}.png', dpi=500)

        cmap_name  = 'viridis' # Colormap to use
        cbar_label = 'Weight'
        var        = np.diag(B)
        n_seg      = 5
        # ticks      = np.linspace(var.min(), var.max(), n_seg + 1)
        alpha      = 1

        # Create colorbar
        cval  = (var - var.min())/(var.max() - var.min()) # Normalized color values
        cmap  = colors.LinearSegmentedColormap.from_list(cmap_name, plt.get_cmap(cmap_name, 265)(np.linspace(0, 1, 265)), n_seg)
        sm    = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=var.min(), vmax=var.max()))
        
        for j in range(len(x_loc)):
            ax.errorbar(x_loc[j]/1000, d_avg[j], d_std[j], marker='.', c=cmap(cval[j]), alpha=0.2, zorder=2)

        fig.savefig(f'{out_dir}/profiles/downsampled/{key}.png', dpi=500)
        # plt.close()

        # Define inputs for inversion
        x     = x_loc
        d     = d_avg * 1e-3
        S_inv = np.diag((d_std*1e-3)**-2)

        # Determine velocity mode - shallow or deep dislocation and/or constant velocity shift
        if (v_mode == 'fixed_dip') | (v_mode == 'fixed_dip_shift'):
            velocity = choose_velocity(v_mode, priors, dip=fault_dip[i])
        else:
            velocity = choose_velocity(v_mode, priors)

        # Solve for initial state using non-linear least squares
        print('Finding starting model...')

        # Set up helper function for optimization
        nll = lambda *args: -cost_function(*args)

        # Choose inijtial guess (means of priors)
        if prior_mode =='Gaussian':
            initial = np.array([priors[prior][0] for prior in priors.keys()])

        else:
            initial = np.array([np.mean(priors[prior]) for prior in priors.keys()])

        # Optimize function & parse results
        m0 = least_squares(nll, initial, args=(x, d, S_inv, B, velocity), bounds=([priors_uniform[prior][0] for prior in priors_uniform.keys()], [priors_uniform[prior][1] for prior in priors_uniform.keys()])).x

        # Assign node-specific parameters to list
        params.append([mode, top_dir, velocity, x, d, S_inv, B, priors, log_prob, n_walkers, n_step, m0, init_mode, labels, units, scales, X0, Y0, Z0, D0, dims, node_r, vlim, l, w, key, out_dir, parallel, moves])

        x_opt = np.linspace(np.min(x_loc), np.max(x_loc), 100)
        z_opt = velocity(m0, x_opt)
        ax.plot(x_opt/1000, z_opt*1000, c='C0', zorder=3)

        fig.savefig(f'{out_dir}/profiles/optimized/{key}.png', dpi=500)
        plt.close()

    h5f.close()

    if mode == 'preview':
        print('Preview finished.')
        return


    # (B) ---------- Stochastic inversion ----------
    print()
    print('Priors:')
    for i, prior in enumerate(priors):
        print(f'{prior}: {[value * scales[i] for value in priors[prior]]} ({units[i]})')
    print()

    # Start timer
    inv_start = time.time()

    # Do actual inversion
    if parallel == 'swaths':
        # Prevents Emcee from parallelizing linear algebra operations? In any case, it's faster this way.
        os.environ["OMP_NUM_THREADS"] = "1"
        pool    = Pool(processes=n_process)
        results = pool.map(run_inversion, params)
        pool.close()
        pool.join()

    else:
        results = []
        for k in range(len(f_nodes)):
            results.append(run_inversion(params[k]))

    inv_end = time.time() - inv_start

    print()
    if inv_end > 3600:
        print(f'Total time: {inv_end/3600:.2f} hr')
    elif inv_end > 120:
        print(f'Total time: {inv_end/60:.2f} min')
    else:
        print(f'Total time: {inv_end:.1f} s')
    
    # Parse results 
    m_key      = np.array([results[i][0]          for i in range(len(results))])
    m_avg      = np.array([results[i][1] * scales for i in range(len(results))])
    m_std      = np.array([results[i][2] * scales for i in range(len(results))])
    m_q1       = np.array([results[i][3] * scales for i in range(len(results))])
    m_q2       = np.array([results[i][4] * scales for i in range(len(results))])
    m_q3       = np.array([results[i][5] * scales for i in range(len(results))])
    m_rms_avg  = np.array([results[i][6]          for i in range(len(results))])
    m_rms_q2   = np.array([results[i][7]          for i in range(len(results))])


    # Create new results dataframe with node info and summary statistics
    summ_dict = {'Longitude': f_nodes['Longitude'], 'Latitude': f_nodes['Latitude'],
                 'UTMx':      f_nodes['UTMx'],      'UTMy':     f_nodes['UTMy'],        
                 'Strike':    f_nodes['Strike'],    'dist':     f_nodes['dist'],}

    # dict1 = {'v0_avg':  m_avg[:, 0],  'v0_std': m_std[:, 0],  'v0_q1': m_q1[:, 0],  'v0_q2': m_q2[:, 0],  'v0_q3': m_q3[:, 0], 
    #          'D_avg':   m_avg[:, 1],   'D_std': m_std[:, 1],   'D_q1': m_q1[:, 1],   'D_q2': m_q2[:, 1],   'D_q3': m_q3[:, 1], 
    #          'dip_avg': m_avg[:, 2], 'dip_std': m_std[:, 2], 'dip_q1': m_q1[:, 2], 'dip_q2': m_q2[:, 2], 'dip_q3': m_q3[:, 2],}

    # Get statistics for each parameter included in prior distribution
    for i, var in enumerate(priors.keys()):
        keys = [var + key for key in keys] 
        vals = m_avg[:, i], m_std[:, i], m_q1[:, i], m_q2[:, i], m_q3[:, i]
        summ_dict.update(dict(zip(keys, vals)))

    # Include RMS for models using parameter expected values
    summ_dict.update({'RMS_avg': m_rms_avg, 'RMS_q2': m_rms_q2})

    # if len(priors) == 4:
        # summ_dict.update({'vc_avg': m_avg[:, 3], 'vc_std': m_std[:, 3], 'vc_q1': m_q1[:, 3], 'vc_q2': m_q2[:, 3], 'vc_q3': m_q3[:, 3],})

    summary = pd.DataFrame(summ_dict, index=m_key)

    # Save dataframe as text file
    summary.to_csv(f'{out_dir}/statistics.txt', sep=' ')

    # Make paramater plot
    # With mean/std
    plot_params(summary['dist'], m_avg, m_std, m_std, summary['RMS_avg'], 'RMS Error',  'YlOrRd', ticks, labels, units, scales, out_dir, f'param_RMS_means.png')
    plot_params(summary['dist'], m_avg, m_std, m_std, summary['Strike'],  'Strike (deg)', 'YlGn', ticks, labels, units, scales, out_dir, f'param_strike_means.png')

    # With median/quantiles
    plot_params(summary['dist'], m_q2, m_q2 - m_q1, m_q3 - m_q2, summary['RMS_avg'], 'RMS Error',  'YlOrRd', ticks, labels, units, scales, out_dir, f'param_RMS_medians.png')
    plot_params(summary['dist'], m_q2, m_q2 - m_q1, m_q3 - m_q2, summary['Strike'],  'Strike (deg)', 'YlGn', ticks, labels, units, scales, out_dir, f'param_strike_medians.png')

    # Move log file, if created
    if os.path.isfile('log.txt'):
        os.rename('log.txt', f'{out_dir}/log.txt')

    # Copy script to output directory

    script     = os.path.basename(__file__)
    run        = out_dir.split('/')[-1]
    new_script = f'{out_dir}/script_{run}.py'

    shutil.copyfile(script, new_script)

    # Open run directory if on computer
    if platform == 'darwin':
        subprocess.call(['open', '-R', out_dir])

    return


#  ---------- READING ---------- 
def read_fault(fault_file):
    """
    Get DataFrame containing fault trace nodes
    """
    fault = pd.read_csv(fault_file, delim_whitespace=True, header=None)
    fault.columns = ['Longitude', 'Latitude']

    fault['Longitude'] = fault['Longitude'] - 360

    return fault


#  ---------- UTILITIES ---------- 
def get_nodes(fault):
    """
    Given verticies of fault trace, extract nodes for profile inversions.
    """
    nodes = {}
    UTMx      = fault['UTMx'].values
    UTMy      = fault['UTMy'].values

    # Initialize node arrays
    UTMx_n   = np.empty(len(fault) - 1)
    UTMy_n   = np.empty(len(fault) - 1)
    strike_n = np.empty(len(fault) - 1)
    keys_n   = []

    for i in range(len(fault) - 1):
        UTMx_n[i]    = (UTMx[i + 1] + UTMx[i])/2
        UTMy_n[i]    = (UTMy[i + 1] + UTMy[i])/2
        strike_n[i] = 360 + np.arctan((UTMx[i + 1] - UTMx[i])/(UTMy[i + 1] - UTMy[i]))*180/np.pi
        keys_n.append(f'F{i}')

    # Write to dataframe
    nodes = pd.DataFrame({'UTMx': UTMx_n, 'UTMy': UTMy_n, 'Strike': strike_n}, index=keys_n)
    nodes = add_ll_coords(nodes, EPSG)

    # nodes = {}
    # lon      = fault['Longitude'].values
    # lat      = fault['Latitude'].values

    # # Initialize node arrays
    # lon_n    = np.empty(len(fault) - 1)
    # lat_n    = np.empty(len(fault) - 1)
    # strike_n = np.empty(len(fault) - 1)
    # keys_n   = []

    # for i in range(len(fault) - 1):
    #     lon_n[i]    = (lon[i + 1] + lon[i])/2
    #     lat_n[i]    = (lat[i + 1] + lat[i])/2
    #     strike_n[i] = 360 + np.arctan((lon[i + 1] - lon[i])/(lat[i + 1] - lat[i]))*180/np.pi
    #     keys_n.append(f'F{i}')

    # # Write to dataframe
    # nodes = pd.DataFrame({'Longitude': lon_n, 'Latitude': lat_n, 'Strike': strike_n}, index=keys_n)

    return nodes


def get_swath(lon, lat, z, node, strike, **kwargs):
    """ 
    INPUT:
    lon    - list of longitude coordinates of data
    lat    - list of latitude coordinates of data
    z      - list of deformation data
    node   - location of profile center (lon, lat)
    strike - local strike at profile center (orthogonal to profile)
    
    (Note: lon, lat, and z should be mutually indexed)

    Optional: (keyword args)
    l - maximum profile half-length (fault-perpendicular direction)
    w - maximum profile width (along-strike)
    
    OUTPUT:
    x_p, y_p, z_p - subset of above x, y, z which correspond to the selected profile
    """

    n = len(lon)

    # Set profile dimensions
    if 'l' in kwargs.keys():
        l = kwargs['l']
    else:
        l = 1500

    if 'w' in kwargs.keys():
        w = kwargs['w']
    else:
        w = 500  
              
    # Set geographic to metric conversion factors
    lon_inc = 111.19e3
    lat_inc = 111.19e3

    # Convert geographic datasets to m
    lon_m  = lon * lon_inc
    lat_m  = lat * lat_inc
    node_m = [node[0] * lon_inc, node[1] * lat_inc]

    # Rotate data by fault strike
    lon_r = np.zeros(lon_m.shape)
    lat_r = np.zeros(lat_m.shape)
    a     = strike * np.pi/180

    # print('Projecting...')
    node_r = rotate(node_m[0], node_m[1], a)
    lon_r, lat_r = rotate(lon_m, lat_m, a)

    # calculate distances
    dl = ((lon_r - node_r[0])**2)*µ*0.5
    dw = ((lat_r - node_r[1])**2)**0.5

    # Prepare outputs
    lon_p = lon_m/lon_inc
    lat_p = lat_m/lat_inc
    z_p   = np.copy(z)

    # Nan-out pixels outside of profile
    z_p[(dl > l) | (dw > w)] = np.nan
    # z_p[(dl > l[-1]) | (dl > l[0]) | (dw > w)] = np.nan

    return lon_p, lat_p, z_p, lon_r, lat_r, node_r


def synth_velocity_profile(v_func, v_args, x_range, n, noise_amp):
    """
    Compute synthetic velocity model
    
    INPUT
    v_func    - handle for velocity model function to use
    v_args    - slip, locking depth, and dip for velocity model (3,)
    x_range   - range of x-values to randomly sample from (2,)
    n         - number of x samples to draw
    noise_amp - amplitude of Gaussian noise to introduce to synthetic velocity 
    
    OUTPUT:
    x       - coordinates of synthetic data
    v_clean - synthetic velocity profile
    v_noise - synthetic velocity profile with noise introduced

    """
    # Generate random sample of locations on xmin to xmax
    x = np.sort(np.random.uniform(x_range[0], x_range[1], n))
    
    # Compute velocities
    v_clean = v_func(v_args, x)
    
    # Introduce random Gaussian noise
    v_noise = v_clean + noise_amp * np.random.normal(size=n)
    
    return x, v_clean, v_noise


def calc_fault_dist(nodes):

    dist = np.empty(len(nodes))

    for i in range(len(nodes)):
        if i == 0:
            dist[i] = 0
        else:
            # dx = (nodes.iloc[i]['Longitude'] - nodes.iloc[i - 1]['Longitude'])*111.19
            # dy = (nodes.iloc[i]['Latitude'] - nodes.iloc[i - 1]['Latitude'])*111.19
            # dist[i] = (dx**2 + dy**2)**0.5 + dist[i - 1]
            dx = (nodes['UTMx'].iloc[i] - nodes['UTMx'].iloc[i - 1])
            dy = (nodes['UTMy'].iloc[i] - nodes['UTMy'].iloc[i - 1])
            dist[i] = (dx**2 + dy**2)**0.5 + dist[i - 1]

    nodes['dist'] = dist/1000

    return nodes


def choose_velocity(v_mode, priors, **kwargs):
    """
    Configure velocity model.
    """

    if v_mode == 'shallow':
        if 'v0' in priors.keys() and 'v0' in priors.keys() and 'dip' in priors.keys():
            global velocity
            if len(priors) == 3:
                velocity = v_shallow
            elif len(priors) == 4:
                velocity = v_shallow_shift
        else:
            print('Error: ')
            print(f'Number of priors: {len(priors)}')
            for key in priors.keys():
                print(f'{key}: {priors[key]}')
    elif v_mode == 'deep':
        velocity = v_deep

    elif v_mode == 'elliptical':
        velocity = v_elliptical

    elif v_mode == 'elliptical_shift':
        velocity = v_elliptical_shift

    elif v_mode == 'elliptical_shift_vert':
        velocity = v_elliptical_shift_vert

    elif v_mode == 'combined':
        velocity = v_combined

    elif v_mode == 'combined_shift':
        velocity = v_combined_shift

    elif v_mode == 'fixed_dip':
        # Get node key
        dip      = kwargs['dip']
        velocity = lambda m, x: v_fixed_dip(m, x, dip)

    elif v_mode == 'fixed_dip_shift':
        # Get node key
        dip      = kwargs['dip']
        velocity = lambda m, x: v_fixed_dip_shift(m, x, dip)

    return velocity


def filter_MA(x, z, width):
    """
    Perform moving-average filter of given width.

    INPUT:
    x     - x coordinates of data
    z     - data values
    width - (one sided) width of filter (m)

    OUTPUT:
    x_filt - coordinates of filtered data
    z_filt - filtered data
    """

    # Remove nans
    x_real = x[~np.isnan(z)]
    z_real = z[~np.isnan(z)]

    # Sort west-to-east
    x_real, z_real = (list(t) for t in zip(*sorted(zip(x_real, z_real))))

    # Perform moving average over filter width
    x_real = np.array(x_real)
    z_real = np.array(z_real)

    # Split into west and east with respect to the fault
    x_real_w = x_real[x_real > 0]
    z_real_w = z_real[x_real > 0]
    x_real_e = x_real[x_real <= 0]
    z_real_e = z_real[x_real <= 0]

    x_filt_w = x_real_w
    z_filt_w = np.empty(len(z_real_w))
    x_filt_e = x_real_e
    z_filt_e = np.empty(len(z_real_e))

    # Perform averaging
    for i in range(len(x_filt_w)):
        dr = abs(x_real_w[i] - x_real_w)
        stencil = z_real_w[(dr <= width)]
        z_filt_w[i] = sum(stencil)/len(stencil)

    for i in range(len(x_filt_e)):
        dr = abs(x_real_e[i] - x_real_e)
        stencil = z_real_e[(dr <= width)]
        z_filt_e[i] = sum(stencil)/len(stencil)

    # Re-merge
    x_filt = np.concatenate((x_filt_w, x_filt_e))
    z_filt = np.concatenate((z_filt_w, z_filt_e))
    return x_filt, z_filt


def fit_fault_spline(fault, bounds, seg_inc, **kwargs):
    """
    Fit linear spline with segments of approximately seg_inc length (km) to fault trace 
    """

    # Define approx. increment for upsampling fault trade
    upsamp_inc = 10

    # Select nodes within specified bounds
    fault_sel  = fault[(fault['Longitude'] >= bounds[0]) & (fault['Longitude'] <= bounds[1]) & (fault['Latitude'] >= bounds[2]) & (fault['Latitude'] <= bounds[3])]

    # Estimate average strike from selected portion of fault 
    p          = np.polyfit(fault_sel['UTMx'], fault_sel['UTMy'], 1)
    strike_avg = np.arctan(p[0])

    # Get approximate spline segment length from average fault strike
    # upsamp_deg   = abs(upsamp_inc * np.sin(strike_avg)/111.19)
    
    # Make spline function
    spl_upsamp    = interp1d(fault_sel['UTMx'], fault_sel['UTMy'])
    
    # Define upsampled longitude coordinates and interpolate latitude coordinates
    lon_upsamp   = np.arange(fault_sel['UTMx'].min(), fault_sel['UTMx'].max(), upsamp_inc)
    lat_upsamp   = spl_upsamp(lon_upsamp)
    fault_upsamp = pd.DataFrame({'UTMx': lon_upsamp[::-1], 'UTMy': lat_upsamp[::-1]})
    fault_upsamp = add_ll_coords(fault_upsamp, EPSG)

    # Calulate strikes from upsampled fault trace
    nodes_upsamp = get_nodes(fault_upsamp)

    # Calculate along strike distance
    nodes_upsamp = calc_fault_dist(nodes_upsamp)    

    # Get approx. fault length 
    f_length = nodes_upsamp['dist'].max()//1

    # Get desired node locations
    d_target = np.arange(0, f_length, seg_inc)
    # d_ind = np.empty(len(d_target))

    # Get index of best-approximation of d
    keys = [nodes_upsamp.iloc[np.argmin(abs(nodes_upsamp['dist'].values - d))].name for i, d in enumerate(d_target)]

    # nodes_spl = nodes_upsamp.iloc[i]
    if 'key_str' in kwargs.keys():
        key_str = kwargs['key_str']
    else:
        key_str = 'F'

    nodes_spl = nodes_upsamp.loc[keys, nodes_upsamp.columns]
    nodes_spl['new_index'] = [f'{key_str}{i}' for i in range(len(nodes_spl))]
    nodes_spl.set_index('new_index', drop=True, inplace=True)

    # Check
    # plt.plot(fault['Longitude'], fault['Latitude'], nodes_upsamp['Longitude'], nodes_upsamp['Latitude'], nodes_spl['Longitude'], nodes_spl['Latitude'], '-o')
    # plt.show()

    return nodes_spl


def gauss_kernel(x, s): 
    """
    Define Gaussian function for weights
    """
    return np.exp(-0.5 * (x/s)**2)


def get_kernel(kernel_mode, max_reg_width, **kwargs):
    """
    Get kernel for weighting sub-swaths along-strike.

    INPUT:
    kernel_mode   - Gaussian, tent, or uniform. 
    max_reg_width - Max. distance from node to include (km) 

    Optional arguments:
    w - width of Gaussian (km)
    m - slope 
        Default for both is 1

    OUTPUT:
    kernel - handle for specified eighting function.
    """

    if kernel_mode == 'Gaussian':
        # Get width
        if 'w' in kwargs.keys():
            w = kwargs['w']
        else:
            w = 1
        return lambda x: np.exp(-0.5 * (x/w)**2)

    elif kernel_mode == 'tent':
        return lambda x: 1 - x * np.sign(x)

    elif kernel_mode == 'uniform':
        return lambda x: np.ones_like(x)

    else:
        print('Error! kernel_mode must be Gaussian, tent, or uniform!')
        return


def bin_swath(X, Z, D, l, l_bin, bin_min, bin_mode, kernel, norm_EW, **kwargs):
    """
    INPUT:
    X        - horizontal coordinates of data 
    Z        - data values
    D        - along strike relative node distances
    l        - fault-perpendicular length
    w        - fault-parallel width
    l_bin    - number of bins in fault perpendicular and fault parallel directions
    bin_min  - minimum number of data points to require for each bin
    bin_mode - bin spacing - linear or log
    norm_EW  - Normalize weighting of East and West sides of fault

    OUTPUT:
    X_loc         - locations of new binned X-data
    Z_avg         - average value of data in  each retainedbin 
    Z_std         - standard deviation of each retainedbin
    B             - weights 
    Z_total_std   - standard deviation of all bins
    Z_total_count - number of data points in each bin
    """

    X_loc         = []
    Z_avg         = []
    Z_std         = []
    B             = []
    Z_total_std   = []
    Z_total_count = []

    for (x, z, d) in zip(X, Z, D):
        # Remove nans
        x_real = x[~np.isnan(z)]
        z_real = z[~np.isnan(z)]

        # Define along-strike bins
        if bin_mode == 'log':
            # Get log spacing
            x_edge_inc = np.logspace(-2, 0, l_bin//2 + 1, base=np.exp(1))

            # Reference to zero
            x_edge_inc -= x_edge_inc[0]

            # Rescale to appropriate length
            x_edge_inc *= l/x_edge_inc[-1]
            x_edge = np.hstack((-x_edge_inc[::-1], x_edge_inc[1:]))

        else:
            x_edge = np.linspace(-l, l, l_bin + 1)

        # Get bin centers
        x_loc  = np.array([np.mean([x_edge[i], x_edge[i + 1]]) for i in range(l_bin)]) # Middle of bin

        # Initialize output arrays
        z_avg         = np.empty(l_bin)
        z_std         = np.empty(l_bin)
        z_total_std   = np.empty(l_bin)
        z_total_count = np.empty(l_bin)

        # Prescribe weights along strike and initialize E/W side weights
        b_x = np.ones(l_bin)
        b_y = kernel(d)

        # Get fault-perpendicular bin values
        for i in range(l_bin):

            # Get all data points in fault-perpendiular bin
            x_bin = x_real[(x_real > x_edge[i]) & (x_real <= x_edge[i + 1])]
            z_bin = z_real[(x_real > x_edge[i]) & (x_real <= x_edge[i + 1])]

            # Check STD of bin
            if len(z_bin) > 0:
                std0 = np.nanstd(z_bin)
            else:
                std0 = np.nan

            z_total_std[i]   = std0
            z_total_count[i] = len(z_bin)

            # Keep if bin is robust, discard if not
            if len(z_bin) >= bin_min and std0 <= std_max:
                # Regular average
                # z_avg[i] = np.nanmean(z_bin)

                # Weighted by distance from x_loc
                z_avg[i] = np.sum(x_bin * z_bin)/np.sum(x_bin)
                z_std[i] = std0
            else:
                z_avg[i]   = np.nan
                z_std[i]   = np.nan

            # If specified, discard bins outside of bounds 
            if 'x_min' in kwargs.keys():
                x_min = kwargs['x_min']
                if x_loc[i] < x_min:
                    z_avg[i]   = np.nan
                    z_std[i]   = np.nan

            if 'x_max' in kwargs.keys():
                x_max = kwargs['x_max']
                if x_loc[i] > x_max:
                    z_avg[i]   = np.nan
                    z_std[i]   = np.nan


        # # Normalize east and west sides of fault
        if norm_EW ==True:
            bin_real = len(x_loc[~np.isnan(z_avg)])
            i_e = x_loc[~np.isnan(z_avg)] > 0
            i_w = x_loc[~np.isnan(z_avg)] <= 0
            # i_0 = x_loc[~np.isnan(z_avg)] == 0

            n_e = sum(i_e)
            n_w = sum(i_w)
            # n_0 = sum(i_0)

            if n_e == 0 or n_w == 0:
                b_x[x_loc > 0]  = 0 
                b_x[x_loc <=0]  = 0 
                # b_x[x_loc == 0] = 0 

            else:        
                b_x[x_loc > 0]  = bin_real/n_e
                b_x[x_loc <= 0]  = bin_real/n_w
                # b_x[x_loc == 0] = bin_real/n_0

        # Apply to weight matrix
        b = b_y * b_x

        # Make diagonal matrix from weights
        i_real = ~np.isnan(z_avg.flatten())

        X_loc.extend(list(x_loc[i_real]))
        Z_avg.extend(list(z_avg[i_real]))
        Z_std.extend(list(z_std[i_real]))
        B.extend(list(b[i_real]))
        Z_total_std.extend(list(z_total_std))
        Z_total_count.extend(list(z_total_count))

    X_loc         = np.array(X_loc)
    Z_avg         = np.array(Z_avg)
    Z_std         = np.array(Z_std)
    B             = np.diag(B)
    Z_total_std   = np.array(Z_total_std)
    Z_total_count = np.array(Z_total_count)

    return X_loc, Z_avg, Z_std, B, Z_total_std, Z_total_count


def extend_line(x_def, z_def, coord_dict):
    """
    Given defining coordinate pair x_def, z_def, extend line to specified coord_dict
    """
    m = (z_def[1] - z_def[0])/(x_def[1] - x_def[0])
    
    if 'x' in coord_dict.keys():
        return m * np.array(coord_dict['x'])
        
    elif 'z' in coord_dict.keys():
        return np.array(coord_dict['z'])/m
    else:
        print('x_new or y_new must be specified!')
        return


def load_fault_results(result_file, H, EPSG):
    """
    Load inversion output file to DataFrame containing pre-computed dislocation geometries.

    result_file - path to inversion output file (usually named  'statistics.txt')
    H           - extrapolation depth (km)
    """

    # Return rotated points
    rot = lambda  x, y, theta: np.array([[np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))], [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))]]) @ np.array([x, y])

    # Load results and add UTM projected coordinates
    results = pd.read_csv(result_file, header=0, delim_whitespace=True) 
    results = add_utm_coords(results, EPSG)
    results = results.set_index('Key')

    # Construct dataframe containing line segment geometries
    subindex  = ['D_low', 'D_avg', 'D_high', 'H_low', 'H_avg', 'H_high']
    i_a      = np.repeat(results.index.values, len(subindex))
    i_b      = np.tile(subindex, len(results))
    geometry = pd.DataFrame(columns=['Longitude', 'Latitude', 'Depth', 'UTMx', 'UTMy'], index=[i_a, i_b])

    for (key, node) in results.iterrows():
        d       = node['D_avg']
        d_std   = node['D_std']
        dip     = node['dip_avg']
        dip_std = node['dip_std']
        strike  = node['Strike']

        z_rng   = np.array([  d +   d_std,   d,   d -   d_std,             H,   H,             H])
        dip_rng = np.array([dip + dip_std, dip, dip - dip_std, dip + dip_std, dip, dip - dip_std])

        # Calculate fault-perpendicular offset
        # r_rng = np.array([z0 * np.cos(np.deg2rad(dip0)) for z0, dip0 in zip(z_rng, dip_rng)])*1000
        r_rng = np.array([z0/np.tan(np.deg2rad(dip0)) for z0, dip0 in zip(z_rng, dip_rng)])*1000

        # Change from fault-perpendicular coordinates to geographic coordinates
        # lon_0 = np.tile(node['Longitude'], 6)
        # lat_0 = np.tile(node['Latitude'], 6)
        # lon_z, lat_z = zip(*[rot(r0, 0, -strike) for r0 in r_rng])
        UTMx_0 = np.tile(node['UTMx'], 6)
        UTMy_0 = np.tile(node['UTMy'], 6)
        UTMx_z, UTMy_z = zip(*[rot(r0, 0, -strike) for r0 in r_rng])

        UTMx_z += UTMx_0 
        UTMy_z += UTMy_0

        # Project to lon/lat
        lon_0, lat_0 = proj_ll2utm(UTMx_0, UTMy_0, EPSG, inverse=True)
        lon_z, lat_z = proj_ll2utm(UTMx_z, UTMy_z, EPSG, inverse=True)

        # Add to dataframe and convert units
        for i, idx in enumerate(subindex):
            geometry.loc[(key, idx), 'UTMx'] = [UTMx_0[i], UTMx_z[i]]
            geometry.loc[(key, idx), 'UTMy'] = [UTMy_0[i], UTMy_z[i]]

            geometry.loc[(key, idx), 'Longitude'] = [lon_0[i], lon_z[i]]
            geometry.loc[(key, idx), 'Latitude']  = [lat_0[i], lat_z[i]]

            # geometry.loc[(key, idx), 'Longitude'] = [lon_0[i], lon_z[i]/111.19 + lon_0[i]]
            # geometry.loc[(key, idx), 'Latitude']  = [lat_0[i], lat_z[i]/111.19 + lat_0[i]]

            geometry.loc[(key, idx), 'Depth']     = [0, z_rng[i]]
        
    return results, geometry


def get_swath_bbox(node, x_min, x_max, w):
    """
    Given fault node and swath dimensions, get bounding coordinate

    INPUT:
    x_min - minimum fault-perpendicular distance
    x_max - maximum fault-perpendicular distance
    width - fault-parallel swath width

    OUTPUT:
    bbox - DataFrame containing UTM and geographic coordinates of swath bounding box
    """

    x = node['UTMx'] # East
    y = node['UTMy'] # North
    a = node['Strike'] * np.pi/180 # Strike
    p0 = np.array([x, y]) 
    
    # Rotate l/w and add to reference coords
    p1 = np.array(rotate(x_max, w,  -a)) + p0
    p2 = np.array(rotate(x_max, -w, -a)) + p0
    p3 = np.array(rotate(x_min, -w, -a)) + p0
    p4 = np.array(rotate(x_min, w,  -a)) + p0

    bbox = np.array([p1, p2, p3, p4, p1])
    
    bbox_df = pd.DataFrame({'UTMx': bbox[:, 0], 'UTMy': bbox[:, 1]})
    bbox_df = add_ll_coords(bbox_df, EPSG)

    return bbox_df


# ---------- INVERSION ----------
def cost_function(m, x, d, S_inv, B, velocity):
    """
    Cost function for initial optimization step (modified version of log. likelihood
    INPUT:
    m        - model parameters
    x        - data locations
    d        - data values
    S_inv    - inverse ovariance matrix
    velocity - function handle for velocity model

    OUTPUT:
    log[p(d|m)] - log likelihood of d given m 
    """
    # Parse parameters
    # v0, D, dip = m

    # Make forward model calculation
    G_m = velocity(m, x)

    return np.hstack((0.5**-0.5) * S_inv @ (G_m - d))


def log_prob_uniform(m, x, d, S_inv, B, velocity, priors):
    """
    Determine log-probaility of model m using a uniform prior
    """

    # Check prior
    if np.all([priors[key][0] <= m[i] <= priors[key][1] for i, key in enumerate(priors.keys())]):
        return log_likelihood(velocity(m, x), d, S_inv, B) # Log. probability of sample is only the log. likelihood
    else:
        return -np.inf                                    # Exclude realizations outside of priors

    # if not np.isfinite(lp):
    #     return -np.inf

    # return log_likelihood(velocity(m, x), d, S_inv, B)


def log_prob_gaussian(m, x, d, S_inv, B, velocity, priors):
    """
    Determine log-probability of model m using a gaussian prior
    """

    # Evaluate prior
    mu    = np.array([priors[key][0] for key in priors.keys()])
    sigma = np.diag(np.array([priors[key][1] for key in priors.keys()])**-2)

    return log_likelihood(velocity(m, x), d, S_inv, B) + log_prior_gaussian(m, mu, sigma) # Log. probability is sum of log. likelihood and log. prior
    
    # if not np.isfinite(lp):
    #     return -np.inf

    # return log_likelihood(velocity(m, x), d, S_inv, B)


def run_hammer(x, d, S_inv, B, velocity, priors, log_prob, n_walkers, n_step, m0, init_mode, labels, units, scales, key, parallel, backend, moves):
    """
    Perform ensemble sampling using the MCMC Hammer.
    """
    print()
    print(f'Performing ensemble sampling for {key}...')

    n_dim = len(priors)

    if parallel == 'swaths':
        progress = False
    else:
        progress = True

    # Initialize walkers around MLE
    # Gaussian ball
    if init_mode == 'gaussian':
        b_size = 1e-1
        pos = m0 + b_size * m0 * np.random.randn(n_walkers, n_dim)

    # Uniform over priors
    elif init_mode == 'uniform':
        pos   = np.empty((n_walkers, n_dim))
        for i in range(n_walkers):
            for j, prior in enumerate(priors.keys()):
                pos[i, j]  = np.random.uniform(low=priors[prior][0], high=priors[prior][1])

    # Run ensemble sampler
    s_start = time.time()

    if parallel == 'sampling':
        print('Parallelizing sampling...')
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob, args=(x, d, S_inv, B, velocity, priors), backend=backend, pool=pool, moves=moves)
            sampler.run_mcmc(pos, n_step, progress=progress)
    else:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob, args=(x, d, S_inv, B, velocity, priors), backend=backend)
        sampler.run_mcmc(pos, n_step, progress=progress)

    s_end   = time.time() - s_start

    if s_end > 120:
        print(f'{key} took {s_end/60:.2f} min')
    else:
        print(f'{key} took {s_end:.2f} s')

    # Get autocorrelation times
    autocorr = sampler.get_autocorr_time(tol=0)
    print(f'Autocorrelation times: {autocorr}')

    # Flatten chain and thin based off of autocorrelation times
    discard = int(2 * np.nanmax(autocorr))
    thin    = int(0.5 * np.nanmin(autocorr))

    print(f'Burn-in:  {discard} samples')
    print(f'Thinning: {thin} samples')
    print()

    # Get samples
    samples      = sampler.get_chain()
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    samp_prob    = sampler.get_log_prob()

    return samples, samp_prob, autocorr, discard, thin


def reload_hammer(result_file):
    # Get mean and standard deviation of posterior distr
    # Load samples
    sampler = emcee.backends.HDFBackend(result_file)

    # Get autocorrelation times
    autocorr = sampler.get_autocorr_time(tol=0)

    # Flatten chain based off of autocorrelation times (rounded to nearest power of 10 * 3)
    discard = int(10**np.ceil(np.log10(np.nanmax(autocorr)))) * 3
    thin    = int(np.nanmax(autocorr)//2)

    # Get samples
    samples       = sampler.get_chain()
    flat_samples  = sampler.get_chain(discard=discard, thin=thin, flat=True)
    samp_prob     = sampler.get_log_prob()

    return samples, samp_prob, autocorr, discard, thin


def run_inversion(params):

    mode, top_dir, velocity, x, d, S_inv, B, priors, log_prob, n_walkers, n_step, m0, init_mode, labels, units, scales, X0, Y0, Z0, D0, dims, node_r, vlim, l, w, key, out_dir, parallel, moves = params

    # Perform ensemble sampling with Emcee or load most recent results
    if mode == 'reload':
        print(f'Reloading {key}...')
        result_file = out_dir + f'/results/{key}_Results.h5'
        samples, samp_prob, autocorr, discard, thin = reload_hammer(result_file)
    else:
        # Set up backend for storing MCMC results
        backend = config_backend(out_dir, key, n_walkers, len(priors))
        samples, samp_prob, autocorr, discard, thin = run_hammer(x, d, S_inv, B, velocity, priors, log_prob, n_walkers, n_step, m0, init_mode, labels, units, scales, key, parallel, backend, moves)

    # Assess consistence of convergence amongst the ensemble members
    samp_prob[np.abs(samp_prob) == np.inf] = np.nan   # Correct for infinite values
    mean_chain_prob = np.nanmean(samp_prob, axis=0)   # Get average prob. for each walker
    std_chain_prob  = np.nanstd(mean_chain_prob)      # Get std of walker means
    mean_prob       = np.nanmean(samp_prob.flatten()) # Get total average
    std_prob        = np.nanstd(samp_prob.flatten())  # Get total STD

    # Discard "lost" walkers
    samples         = samples[:, abs(mean_chain_prob - mean_prob) <= std_prob]   
    samp_prob       = samp_prob[:, abs(mean_chain_prob - mean_prob) <= std_prob] 
    flat_samples    = samples[discard::thin, :, :].reshape(len(samples[discard::thin, 0, 0])*len(samples[0, :, 0]), len(samples[0, 0, :]))
    discard_walkers = n_walkers - samples.shape[1]

    print(f'({key}) Average log(p(m|d)) = {mean_prob} +/- {std_prob}')
    print(f'({key}) Chain  log(p(m|d))  = {mean_chain_prob} +/- {std_chain_prob}')
    print(f'({key}) Number of discarded ensemble members = {discard_walkers}')
    print(f'({key}) Number of effective samples = {len(flat_samples)}')
    
    # Compute mean and standard deviation of flat samples
    m_avg = np.mean(flat_samples,           axis=0)
    m_std = np.std(flat_samples,            axis=0)
    m_q1  = np.quantile(flat_samples, 0.16, axis=0)
    m_q2  = np.quantile(flat_samples, 0.50, axis=0)
    m_q3  = np.quantile(flat_samples, 0.84, axis=0)

    # Compute RMSE for representative models
    m_rms_avg  = wRMSE(velocity(m_avg, x), d, S_inv, B)
    m_rms_q2   = wRMSE(velocity(m_q2, x),  d, S_inv, B)

    # Plot Markov chains for each parameter
    plot_chains(samples, samp_prob, discard, labels, units, scales, key, out_dir)

    # Plot parameter marginals and correlations
    plot_triangle(flat_samples, priors, labels, units, scales, key, out_dir)

    # Plot everything
    plot_profile(X0, Y0, Z0, D0, l, w, node_r, key, [x, d], S_inv, B, stats=[m_avg, m_std, velocity, labels, units, scales], samples=flat_samples, vlim=vlim, file_name=f'{out_dir}/profiles/modeled/{key}')

    # Plot raw data
    # plot_profile(X0, Y0, Z0, D0, l, w, node_r, key, [[], []], S_inv, B, vlim=vlim, file_name=f'{out_dir}/profiles_raw/{key}_profile')

    # Plot smoothed data and raw data
    # plot_profile(X0, Y0, Z0, D0, l, w, node_r, key, [x, d], S_inv, B, vlim=vlim, stats=[m_avg, m_std, velocity, labels, units, scales], file_name=f'{out_dir}/profiles_smooth/{key}_profile')

    return key, m_avg, m_std, m_q1, m_q2, m_q3, m_rms_avg, m_rms_q2


# ---------- PLOTS ---------- 
def plot_profile_simple():

    return


def plot_profile(X, Y, Z, D, l, w, node, key, avg, S_inv, B, **kwargs):
    """
    Plot profile in map view anr/or cross-section view
    """

    # Define labels 
    xlabel = 'Fault-perpendicular distance (m)'
    ylabel = 'Along-strike dist. (m)'
    zlabel = 'Velocity (mm/yr)'

    # Initialize figure    
    fig = plt.figure(figsize=(6, 4), constrained_layout=False)
    fig.suptitle(key)
    axd = fig.subplot_mosaic(
        """
        B
        D
        """,
        gridspec_kw={'height_ratios': [1, 2], 'hspace': 0.1}, )
        # 'width_ratios': [1, 5], 'wspace': 0.075

    fontsize = 6

    # Fix axis limits
    for axkey in ['B', 'D']:
        axd[axkey].set_xlim([-l, l])

    # for key in ['A', 'B']:
    axd['B'].set_ylim([-w - w/10, w + w/10])


    # B) ---------- Plot map ----------
    if 'vlim' in kwargs.keys():
        vlim = kwargs['vlim']
        for i in range(len(Z)):
            im0 = axd['B'].scatter(X.iloc[i], Y.iloc[i], c=Z.iloc[i], s=2, marker='.', cmap='coolwarm', vmin=-vlim, vmax=vlim, alpha=0.5)
    else:
        for i in range(len(Z)):
            im0 = axd['B'].scatter(X.iloc[i], Y.iloc[i], c=Z.iloc[i], s=3, marker='.', cmap='coolwarm')

    # Plot fault trace
    im1 = axd['B'].plot([0, 0], [-w, w], '--k')

    # Axes
    axd['B'].set_xticklabels([])
    axd['B'].tick_params(direction='in')

    # Colorbar
    divider = make_axes_locatable(axd['B'])
    cax     = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im0, cax=cax, label=zlabel)


    # D) ---------- Plot profile ----------
    divider = make_axes_locatable(axd['D'])
    cax     = divider.append_axes('right', size='5%', pad=0.05)

    # Plot raw data
    c     = 'grey'
    alpha = 0.1

    if len(avg[0]) == 0:
        cax.set_axis_off()

    for i in range(len(Z)):
        im2 = axd['D'].scatter(X.iloc[i], Z.iloc[i], c=c, s=1, marker='.', zorder=0, alpha=alpha)

    # Plot downsampled data
    x_avg = avg[0]
    z_avg = avg[1]

    if len(x_avg) > 0:
        x_samp = np.linspace(min(x_avg), max(x_avg), 1000)

    # Set colorbar parameters
    n_ticks    = 5
    n_seg      = 20
    vmin       = 0
    vmax       = 1
    cval       = (np.diag(B) - vmin)/abs(vmax - vmin)
    cmap_name  = 'GnBu'
    cbar_label = 'Weight'

    # Take care of colorbar 
    # ticks    = 
    cmap     = plt.get_cmap(cmap_name)
    crange   = np.sort(np.linspace(0, vmax - vmin, n_seg)/abs(vmax - vmin))
    cmap_lin = colors.LinearSegmentedColormap.from_list(cmap_name, cmap(crange), n_seg)
    sm       = plt.cm.ScalarMappable(cmap=cmap_lin, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A    = []

    z_sigma = np.diag(S_inv)**-0.5

    for i in range(len(x_avg)):
        axd['D'].errorbar(x_avg[i], z_avg[i] * 1e3, yerr=z_sigma[i] * 1e3, marker='.', color=cmap_lin(cval[i]), markersize=3, linestyle='', linewidth=0.5, capsize=0, zorder=1 + cval[i])

    if len(x_avg) > 0:
        plt.colorbar(sm, cax=cax, label=cbar_label)

    # Plot posterior stats if given
    if 'stats' in kwargs.keys():
        m_mean   = kwargs['stats'][0]
        m_std    = kwargs['stats'][1]
        velocity = kwargs['stats'][2]
        labels   = kwargs['stats'][3]
        units    = kwargs['stats'][4]
        scales   = kwargs['stats'][5]
        textstr  = ''

        for i in range(len(m_mean)):
            textstr += '{} = {:.2f} '.format(labels[i], m_mean[i]*scales[i]) + r'$\pm$ {:.2f} '.format(m_std[i]*scales[i]) + '({})'.format(units[i])
            if i != len(m_mean) - 1:
                textstr += '\n'

        props = dict(edgecolor='k', facecolor='w', alpha=0.85)
       
        if len(m_mean) == 4:
            yloc = 0.385
        else:
            yloc = 0.265

        axd['D'].text(0.03, yloc, textstr, transform=axd['D'].transAxes, fontname='monospace', fontsize=6, verticalalignment='top', bbox=props, linespacing=2)


    # Plot ensemble of models from Markov chain
    if 'samples' in kwargs.keys():
        samples = kwargs['samples']
        inds = np.random.randint(len(samples), size=100)

        for ind in inds:
            v   = velocity(samples[ind], x_samp)
            axd['D'].plot(x_samp, v*scales[0], 'C1', alpha=0.1, zorder=len(cval) + 1)

    # Fix vertical axis
    if 'vlim' in kwargs.keys():
        vlim = kwargs['vlim']
        axd['D'].set_ylim([-vlim, vlim])

    # Figure settings
    axd['D'].tick_params(direction='in')
    axd['D'].set_xlabel(xlabel, fontsize=fontsize)
    axd['D'].set_ylabel(zlabel, fontsize=fontsize)

    if 'file_name' in kwargs.keys():
        file_name = kwargs['file_name'] + '.png'
        plt.savefig(file_name, dpi=dpi)
        plt.close()

    else:
        plt.show()

    return


def plot_overview(x, y, z, fault, **kwargs):

    if 'cmap' in kwargs.keys():
        cmap = kwargs['cmap']
    else:
        cmap = 'coolwarm'
    

    extent = [np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y)]

    fig, ax = plt.subplots(figsize=(13, 8))

    # Plot data
    if 'vlim' in kwargs.keys():
        vlim = kwargs['vlim']
        im = ax.imshow(z[::-1, :], cmap=cmap, extent=extent, vmin=-vlim, vmax=vlim)
    else:
        im = ax.imshow(z[::-1, :], cmap=cmap, extent=extent)

    # Plot fault
    ax.plot(fault['Longitude'], fault['Latitude'], 'k-')
    
    # Plot fault parameter
    if 'param' in kwargs.keys():
        param = kwargs['param']

        # Set plot parameters
        n_ticks    = 5
        n_seg      = 36
        vmin       = param[4]
        vmax       = param[5]
        cval       = (param[2] - vmin)/abs(vmax - vmin)
        cmap_name  = param[6]
        cbar_label = param[3]
        
        # Take care of colorbar bullshit
        ticks    = np.linspace(vmin, vmax, n_ticks)
        cmap     = plt.get_cmap(cmap_name)
        crange   = np.sort(np.linspace(0, vmax - vmin, n_seg)/abs(vmax - vmin))
        cmap_lin = colors.LinearSegmentedColormap.from_list(cmap_name, cmap(crange), n_seg)
        sm       = plt.cm.ScalarMappable(cmap=cmap_lin, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A    = []
        

        for i, (lon0, lat0) in enumerate(zip(param[0], param[1])):

            lon_less = fault[fault['Longitude'] < lon0]['Longitude']
            lon_more = fault[fault['Longitude'] > lon0]['Longitude']
            lon = [lon_more.iloc[-1], lon_less.iloc[0]]

            lat_less = fault[fault['Latitude'] < lat0]['Latitude']
            lat_more = fault[fault['Latitude'] > lat0]['Latitude']
            lat = [lat_less.iloc[-1], lat_more.iloc[0]]

            c = cmap_lin(cval[i])

            ax.plot(lon, lat, c=c, linewidth=3)

        cbar = fig.colorbar(sm, ticks=ticks, label=cbar_label)
        # cbar = fig.colorbar(sm, ticks=np.linspace(0, 1, ticks), label=cbar_label)
        # cbar.ax.set_yticklabels(labels)

    # Figure settings
    if 'bounds' in kwargs.keys():
        bounds = kwargs['bounds']

        xlim = bounds[:2]
        ylim = bounds[2:]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    ax.tick_params(direction='in')
    plt.colorbar(im, label='Fault-parallel velocity (mm/yr)')
    fig.tight_layout()

    if 'file_name' in kwargs.keys():
        file_name = kwargs['file_name'] + '.png'
        plt.savefig(file_name, dpi=dpi)
        plt.close()

    else:
        plt.show()
    return


def plot_chains(samples, samp_prob, discard, labels, units, scales, key, out_dir):
    """
    Plot Markov chains
    """
    n_dim = len(labels)

    fig, axes = plt.subplots(n_dim + 1, figsize=(6.5, 4), sharex=True)
    
    for i in range(n_dim):
        ax = axes[i]
        ax.plot(samples[discard:, :, i] * scales[i], "k", alpha=0.3, linewidth=0.5)
        # ax.plot(samples[:, :, i] * scales[i], "k", alpha=0.3, linewidth=0.5)
        # ax.plot(samples[:discard, :, i] * scales[i], color='tomato', linewidth=0.5)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    # Plot log-probability
    ax = axes[n_dim]
    ax.plot(samp_prob[discard:], "k", alpha=0.3, linewidth=0.5) # log(p(d|m))
    # ax.plot(samp_prob, "k", alpha=0.3, linewidth=0.5)
    # ax.plot(samp_prob[:discard], color='tomato', linewidth=0.5)
    ax.set_xlim(0, len(samples))
    ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.set_ylabel(r'log(p(m|d))') # log(p(d|m))
    axes[-1].set_xlabel("Step");
    fig.tight_layout()
    fig.savefig(f'{out_dir}/chains/{key}_chains.png', dpi=dpi)
    plt.close()
    
    return


def plot_triangle(samples, priors, labels, units, scales, key, out_dir, **kwargs):
    # Make corner plot
    font = {'size'   : 6}

    matplotlib.rc('font', **font)

    if 'figsize' in kwargs.keys():
        figsize = kwargs['figsize']
    else:
        figsize=(6.5, 4)

    prior_vals = [[prior * scales[i] for prior in priors[key]] for i, key in enumerate(priors.keys())]

    fig = plt.figure(figsize=figsize, tight_layout={'h_pad':0.1, 'w_pad': 0.1})
    fig.suptitle(key)
    fig = corner.corner(samples * scales, 
                        quantiles=[0.16, 0.5, 0.84], 
                        range=prior_vals,
                        labels=[f'{label} ({unit})' for label, unit in zip(labels, units)], 
                        label_kwargs={'fontsize': 8},
                        show_titles=True,
                        title_kwargs={'fontsize': 8},
                        fig=fig, 
                        labelpad=0.1
                        )

    # fig.tight_layout(pad=1.5)
    fig.savefig(f'{out_dir}/triangles/{key}_triangle.png', dpi=dpi)
    plt.close()
    return


def plot_params(dist, m_pref, m_low, m_high, c, clabel, cmap, ticks, labels, units, scales, out_dir, file_name):

    # Plot model parameters along-strike 
    fig, axes = plt.subplots(nrows=len(units), figsize=(6, 4.5))

    ylabels = [f'{label} ({unit})' for label, unit in zip(labels, units)]

    # Plot results
    for i, ax in enumerate(axes):

        # Get data vector
        im0 = ax.fill_between(dist, m_pref[:, i] - m_low[:, i], m_pref[:, i] + m_high[:, i], facecolor='lightgray', edgecolor='lightgray', label='Posterior $\sigma$')
        im1 = ax.plot(dist, m_pref[:, i], 'k', label=r'Posterior mean')
        im2 = ax.scatter(dist, m_pref[:, i], c=c, marker='.', s=15, cmap=cmap, zorder=100)

        # Add legend
        if i == 0:
            ax.legend(loc='lower right', fontsize=7)

        # Other settings
        ax.set_xlim([min(dist), max(dist)])
        ax.set_ylabel(ylabels[i], fontsize=8)
        ax.tick_params(direction='in')
        ax.set_axisbelow(True)
        ax.grid(zorder=0)

        # Tick font size
        for xtick, ytick in zip(ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()):
            xtick.label.set_fontsize(8) 
            ytick.label.set_fontsize(8) 

    # Adjust yticks
    for i in range(3):
        axes[i].set_ylim([ticks[i][0], ticks[i][-1]])
        axes[i].set_yticks(ticks[i])

    # Adjust xticks
    axes[0].set_xticklabels([])
    axes[1].set_xticklabels([])
    axes[2].set_xlabel('Fault distance (km)', fontsize=8)

    # Adjust spacing and add colorbar
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.2)
    cax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    fig.colorbar(im2, cax=cax, label=clabel)

    # Save
    fig.savefig(f'{out_dir}/{file_name}', dpi=dpi)

    return


def plot_data_hist(d_total_std, std_max, d_total_count, bin_min, out_dir, key):
    """
    Plot histogram of bin standard deviations and counts.
    """
    fig, ax = plt.subplots(2, 1)

    ax[0].hist(d_total_std[d_total_std <= std_max], range=[np.nanmin(d_total_std), np.nanmax(d_total_std)], color='k')
    ax[0].hist(d_total_std[d_total_std > std_max],  range=[np.nanmin(d_total_std), np.nanmax(d_total_std)], color='r')
    ax[0].set_xlabel('Standard deviation')
    ax[0].set_ylabel('Count')

    ax[1].hist(d_total_count[d_total_count >= bin_min], range=[np.nanmin(d_total_count), np.nanmax(d_total_count)], color='k')
    ax[1].hist(d_total_count[d_total_count < bin_min],  range=[np.nanmin(d_total_count), np.nanmax(d_total_count)], color='r')
    ax[1].set_xlabel('Bin count')
    ax[1].set_ylabel('Count')

    fig.tight_layout()
    fig.savefig(f'{out_dir}/hist/{key}_hist.png')
    plt.close()

    return 


def plot_kernel(max_reg_width, f_kernel, out_dir):
    """
    Plot kernel used in along-strike regularization
    """
    fig, ax = plt.subplots()

    x_kernel = np.linspace(-max_reg_width, max_reg_width)
    y_kernel = f_kernel(x_kernel)

    ax.plot(x_kernel, y_kernel, color='k')
    ax.set_xlabel('Fault distance (km)')
    ax.set_ylabel('Weight')
    ax.tick_params(direction='in')

    fig.tight_layout()
    fig.savefig(f'{out_dir}/kernel.png')
    plt.close()
    
    return


# ---------- WRITING ---------- 
def prep_out_dir(param_names, param_values, mode, top_dir):
    """
    Prepare directory for output products. 
    Automatically infers run number to avoid overwriting.
    """

    # Initiate parameters
    run_count = 0
    written   = False

    if os.path.isdir(top_dir) is not True:
        os.mkdir(top_dir)

    # Make directory
    if mode == 'preview':
        out_dir = top_dir + '/' + 'preview'
        if os.path.isdir(out_dir):
            try:
                shutil.rmtree(out_dir)
            except OSError:
                os.remove(out_dir)
        
        os.mkdir(out_dir)
        os.mkdir(out_dir + '/' + 'hist')
        os.mkdir(out_dir + '/' + 'profiles')
        os.mkdir(out_dir + '/' + 'profiles/raw')
        os.mkdir(out_dir + '/' + 'profiles/downsampled')
        os.mkdir(out_dir + '/' + 'profiles/optimized')

    elif mode == 'reload':
        if len(sys.argv) == 2:
            run_count = np.sort([int(dir.split('/')[-1].split('_')[-1]) for dir in glob.glob(f'{top_dir}/Run_*')])[-1]
            
            if 0 <= run_count < 10:
                n_run = '00' + str(run_count)
            elif 10 <= run_count < 100:
                n_run = '0' + str(run_count)
            else:
                n_run = str(run_count)

            out_dir = f'{top_dir}/Run_{n_run}'
            written = True
        elif len(sys.argv) == 3:
            out_dir = f'{top_dir}/{sys.argv[2]}'
    
    else:
        while written is False:
            if 0 <= run_count < 10:
                n_run = '00' + str(run_count)
            elif 10 <= run_count < 100:
                n_run = '0' + str(run_count)
            else:
                n_run = str(run_count)

            out_dir = f'{top_dir}/Run_{n_run}'

            # Check if output directory exists
            if os.path.isdir(out_dir):
                run_count += 1
            else:
                os.mkdir(out_dir)
                os.mkdir(out_dir + '/' + 'swaths')
                os.mkdir(out_dir + '/' + 'results')
                os.mkdir(out_dir + '/' + 'profiles')
                os.mkdir(out_dir + '/' + 'profiles/raw')
                os.mkdir(out_dir + '/' + 'profiles/downsampled')
                os.mkdir(out_dir + '/' + 'profiles/optimized')
                os.mkdir(out_dir + '/' + 'profiles/modeled')
                os.mkdir(out_dir + '/' + 'chains')
                os.mkdir(out_dir + '/' + 'triangles')
                os.mkdir(out_dir + '/' + 'hist')
                written = True

    # Write parameter file
    # param_names = ['file', 'std', 'l', 'w', 'top_dir', 'v_mode', 'priors', 'init', 'n_walkers', 'n_step']  
    # pad = max([len(name) for name in param_names]) + 1

    with open(f'{out_dir}/params.txt', 'w') as f:
        f.write('# ----------- Inversion parameters ----------- \n')
        for name, value in zip(param_names, param_values):
            f.write(f'{name} = {value} \n')


    return out_dir


def config_backend(out_dir, key, n_walkers, n_dim):

    file_name = f'{out_dir}/results/{key}_Results.h5'
    backend   = emcee.backends.HDFBackend(file_name)
    backend.reset(n_walkers, n_dim)

    return backend


# ---------- NUMBA-FIED FUNCTIONS ---------- 
@numba.jit(nopython=True)
def v_shallow(m, x):
    """
    Compute 2D velocity profile due to a shallow screw dislocation
    (See Equation 2 in Tymofyeyeva et al. 2019, JGR)

    INPUT:
        m is a vector (3,) containing:
            v0  - slip rate on fault (m/yr)
            D   - locking depth (m)
            dip - dip of fault (right-hand rule relative to fault trace) (deg)
        x  - fault-perpendicular distance of observation point (m)

    OUTPUT:
        v - surface velocity (m/yr)
    """

    return m[0]/2*(x/np.abs(x)) - m[0] * np.arctan((x - m[1]/np.tan(np.pi*m[2]/180))/m[1]) / np.pi


@numba.jit(nopython=True)
def v_shallow_shift(m, x):
    """
    Compute 2D velocity profile due to a shallow screw dislocation
    (See Equation 2 in Tymofyeyeva et al. 2019, JGR)

    INPUT:
    INPUT:
        m is a vector (3,) containing:
            v0   - slip rate on fault (m/yr)
            D    - locking depth (m)
            dip  - dip of fault (right-hand rule relative to fault trace) (deg)
            vs   - uniform velocity shift
        x  - fault-perpendicular distance of observation point (m)

    OUTPUT:
        v - surface velocity (m/yr)
    """

    return m[0]/2*(x/np.abs(x)) - m[0] * np.arctan((x - m[1]/np.tan(np.pi*m[2]/180))/m[1]) / np.pi + m[3]


@numba.jit(nopython=True)
def v_deep(m, x):
    """
    Compute 2D velocity profile due to a shallow screw dislocation
    (See Equation 2 in Tymofyeyeva et al. 2019, JGR)

    INPUT:
        m is a vector (3,) containing:
            v0  - slip rate on fault (m/yr)
            D   - locking depth (m)
            dip - dip of fault (right-hand rule relative to fault trace) (deg)
        x  - fault-perpendicular distance of observation point (m)

    OUTPUT:
        v - surface velocity (m/yr)
    """ 

    return (m[0]/np.pi) * np.arctan((x - m[1]/np.tan(np.pi*m[2]/180))/m[1])


@numba.jit(nopython=True)
def v_elliptical(m, x):

    v0, D, dip = m
    dip_r = dip * np.pi/180

    # Get edges of patches
    d = D * patch_edges
    
    # Determine slip distribution
    s = np.array([v0 * (1 - (d[i]/D)**2)**0.5 for i in range(n_patch + 1)])

    # Compute velocities
    v = np.empty((n_patch, len(x)))
    
    # First compute surface dislocation
    v[0, :] = s[0]/2*(x/np.abs(x)) - s[0] * np.arctan((x - d[1]/np.tan(dip_r))/d[1]) / np.pi
    
    # Then the finite ones at depth
    for i in range(1, n_patch):
        v[i, :] = -(s[i]/np.pi) * (np.arctan((x - d[i + 1]/np.tan(dip_r))/d[i + 1]) - np.arctan((x - d[i]/np.tan(dip_r))/d[i]))
        
    return -np.sum(v, axis=0)


@numba.jit(nopython=True)
def v_elliptical_shift(m, x):

    v0, D, dip, vc = m
    dip_r = dip * np.pi/180

    # Get edges of patches
    d = np.linspace(0, D, n_patch + 1)
    
    # Determine slip distribution
    s = np.array([v0 * (1 - (d[i]/D)**2)**0.5 for i in range(n_patch + 1)])

    # Compute velocities
    v = np.empty((n_patch, len(x)))
    
    # Get tan of dip
    theta = np.tan(dip_r)
    # First compute surface dislocation
    v[0, :] = s[0]/2*(x/np.abs(x)) - s[0] * np.arctan((x - d[1]/theta)/d[1]) / np.pi
    
    # Then the finite ones at depth
    for i in range(1, n_patch):
        v[i, :] = -(s[i]/np.pi) * (np.arctan((x - d[i + 1]/theta)/d[i + 1]) - np.arctan((x - d[i]/theta)/d[i]))
        
    return -np.sum(v, axis=0) + vc


@numba.jit(nopython=True)
def v_elliptical_shift_vert(m, x):
    """
    Verfical fault with ellipical slip distribution and jitter shift.
    """

    v0, D, vc = m

    # Get edges of patches
    d = np.linspace(0, D, n_patch + 1)
    
    # Determine slip distribution
    s = np.array([v0 * (1 - (d[i]/D)**2)**0.5 for i in range(n_patch + 1)])

    # Compute velocities
    v = np.empty((n_patch, len(x)))
    
    # First compute surface dislocation
    v[0, :] = s[0]/2*(x/np.abs(x)) - s[0] * np.arctan(x/d[1]) / np.pi
    
    # Then the finite ones at depth
    for i in range(1, n_patch):
        v[i, :] = -(s[i]/np.pi) * (np.arctan(x/d[i + 1]) - np.arctan(x/d[i])) 
        
    return -np.sum(v, axis=0) + vc


@numba.jit(nopython=True)
def v_fixed_dip(m, x, dip):

    v0, D = m
    dip_r = dip * np.pi/180

    # Get edges of patches
    d = np.linspace(0, D, n_patch + 1)
    
    # Determine slip distribution
    s = np.array([v0 * (1 - (d[i]/D)**2)**0.5 for i in range(n_patch + 1)])

    # Compute velocities
    v = np.empty((n_patch, len(x)))
    
    # Get tan of dip
    theta = np.tan(dip_r)

    # First compute surface dislocation
    v[0, :] = s[0]/2*(x/np.abs(x)) - s[0] * np.arctan((x - d[1]/theta)/d[1]) / np.pi
    
    # Then the finite ones at depth
    for i in range(1, n_patch):
        v[i, :] = -(s[i]/np.pi) * (np.arctan((x - d[i + 1]/theta)/d[i + 1]) - np.arctan((x - d[i]/theta)/d[i]))
        
    return -np.sum(v, axis=0)


@numba.jit(nopython=True)
def v_fixed_dip_shift(m, x, dip):

    v0, D, vc = m
    dip_r = dip * np.pi/180

    # Get edges of patches
    d = np.linspace(0, D, n_patch + 1)
    
    # Determine slip distribution
    s = np.array([v0 * (1 - (d[i]/D)**2)**0.5 for i in range(n_patch + 1)])

    # Compute velocities
    v = np.empty((n_patch, len(x)))
    
    # Get tan of dip
    theta = np.tan(dip_r)

    # First compute surface dislocation
    v[0, :] = s[0]/2*(x/np.abs(x)) - s[0] * np.arctan((x - d[1]/theta)/d[1]) / np.pi
    
    # Then the finite ones at depth
    for i in range(1, n_patch):
        v[i, :] = -(s[i]/np.pi) * (np.arctan((x - d[i + 1]/theta)/d[i + 1]) - np.arctan((x - d[i]/theta)/d[i]))
        
    return -np.sum(v, axis=0) + vc


@numba.jit(nopython=True)
def v_combined(m, x):
    """
    Model for velocity profile due to both shallow and deep slip.
    """
    s_0, D_0, dip_0, s_d, D_d, dip_d = m
    dip_0_r = dip_0 * np.pi/180
    dip_d_r = dip_d * np.pi/180

    # Get edges of patches
    d = D_0 * patch_edges
    
    # Determine slip distribution
    s = np.array([s_0 * (1 - (d[i]/D_0)**2)**0.5 for i in range(n_patch + 1)])

    # Compute velocities
    v = np.empty((n_patch, len(x)))
    
    # First compute surface dislocation
    v[0, :] = s[0]/2*(x/np.abs(x)) - s[0] * np.arctan((x - d[1]/np.tan(dip_0_r))/d[1]) / np.pi
    
    # Then the finite ones at depth
    for i in range(1, n_patch):
        v[i, :] = -(s[i]/np.pi) * (np.arctan((x - d[i + 1]/np.tan(dip_0_r))/d[i + 1]) - np.arctan((x - d[i]/np.tan(dip_0_r))/d[i]))
    
    # Sum shallow and deep dislocations
    return -np.sum(v, axis=0) - (s_d/np.pi) * (np.arctan((x - D_d/np.tan(dip_d_r))/D_d))


@numba.jit(nopython=True)
def v_combined_shift(m, x):
    """
    Model for velocity profile due to both shallow and deep slip with reference velocity shift included.
    """
    s_0, D_0, dip_0, s_d, D_d, dip_d, v_ref = m
    dip_0_r = dip_0 * np.pi/180
    dip_d_r = dip_d * np.pi/180

    # Get edges of patches
    d = D_0 * patch_edges
    
    # Determine slip distribution
    s = np.array([s_0 * (1 - (d[i]/D_0)**2)**0.5 for i in range(n_patch + 1)])

    # Compute velocities
    v = np.empty((n_patch, len(x)))
    
    # First compute surface dislocation
    v[0, :] = s[0]/2*(x/np.abs(x)) - s[0] * np.arctan((x - d[1]/np.tan(dip_0_r))/d[1]) / np.pi
    
    # Then the finite ones at depth
    for i in range(1, n_patch):
        v[i, :] = -(s[i]/np.pi) * (np.arctan((x - d[i + 1]/np.tan(dip_0_r))/d[i + 1]) - np.arctan((x - d[i]/np.tan(dip_0_r))/d[i]))
    
    # Sum shallow and deep dislocations
    return -np.sum(v, axis=0) - (s_d/np.pi) * (np.arctan((x - D_d/np.tan(dip_d_r))/D_d)) + v_ref


@numba.jit(nopython=True)
def rotate(x, y, a):
    """
    Rotate a point (x, y) to a new coordinate system specified by angle a
    # R     = np.array([[np.cos(a), -np.sin(a)], 
    #                   [np.sin(a), np.cos(a)]])
    """
    x_r = np.cos(a)*x - np.sin(a)*y
    y_r = np.sin(a)*x + np.cos(a)*y

    return x_r, y_r


@numba.jit(nopython=True)
def log_likelihood(G_m, d, S_inv, B):
    """
    Speedy version of log. likelihood function

    INPUT:
    G_m   - model realization
    d     - data values
    S_inv - inverse covariance matrix
    B     - data weights
    """

    r = d - G_m

    return -0.5 * r.T @ S_inv @ B @ r


@numba.jit(nopython=True)
def log_prior_gaussian(m, mu, S_inv):
    """
    Evaluate Gaussian log-prior.
    
    INPUT:
    m     - model parameters (n,)
    mu    - (n,) vector defining the means of the Gaussian prior distribution
    S_inv - (n,n) diagonal covariance matrix of the Gaussian prior distribution

    OUTPUT:
    lp     - log. prob. of m given priors
    """

    r = m - mu

    return -0.5 * r.T @ S_inv @ r


@numba.jit(nopython=True)
def RMSE(G_m, d, S_inv):

    N = len(d)

    return (((d - G_m).T @ S_inv @ (d - G_m))/N)**0.5


@numba.jit(nopython=True)
def wRMSE(G_m, d, S_inv, B):
    """
    Weighted root-mean-square error
    G_m   (n,)  - model vector
    d     (n,)  - data vector
    S_inv (n,n) - covariance matrix (diagonal)
    B     (n,n) - matrix of data weights (diagonal)
    """

    r = G_m - d # Residuals

    return ((r.T @ S_inv @ B @ r)/np.sum(B))**0.5


if __name__ == '__main__':
    main()


