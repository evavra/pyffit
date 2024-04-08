import numpy as np
import matplotlib.pyplot as plt
import numba
import os
import gc
import glob
import time
import h5py
import shutil
import psutil
import pyffit
import pandas as pd
import cutde.halfspace as HS
import matplotlib.tri as tri
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.optimize import lsq_linear
from itertools import combinations
from scipy.sparse import csr_array
from scipy.sparse.linalg import eigsh
import multiprocessing 
from matplotlib import colors

import cmcrameri.cm as cmc

markersize = 30

def main():    
    # run()
    plot_fault()
    # edge_sensitivity()
    # smoothness_sensitivity()
    # grid_search_plot()

    # test_synthetic_intf()
    # make_synthetic_intfs()
    return

# ------------------ Drivers -----------------
def run():
    # -------------------------- Settings -------------------------------------------------------------------------------------------------------------------------
    # Files
    mesh_version    = 2
    # mesh_version    = 'mapped_short'
    mesh_file       = f'/Users/evavra/Projects/SHF/Analysis/Mesh/Geometry/mesh_points_{mesh_version}_10km.txt'
    triangle_file   = f'/Users/evavra/Projects/SHF/Analysis/Mesh/Geometry/mesh_connectivity_{mesh_version}_10km.txt' 
    # insar_file      = '/Users/evavra/Projects/SHF/Data/Sentinel_1/Version_1_2023-07-17/unwrap2_ll_edit.grd'
    insar_file      = '/Users/evavra/Projects/SHF/Data/Sentinel_1/20230503/20230726/unwrap_2_ll_edit.grd'
    look_e_file     = '/Users/evavra/Projects/SHF/Data/Sentinel_1/look_e.grd'
    look_n_file     = '/Users/evavra/Projects/SHF/Data/Sentinel_1/look_n.grd'
    look_u_file     = '/Users/evavra/Projects/SHF/Data/Sentinel_1/look_u.grd'
    salton_sea_file = '/Users/evavra/Projects/SSAF/Data/Salton_Sea/salton_sea.txt'

    # Data parameters:
    version     = 1
    units       = 'rad'
    remove_mean = True
    wavelength  = 55.465763
    proj_los    = False

    # Geographic parameters
    EPSG        = '32611' 
    ref_point   = [-115.7009,  32.9301]   # Bilham reepmeter coordinates  
    ref_point   = [-115.70124, 32.93049] # Czech reepmeter coordinates  
    avg_strike  = 302
    data_region = [-30, 20, -20, 25] # local coordinate system

    # Quadtree parameters
    rms_min      = 0.2  # RMS threshold (data units)
    nan_frac_max = 0.9  # Fraction of NaN values allowed per cell
    width_min    = 0.1  # Minimum cell width (km)
    width_max    = 2    # Maximum cell width (km)
    
    # Forward model parameters
    poisson_ratio = 0.25       # Poisson ratio
    shear_modulus = 6 * 10**9
    init_model    = 'read'     # 'generate' to compute initial model for downsampling or 'read' to load existing model 
    # init_model    = 'generate'     # 'generate' to compute initial model for downsampling or 'read' to load existing model 
    n_disp_chunks = 1

    # Inversion parameters
    test         = False
    init_detrend = True
    mu           = 0.6951927962 
    eta          = 1.4384498883
    slip_lim     = (0, 10000)

    # Sensitivity tests
    # mode   = 'regular'
    n_iter = 3
    # mode = 'grid_search'
    # n_iter           = 1
    eta_range = np.logspace(-3, 3, 10)
    mu_range  = np.logspace(-1, 1, 10)

    # Synthetic uncertainty quantification test parameters
    mode             = 'syn_UQ'
    input_model_file = 'Results/mu-0.6951927962_eta-1.4384498883/results.h5'
    n_aps            = 10
    n_aps_iter       = 100
    aps_amp          = 1.5

    # Plot parameters
    xlim      = [-15, 7]
    ylim      = [-7, 13]
    xticks    = np.arange(np.ceil(data_region[0]/10)*10, np.floor(data_region[1]/10) + 1, 10)
    yticks    = np.arange(np.ceil(data_region[2]/10)*10, np.floor(data_region[3]/10) + 1, 10)
    vlim_disp = [-10, 10]
    vlim_slip = [0,   18]
    cmap_disp = 'coolwarm'
    cmap_slip = 'viridis_r'


    # -------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------- Prepare data --------------------------
    
    # Load fault data
    mesh         = np.loadtxt(mesh_file,     delimiter=',', dtype=float)
    triangles    = np.loadtxt(triangle_file, delimiter=',', dtype=int)
    fault_info   = get_fault_info(mesh, triangles, verbose=True)

    # Load InSAR data
    _, _, look_e       = pyffit.data.read_grd(look_e_file)
    _, _, look_n       = pyffit.data.read_grd(look_n_file)
    _, _, look_u       = pyffit.data.read_grd(look_u_file)
    x_rng, y_rng, data = pyffit.data.read_grd(insar_file)
    look = np.vstack((look_e.flatten(), look_n.flatten(), look_u.flatten())).T

    # Get full gridded coordinates
    x, y = np.meshgrid(x_rng, y_rng)

    # Apply corrections if specified
    if units == 'rad':
        data *= wavelength/(4*np.pi) # Convert from rad to mm

    if remove_mean:
        data -= np.nanmean(data)

    if proj_los:
        data /= (look_e * np.sin(np.deg2rad(avg_strike)))

    # Get reference point and convert coordinates to km
    x_utm, y_utm = pyffit.utilities.get_local_xy_coords(x, y, ref_point) 

    # Filter based off of region specified region
    x_utm, y_utm, data, extent = pyffit.utilities.clip_grid(x_utm, y_utm, data, data_region, extent=True)

    # Load or compute initial model
    if init_model == 'generate':
        saveprint(f'')
        saveprint(f'# -------------------- Generating starting model --------------------')

        # Get guess slip distribution
        init_slip_model   = get_synthetic_slip(mesh, triangles, max_slip=-20,  strike_scale=0.7, dip_scale=0.5, x_shift=0.4)
        init_slip_model  += get_synthetic_slip(mesh, triangles, max_slip=-30,  strike_scale=0.3, dip_scale=1.5, x_shift=-0.4)

        # Save to file
        f = h5py.File(f'Results/initial_model.h5', 'w')
        f.create_dataset('slip', data=init_slip_model)

        # Get Green's functions
        GF = get_fault_greens_functions(x_utm, y_utm, 0 * y_utm, mesh, triangles, nu=poisson_ratio, verbose=True)
        # f.create_dataset('GF', data=GF)

        # Project to LOS
        GF_proj = proj_greens_functions(GF, look, verbose=True)
        # f.create_dataset('GF_proj', data=GF_proj)

        # Extract elements corresponding to real-valued displacements and strike-slip motion  
        i_nans = np.isnan(data.flatten())
        A = GF_proj[~i_nans, :, 0] 
        f.create_dataset('A', data=A)

        # Compute model prediction and save to file
        init_disp_model = A.dot(init_slip_model[:, 0].flatten())
        init_disp_model_grid = np.empty_like(data.flatten())
        init_disp_model_grid[i_nans] = np.nan
        init_disp_model_grid[~i_nans] = init_disp_model
        init_disp_model_grid = init_disp_model_grid.reshape(data.shape)
        f.create_dataset('disp', data=init_disp_model)

        f.close()
        
        return
        # # Compute forward model prediction
        # disp            = get_fault_displacements(x_utm, y_utm, 0 * y_utm, mesh, triangles, init_slip_model, nu=nu, verbose=True)
        # disp_grid       = disp.reshape((*data.shape, 3)) # Get gridded version
        # init_disp_model = np.array([np.dot(disp[i, :], look[i, :]) for i in range(len(disp))]) # project to LOS

        # # Save to file
        # f = h5py.File(f'Results/initial_model.h5', 'w')
        # f.create_dataset('disp', data=init_disp_model)
        # f.create_dataset('slip', data=init_slip_model)
        # f.close()

        # return

        # Plot LOS displacements
        fig, ax = plt.subplots(figsize=(14, 8.2))
        im = ax.imshow(init_disp_model_grid, extent=extent, vmin=vlim_disp[0], vmax=vlim_disp[1], cmap='coolwarm', interpolation='none')
        ax.set_ylabel('North (km)')
        ax.set_xlabel('East (km)')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[3], extent[2])
        ax.set_aspect('equal')
        fig.colorbar(im, label='LOS Displacement (mm)', shrink=0.3)
        plt.savefig(f'Results/Starting_model_LOS.png', dpi=500)

        # # Plot true synthetic displacements
        # panels = [
        #           dict(extent=extent, data=disp_grid[:, :, 0], label=r'$U_{E}$'),
        #           dict(extent=extent, data=disp_grid[:, :, 1], label=r'$U_{N}$'),
        #           dict(extent=extent, data=disp_grid[:, :, 2], label=r'$U_{z}$'),
        #          ]
        # pyffit.figures.plot_fault_panels(panels, mesh, triangles, init_slip_model[:, 0], cmap_slip=cmap_slip, vlim_disp=vlim_disp, vlim_slip=vlim_slip, xlim=xlim, ylim=ylim, show=False, file_name=f'Results/Starting_model.png') 
        
        return

    elif init_model == 'read':

        # Load saved arrays
        f = h5py.File(f'Results/initial_model.h5')
        init_slip_model = f['slip'][()]
        GF              = f['A'][()]
        f.close()

    # Generate initial ramp
    if init_detrend:
        ramp = pyffit.corrections.fit_ramp(x_utm, y_utm, data, deg=1)

    else:
        ramp = np.zeros_like(data)

    # -------------------------- Inversion with model-based resampling --------------------------
    # Get regularization matrix
    R = get_smoothing_matrix(mesh, triangles)

    # Get zero-slip matrix
    E = get_edge_matrix(mesh, triangles, R)
    pyffit.figures.plot_fault_3d(mesh, triangles, c=np.diag(E), edges=True, filename='Results/zero_slip_BC_plot.png', cbar_label='Weight', show=False)

    gc.collect()

    if test:
        out_dir = f'Results/Test'
        pyffit.utilities.check_dir_tree(out_dir)

        # Clear figures
        fig_list = glob.glob(f'{a}/*.png')

        for fig in fig_list:
            os.remove(fig)
    else:
        out_dir = '.'

    if mode == 'syn_UQ':
        out_dir = 'UQ'
        synthetic_uncertainty_quantification(mesh, triangles, x_utm, y_utm, data, look, GF, input_model_file, mu, R, eta, E, out_dir, poisson_ratio, shear_modulus, avg_strike, 
                                             rms_min, nan_frac_max, width_min, width_max, vlim_disp, vlim_slip, xlim, ylim, extent, slip_lim, n_aps_iter, n_aps, aps_amp)
    
    elif mode == 'edge_slip':
        edge_sensitivity_test(mesh, triangles, x_utm, y_utm, data, look, GF, init_slip_model, ramp, mu, R, eta_range, E, out_dir, poisson_ratio, shear_modulus, avg_strike, 
                                  rms_min, nan_frac_max, width_min, width_max, n_iter, vlim_disp, vlim_slip, xlim, ylim, extent, slip_lim)
    elif mode == 'smoothness':
        smoothness_sensitivity_test(mesh, triangles, x_utm, y_utm, data, look, GF, init_slip_model, ramp, mu_range, R, eta, E, out_dir, poisson_ratio, shear_modulus, avg_strike, 
                                rms_min, nan_frac_max, width_min, width_max, n_iter, vlim_disp, vlim_slip, xlim, ylim, extent, slip_lim)
    elif mode == 'grid_search':
        grid_search(mesh, triangles, x_utm, y_utm, data, look, GF, init_slip_model, ramp, mu_range, R, eta_range, E, out_dir, poisson_ratio, shear_modulus, avg_strike, 
                                rms_min, nan_frac_max, width_min, width_max, n_iter, vlim_disp, vlim_slip, xlim, ylim, extent, slip_lim)
    else:
        out_dir = f'Results/mu-{mu:.10f}_eta-{eta:.10f}'
        pyffit.utilities.check_dir_tree(out_dir)
        params = mesh, triangles, x_utm, y_utm, data, look, GF, init_slip_model, ramp, mu, R, eta, E, out_dir, poisson_ratio, shear_modulus, avg_strike, rms_min, nan_frac_max, width_min, width_max, n_iter, vlim_disp, vlim_slip, xlim, ylim, extent, slip_lim

        # Perform iterative solving
        solve_with_resampling(params)


def grid_search_plot():
    grid_search_dir = 'Results/0726/grid_search'
    mu_pref  = 0.6951927962
    eta_pref = 1.4384498883

    results   = pd.read_csv(f'{grid_search_dir}/grid_search_results.txt', delim_whitespace=True, header=0)
    results.columns = ['mu', 'eta', 'rms', 'cost']
    results   = results.sort_values('mu')
    rms       = results['rms'].values
    mu        = results['mu'].values
    eta       = results['eta'].values
    cost      = results['cost'].values
    # magnitude = results['magnitude'].values

    # plot_grid_search(magnitude,  mu, eta, mu_pref=mu_pref, eta_pref=eta_pref, label=r'$M_w$',         log=False, show=True, file_name=f'{grid_search_dir}/grid_search_Mw.png')
    plot_grid_search(rms,        mu, eta, mu_pref=mu_pref, eta_pref=eta_pref, label='RMS (mm)',       log=False, show=True, file_name=f'{grid_search_dir}/grid_search_rms.png')
    plot_grid_search(cost,       mu, eta, mu_pref=mu_pref, eta_pref=eta_pref, label='Cost (mm)',      log=False, show=True, file_name=f'{grid_search_dir}/grid_search_cost.png')
    plot_grid_search(rms,        mu, eta, mu_pref=mu_pref, eta_pref=eta_pref, label='log(RMS) (mm)',  log=True,  show=True, file_name=f'{grid_search_dir}/grid_search_rms_log.png')
    plot_grid_search(cost,       mu, eta, mu_pref=mu_pref, eta_pref=eta_pref, label='log(Cost) (mm)', log=True,  show=True, file_name=f'{grid_search_dir}/grid_search_cost_log.png')

    return


def plot_fault():
    mesh_version    = 2
    mesh_file       = f'/Users/evavra/Projects/SHF/Analysis/Mesh/Geometry/mesh_points_{mesh_version}_10km.txt'
    triangle_file   = f'/Users/evavra/Projects/SHF/Analysis/Mesh/Geometry/mesh_connectivity_{mesh_version}_10km.txt' 
    

    # Model runs
    result_dir      = '/Users/evavra/Projects/SHF/Analysis/Inversion/Results/mu-0.6951927962_eta-1.4384498883'
    # result_dir      = '/Users/evavra/Projects/SHF/Analysis/Inversion/Results/Shear-Modulus_33-GPa/0726/grid_search/mu-1.4384498883_eta-1.4384498883'
    # result_dir      = '/Users/evavra/Projects/SHF/Analysis/Inversion/Results/Shear-Modulus_33-GPa/0726/grid_search/mu-0.3359818286_eta-1.4384498883'
    # result_dir      = '/Users/evavra/Projects/SHF/Analysis/Inversion/Results/Shear-Modulus_33-GPa/0726/grid_search/mu-0.6951927962_eta-1.4384498883'
    # result_dir      = '/Users/evavra/Projects/SHF/Analysis/Inversion/Results/Shear-Modulus_33-GPa/0726/grid_search/mu-0.0784759970_eta-1.4384498883'
    # result_dir      = '/Users/evavra/Projects/SHF/Analysis/Inversion/Results/Shear-Modulus_33-GPa/0726/grid_search/mu-6.1584821107_eta-1.4384498883'
    # result_file     = 'results.h5'
    iteration       = 0
    mode            = 'Iteration_2/slip_model'
    out_name        = 'slip_model'
    result_file     = 'results.h5'

    # # Synthetic test
    # result_dir      = '/Users/evavra/Projects/SHF/Analysis/Inversion/UQ'
    # result_file     = 'UQ_results.h5'
    # # mode            = 'slip_model_mean'
    # # mode            = 'slip_model_std'
    # mode            = 'residual_means'
    # # mode            = 'residual_stds'


    label           = 'Dextral slip (mm)'
    n_seg           = 100
    n_tick          = 7
    vlim_slip       = [0, 30]

    # Load fault data
    mesh         = np.loadtxt(mesh_file,     delimiter=',', dtype=float)
    triangles    = np.loadtxt(triangle_file, delimiter=',', dtype=int)
    f = h5py.File(f'{result_dir}/{result_file}')
    print(f.keys())
    # slip_model = f[f'Iteration_{iteration}/slip_model'][()]
    # magnitude  = f[f'Iteration_{iteration}/magnitude'][()]
    slip_model = f[mode][()]

    f.close()

    print(np.nanmin(slip_model), np.nanmax(slip_model))

    # Plot
    file_name = f'{result_dir}/{out_name}.png'
    fig, ax = pyffit.figures.plot_fault_3d(mesh, triangles, c=slip_model, edges=True, cmap_name='viridis', 
                            cbar_label=label, 
                            # cbar_label='Dextral slip (mm)', 
                            vlim_slip=vlim_slip, labelpad=10, azim=235, elev=17, n_seg=n_seg, n_tick=n_tick, alpha=1,
                            show=False, figsize=(7, 5), cbar_kwargs=dict(location='right', pad=0.05, shrink=0.4))

    # ax.scatter(0, 0, 0.33, marker='^', s=50, facecolor='gold', edgecolor='k')
    fig.savefig(file_name, dpi=500)


def edge_sensitivity():

    # Load data
    results = pd.read_csv('Results/edge_slip_test/sensitivity_results.txt', delim_whitespace=True, header=None)
    results.columns = ['eta', 'rms']
    results = results.sort_values('eta')

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.semilogx(results['eta'], results['rms'], marker='.')
    
    ax.invert_xaxis()
    ax.set_xlabel(r'Zero slip weight $\eta$')
    ax.set_ylabel('Root-mean-square (mm)')
    ax.tick_params(direction='in')
    fig.savefig('Results/edge_slip_test/edge_slip_sensitivity.png', dpi=500)
    plt.show()
    

def smoothness_sensitivity():
    # Load data
    results = pd.read_csv('Results/smoothness_test/sensitivity_results.txt', delim_whitespace=True, header=None)
    results.columns = ['mu', 'rms']
    results = results.sort_values('mu')

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.semilogx(results['mu'], results['rms'], marker='.')
    
    ax.set_xlabel(r'Smoothness weight $\mu$')
    ax.set_ylabel('Root-mean-square (mm)')
    ax.tick_params(direction='in')
    fig.savefig('Results/smoothness_test/smoothness_sensitivity.png', dpi=500)
    plt.show()
    return


# ------------------ Methods ------------------
def load_mesh(filename, **kwargs):
    """
    Load csv file of mesh (x, y, z format) to array
    """

    return np.loadtxt(filename, delimiter=',', dtype=float)


@numba.jit
def fast_compute_cov(x, y, C, corr_length):
    """
    Speed up generation of covariance matrix using Numba.
    Uses method of Lohman and Simons (2005).

    INPUT:
    x, y (m, n) - data coordinates
    corr_length - length scale for spatial correlations (same units as x/y)

    OUTPUT:
    C - (m*n, m*n) - covariance matrix

    """
    for i in range(x.size):
        for j in range(x.size):
            # Compute distance between points
            l = ((x[i] - x[j])**2 + (y[i] - y[j])**2)**0.5

            # Compute covariance
            C[i, j] = np.exp(-l/corr_length)
    return C


def get_correlated_cov(x, y, corr_length, file_name='', verbose=True):
    """
    Compute spatial covariance matrix for a gridded dataset. Correlation decays exponetially 
    with wavelenght specified by corr_length.

    Uses method of Lohman and Simons (2005).

    INPUT:
    x, y (m, n) - data coordinates
    corr_length - length scale for spatial correlations (same units as x/y)

    OUTPUT:
    C - (m*n, m*n) - covariance matrix
    """
    start = time.time()

    # Compute correlated covariance matrix C
    C = np.zeros((x.size, x.size))

    # Flatten
    x0 = x.flatten()
    y0 = y.flatten()

    C = fast_compute_cov(x0, y0, C, corr_length)

    end = time.time() - start


    if verbose:
        saveprint(f'Covariance matrix computation time: {end:.2f}')

    # Save if file name is specified
    if len(file_name) > 0:
        pyffit.data.write_grd(x.flatten(), y.flatten(), C, file_name)

    return C


def get_correlated_noise(x, y, C, noise_level, uncorr_level=0.02, k=0, file_name='', verbose=True):
    """
    Calculate a synthetic covariance matix with exponential spatial decay.
    Uses technique of Lohman & Simons (2005)
    
    INPUT:
    x, y (m, n)   - coordinates of data
    C (m*n, m*n)  - covariance matrix    
    noise_level   - standard deviation of noise (data units)
    k             - number of eigenvalues to use in generating noise
                    k = 0 will use all eigenvaules
                    Uncorrelated high-frequency noise will be added for k > 0

    OUTPUT:
    N_c - corrrelated noise
    """
    start = time.time()

    # Get eigenvals and eigenvectors
    if k == 0: 
        u, V = np.linalg.eig(C)
    else:
        u, V = eigsh(C, k=k) # can do much faster because C is symmetric 
    U = csr_array(np.diag(u**0.5)) # Convert to sparse diagonal matrix

    # Generate uncorrelated noise
    noise_uncorr = np.random.normal(loc=0, scale=noise_level, size=len(u)).reshape(-1, 1)
    
    # Multiply to get correlated noise
    noise_corr = V @ U @ noise_uncorr
    # print(V.shape, U.shape, noise_uncorr.shape)
    # product = np.matmul(V, U)
    # noise_corr = np.matmul(product, noise_uncorr)
    noise_corr = noise_corr.reshape(x.shape) # reshape to gridded format
    noise_corr += np.random.normal(loc=0, scale=uncorr_level, size=(noise_corr.shape))
    end = time.time() - start

    if verbose:
        saveprint(f'Noise computation time: {end:.2f}')

    if len(file_name) > 0:
        pyffit.data.write_grd(x[0, :], y[:, 0], noise_corr, file_name)

    return noise_corr


def get_smoothing_matrix(mesh, triangles):
    """
    Form first-difference regularization matrix for triangular fault mesh.

    Each triangular element has three forward first-order differences:\

        ds_i/dr_i = (s_i - s_0)/||r_i - r_0||**2

    where s is the slip associate with each element, r are the element centroid coordinates, 
    index 0indicates the reference element, and index i indicates a neighboring element.


    INPUT:
    mesh (m, 3)      - x/y/z coordinates for mesh vertices
    triangles (n, 3) - row indices for mesh vertices corresponding to the nth element
    
    OUTPUT:
    R (n, n) - finite-difference matrix where each row encodes the finite-difference operator
               corresponding to the nth element.
    """

    R = np.zeros((len(triangles), len(triangles)))
    
    # Loop over each fault patch
    for i, tri in enumerate(triangles):
        pts = mesh[tri]

        # Get neighboring patches (shared edge)
        for cmb in combinations(tri, 2):

            # Check if each row contains the values
            for j, tri0 in enumerate(triangles):
                result = np.isin(cmb, tri0)

                # Neighbor triangle if two vertices are shared and not the original triangle
                if (sum(result) == 2) & (i != j):

                    # Nearest-neighbor only
                    R[i, j] -= 1
                    R[i, i] += 1
                    break

    return R


def get_edge_matrix(mesh, triangles, R):
    """
    Place zero-slip condition along bottom and side edges of fault.
    """

    E = np.zeros_like(R)

    # Smoothing-matrix method
    # Get number of finite differences for each element
    # n_diff = np.sum(np.abs(R) > 0, axis=1)

    # for i in range(len(R)):

        # Check if edge element by counting number of finite differences
        # 4 if fully surrounded by neighboring elements
        # 3 if one side is an edge
        # 2 if two sides are edges (corner element)

        # if n_diff[i] != 4:
        #     E[i, i] = n_diff[i]

        #     pts = mesh[triangles[i]][:, 2]
        #     print(pts)
            # If not at surface
            # if 0 in z_pts:
            #     print(i, z_pts)

            # if 0 not in z_pts:
                
                # E[i, i] = 1


    # Geometric method
    xmin = mesh[:, 0].min()
    xmax = mesh[:, 0].max()
    ymin = mesh[:, 1].min()
    ymax = mesh[:, 1].max()
    zmin = mesh[:, 2].min()
    zmax = mesh[:, 2].max()
    print(zmax)

    for i in range(len(R)):
        # Get vertices of each triangle
        pts = mesh[triangles[i]]

        # If any vertex is on the mesh edge (except for the top), add boundary constraint
        cond = ((xmin in pts[:, 0]) | (xmax in pts[:, 0]) | (ymin in pts[:, 1]) | (ymax in pts[:, 1]) | (zmin in pts[:, 2])) & (zmax not in pts[:, 2])
        # cond = (zmax not in pts[:, 2])

        if cond:
            E[i, i] = 1

    return E


def rotate_greens_functions(GF, avg_strike):
    """
    Rotate fault Greens functions to fault-oriented coordinate system
    """

    GF_perp, GF_para = pyffit.utilities.rotate(GF[:, 0, :, :].flatten(), GF[:, 1, :, :].flatten(), np.deg2rad(avg_strike))
    GF_proj = np.copy(GF)
    GF_proj[:, 0, :, :] = GF_para.reshape(GF.shape[0], *GF.shape[2:]) # need to reshape for reassignment
    GF_proj[:, 1, :, :] = GF_perp.reshape(GF.shape[0], *GF.shape[2:])

    return GF_proj


def proj_greens_functions(G, U, verbose=True):
    """
    Project fault Greens functions into direction specfied by input vectors

    INPUT:
    G (n_obs, 3, n_patch, 3) - array of Greens functions
    U  (n_obs, 3) - array of unit vector components

    OUTPUT:
    G_proj (n_obs, n_patch, 3)
    """ 
    start = time.time()

    G_proj = np.empty((G.shape[0], G.shape[2], G.shape[3]))

    for i in range(G.shape[0]):
        for j in range(G.shape[2]):
            for k in range(G.shape[3]):

                G_proj[i, j, k] = np.dot(G[i, :, j, k], U[i, :])
    
    end = time.time() - start

    if verbose:
        saveprint(f'LOS Greens function array size:      {G_proj.shape} {G_proj.size:.1e} elements')
        saveprint(f'LOS Greens function computation time: {end:.2f}')

    return G_proj


def read_fault(fault_file):
    """
    Get DataFrame containing fault trace nodes
    """
    fault = pd.read_csv(fault_file, delim_whitespace=True, header=None)
    fault.columns = ['Longitude', 'Latitude']

    fault['Longitude'] = fault['Longitude'] - 360

    return fault


def disp_wrapper(params):
    """
    Wrapper for matrix-free displacement computation.
    """
    pts, tris, slip, nu = params
    return  HS.disp_free(pts, tris, slip, nu)


def get_fault_info(mesh, triangles, verbose=True):
    """
    Get information about fault mesh construction.

    INPUT:
    mesh (m, 3)       - x/y/z coordinates of mesh vertices
    triangles (mn, 3) - indicies of vertices for each nth triangular element

    OUTPUT:
    fault_info - dictionary containing the folowing attributes:
                n_vertex          - number of vertices
                n_patch           - number of patches
                depths            - depths of each layer
                layer_thicknesses - thicknesses of each layer
                trace             - x/y/z/ coordinates of fault surface trace
                n_top_patch       - number of surface patches
                l_top_patch       - along-strike length of surface patches
    """

    n_vertex = len(mesh)
    n_patch  = len(triangles)

    depths = abs(np.unique(mesh[:, 2])[::-1])
    depths_formatted = ", ".join(f"{d:.2f}" for d in depths)
    layer_thicknesses = np.diff(depths)
    layer_thicknesses_formatted = ", ".join(f"{l:.2f}" for l in layer_thicknesses)

    trace       = mesh[mesh[:, 2] == 0]
    n_top_patch = len(trace) - 1
    l_top_patch = np.linalg.norm(trace[1:, :] - trace[:-1, :], axis=1)

    if verbose:
        saveprint(f'# -------------------- Mesh info -------------------- ')
        saveprint(f'Mesh vertices:           {n_vertex}')
        saveprint(f'Mesh elements:           {n_patch}')
        saveprint(f'Depths:                  {depths_formatted}')
        saveprint(f'Layer thicknesses:       {layer_thicknesses_formatted}')
        saveprint(f'Surface elements:        {n_top_patch}')
        saveprint(f'Surface element lengths: {l_top_patch.min():.2f} - {l_top_patch.max():.2f}')

    return dict(n_vertex=n_vertex, n_patch=n_patch, depths=depths, layer_thicknesses=layer_thicknesses, trace=trace, n_top_patch=n_top_patch, l_top_patch=l_top_patch)


def disp_parallel(pts, grid_dims, mesh, triangles, n_disp_chunks, slip, nu):
    """
    Parallelize computation of surface displacements due to finite-fault model
    
    INPUT:
    pts - observation coordinates into row-matrix
    n_disp_chunks

    """

    grid_size = len(pts)
    params = []
    i_chunk = np.linspace(0, len(pts), n_disp_chunks + 1, dtype=int) # approximately split into n_disp_chunks pieces

    saveprint(f'Number of chunks: {n_disp_chunks}')
    saveprint(f'Number of points: {grid_size}')
    saveprint(f'Chunk sizes:      {np.diff(i_chunk)}')
    saveprint(f'Chunk indices:    {i_chunk}')

    # Split up chunks
    for i in range(n_disp_chunks):
        params.append((pts[i_chunk[i]:i_chunk[i + 1], :], mesh[triangles], slip, nu))

    # Compute chunk displacements in parallel
    os.environ["OMP_NUM_THREADS"] = "1"
    start       = time.time()
    # n_processes = 4
    n_processes = multiprocessing.cpu_count()
    pool        = multiprocessing.Pool(processes=n_processes)
    results     = pool.map(disp_wrapper, params)
    pool.close()
    pool.join()
    end         = time.time() - start
    saveprint(f'Displacements computation time: {end:.2f}')

    # Aggregate and reshape chunk results
    disp = np.vstack(results)
    disp_grid = disp.reshape((*grid_dims, 3))

    return disp_grid


def get_synthetic_slip(mesh, triangles, plane='east', max_slip=1, strike_scale=0.5, dip_scale=0.5, x_shift=0):
    """
    Get Gaussian slip distribution
    """
    
    if plane == 'east':
        i = 0
    elif plane == 'north':
        i = 1

    X = 2*(mesh[:, i] - mesh[:, i].min())/(mesh[:, i].max() - mesh[:, i].min()) - 1

    # Z = (mesh[:, 2] - mesh[:, 2].min() )/(mesh[:, 2].max() - mesh[:, 2].min())
    Z = mesh[:, 2]

    sigma       = np.array([strike_scale, dip_scale])              # widths for Gaussian slip distribution
    slip_pts    = max_slip * np.array([np.exp(-0.5*np.array([[x0 + x_shift, z0]]) @ np.diag(1/sigma**2) @ np.array([[x0 + x_shift, z0]]).T) for x0, z0 in zip(X.flatten(), Z.flatten())]) # make gridded slip distribution, corresponding to triangle vertices
    slip_tri    = np.mean(slip_pts[triangles], axis=1).reshape(-1) # average slip at verticies to get triangle slip
    slip        = np.zeros((len(triangles), 3))                    # (strike-slip, dip-slip, opening)
    slip[:, 0]  = slip_tri

    return slip


def saveprint(arg=None, mode='a', log_file='log.txt'):
    """
    Print text to sys.stdout and output to specified log file.
    
    Options:
    mode - 'a' ot append
    """

    if arg is None:
        text = ''
    else:
        text = arg

    with open(log_file, 'a') as f:
        print(text, file=f)

    print(text)

    return


def get_fault_greens_functions(x, y, z, mesh, triangles, nu=0.25, verbose=True):
    """
    Compute fault Greens functions for given data coordinates and fault model.

    INPUT:


    OUTPUT:
    GF - array containing Greens functions for each fault element for each data point 
         Dimensions: N_OBS_PTS, 3 (E/N/Z), N_SRC_TRIS, 3 (srike-slip/dip-slip/opening)

    """
    # Start timer
    start = time.time()

    # Prep coordinates and generate Greens functions
    pts = np.array([x, y, z]).reshape((3, -1)).T.copy() # Convert observation coordinates into row-matrix
    GF  = HS.disp_matrix(obs_pts=pts, tris=mesh[triangles], nu=nu)    # (N_OBS_PTS, 3, N_SRC_TRIS, 3)

    # Stop timer
    end = time.time() - start

    # Display info
    if verbose:
        saveprint(f'Greens function array size:      {GF.reshape((-1, triangles.size)).shape} {GF.size:.1e} elements')
        saveprint(f'Greens function computation time: {end:.2f}')

    return GF


def get_fault_displacements(x, y, z, mesh, triangles, slip, nu=0.25, verbose=True):
    """
    Get surface displacements for given fault mesh and slip distribution.
    """

    # Start timer
    start      = time.time()

    # Prepare coordinates
    pts = np.array([x, y, z]).reshape((3, -1)).T.copy() 

    # Compute displacements
    disp = HS.disp_free(pts, mesh[triangles], slip, nu)

    # disp_grid  = disp.reshape((*data.shape, 3))
    # disp_model = np.array([np.dot(disp[i, :], look[i, :]) for i in range(len(disp))]) # project to LOS

    # Stop timer
    end        = time.time() - start

    if verbose:
        saveprint(f'Full displacements computation time: {end:.2f}')

    return disp


def get_full_disp(data, GF, slip, grid=False):
    """
    Compute full displacement field with original NaN values.
    """
    # Get nan locations
    i_nans = np.isnan(data.flatten())

    # Compute displacements
    disp = GF.dot(slip[:, 0].flatten())

    # Form output array
    disp_full = np.empty_like(data.flatten())
    disp_full[i_nans] = np.nan
    disp_full[~i_nans] = disp

    return disp_full


# -------------------------- Inversion routines --------------------------
def synthetic_uncertainty_quantification(mesh, triangles, x_data, y_data, data, look, GF, input_model_file, mu, R, eta, E, out_dir, poisson_ratio, shear_modulus, avg_strike, rms_min, nan_frac_max, width_min, width_max, vlim_disp, vlim_slip, xlim, ylim, extent, slip_lim, n_aps_iter, n_aps, aps_amp):
    """
    Perform uncertainty/resolution test using synthetic atmosperic noise
    """

    # Start tracking
    start = time.time()
    mem_before = psutil.Process().memory_info().rss / 1024 / 1024

    # Initialize output stuff
    pyffit.utilities.check_dir_tree(out_dir)
    out_file = h5py.File(f'{out_dir}/UQ_results.h5', 'w')

    # Load saved slip model
    f = h5py.File(input_model_file)
    input_slip_model = f['Iteration_2/slip_model'][()]
    f.close()

    # Compute forward model prediction
    disp_model = get_full_disp(data, GF, input_slip_model).reshape(data.shape)

    # Convert observation coordinates into column-matrix (x, y, z)
    pts = np.array([x_data, y_data, 0 * y_data]).reshape((3, -1)).T.copy() 

    # Get nans from real dataset
    i_nans = np.where(np.isnan(data))

    # Compute forward model prediction
    # disp_model = get_full_disp(data, GF, input_model)

    # Downsample model
    data_extent = [x_data.min(), x_data.max(), y_data.min(), y_data.max()]
    data_index  = np.arange(0, data.size)
    x_samp, y_samp, model_samp, data_samp_std, data_tree, cell_dims, cell_extents, nan_frac = pyffit.quadtree.quadtree_unstructured(x_data.flatten(), y_data.flatten(), disp_model.flatten(), data_index, data_extent, 
                                                                                                                    rms_min=rms_min, nan_frac_max=nan_frac_max, width_min=width_min, width_max=width_max,
                                                                                                                    x_samp=[], y_samp=[], data_samp=[], data_samp_std=[], data_tree=[], data_dims=[], data_extents=[], nan_frac=[])
    # Downsample data to get nan locations
    x_samp, y_samp, data_samp, data_samp_std, nan_frac = pyffit.quadtree.apply_unstructured_quadtree(x_data.flatten(), y_data.flatten(), data.flatten(), data_tree, nan_frac_max)
    i_nans_samp = np.isnan(data_samp)

    # Downsample look vectors
    _, _, look_e_samp, _, _ = pyffit.quadtree.apply_unstructured_quadtree(x_data.flatten(), y_data.flatten(), look[:, 0], data_tree, nan_frac_max)
    _, _, look_n_samp, _, _ = pyffit.quadtree.apply_unstructured_quadtree(x_data.flatten(), y_data.flatten(), look[:, 1], data_tree, nan_frac_max)
    _, _, look_u_samp, _, _ = pyffit.quadtree.apply_unstructured_quadtree(x_data.flatten(), y_data.flatten(), look[:, 2], data_tree, nan_frac_max)
    look_samp = np.vstack((look_e_samp, look_n_samp, look_u_samp)).T

    # Generate Greens functions for dowmsampled coordinates
    GF_samp = get_fault_greens_functions(x_samp, y_samp, 0 * y_samp, mesh, triangles, nu=poisson_ratio, verbose=True)

    # Project Greens functions into LOS
    GF_samp_proj = pyffit.finite_fault.proj_greens_functions(GF_samp, look_samp)

    # Include only strike-slip contribution
    A = GF_samp_proj[~i_nans_samp, :, 0] 

    # Form full design matrix
    G = np.vstack((A, mu*R, eta*E)) # Add regularization        

    # Initialize results matrix
    slip_models       = np.empty((n_aps_iter, len(input_slip_model)))
    slip_model_resids = np.empty((n_aps_iter, len(input_slip_model)))


    saveprint(f'Quadtree points: {len(model_samp)}')
    saveprint('Beginning synthetic inversions...')

    # Perform synthetic inversions
    for k in range(n_aps_iter):
        saveprint(f'')
        saveprint(f'# -------------------- Iteration {k} --------------------')

        # ------------------ Generate APS ------------------ 
        aps_resid = np.zeros_like(data)

        for i in range(n_aps):
            aps_resid += pyffit.atmosphere.make_synthetic_aps(x_data[0, :], y_data[:, 0])

        aps_resid /= n_aps
        aps_resid[i_nans] = np.nan
        aps_resid -= np.nanmean(aps_resid)
        aps_resid *= aps_amp/np.nanmean(np.abs(aps_resid))

        # Add to data
        disp_model_noisy = disp_model + aps_resid

        # Fit ramp
        ramp = pyffit.corrections.fit_ramp(x_data, y_data, disp_model_noisy, deg=1)

        # ------------------ Prep and perform inversion ------------------ 
        # Apply quadtree to synthetic noisy data
        x_samp, y_samp, data_samp, data_samp_std, nan_frac = pyffit.quadtree.apply_unstructured_quadtree(x_data.flatten(), y_data.flatten(), disp_model_noisy.flatten() - ramp.flatten(), data_tree, nan_frac_max)
        i_nans_samp = np.isnan(data_samp)

        # Form data matrix
        d = np.hstack((data_samp[~i_nans_samp], np.zeros(R.shape[0] + E.shape[0]))) # Pad data vector with zeros

        # Solve using bounded least squares to enforce slip direction constraint
        results         = lsq_linear(G, d, bounds=slip_lim)
        slip_model      = np.column_stack((results.x, np.zeros_like(results.x), np.zeros_like(results.x)))
        data_samp_model = A.dot(slip_model[:, 0].flatten())
        
        # ------------------ Process results ------------------
        # Get residuals
        resid = results.fun[:len(data_samp[~i_nans_samp])]
        cost  = results.cost
        rms   = np.sqrt(np.sum(resid**2)/len(resid))

        # Compute full model residuals
        # full_resid = data - disp_model.reshape(data.shape)

        # Compute magnitude
        patch_area     = np.empty(len(slip_model))
        patch_slip     = np.empty(len(slip_model))
        patch_potency  = np.empty(len(slip_model))
        patch_moment   = np.empty(len(slip_model))
        patch_residual = np.empty(len(slip_model))

        for i in range(len(slip_model)):
            patch = mesh[triangles[i]]

            # Get patch area via cross product
            ab = patch[0, :] - patch[1, :]
            ac = patch[0, :] - patch[2, :]
            patch_area[i] = np.linalg.norm(np.cross(ab, ac), ord=2)/2 * 1e6 # convert from km^3 to m^3

            # Get slip magnitude
            patch_slip[i] = np.linalg.norm(slip_model[i, :], ord=2) * 1e-3 # convert from mm to m

            # Compute potency
            patch_potency[i] = patch_area[i] * patch_slip[i]

            # Compute magnitude
            patch_moment[i] = shear_modulus * patch_potency[i] * 1e7 # convert from N-m to dyne-cm

            # Compute residual with respect to input model
            patch_residual[i] = abs(1000*patch_slip[i] - np.linalg.norm(input_slip_model[i, :], ord=2))

        # Get potency, moment, and moment magnitude
        potency   = np.sum(patch_potency)
        moment    = np.sum(patch_moment)
        magnitude = (2/3) * np.log10(moment) - 10.7

        slip_models[k, :]       = patch_slip*1000
        slip_model_resids[k, :] = patch_residual

        # Display model info
        saveprint()
        saveprint(f'Model information:')
        saveprint(f'Slip range  = {slip_model[:, 0].min():.1f} {slip_model[:, 0].max():.1f}')
        saveprint(f'Moment: {moment:.2e} dyne-cm')
        saveprint(f'M_w: {magnitude:.1f}')
        saveprint(f'Mean residual = {np.nanmean(np.abs(resid)):.2f}')
        saveprint()
        gc.collect()
        
        # Save some things to disk
        out_file.create_dataset(f'{k}/slip_model', data=slip_model)
        out_file.create_dataset(f'{k}/potency',    data=potency)
        out_file.create_dataset(f'{k}/moment',     data=moment)
        out_file.create_dataset(f'{k}/magnitude',  data=magnitude)

        # # -------------------------- Plots --------------------------
        # Plot results
        panels = [
                  dict(x=x_samp,               y=y_samp,               data=data_samp,                                 label='Downsampled data'),
                  dict(x=x_samp[~i_nans_samp], y=y_samp[~i_nans_samp], data=data_samp_model,                           label=rf'Model ($\mu$ = {mu:.1e}, $\eta$ = {eta:.1e})'),
                  dict(x=x_samp[~i_nans_samp], y=y_samp[~i_nans_samp], data=data_samp[~i_nans_samp] - data_samp_model, label=rf'Residuals ({np.nanmean(np.abs(resid)):.2f} $\pm$ {np.nanstd(np.abs(resid)):.2f}) mm'),
                 ]

        file_name      = out_dir + f'/{k}_results.png'
        file_name_zoom = out_dir + f'/{k}_results_zoom.png'

        # pyffit.figures.plot_fault_panels(panels, mesh, triangles, slip_model[:, 0], figsize=(14.2, 14.2), vlim_disp=vlim_disp, xlim=xlim, ylim=ylim, dpi=300, show=False, file_name=file_name_zoom) 
        pyffit.figures.plot_fault_panels(panels, mesh, triangles, slip_model[:, 0], figsize=(14.2, 14.2), vlim_disp=vlim_disp, dpi=300, show=False, markersize=30, file_name=file_name) 
        # gc.collect()

        # ------------------ Plot ------------------ 
        vlim      = [-10, 10]
        cbar_inc  = 5
        cmap      = cmc.vik
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(disp_model_noisy - ramp, cmap=cmap, vmin=-10, vmax=10, extent=extent,interpolation='none')
        ax.set_ylabel('North (km)')
        ax.set_xlabel('East (km)')
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[-1:-3:-1])
        fig.colorbar(im, ticks=np.arange(vlim[0], vlim[1] + cbar_inc, cbar_inc), label='LOS Displacement (mm)', shrink=0.5)
        fig.savefig(f'{out_dir}/{k}_noisy_data.png', dpi=300)


    # Compute residual statistics
    slip_model_mean        = np.mean(slip_models, axis=0)
    slip_model_std         = np.std(slip_models, axis=0)
    residual_means         = np.mean(slip_model_resids, axis=0)
    residual_stds          = np.std(slip_model_resids, axis=0)
    residual_means_percent = 100*residual_means/slip_model[:, 0]
    residual_stds_percent  = 100*residual_stds/slip_model[:, 0]
    out_file.create_dataset('residual_means',         data=residual_means)
    out_file.create_dataset('residual_stds',          data=residual_stds)
    out_file.create_dataset('residual_means_percent', data=residual_means_percent)
    out_file.create_dataset('residual_stds_percent',  data=residual_stds_percent)
    out_file.create_dataset('slip_model_mean',        data=slip_model_mean)
    out_file.create_dataset('slip_model_std',         data=slip_model_std)

    # Plot 3D fault
    file_name = out_dir + '/slip_mean.png'
    fig, ax = pyffit.figures.plot_fault_3d(mesh, triangles, c=slip_model_mean, edges=True, cmap_name='viridis', cbar_label='Dextral slip (mm)', 
                  vlim_slip=[0, 30], labelpad=10, azim=235, elev=17, n_seg=10, n_tick=6, alpha=1,
                show=False, figsize=(7, 5), cbar_kwargs=dict(location='right', pad=0.05, shrink=0.4))
    fig.savefig(file_name, dpi=500)

    file_name = out_dir + '/slip_std.png'
    fig, ax = pyffit.figures.plot_fault_3d(mesh, triangles, c=slip_model_std, edges=True, cmap_name='viridis', cbar_label='Dextral slip (mm)', 
                  vlim_slip=[0, 30], labelpad=10, azim=235, elev=17, n_seg=10, n_tick=6, alpha=1,
                show=False, figsize=(7, 5), cbar_kwargs=dict(location='right', pad=0.05, shrink=0.4))
    fig.savefig(file_name, dpi=500)

    file_name = out_dir + '/residuals_mean.png'
    fig, ax = pyffit.figures.plot_fault_3d(mesh, triangles, c=residual_means, edges=True, cmap_name='viridis', cbar_label='Dextral slip (mm)', 
                  vlim_slip=[0, 15], labelpad=10, azim=235, elev=17, n_seg=10, n_tick=6, alpha=1,
                show=False, figsize=(7, 5), cbar_kwargs=dict(location='right', pad=0.05, shrink=0.4))
    fig.savefig(file_name, dpi=500)

    file_name = out_dir + '/residuals_std.png'
    fig, ax = pyffit.figures.plot_fault_3d(mesh, triangles, c=residual_stds, edges=True, cmap_name='viridis', cbar_label='Dextral slip (mm)', 
                  vlim_slip=[0, 15], labelpad=10, azim=235, elev=17, n_seg=10, n_tick=6, alpha=1,
                show=False, figsize=(7, 5), cbar_kwargs=dict(location='right', pad=0.05, shrink=0.4))
    fig.savefig(file_name, dpi=500)

    # file_name = out_dir + '/residuals_mean_percent.png'
    # fig, ax = pyffit.figures.plot_fault_3d(mesh, triangles, c=residual_means_compare, edges=True, cmap_name='Reds', cbar_label='Residual mean (%)', 
    #               vlim_slip=[0, 100], labelpad=10, azim=235, elev=17, n_seg=100, n_tick=6, alpha=1,
    #             show=False, figsize=(7, 5), cbar_kwargs=dict(location='right', pad=0.05, shrink=0.4))
    # fig.savefig(file_name, dpi=500)

    # file_name = out_dir + '/residuals_std_percent.png'
    # fig, ax = pyffit.figures.plot_fault_3d(mesh, triangles, c=residual_stds_compare, edges=True, cmap_name='Reds', cbar_label='Residual standard deviation (%)', 
    #               vlim_slip=[0, 100], labelpad=10, azim=235, elev=17, n_seg=100, n_tick=6, alpha=1,
    #             show=False, figsize=(7, 5), cbar_kwargs=dict(location='right', pad=0.05, shrink=0.4))
    # fig.savefig(file_name, dpi=500)

    # Plot synthetic data
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(disp_model, extent=extent, vmin=vlim_disp[0], vmax=vlim_disp[1], cmap='coolwarm')
    ax.set_ylabel('North (km)')
    ax.set_xlabel('East (km)')
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[3], extent[2])
    ax.set_aspect('equal')
    fig.colorbar(im, label='LOS Displacement (mm)', shrink=0.3)
    fig.savefig(f'{out_dir}/input_model.png', dpi=500)

    # Get the memory usage after the computation
    mem_after = psutil.Process().memory_info().rss / 1024 / 1024

    # Calculate the memory used by the computation
    mem_used = mem_after - mem_before
    saveprint(f"Memory used by process_data: {mem_used:.2f} MB")

    out_file.close()
    end = time.time() - start

    # Clean up
    plt.close()
    gc.collect()

    # Get the memory usage after the computation
    mem_after = psutil.Process().memory_info().rss / 1024 / 1024
    mem_used = mem_after - mem_before
    saveprint(f"Memory used by process_data after: {mem_used:.2f} MB")

    if end > 120:
        saveprint(f'Total time for {n_aps_iter} iterations: {end/60:.2f} min')
    else:
        saveprint(f'Total time for {n_aps_iter} iterations: {end:.2f} s')

    return mu, eta, resid, rms, cost


def solve_with_resampling(params):
                          # nu=0.25, avg_strike=0, n_iter=3, rms_min=0.1, nan_frac_max=0.9, width_min=3, width_max=1000, 
                          # vlim_disp=[], vlim_slip=[], xlim=[], ylim=[]):
    """
    # Solve for slip using least-squares
    #    G      m  =  d
    #   ⎡A    ⎤⎡ ⎤   ⎡data_samp⎤
    #   ⎢mu*R ⎥⎢m⎥ = ⎢    0    ⎥
    #   ⎣eta*E⎦⎣ ⎦   ⎣    0    ⎦
    """

    # Start tracking
    start = time.time()
    mem_before = psutil.Process().memory_info().rss / 1024 / 1024

    # Unpack input parameters
    mesh, triangles, x_data, y_data, data, look, GF, slip_model, ramp, mu, R, eta, E, out_dir, poisson_ratio, shear_modulus, avg_strike, rms_min, nan_frac_max, width_min, width_max, n_iter, vlim_disp, vlim_slip, xlim, ylim, extent, slip_lim = params

    # Initialize output stuff
    print(out_dir)
    out_file = h5py.File(f'{out_dir}/results.h5', 'w')

    # Convert observation coordinates into column-matrix (x, y, z)
    pts = np.array([x_data, y_data, 0 * y_data]).reshape((3, -1)).T.copy() 

    saveprint('Beginning iterative inversion...')

    # Perform model-based resampling
    for k in range(n_iter):
        saveprint(f'')
        saveprint(f'# -------------------- Iteration {k} --------------------')

        # Compute forward model prediction
        disp_model = get_full_disp(data, GF, slip_model)

        # Downsample model
        data_extent = [x_data.min(), x_data.max(), y_data.min(), y_data.max()]
        data_index  = np.arange(0, data.size)
        x_samp, y_samp, model_samp, data_samp_std, data_tree, cell_dims, cell_extents, nan_frac = pyffit.quadtree.quadtree_unstructured(x_data.flatten(), y_data.flatten(), disp_model.flatten() - ramp.flatten(), data_index, data_extent, 
                                                                                                                        rms_min=rms_min, nan_frac_max=nan_frac_max, width_min=width_min, width_max=width_max,
                                                                                                                        x_samp=[], y_samp=[], data_samp=[], data_samp_std=[], data_tree=[], data_dims=[], data_extents=[], nan_frac=[])
        saveprint(f'Quadtree points: {len(model_samp)}')

        # Apply quadtree to data
        x_samp, y_samp, data_samp, data_samp_std, nan_frac = pyffit.quadtree.apply_unstructured_quadtree(x_data.flatten(), y_data.flatten(), data.flatten() - ramp.flatten(), data_tree, nan_frac_max)
        _, _, look_e_samp, _, _ = pyffit.quadtree.apply_unstructured_quadtree(x_data.flatten(), y_data.flatten(), look[:, 0], data_tree, nan_frac_max)
        _, _, look_n_samp, _, _ = pyffit.quadtree.apply_unstructured_quadtree(x_data.flatten(), y_data.flatten(), look[:, 1], data_tree, nan_frac_max)
        _, _, look_u_samp, _, _ = pyffit.quadtree.apply_unstructured_quadtree(x_data.flatten(), y_data.flatten(), look[:, 2], data_tree, nan_frac_max)
        look_samp = np.vstack((look_e_samp, look_n_samp, look_u_samp)).T
        i_nans = np.isnan(data_samp)

        # Generate Greens functions for dowmsampled coordinates
        GF_samp = get_fault_greens_functions(x_samp, y_samp, 0 * y_samp, mesh, triangles, nu=poisson_ratio, verbose=True)

        # Project Greens functions into LOS
        GF_samp_proj = proj_greens_functions(GF_samp, look_samp)

        # Include only strike-slip contribution
        A = GF_samp_proj[~i_nans, :, 0] 

        # Form full design matrix
        regularize = True

        if regularize:
            G = np.vstack((A, mu*R, eta*E)) # Add regularization        
            # G = np.hstack((G, np.ones((G.shape[0], 1)))) # Add translation
            d = np.hstack((data_samp[~i_nans], np.zeros(R.shape[0] + E.shape[0]))) # Pad data vector with zeros
        else:
            G = A
            d = data_samp[~i_nans]

        saveprint(f'G: {G.shape}')
        saveprint(f'd: {d.shape}')

        # Get model bounds
        # bounds    = (model_min, model_max)
        # bounds    = (0, 1000)
        bounds = slip_lim

        # Solve using bounded least squares to enforce slip direction constraint
        results         = lsq_linear(G, d, bounds=bounds)
        slip_model      = np.column_stack((results.x, np.zeros_like(results.x), np.zeros_like(results.x)))
        data_samp_model = A.dot(slip_model[:, 0].flatten())
        
        # Get residuals
        resid = results.fun[:len(data_samp[~i_nans])]
        cost  = results.cost
        rms   = np.sqrt(np.sum(resid**2)/len(resid))

        # Compute full model residuals
        full_resid = data - disp_model.reshape(data.shape)

        # Update ramp
        ramp = pyffit.corrections.fit_ramp(x_data, y_data, full_resid)

        # Compute magnitude
        patch_area   = np.empty(len(slip_model))
        patch_slip   = np.empty(len(slip_model))
        patch_moment = np.empty(len(slip_model))

        for i in range(len(slip_model)):
            patch = mesh[triangles[i]]

            # Get patch area via cross product
            ab = patch[0, :] - patch[1, :]
            ac = patch[0, :] - patch[2, :]
            patch_area[i] = np.linalg.norm(np.cross(ab, ac), ord=2)/2 * 1e6 # convert from km^3 to m^3

            # Get slip magnitude
            patch_slip[i] = np.linalg.norm(slip_model[i, :], ord=2) * 1e-3 # convert from mm to m

            # Compute magnitude
            patch_moment[i] = shear_modulus * patch_area[i] * patch_slip[i] * 1e7 # convert from N-m to dyne-cm


        print(f'{shear_modulus*1e-9} GPa')


        # Get moment and moment magnitude
        moment = np.sum(patch_moment)
        magnitude = (2/3) * np.log10(moment) - 10.7


        # Display model info
        saveprint()
        saveprint(f'Model information:')
        saveprint(f'Slip range  = {slip_model[:, 0].min():.1f} {slip_model[:, 0].max():.1f}')
        print(f'Moment: {moment:.2e} dyne-cm')
        print(f'M_w: {magnitude:.1f}')
        # saveprint(f'Translation = {shift:.1f}')
        saveprint(f'Mean residual = {np.nanmean(np.abs(resid)):.2f}')
        saveprint()
        gc.collect()
        
        # Save some things to disk
        # out_file.create_dataset(f'Iteration_{k}/init_disp', data=disp_model)
        # out_file.create_dataset(f'Iteration_{k}/init_slip', data=slip_model)
        # out_file.create_dataset(f'Iteration_{k}/quadtree/x_model', data=x_samp)
        # out_file.create_dataset(f'Iteration_{k}/quadtree/y_model', data=y_samp)
        # out_file.create_dataset(f'Iteration_{k}/quadtree/model',  data=model_samp)
        # out_file.create_dataset(f'Iteration_{k}/quadtree/x_data', data=x_samp)
        # out_file.create_dataset(f'Iteration_{k}/quadtree/y_data', data=y_samp)
        # out_file.create_dataset(f'Iteration_{k}/quadtree/data',   data=data_samp)
        out_file.create_dataset(f'Iteration_{k}/slip_model', data=slip_model)
        out_file.create_dataset(f'Iteration_{k}/magnitude', data=magnitude)
        # out_file.create_dataset(f'Iteration_{k}/quadtree/greens_functions', data=GF_samp_proj)
        # out_file.create_dataset(f'Iteration_{k}/disp_model', data=data_samp_model)


        # -------------------------- Plots --------------------------
        # # Plot true synthetic displacements
        # panels = [
        #           dict(x=x_data.flatten(), y=y_data.flatten(), data=data.flatten() - ramp.flatten(), label=r'Data'),
        #           dict(x=x_data.flatten(), y=y_data.flatten(), data=disp_model.flatten(), label=r'Model'),
        #           dict(x=x_data.flatten(), y=y_data.flatten(), data=data.flatten() - ramp.flatten() - disp_model.flatten() , label=r'Residual'),
        #          ]

        # pyffit.figures.plot_fault_panels(panels, mesh, triangles, slip_model[:, 0], vlim_disp=vlim_disp, xlim=xlim, ylim=ylim, file_name=f'{out_dir}/Forward_model_{k}_zoom.png') 
        # pyffit.figures.plot_fault_panels(panels, mesh, triangles, slip_model[:, 0], show=False, vlim_disp=vlim_disp, file_name=f'{out_dir}/Forward_model_{k}.png') 

        # # Plot ramp
        # fig, ax = plt.subplots(figsize=(14, 8.2))
        # im = ax.imshow(ramp, extent=extent, vmin=vlim_disp[0], vmax=vlim_disp[1], cmap='coolwarm')
        # ax.set_ylabel('North (km)')
        # ax.set_xlabel('East (km)')
        # ax.set_xlim(extent[0], extent[1])
        # ax.set_ylim(extent[3], extent[2])
        # ax.set_aspect('equal')

        # fig.colorbar(im, label='LOS Displacement (mm)', shrink=0.3)
        # plt.savefig(f'{out_dir}/Ramp_{k}.png', dpi=300)

        # # Plot quadtree
        # plot_quadtree(data - ramp, extent, (x_samp, y_samp), data_samp, mesh, vlim_disp=vlim_disp, file_name=f'{out_dir}/Quadtree_{k}.png',)

        # Plot results
        panels = [
                      dict(x=x_samp, y=y_samp,                   data=data_samp,                            label='Downsampled data'),
                      dict(x=x_samp[~i_nans], y=y_samp[~i_nans], data=data_samp_model,                      label=rf'Model ($\mu$ = {mu:.1e}, $\eta$ = {eta:.1e})'),
                      dict(x=x_samp[~i_nans], y=y_samp[~i_nans], data=data_samp[~i_nans] - data_samp_model, label=rf'Residuals ({np.nanmean(np.abs(resid)):.2f} $\pm$ {np.nanstd(np.abs(resid)):.2f}) mm'),
                     ]

        file_name      = out_dir + f'/Results_{k}_eta-{eta:.10f}_mu-{mu:.10f}.png'
        file_name_zoom = out_dir + f'/Results_{k}_eta-{eta:.10f}_mu-{mu:.10f}_zoom.png'

        pyffit.figures.plot_fault_panels(panels, mesh, triangles, slip_model[:, 0], figsize=(14.2, 14.2), vlim_disp=vlim_disp, xlim=xlim, ylim=ylim, dpi=300, show=False, file_name=file_name_zoom) 
        pyffit.figures.plot_fault_panels(panels, mesh, triangles, slip_model[:, 0], figsize=(14.2, 14.2), vlim_disp=vlim_disp, dpi=300, show=False, markersize=30, file_name=file_name) 
        
        gc.collect()


    # # Plot 3D fault
    file_name = out_dir + f'/Fault_3D_{k}_mu-{mu:.1e}_eta-{eta:.1e}.png'.replace('e+0', 'e+').replace('e-0', 'e-')
    # pyffit.figures.plot_fault_3d(mesh, triangles, c=slip_model[:, 0], edges=True, cmap_name='viridis', cbar_label='Slip (mm/yr)', 
    #               labelpad=20, azim=240, elev=17, n_seg=100, n_tick=7, alpha=1,
    #               filename=file_name, show=False, dpi=300)
    # file_name = f'{result_dir}/fault.png'
    fig, ax = pyffit.figures.plot_fault_3d(mesh, triangles, c=slip_model[:, 0], edges=True, cmap_name='viridis', cbar_label='Dextral slip (mm)', 
                  vlim_slip=[0, 35], labelpad=10, azim=235, elev=17, n_seg=100, n_tick=8, alpha=1,
                show=False, figsize=(7, 5), cbar_kwargs=dict(location='right', pad=0.05, shrink=0.4))

    # ax.scatter(0, 0, 0.33, marker='^', s=50, facecolor='gold', edgecolor='k')
    fig.savefig(file_name, dpi=500)

    # Get the memory usage after the computation
    mem_after = psutil.Process().memory_info().rss / 1024 / 1024

    # Calculate the memory used by the computation
    mem_used = mem_after - mem_before
    saveprint(f"Memory used by process_data: {mem_used:.2f} MB")

    out_file.close()
    end = time.time() - start

    # Clean up
    plt.close()
    del mesh
    del triangles 
    del x_data 
    del y_data 
    del data 
    del look 
    del GF
    del GF_samp
    del GF_samp_proj
    del A
    del G
    del d
    del pts
    del x_samp
    del y_samp
    del model_samp
    del look_samp
    del data_samp_std
    del data_tree
    del cell_dims
    del cell_extents
    del nan_frac
    del full_resid
    # del resid
    del disp_model
    gc.collect()

    # Get the memory usage after the computation
    mem_after = psutil.Process().memory_info().rss / 1024 / 1024
    mem_used = mem_after - mem_before
    saveprint(f"Memory used by process_data after: {mem_used:.2f} MB")



    if end > 120:
        saveprint(f'Total time for {n_iter} iterations: {end/60:.2f} min')
    else:
        saveprint(f'Total time for {n_iter} iterations: {end:.2f} s')

    return mu, eta, resid, rms, cost


def edge_sensitivity_test(mesh, triangles, x_utm, y_utm, data, look, init_disp_model, init_slip_model, ramp, mu, R, eta_range, E, out_dir, nu, avg_strike, 
                          rms_min, nan_frac_max, width_min, width_max, n_iter, vlim_disp, vlim_slip, xlim, ylim, extent, slip_lim):

    # Perform sensitivity test for eta
    params = []
    for eta in eta_range:
        out_dir = f'Results/edge_slip_test/mu-{mu:.1e}_eta-{eta:.1e}'.replace('e+0', 'e+').replace('e-0', 'e-')
        pyffit.utilities.check_dir_tree(out_dir)
        params.append((mesh, triangles, x_utm, y_utm, data, look, init_disp_model, init_slip_model, ramp, mu, R, eta, E, out_dir, nu, avg_strike, rms_min, nan_frac_max, width_min, width_max, n_iter, vlim_disp, vlim_slip, xlim, ylim, extent, slip_lim))

    os.environ["OMP_NUM_THREADS"] = "1"
    start       = time.time()
    n_processes = multiprocessing.cpu_count()
    pool        = multiprocessing.Pool(processes=n_processes)
    results     = pool.map(solve_with_resampling, params)
    pool.close()
    pool.join()
    end         = time.time() - start
    saveprint(f'Sensitivity test computation time for {len(eta_range)} models: {end:.2f}')

    mu     = np.array([result[0] for result in results])
    eta    = np.array([result[1] for result in results])
    resids = np.array([result[2] for result in results])
    rms    = np.array([result[3] for result in results])

    output = np.vstack((eta, rms)).T

    np.savetxt(f'Results/edge_slip_test/sensitivity_results.txt', output)
    np.savetxt(f'Results/edge_slip_test/rms.txt', rms)
    np.savetxt(f'Results/edge_slip_test/eta.txt', eta)
    return


def smoothness_sensitivity_test(mesh, triangles, x_utm, y_utm, data, look, init_disp_model, init_slip_model, ramp, mu_range, R, eta, E, out_dir, nu, avg_strike, 
                                rms_min, nan_frac_max, width_min, width_max, n_iter, vlim_disp, vlim_slip, xlim, ylim, extent, slip_lim):

    # Perform sensitivity test for eta
    params = []
    for mu in mu_range:
        # out_dir = f'Results/smoothness_test/mu-{mu:.1e}_eta-{eta:.1e}'.replace('e+0', 'e+').replace('e-0', 'e-')
        out_dir = f'Results/smoothness_test/mu-{mu:.10f}_eta-{eta:.10f}'
        pyffit.utilities.check_dir_tree(out_dir)
        params.append((mesh, triangles, x_utm, y_utm, data, look, init_disp_model, init_slip_model, ramp, mu, R, eta, E, out_dir, nu, avg_strike, rms_min, nan_frac_max, width_min, width_max, n_iter, vlim_disp, vlim_slip, xlim, ylim, extent, slip_lim))

    os.environ["OMP_NUM_THREADS"] = "1"
    start       = time.time()
    n_processes = multiprocessing.cpu_count()
    n_processes = 4
    pool        = multiprocessing.Pool(processes=n_processes)
    results     = pool.map(solve_with_resampling, params)
    pool.close()
    pool.join()
    end         = time.time() - start
    saveprint(f'Sensitivity test computation time for {len(mu_range)} models: {end:.2f}')

    mu     = np.array([result[0] for result in results])
    eta    = np.array([result[1] for result in results])
    resids = np.array([result[2] for result in results])
    rms    = np.array([result[3] for result in results])

    output = np.vstack((mu, rms)).T

    np.savetxt(f'Results/smoothness_test/sensitivity_results.txt', output)
    np.savetxt(f'Results/smoothness_test/rms.txt', rms)
    np.savetxt(f'Results/smoothness_test/mu.txt', mu)
    return


def grid_search(mesh, triangles, x_utm, y_utm, data, look, GF, init_slip_model, ramp, mu_range, R, eta_range, E, out_dir, poisson_ratio, shear_modulus, avg_strike, 
                                rms_min, nan_frac_max, width_min, width_max, n_iter, vlim_disp, vlim_slip, xlim, ylim, extent, slip_lim):
    

    pyffit.utilities.check_dir_tree(f'Results/grid_search/results')
    pyffit.utilities.check_dir_tree(f'Results/grid_search/results/zoom')

    # Perform sensitivity test for eta
    params = []
    for mu in mu_range:
        for eta in eta_range:
            # print(mu, eta)
            # out_dir = f'Results/smoothness_test/mu-{mu:.1e}_eta-{eta:.1e}'.replace('e+0', 'e+').replace('e-0', 'e-')
            out_dir = f'Results/grid_search/mu-{mu:.10f}_eta-{eta:.10f}'
            pyffit.utilities.check_dir_tree(out_dir)
            params.append((mesh, triangles, x_utm, y_utm, data, look, GF, init_slip_model, ramp, mu, R, eta, E, out_dir, poisson_ratio, shear_modulus, avg_strike, rms_min, nan_frac_max, width_min, width_max, n_iter, vlim_disp, vlim_slip, xlim, ylim, extent, slip_lim))

    saveprint('Initiating grid search:')
    saveprint(rf'mu  = {mu_range.min():.2e} - {mu_range.max():.2e} ({len(mu_range)}) samples')
    saveprint(rf'eta = {eta_range.min():.2e} - {eta_range.max():.2e} ({len(eta_range)}) samples')
    saveprint(rf'{len(mu_range)*len(eta_range)} total samples')

    # Perform grid search
    os.environ["OMP_NUM_THREADS"] = "1"
    start       = time.time()
    n_processes = multiprocessing.cpu_count()
    n_processes = 4
    pool        = multiprocessing.Pool(processes=n_processes)
    results     = pool.map(solve_with_resampling, params)
    pool.close()
    pool.join()
    end         = time.time() - start
    saveprint(f'Grid search computation time for {len(mu_range)*len(eta_range)} models: {end:.2f} ({end/(len(mu_range)*len(eta_range))} per model)')

    mu     = np.array([result[0] for result in results])
    eta    = np.array([result[1] for result in results])
    resids = np.array([result[2] for result in results])
    rms    = np.array([result[3] for result in results])
    cost   = np.array([result[4] for result in results])

    # Organize plots
    result_files = glob.glob('Results/grid_search/mu*/Result*png')
    for file in result_files:
        shutil.copy(file, 'Results/grid_search/results/' + file.split('/')[-1])

    zoom_result_files = glob.glob('Results/grid_search/results/*zoom.png')
    for file in zoom_result_files:
        shutil.move(file, 'Results/grid_search/results/zoom/' + file.split('/')[-1])


    # Save results
    df = pd.DataFrame({'mu': mu, 'eta': eta, 'rms': rms, 'cost': cost})
    df.to_csv(f'Results/grid_search/grid_search_results.txt', index=False, sep=' ')

    # Make result plots
    plot_grid_search(rms, mu, eta, label='RMS (mm)', show=True, file_name='Results/grid_search/grid_search_rms.png')
    plot_grid_search(cost, mu, eta, label='Cost (mm)', show=True, file_name='Results/grid_search/grid_search_cost.png')

    return


# -------------------------- Plots ----------------------------------
def plot_noise_panels(x, y, noise, corr_length, projection='geographic', file_name='', show=False, dpi=300):

    """
    Plot east/north/vertical noise components.
    """

    if projection == 'geographic':
        labels = ['East noise', 'North noise', 'Vertical noise']
        xlabel = 'East (km)'
        ylabel = 'North (km)'

    elif projection == 'fault':
        labels = ['Fault parallel noise', 'Fault perpendicular noise', 'Vertical noise']
        xlabel = 'East (km)'
        ylabel = 'North (km)'

    else:
        labels = ['X noise', 'Y noise', 'Z noise']
        xlabel = 'X (km)'
        ylabel = 'Y (km)'

    extent = [x.min(), x.max(), y.max(), y.min()]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, N, label in zip(axes, noise, labels):
        im = ax.imshow(N, extent=extent)
        ax.set_title(label + r' ($L_c = $' + f'{corr_length})')
        ax.set_xlabel(xlabel)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())

    axes[0].set_ylabel(ylabel)
    axes[1].set_yticks([])
    axes[2].set_yticks([])

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.25, 0.02, 0.5])
    fig.colorbar(im, cax=cbar_ax)

    if len(file_name) > 0:
        fig.savefig(file_name, dpi=dpi)

    if show:
        plt.show()

    return fig, axes


def plot_quadtree(data, extent, samp_coords, samp_data, fault_mesh, original_data=[], cell_extents=[], cmap_disp='coolwarm', vlim_disp=[], figsize=(14, 8.2), file_name='', show=False, dpi=300):
    """
    Compare gridded and quadtree downsampled displacements
    """

    if len(vlim_disp) == 0:
        vlim_disp = 0.7*np.nanmax(np.abs(data))

    fig   = plt.figure(figsize=figsize)
    gs    = fig.add_gridspec(1, 3, width_ratios=(1, 1, 0.05), height_ratios=(1,))
    ax0   = fig.add_subplot(gs[0, 0])
    ax1   = fig.add_subplot(gs[0, 1])
    cax   = fig.add_subplot(gs[0, 2])
    axes  = [ax0, ax1]


    trace = fault_mesh[:, :2][fault_mesh[:, 2] == 0]

    im = axes[0].imshow(data, extent=extent, vmin=vlim_disp[0], vmax=vlim_disp[1], cmap=cmap_disp)
    axes[0].plot(trace[:, 0], trace[:, 1], linewidth=2, c='k')

    if len(original_data) > 0:
        axes[1].scatter(original_data[0].flatten(), original_data[1].flatten(), c='k', marker='.', s=1)

    im = axes[1].scatter(samp_coords[0], samp_coords[1], c=samp_data, vmin=vlim_disp[0], vmax=vlim_disp[1], cmap=cmap_disp, marker='.', s=markersize)
    axes[1].plot(trace[:, 0], trace[:, 1], linewidth=2, c='k')

    if len(cell_extents) > 0:
        for cell_extent in cell_extents:
            for ax in axes:
                # Create a rectangle patch
                cell = Rectangle((cell_extent[0], cell_extent[2]), cell_extent[1] - cell_extent[0], cell_extent[3] - cell_extent[2], fill=False, edgecolor='k', linewidth=0.25)

                # Add the rectangle patch to the axes
                ax.add_patch(cell)



    axes[0].set_ylabel('North (km)')
    axes[0].set_xlabel('East (km)')
    axes[0].set_xlim(extent[0], extent[1])
    axes[0].set_ylim(extent[3], extent[2])

    axes[1].set_xlabel('East (km)')
    axes[1].set_ylabel('')
    axes[1].set_yticks([])
    axes[1].set_xlim(extent[0], extent[1])
    axes[1].set_ylim(extent[3], extent[2])

    ax1.set_aspect('equal')
    ax0.set_aspect('equal')

    fig.colorbar(im, cax=cax, label='LOS Displacement (mm)', shrink=0.3)

    if len(file_name) > 0:
        fig.savefig(file_name, dpi=dpi)

    if show:
        plt.show()

    plt.close()

    return fig, axes


def plot_mesh_yz(mesh, triangles, selected, file_name='', show=False,dpi=300):

    R = np.zeros((len(triangles), len(triangles)))

    fig, ax = plt.subplots(figsize=(14, 8.2))

    # Plot edges
    for tri in triangles:
        pts   = mesh[tri]
        edges = Polygon(list(zip(pts[:, 1], pts[:, 2])), edgecolor='k', facecolor='none', alpha=1, linewidth=0.5)
        ax.add_patch(edges)

    # Loop over each fault patch
    for i, tri in enumerate(selected):

        # Plot patch
        pts = mesh[tri]
        face = Polygon(list(zip(pts[:, 1], pts[:, 2])), color='C0', alpha=1, linewidth=0.1)
        # ax.add_patch(face)

        # Plot centroid
        (x, y, z) = np.mean(pts, axis=0)
        ax.scatter(y, z, c='C1', s=5, zorder=10)
        ax.text(y, z, str(i))

        # Get neighboring patches (share edge)
        neighbors = []
        for cmb in combinations(tri, 2):

            # Check if each row contains the values
            for j, tri0 in enumerate(triangles):
                result = np.isin(cmb, tri0)

                # Neighbor triangle if two vertices are shared and not the original triangle
                if (sum(result) == 2) & (i != j):
                    # Compute centroid distance
                    pts0 = mesh[tri0]
                    dr = (np.sum((np.mean(pts0, axis=0) - np.mean(pts, axis=0))**2))**0.5

                    # Assign finite differences
                    R[i, j] += 1/dr
                    R[i, i] -= 1/dr


                    # # Plot centroid
                    # (x0, y0, z0) = np.mean(pts0, axis=0)
                    # ax.scatter(y0, z0, c='C1',  zorder=10)
                    # ax.text(y0, z0, str(j))

        # for j in neighbors:
        #     pts = mesh[triangles[j]]
        #     face = Polygon(list(zip(pts[:, 1], pts[:, 2])), color='C1', alpha=1, linewidth=0.1)
            # ax.add_patch(face)

    ax.set_xlim(mesh[:, 1].min(), mesh[:, 1].max())
    ax.set_ylim(mesh[:, 2].min(),  mesh[:, 2].max())

    if len(file_name) > 0:
        fig.savefig(file_name, dpi=dpi)
    
    if show:
        plt.show()


def plot_grid_search(cost, mu, eta, label='', log=False, file_name='', show=False, dpi=300, vlim=[], mu_pref=0.6951927962, eta_pref=1.4384498883):
    """
    Plot heatmap and marginal distributionss of grid search.
    """

    if log:
        cost_heatmap = np.log(cost)
    else:
        cost_heatmap = cost

    if len(vlim) != 2:
        vlim = [np.min(cost_heatmap), np.max(cost_heatmap)]

    # Plot
    fig = plt.figure(figsize=(8, 8))

    grid = fig.add_gridspec(2, 4,  width_ratios=(3, 1, 0.5, 0.1), height_ratios=(3, 1),
    #                       left=0, right=1, bottom=0, top=1,
                          # left=grid_bbox[0], right=grid_bbox[1], bottom=grid_bbox[2], top=grid_bbox[3] ,
                          wspace=0.1, hspace=0.1)
    # legend_elements = []

    # Create the Axes.
    ax2 = fig.add_subplot(grid[1, 0])
    ax0 = fig.add_subplot(grid[0, 0], sharex=ax2)
    ax1 = fig.add_subplot(grid[0, 1], sharey=ax0)
    cax = fig.add_subplot(grid[0, 3])

    im = ax0.scatter(mu, eta, c=cost_heatmap, cmap='plasma_r', marker='s', s=1200, vmin=vlim[0], vmax=vlim[1])

    n_eta       = len(np.unique(eta))
    n_mu        = len(np.unique(eta))
    eta_rms_sum = np.zeros(n_eta)
    mu_rms_sum  = np.zeros(n_mu)

    for mu0 in np.unique(mu):  

        mu_select = mu.flatten()[mu.flatten() == mu0]
        rms_select = cost.flatten()[mu.flatten() == mu0]

        rms_sort    = np.array([rms0 for _, rms0 in sorted(zip(mu_select, rms_select))])
        eta_sort    = np.sort(np.unique(eta))
        mu_rms_sum += rms_sort

        ax1.plot(rms_sort, eta_sort, c='gainsboro')


    for eta0 in np.unique(eta):  

        eta_select = eta.flatten()[eta.flatten() == eta0]
        rms_select = cost.flatten()[eta.flatten() == eta0]

        rms_sort     = np.array([rms0 for _, rms0 in sorted(zip(eta_select, rms_select))])
        mu_sort      = np.sort(np.unique(mu))
        eta_rms_sum += rms_sort

        ax2.plot(mu_sort, rms_sort, c='gainsboro')

    ax1.plot(mu_rms_sum/n_eta, eta_sort, c='k')
    ax2.plot(mu_sort, eta_rms_sum/n_mu, c='k')
    ax1.scatter((mu_rms_sum/n_eta)[np.argmin(np.abs(eta_sort - eta_pref))], eta_pref, c='C0', marker='o', zorder=100)
    ax2.scatter(mu_pref, (eta_rms_sum/n_mu)[np.argmin(np.abs(mu_sort - mu_pref))],  c='C0', marker='o', zorder=100)
    ax0.scatter(mu_pref, eta_pref, c='C0', marker='o')

    # Settings
    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.xaxis.set_label_position("top")
    ax0.xaxis.tick_top()
    ax0.set_xlabel(r'Smoothing weight $\mu$')
    ax0.set_ylabel(r'Boundary slip weight $\eta$')

    ax1.set_yscale('log')
    ax1.set_xlabel(label)
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.set_ylabel(r'Boundary slip weight $\eta$')

    ax2.set_xscale('log')
    ax2.set_xlabel(r'Smoothing weight $\mu$')
    ax2.set_ylabel(label)

    if log:
        ax1.set_xscale('log')
        ax2.set_yscale('log')

    # ax0.set_aspect('equal')
    # ax1.set_aspect('equal')
    # ax2.set_aspect('equal')


    ax0.set_facecolor('gainsboro')
    fig.colorbar(im, cax=cax, label=label, shrink=0.1)

    if len(file_name) > 0:
        fig.savefig(file_name, dpi=dpi)
    
    if show:
        plt.show()

    return 


# fig = plt.figure(figsize=(10, 8.2))
# projection = ccrs.UTM(11)
# ax = fig.add_subplot(1, 1, 1)
# im = ax.imshow(data, extent=extent, interpolation='None', vmin=vlim_disp[0], vmax=vlim_disp[1], cmap=cmap_disp, )
# ax.plot(mesh[:n_top_vertex, 0], mesh[:n_top_vertex, 1], linewidth=2, c='k', )
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# ax.set_xticks(xticks)
# ax.set_yticks(yticks)
# ax.set_xlabel('East (km)')
# ax.set_ylabel('North (km)')
# ax.set_aspect('equal')
# fig.colorbar(im, label='LOS Displacement (mm/yr)')
# fig.savefig('Figures/quadtree_data.png')

# fig = plt.figure(figsize=(10, 8.2))
# projection = ccrs.UTM(11)
# ax  = fig.add_subplot(1, 1, 1)
# im = ax.scatter(x_samp, y_samp, c=data_samp, marker='.', s=20, vmin=vlim_disp[0], vmax=vlim_disp[1], cmap=cmap_disp)
# ax.plot(mesh[:n_top_vertex, 0], mesh[:n_top_vertex, 1], linewidth=2, c='k')
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# ax.set_xticks(xticks)
# ax.set_yticks(yticks)
# ax.set_xlabel('East (km)')
# ax.set_ylabel('North (km)')
# ax.set_aspect('equal')
# fig.colorbar(im, label='LOS Displacement (mm/yr)')
# fig.savefig('Figures/quadtree_samp.png')

# fig = plt.figure(figsize=(10, 8.2))
# projection = ccrs.UTM(11)
# ax  = fig.add_subplot(1, 1, 1, projection=projection)
# im = ax.scatter(x_samp, y_samp, c=data_samp_std, marker='.', cmap='viridis', vmin=0, vmax=3)
# ax.plot(mesh[:n_top_vertex, 0], mesh[:n_top_vertex, 1], linewidth=2, c='k')
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# ax.set_xticks(xticks)
# ax.set_yticks(yticks)
# ax.set_xlabel('East (km)')
# ax.set_ylabel('North (km)')
# ax.set_aspect('equal')
# fig.colorbar(im, label='Standard deviation (mm/yr)')
# fig.savefig('Figures/quadtree_std.png')


if __name__ == '__main__':
    main()