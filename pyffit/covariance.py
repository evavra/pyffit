import os
import time
import h5py
import pyffit
import datetime
import numpy as np
import multiprocessing
import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from matplotlib import colors
from numba import jit

# Pyffit imports
from pyffit.utilities import check_dir_tree
from pyffit.figures import plot_grid
from pyffit.data import modify_hdf5 


def main():

    return

def plot_covariance_evolution():
    cov_dir = '/Users/evavra/Projects/SSAF/Data/InSAR/Sentinel_1/timeseries/covariance'
    sv_file = h5py.File(f'{cov_dir}/semivariogram_params.h5', 'r')

    r      = sv_file['dist'][()]
    n_date = len(sv_file['mask_dist_3_km'].keys())
    sv_params = np.empty((n_date, 3))

    for i, date in enumerate(sv_file['mask_dist_3_km'].keys()):
        sv_params[i, :] = sv_file[f'mask_dist_3_km/{date}'][()]

    fig, ax = plt.subplots(figsize=(5,4))
    axr = ax.twinx()

    ax.set_title(r'Covariance parameters: $s_0 + (s - s_0) \exp(-\frac{h}{r})$')
    line0, = ax.plot(sv_params[:, 0], c='C0', label=r'$s_0$')
    line1, = ax.plot(sv_params[:, 1], c='C3', label=r'$s$')
    line2, = axr.plot(sv_params[:, 2], c='k', label=r'$r$')

    ax.set_xlabel(r'Distance $h$ (km)')
    ax.set_ylabel(r'Covariance ($mm^2$)')
    axr.set_ylabel(r'Range (km)')

    plt.legend(handles=[line0, line1, line2])
    plt.show()

    return


def estimate_covariances():
    # Data info
    mesh_file     = '/Users/evavra/Projects/SSAF/Analysis/Finite_Fault_Modeling/Mesh/Geometry/mesh_points_simple.txt'
    triangle_file = '/Users/evavra/Projects/SSAF/Analysis/Finite_Fault_Modeling/Mesh/Geometry/mesh_connectivity_simple.txt'
    insar_dir     = '/Users/evavra/Projects/SSAF/Data/InSAR/Sentinel_1/timeseries/decomposed/filt/'
    mask_dir      = '/Users/evavra/Projects/SSAF/Data/InSAR/Sentinel_1/timeseries/masks'
    cov_dir       = '/Users/evavra/Projects/SSAF/Data/InSAR/Sentinel_1/timeseries/covariance'

    # file_format = 'asc_spline_2021121*_interp_filt_10km.grd'
    # file_format = 'asc_spline_*_interp_filt_10km.grd'
    # date_index_range    = [-7, -4]
    file_format = 'u_para_*_filt_10km.grd'
    date_index_range    = [-22, -14]
    xkey                = 'lon'

    # Geographic parameters
    EPSG        = '32611' 
    ref_point   = [-116, 33.5]
    data_region = [-116.4, -115.7, 33.25, 34.0]
    avg_strike  = 315.8
    trace_inc   = 0.01
    
    # Fault parameters
    poisson_ratio   = 0.25       # Poisson ratio
    shear_modulus   = 6 * 10**9  # Shear modulus (Pa)
    slip_components = [0] # slip components to use [0 for strike-slip, 1 for dip-slip, 2 for opening]

    # Covariance estimation parameters
    # mask_dists = np.arange(0, 16, 1) 
    mask_dists = [3]
    n_samp = 2*10**7
    r_inc  = 0.2
    r_max  = 80
    c_max  = 2
    sv_max = 2
    
    # ---------------------------------------- Analysis ----------------------------------------
    # Check/create directories
    check_dir_tree(mask_dir)
    check_dir_tree(cov_dir)

    # Load fault model and regularization matrices
    fault = pyffit.finite_fault.TriFault(mesh_file, triangle_file, slip_components=slip_components, poisson_ratio=poisson_ratio, shear_modulus=shear_modulus, verbose=False, trace_inc=trace_inc)

    # Load data
    dataset = pyffit.data.load_insar_dataset(insar_dir, file_format, 'data', ref_point, xkey=xkey, coord_type='geographic', date_index_range=date_index_range, check_lon=False, reference_time_series=True, incremental=False, use_dates=False, use_datetime=False)
    x       = dataset.coords['x'].compute().data.flatten()
    y       = dataset.coords['y'].compute().data.flatten()
    dims    = dataset['z'].isel(date=-1).shape
    

def estimate(x, y, fault, dataset, mask_dists=[3], n_samp=2*10**7, r_inc=0.2, r_max=80, m0=[0.9, 1.5, 5], c_max=2, sv_max=2, mask_dir='.', cov_dir='.', ):
    """
    Estimate covariances from data.
    """

    check_dir_tree(mask_dir)
    check_dir_tree(cov_dir)

    sv_file = h5py.File(f'{cov_dir}/semivariogram_params.h5', 'w')

    for k, date in enumerate(dataset['date']):
        # Format if datetime object
        try: 
            date = date.dt.strftime('%Y-%m-%d')
        except AttributeError:
            continue
        
        data = dataset['z'].sel(date=date).compute().data.flatten()

        print(f'\nWorking on {date} ({k+1}/{len(dataset["date"])}')
        
        SV         = []
        SV_params  = []
        
        for mask_dist in mask_dists:
            # Get fault mask    
            mask_file = f'{mask_dir}/fault_mask_{mask_dist}_km.h5'

            if os.path.exists(mask_file):
                # print(f'##### Loading d = {mask_dist} mask #####')

                with h5py.File(mask_file, 'r') as file:
                    mask = file['mask'][()]
            else:
                mask = get_fault_mask(x, y, fault, mask_dist=mask_dist, out_dir=mask_dir)

                file = h5py.File(mask_file, 'w')
                file.create_dataset('mask', data=mask)
                file.close()

            # # Plot mask
            # mask_inv = np.ones_like(mask) * 0.5
            # mask_inv[~np.isnan(mask)] = np.nan
            # file_name = f'{mask_dir}/plot_fault_mask_{mask_dist}_km.png'

            # fig, ax = plot_grid(x, y, (data*mask).reshape(dims), figsize=(7, 6), cmap='coolwarm', vlim=[-2, 2], cbar=True, 
            #           title=f'Mask distance = {mask_dist} km', xlabel='East (km)', ylabel='North (km)', clabel='Displacement (mm)')
            # ax.imshow(mask_inv.reshape(dims), interpolation='none', extent=[np.min(x), np.max(x), np.max(y), np.min(y)], cmap='Greys_r', vmin=0, vmax=1, alpha=0.5)
            # plt.savefig(file_name, dpi=300)

            # Compute the semivariogram
            r, sv, sv_count = semivariogram(x, y, data * mask, n_samp=n_samp, r_inc=r_inc, r_max=r_max)

            # Fit exponential model
            sv_params = fit_semivariogram(r[:-1], sv, m0=m0, r_max=r_max).x

            SV.append(sv)
            SV_params.append(sv_params)

            modify_hdf5(sv_file, f'mask_dist_{mask_dist}_km/{date}', sv_params)

            v0  = r'$s_0$'
            v1  = r'$s$'
            v2  = r'$r$'
            u0  = r'$km$'
            u1  = r'$mm^2$'

            # Make plots
            plot_covariance(r, sv, sv_params, title=f'Mask dist. = {mask_dist} {u0} | {v0} = {sv_params[0]:.2f} {u1} | {v1} = {sv_params[1]:.2f} {u1} | {v2} = {sv_params[2]:.2f} {u0}',
                            file_name=f'{cov_dir}/semivariogram_mask_{mask_dist}_km_{date}.png', c_max=c_max, sv_max=sv_max)

            # Make covariance matrix


    modify_hdf5(sv_file, 'dist', r)

    sv_file.close()

    # plot_covariance_overlay(r, SV, SV_model, mask_dists, ymax=ymax, file_name=f'{date_dir}/semivariograms_mask_dist.png')

    # ---------------------------------------- Plot ---------------------------------------- 
    # fig = plt.figure(figsize=(6, 6))
    # grid = fig.add_gridspec(2, 1, height_ratios=(1, 3), hspace=0.1)

    # # Create the axes
    # ax0 = fig.add_subplot(grid[0, 0])
    # ax1 = fig.add_subplot(grid[1, 0], sharex=ax0)

    # ax0.plot(r[:-1]+ r_inc/2, sv_count)
    # ax0.set_yscale('log')

    # ax0.set_ylabel('Count')
    # ax0.xaxis.tick_top()
    # ax0.set_xlabel(r'Distance (km)')
    # ax0.xaxis.set_label_position("top")
    return
        
def plot_covariance(r, sv, sv_params, figsize=(6, 3), title='', c_max=2, sv_max=2, dpi=500, file_name='', show=False):
    """
    Plot semivariogram.
    """

    r_inc = r[1] - r[0]

    # Plot
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(1, 2, width_ratios=(1, 1))

    # # Create the axes
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    axes = [ax0, ax1]

    fig.suptitle(title, fontsize=10)

    # Get semivariogram funciton
    sv_model = exp_semivariogram(sv_params[0], sv_params[1], sv_params[2], r)

    # Get covariance
    c       = sv_params[0] + sv_params[1] - sv
    c_model = sv_params[0] + sv_params[1] - sv_model

    # c_model = exp_covariance(sv_model[0], sv_model[1], sv_model[2], r)

    ax0.scatter(r[:-1] + r_inc/2, c, marker='.', color='k', alpha=0.2)
    ax0.plot(r, c_model, c='k')
    ax0.set_ylabel(r'Covariance ($mm^2$)')

    ax1.scatter(r[:-1] + r_inc/2, sv, marker='.', color='k', alpha=0.2)
    ax1.plot(r, sv_model, c='k')
    ax1.set_ylabel(r'Semivariance ($mm^2$)')

    # ax1.hlines(results.x[0], 0, r[-1], linestyle='--', color='gray')
    # ax1.hlines(results.x[1], 0, r[-1], linestyle='--', color='gray')
    # ax1.vlines(results.x[2], 0, results.x[1], linestyle='--', color='gray')

    # ax1.set_xlim(r[0], r[-1])
    # ax1.set_ylim(r[0], results.x[1] * 1.5)



    # if ymax <= 0:
        # ymax = 2 * sv_model[1]

    ax0.set_ylim(0, c_max)
    ax1.set_ylim(0, sv_max)

    for ax in axes:
        ax.set_xlabel(r'Distance (km)')
        ax.set_xlim(0, np.max(r))


    fig.tight_layout()

    if len(file_name) > 0:
        plt.savefig(file_name, dpi=dpi)
    if show:
        plt.show()
    return


def plot_covariance_overlay(r, SV, SV_model, mask_dists, figsize=(5, 4), cmap=cmc.roma, ymax=2, dpi=500, file_name='', show=False):
    """
    Plot semivariogram.
    """

    r_inc = r[1] - r[0]

    # Make colorbar
    vmin   = 0
    vmax   = max(mask_dists)
    n_tick = len(mask_dists)
    n_seg  = n_tick - 1
    cmap_name = cmc.roma

    cvar   = mask_dists
    cval   = (cvar - vmin)/(vmax - vmin) # Normalized color values
    ticks  = np.linspace(vmin, vmax, n_tick)

    cmap  = colors.LinearSegmentedColormap.from_list(cmap_name, plt.get_cmap(cmap_name, 265)(np.linspace(0, 1, 265)), n_seg)
    sm    = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    c     = cmap(cval)

    # Plot
    fig, ax1 = plt.subplots(figsize=figsize)

    for i in range(len(SV)):
        ax1.scatter(r[:-1] + r_inc/2, SV[i], marker='.', color=c[i], alpha=0.2)
        ax1.plot(r, SV_model[i], c=c[i])

    # ax1.hlines(results.x[0], 0, r[-1], linestyle='--', color='gray')
    # ax1.hlines(results.x[1], 0, r[-1], linestyle='--', color='gray')
    # ax1.vlines(results.x[2], 0, results.x[1], linestyle='--', color='gray')

    ax1.set_ylabel(r'Semivariogram ($mm^2$)')
    ax1.set_xlabel(r'Distance (km)')
    # ax1.set_xlim(r[0], r[-1])
    # ax1.set_ylim(r[0], results.x[1] * 1.5)
    ax1.set_xlim(0, np.max(r))

    # if ymax > 0:
        # ymax = 2 * SVmodel
    ax1.set_ylim(0, ymax)

    fig.colorbar(sm, label='Mask distance (km)')
    fig.tight_layout()

    if len(file_name) > 0:
        plt.savefig(file_name, dpi=dpi)
    if show:
        plt.show()
    return

    # return 
    # # print(np.nanmin(data), np.nanmax(data))
    # # print(dataset)
    # # fig, ax = plt.subplots(figsize=(14, 8.2))
    # # ax.imshow(data, cmap='coolwarm', interpolation='none', vmin=-3, vmax=3)
    # # ax.invert_yaxis()
    # # plt.show()

    # # return
    # start = time.time()
    # now = datetime.datetime.now()
    # print('Time: ', now)
    # C, S, R = covariance(x, y, data)
        
    # end = time.time() - start
    # print(f'Run time: {end} s for {data.size} points')

    # # Compute averages
    # r_inc = 0.050
    # r_max = np.max(R)
    # r = np.linspace(0, r_max + (r_max % r_inc), int(r_max/r_inc) + 1)

    # C_avg = np.empty_like(r)
    # S_avg = np.empty_like(r)

    # for i in range(len(r) - 1):
    #     sel = (r[i + 1] > R) & (R >= r[i])
    #     C_avg[i] = np.nanmean(C[sel])
    #     S_avg[i] = np.nanmean(S[sel])

    # fig, axes = plt.subplots(2, 1, figsize=(4, 8))
    # axes[0].scatter(R.flatten(), C.flatten(), marker='.', c='k', alpha=0.2)
    # axes[0].plot(r, C_avg, c='C3')
    # axes[1].scatter(R.flatten(), S.flatten(), marker='.', c='k', alpha=0.2)
    # axes[1].plot(r, S_avg, c='C3')

    # axes[0].set_ylabel(r'Covariance ($mm^2$)')
    # axes[1].set_ylabel(r'Structure function ($mm^2$)')
    # axes[1].set_xlabel(r'Distance (km)')
    # plt.show()

    # # pyffit.data.write_grd(x, y, data, '/Users/evavra/Software/pyffit/tests/covariance/A.grd')
    # return


def semivariogram(x, y, M, r_max=80, n_samp=5000, r_inc=0.1):
    """
    Estimate an experimental semivariogram from dataset M.
    """

    diff  = np.zeros(n_samp)
    r     = np.zeros(n_samp)

    # Get real values
    i_nans = np.isnan(M)
    x      = x[~i_nans]
    y      = y[~i_nans]
    M      = M[~i_nans]

    # Select random pairs
    i = np.random.randint(0, high=M.size, size=n_samp)
    j = np.random.randint(0, high=M.size, size=n_samp)

    # Compute pairs
    start = time.time()
    r    = dist(x[i], x[j], y[i], y[j])
    diff = diffsq(M[i], M[j])  
    end = time.time() - start

    print(f'Time for {n_samp:1e} pairs: {end:.1f}')

    # Prep averaging bins
    # r_max    = np.sqrt((x.min() - x.max())**2 + (y.min() - y.max())**2)
    r_bins   = np.linspace(0, r_max + (r_max % r_inc), int(r_max/r_inc) + 1)
    n_bin    = len(r_bins) - 1
    sv_sum   = np.zeros(n_bin)
    sv_count = np.zeros(n_bin)

    # For each distance bin, perform sum
    for k in range(n_bin):
        # Get pairs within bin
        sel = (r < r_bins[k + 1]) & (r >= r_bins[k])
        # sv_sum[k]   += np.sum(diff[sel])
        # sv_count[k] += len(diff[sel])
        sv_sum[k] =  np.mean(diff[sel])

    # Average to compute the semivariogram
    # sv     = 0.5 * sv_sum/sv_count
    sv = 0.5 * sv_sum

    end = time.time() - start
    print(f'Total time for {n_samp:.1e} pairs: {end:.1f}')

    return r_bins, sv, sv_count


def exp_semivariogram(s0, s, r, h):
    """
    Exponential semivariogram function.
    """
    return s0 + (s - s0) * (1 - np.exp(-h/r))


def exp_covariance(s0, s, r, h):
    """
    Exponential covariance function.
    """
    return s0 + (s - s0) * np.exp(-h/r)


def fit_semivariogram(r, sv, m0=[1, 1, 1], r_max=100):
    i = (~np.isnan(sv)) & (r <= r_max)

    fun = lambda m: sv[i] - exp_semivariogram(m[0], m[1], m[2], r[i])

    start   = time.time()
    results = least_squares(fun, m0)
    end     = time.time() - start
    print(f'Optimization time: {end:.1f} s')
    
    return results


@jit(nopython=True)
def covariance(x, y, M, n_processes=8):

    A       = np.zeros_like(M)
    S       = np.zeros_like(M)
    R       = np.zeros_like(M)
    nx      = A.shape[0]
    ny      = A.shape[1]

    # start = time.time()
    x_inc = x[0, 1] - x[0, 0]
    y_inc = y[1, 0] - y[0, 0]

    # Prep parameters
    for dx in range(nx):
        # print(f'Working on dx = {dx}')
        for dy in range(ny):

            # Compute distance
            R[dx, dy] = ((dx * x_inc)**2 + (dy * x_inc)**2)**0.5

            # Compute autocorrelation
            # A[dx, dy], S[dx, dy] = autocorrelation(dx, dy, nx, ny, M)
            a = np.nan
            s = np.nan
            value = 0
            diff  = 0
            count = 0

            for k in range(dx, nx):
                for l in range(dy, ny):
                    # Check to see both pixels are real valued
                    if ~np.isnan(M[k, l]) & ~np.isnan(M[k - dx, l - dy]):
                        value +=  M[k, l] * M[k - dx, l - dy]
                        diff  += (M[k, l] - M[k - dx, l - dy])**2
                        count += 1
            
            if count > 0:
                A[dx, dy] = value/count
                S[dx, dy] = diff/count


    # end = time.time() - start
    # print(f'Run time: {end} s for {M.size} points')

    # pyffit.data.write_grd(x, y, A, '/Users/evavra/Software/pyffit/tests/covariance/A.grd')
    # pyffit.data.write_grd(x, y, S, '/Users/evavra/Software/pyffit/tests/covariance/S.grd')


    # Compute covarince 
    C = A**2 - S/2

    return C, S, R


@jit(nopython=True)
def autocorrelation(dx, dy, nx, ny, M):
    """
    Compute autocorrelation for a given lags of dx & dy
    """
    # dx, dy, nx, ny, M = params

    print(f'Working on ({dx}, {dy})')

    a = np.nan
    s = np.nan
    value = 0
    diff  = 0
    count = 0

    for k in range(dx, nx):
        for l in range(dy, ny):
            # Check to see both pixels are real valued
            if ~np.isnan(M[k, l]) & ~np.isnan(M[k - dx, l - dy]):
                value +=  M[k, l] * M[k - dx, l - dy]
                diff  += (M[k, l] - M[k - dx, l - dy])**2
                count += 1
    
    if count > 0:
        a = value/count
        s = diff/count

    return a, s


@jit(nopython=True)
def dist(x0, x1, y0, y1):
    return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)

@jit(nopython=True)
def diffsq(a0, a1):
    return(a0 - a1)**2


def get_fault_mask(x, y, fault, mask_dist=5, out_dir=''):
    """
    Mask data within mask_dist (km) of fault trace.
    """

    # Initialize objects
    r     = np.empty_like(x)
    mask  = np.ones_like(x)
    trace = fault.trace_interp

    # Compute min. fault distance for each point
    for k in range(len(x)):
        r[k] = np.min(dist(trace[:, 0], x[k], trace[:, 1], y[k]))

    # Form mask
    mask[r <= mask_dist] = np.nan

    # Save
    if len(out_dir) > 0:
        check_dir_tree(out_dir)
        with h5py.File(out_dir + f'/fault_mask_{mask_dist}_km.h5', 'w') as file:
            file.create_dataset('mask', data=mask)

    return mask


if __name__ == '__main__':
    main()
