import os
import time
import emcee
import numba
import pyffit
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.optimize import least_squares
from multiprocessing import Pool
from matplotlib import colors
from example_setup import priors, labels, units, scales, log_prob


def main():
    # prepare_synthetic_data()
    inversion()
    return

def inversion():
    # ---------------------------------- PARAMETERS ----------------------------------
    # Inversion parameters
    progress  = True
    parallel  = True                                 # Parallelize sampling (NOTE: may NOT always result in performance increase)
    n_process = 8                                    # number of threads to use for parallelization
    n_walkers = 50                                   # number of walkers in ensemble (must be at least 2*n + 1 for n free parameters)
    n_step    = 50000                                # number of steps for each walker to take
    moves     = [
                 # (emcee.moves.RedBlueMove(), 1.0),        # Choice of walker moves 
                 (emcee.moves.DEMove(), 0.6),        # Choice of walker moves 
                 (emcee.moves.DESnookerMove(), 0.4),
                 # (emcee.moves.StretchMove(), 1.0),
                ] # emcee default is [emcee.moves.StretchMove(), 1.0]
    init_mode = 'gaussian'

    # Define uniform prior 
    out_dir         = 'results'
    inversion_mode  = 'run' # 'check_downsampling' to only  make quadtree downsampling plots, 'run' to run inversion, 'reload' to load previous results and prepare output products

    # ---------------------------------- INVERSION ----------------------------------  #
    # pyffit.inversion.mcmc(priors, log_prob, out_dir, init_mode=init_mode, moves=moves, n_walkers=n_walkers, 
    #                       n_step=n_step, parallel=parallel, n_process=n_process, progress=progress, inversion_mode=inversion_mode)

    # Get starting model parameters
    m0 = np.array([np.mean(priors[prior]) for prior in priors.keys()])
    m0 = np.array([12.6, -40.9, 149.8, 5.0, 330.3, 107.9, -2.8, 0.6,])

    # Run inversion or reload previous results
    if inversion_mode == 'run':
        # Set up backend for storing MCMC results
        backend = pyffit.inversion.config_backend(f'{out_dir}/results.h5', n_walkers, len(priors)) 
        samples, samp_prob, autocorr, discard, thin = pyffit.inversion.run_hammer((), priors, log_prob, n_walkers, n_step, m0, backend, moves, 
                                                                                  progress=progress, init_mode=init_mode, parallel=parallel, processes=n_process)
    elif inversion_mode == 'reload':
        result_file = f'{out_dir}/results.h5'
        samples, samp_prob, autocorr, discard, thin = pyffit.inversion.reload_hammer(result_file)

    # Assess consistence of convergence amongst the ensemble members
    samp_prob[np.abs(samp_prob) == np.inf] = np.nan   # Correct for infinite values
    mean_chain_prob = np.nanmean(samp_prob, axis=0)   # Get average prob. for each walker
    std_chain_prob  = np.nanstd(mean_chain_prob)      # Get std of walker means
    mean_prob       = np.nanmean(samp_prob.flatten()) # Get total average
    std_prob        = np.nanstd(samp_prob.flatten())  # Get total STD

    # Discard "lost" walkers
    # samples         = samples[:, abs(mean_chain_prob - mean_prob) <= std_prob]   
    samp_prob       = samp_prob[:, abs(mean_chain_prob - mean_prob) <= std_prob] 
    flat_samples    = samples[discard::thin, :, :].reshape(len(samples[discard::thin, 0, 0])*len(samples[0, :, 0]), len(samples[0, 0, :]))
    # discard_walkers = n_walkers - samples.shape[1]

    print(f'Average log(p(m|d)) = {mean_prob} +/- {std_prob}')
    print(f'Chain  log(p(m|d))  = {mean_chain_prob} +/- {std_chain_prob}')
    # print(f'Number of discarded ensemble members = {discard_walkers}')
    print(f'Number of effective samples = {len(flat_samples)}')

    # Compute mean and standard deviation of flat samples
    m_avg = np.mean(flat_samples,           axis=0)
    m_std = np.std(flat_samples,            axis=0)
    m_q1  = np.quantile(flat_samples, 0.16, axis=0)
    m_q2  = np.quantile(flat_samples, 0.50, axis=0)
    m_q3  = np.quantile(flat_samples, 0.84, axis=0)

    print(priors.keys())
    print(m_avg)
    print(m_std)

    # # Compute RMSE for representative models
    # m_rms_avg  = wRMSE(model(m_avg, x), d, S_inv, B)
    # m_rms_q2   = wRMSE(model(m_q2, x),  d, S_inv, B)

    # Plot Markov chains for each parameter
    pyffit.figures.plot_chains(samples, samp_prob, priors, discard, labels, units, scales, out_dir)

    # Plot parameter marginals and correlations
    pyffit.figures.plot_triangle(flat_samples, priors, labels, units, scales, out_dir)    


    return  


def prepare_synthetic_data():
    # Example SAR files
    insar_file = '/Users/evavra/Projects/SHF/Data/Sentinel_1/20230503/20230726/unwrap_ll_edit.grd'
    look_dir   = '/Users/evavra/Projects/SHF/Data/Sentinel_1'
    out_file   = 'synthetic_data_1.grd'

    # ----------------------------------------------------------------------------- #
    # WARNING:                                                                      #
    # Singularties will occur when observation points lie on fault edges. This will #
    # cause dc3dwrapper to produce an AssertionError. This is common for idealized  #
    # examples and can usually be avoided by making small adjustments to the fault  #
    # geometry and/or observation discretization.                                   #
    # ----------------------------------------------------------------------------- #

    # Geographic parameters
    ref_point   = [-115.929, 33.171]
    data_region = [-30, 20, -20, 25] # local coordinate system

    # Fault patch parameters
    origin = (-10, -10, 0) # x/y/z origin coordinates (km)
    strike = 317           # strike angle (deg)
    dip    = 70            # dip angle (deg)
    l      = 80            # along-strike length (km)
    d      = 15            # down dip width (km)
    slip   = [-3, 0, 0]    # Slip vector (m)

    # Constituitive parameters
    poisson_ratio  = 0.25
    shear_modulus  = 30 * 10**9 # From Turcotte & Schubert

    # Synthetic data parameters
    nan_frac         = 0.0 # fraction of Nans in synthetic data 
    rupture_nan_frac = 0.98 # Surface rupture decorrelation fraction
    width            = 1.0 # Surface rupture half-width (km)
    aps_amp          = 0.4 # Scaling factor for synthetic atmospheric noise (m)

    # Plotting parameters
    xticks    = np.arange(np.ceil(data_region[0]/10)*10, np.floor(data_region[1]/10) + 1, 10)
    yticks    = np.arange(np.ceil(data_region[2]/10)*10, np.floor(data_region[3]/10) + 1, 10)
    vlim_disp = [-1, 1]
    cmap_disp = 'coolwarm'

    # ---------------------------------- CONFIGURATION ----------------------------------  #
    # Read data
    x_rng, y_rng, data = pyffit.data.read_grd(insar_file)
    look = pyffit.data.load_look_vectors(look_dir)

    # Get full gridded coordinates
    x, y = np.meshgrid(x_rng, y_rng)

    # Get reference point and convert coordinates to km
    X, Y = pyffit.utilities.get_local_xy_coords(x, y, ref_point) 

    # Define constituitive parameters
    lmda  = 2 * shear_modulus * poisson_ratio / (1 - 2 * poisson_ratio)
    alpha = (lmda + shear_modulus) / (lmda + 2 * shear_modulus)    

    # Generate fault patch
    patch = pyffit.finite_fault.Patch()
    patch.add_self_geometry(origin, strike, dip, l, d, slip=slip)

    # Complute full displacements
    disp     = patch.disp(X.flatten(), Y.flatten(), 0, alpha, slip=slip).reshape(X.size, 3, 1, 1)
    disp_LOS = pyffit.finite_fault.proj_greens_functions(disp, look)[:, :, 0]

    # Generate artificial atmopsheric phase screen (APS)
    aps = pyffit.synthetics.make_synthetic_aps(X[0, :], Y[:, 0], manual_amp=aps_amp)
    aps -= aps.mean()

    # Introduce random NaNs
    # decorr_mask = pyffit.synthetics.get_decorr_mask(X, nan_frac) # Generate random noise
    decorr_mask = np.ones_like(data) # Use example interferogram
    decorr_mask[np.isnan(data)] = np.nan

    # Mask fault trace to simulate coseismic rupture
    trace = np.vstack((patch.x[:2], patch.y[:2])).T
    rupture_mask = pyffit.synthetics.get_fault_mask(X, Y, trace, rupture_nan_frac=rupture_nan_frac, width=width)

    # Combine to make synthetic interferogram
    synthetic_intf = (disp_LOS.reshape(X.shape) + aps) * rupture_mask * decorr_mask

    # Save file
    pyffit.data.write_grd(x_rng, y_rng, synthetic_intf, out_file, T=True, V=False)

    # # ---------------------------------- INVERSION ----------------------------------
    # run_hammer(x, d, S_inv, model, priors, log_prob, n_walkers, n_step, m0, backend, moves, 
    #        progress=False, init_mode='uniform', run_name='Sampling', parallel=False, processes=8)

    # # Do actual inversion


    # ---------------------------------- PLOTS ---------------------------------- #

    # Plot displacement components
    fig = plt.figure(figsize=(14, 8.2))
    axes = ImageGrid(fig, 111,
                     nrows_ncols=(1, 3), axes_pad=0.2, label_mode="L", share_all=True,
                     cbar_location='right', cbar_mode="single")

    labels = [r'$U_x$', r'$U_y$', r'$U_z$']
    for i, ax in enumerate(axes):
        # ax.scatter(X.flatten(), Y.flatten(), c=disp[:, i], cmap='coolwarm', vmin=vmin, vmax=vmax)
        # im = ax.imshow((disp[:, i].reshape(X.shape) + aps) * rupture_mask * decorr_mask, extent=(X.min(), X.max(), Y.max(), Y.min()), cmap='coolwarm', vmin=vlim_disp[0], vmax=vlim_disp[1], interpolation='none')
        im = ax.imshow((disp[:, i, 0, 0].reshape(X.shape)), extent=(X.min(), X.max(), Y.max(), Y.min()), cmap='coolwarm', vmin=vlim_disp[0], vmax=vlim_disp[1], interpolation='none')
        ax.add_collection(patch.poly(kind='edges', mode='2d', color='gray', linewidth=1))
        ax.plot(trace[:, 0], trace[:, 1], c='k')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.invert_yaxis()
        ax.set_aspect(1)
        ax.set_title(labels[i])

    axes.cbar_axes[0].colorbar(im, label='Displacement (m)')
    fig.savefig('synthetics.png', dpi=500)

    # Plot noise
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(aps * rupture_mask * decorr_mask, extent=(X.min(), X.max(), Y.max(), Y.min()), cmap='coolwarm', vmin=vlim_disp[0], vmax=vlim_disp[1], interpolation='none')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.invert_yaxis()
    ax.set_aspect(1)
    ax.set_title('Noise')
    fig.colorbar(im, label='LOS Displacement (m)')
    fig.savefig('noise.png', dpi=500)

    # Plot LOS displacements
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(disp_LOS.reshape(X.shape), extent=(X.min(), X.max(), Y.max(), Y.min()), cmap='coolwarm', vmin=vlim_disp[0], vmax=vlim_disp[1], interpolation='none')
    ax.add_collection(patch.poly(kind='edges', mode='2d', color='gray', linewidth=1))
    ax.plot(trace[:, 0], trace[:, 1], c='k')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.invert_yaxis()
    ax.set_aspect(1)
    ax.set_title('Synthetic interferogram')
    fig.colorbar(im, label='LOS Displacement (m)')
    fig.savefig('LOS_clean.png', dpi=500)

    # Plot synthetic interferogram with noise
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(synthetic_intf, extent=(X.min(), X.max(), Y.max(), Y.min()), cmap='coolwarm', vmin=vlim_disp[0], vmax=vlim_disp[1], interpolation='none')
    ax.add_collection(patch.poly(kind='edges', mode='2d', color='gray', linewidth=1))
    ax.plot(trace[:, 0], trace[:, 1], c='k')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.invert_yaxis()
    ax.set_aspect(1)
    ax.set_title('Synthetic interferogram with noise')
    fig.colorbar(im, label='LOS Displacement (m)')
    fig.savefig('LOS_noisy.png', dpi=500)

    return


    # # Plot fault
    # cmap_name  = 'viridis'        # colormap to use
    # cbar_label = 'Slip (m)'       # ccolorbar label
    # var        = np.array([0, 1]) # Fake data to get color range
    # n_seg      = 5                # nunmber of colorbar segments
    
    # # Create colorbar
    # ticks = np.linspace(var.min(), var.max(), n_seg + 1)
    # cval  = (var - var.min())/(var.max() - var.min()) # Normalized color values
    # cmap  = colors.LinearSegmentedColormap.from_list(cmap_name, plt.get_cmap(cmap_name, 265)(np.linspace(0, 1, 265)), n_seg)
    # sm    = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=var.min(), vmax=var.max()))
    
    
    # fig = plt.figure(figsize=(14, 8.2))
    # ax  = fig.add_subplot(projection='3d')
    # ax.add_collection3d(patch.poly(kind='edges', edgecolor='k'))
    # ax.add_collection3d(patch.poly(kind='face', facecolor=cmap(np.linalg.norm(patch.slip))))
    # ax.set_xlim(-100, 100)
    # ax.set_ylim(-100, 100)
    # ax.set_zlim(20, 0)
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # zlim = ax.get_zlim()
    # ranges = [xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[0] - zlim[1]]
    # ax.set_box_aspect(ranges) 
    # adjust = 0.2
    # cax = inset_axes(ax, bbox_to_anchor=(0, 0, 1, 1),
    #                 width="25%",  
    #                 height="3%",
    #                 loc='center',
    #                 borderpad=0
    #                )
    # fig.colorbar(sm, location='bottom', orientation='horizontal', shrink=0.25, label=cbar_label, ticks=ticks)
    # fig.subplots_adjust(left=-adjust, right=1+adjust, bottom=-adjust, top=1+adjust)
    # plt.show()

    return


if __name__ == '__main__':
    main()