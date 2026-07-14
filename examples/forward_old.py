import sys
#sys.path.append('/raid/class239/xiaoyu/Pamir/pyffit-main')
#sys.path.append('../../pyffit-main')
sys.path.append('/home/cheng/work/new_sampling/pyffit-main')

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


#----------------------------Define what kind of plot you want-----------------------------#
vector=1 #vector field (0: no I don't want it ; 1: yes I want)
mis=0 # full resolution misfit
full_forward_data=0 #save the forward model data 
forward=0 #forward model
forward_full=0 #full resolution forward model
subsampled_data=0 # Subsampled data
#------------------------------------------------------------------------------------------#



def main():
    # prepare_synthetic_data()
    try:
        inversion()
    except ValueError:
        print('Value Error occurred.Try it again')
        exit()
    return

def inversion():
    # Files
    insar_files = [ # Paths to InSAR datasets
               #'real_data/ALOS/asc/alos_asc_los_ll.grd',
               'real_data/ALOS/des/alos_des_los_ll.grd',
               ]

    smooth_files = [ # Paths to smoothed input data for enhanced sampling
               #'real_data/ALOS/asc/smooth.grd',
               'real_data/ALOS/des/smooth.grd',
               ]

    look_dirs   = [ # Paths to InSAR look vectors
               #'real_data/ALOS/asc/look',
               'real_data/ALOS/des/look',
               ]
    weights     = [ # Relative inversion weights for datasets
               1, 
               #1,
               ]
               
    model_dirs   = [ # Paths to models
               #'forward1/disp_sen_asc_los_ll.txt',
               'forward1/disp_sen_des_los_ll.txt',
               ]

    sampling_mode = 1 # 0: no smoothing; 1: yes, smooth it; 2: sample from the forward model

    # Geographic parameters
    ref_point   = [73.1603, 38.1025] # Cartesian coordinate reference point
    EPSG        = '32643'            # EPSG code for relevant UTM zone (zone 43 for the Pamir event)

    # Constituitive parameters
    poisson_ratio  = 0.25
    shear_modulus  = 30 * 10**9 # From Turcotte & Schubert
    lmda           = 2 * shear_modulus * poisson_ratio / (1 - 2 * poisson_ratio)
    alpha          = (lmda + shear_modulus) / (lmda + 2 * shear_modulus)    

	# Quadtree parameters
    rms_min      = 0.007  # RMS threshold (data units), default 0.006
    rms_min2     = 0.007 # maximum RMS Threshold for downsampling lobes of data,default 0.006
    nan_frac_max = 0.7  # Fraction of NaN values allowed per cell, default 0.7
    width_min    = 0.3  # Minimum cell width (km), default 0.3
    width_max    = 7   # Maximum cell width (km), default 7
    mean_low     = 0.07  # minimum threshold for the mean data value for downsampling 'lobes' of data, defaut 0.07
    

    # Inversion parameters
    parallel  = False                                # Parallelize sampling (NOTE: may NOT always result in performance increase)
    n_process = 8                                    # number of threads to use for parallelization
    n_walkers = 40                                   # number of walkers in ensemble (must be at least 2*n + 1 for n free parameters)
    n_step    = 10000                                # number of steps for each walker to take, default 10000
    moves     = [(emcee.moves.DEMove(), 0.8),        # Choice of walker moves 
                 (emcee.moves.DESnookerMove(), 0.2)] # emcee default is [emcee.moves.StretchMove(), 1.0]
    init_mode = 'uniform'

    # Define uniform prior 
    out_dir         = 'ALOS_result'
    inversion_mode  = 'run' # 'run' to run inversion, 'reload' to load previous results and prepare output products

    # NOTE: Data coordinates and fault coordinates should be the same units (km or m) and 
    #       LOS displacement units and slip units should be same units (m, cm, or km), but
    #       spatial and slip units do not need to be the same.

    # Prior limits
    x_lim           = [-20,  20]     # (km) 
    y_lim           = [-20,  20]     # (km)
    z_lim           = [8  ,  12]     # (km)
    l_lim           = [ 0, 50]     # (km)
    w_lim           = [  5,  20]     # (km)
    strike_lim      = [20, 340] # (deg)
    dip_lim         = [  30, 150]     # (deg)
    strike_slip_lim = [  -3,  3]     # (m)
    dip_slip_lim    = [  -3,  3]     # (m)

    # Construct prior
    priors  = {
               'x':           x_lim,
               'y':           y_lim,
               'z':           z_lim,
               'l':           l_lim,
               'w':           w_lim,
               'strike':      strike_lim,
               'dip':         dip_lim,
               'strike_slip': strike_slip_lim,
               'dip_slip':    dip_slip_lim,
               }

    labels = ['x',   'y', 'z' , 'l',  'w', 'strike', 'dip', 'strike_slip', 'dip_slip',] # Labels for plotting                        
    units  = ['km', 'km', 'km',  'km', 'km',    'deg', 'deg',           'm',        'm',] # Unit labels for plotting
    scales = [1   , 1   , 1   ,   1   , 1   ,        1,     1,             1,          1,] # Unit scaling factors for plotting

    # Plotting parameters
    vlim_disp = [-0.15,0.15]

    # Define the elliptical tapering function

    def elliptical_tapering(m,n,N,u0):
        central = (N+1)/2 -1
        axis = N/2
        scale1 = np.sqrt(1 - (m - central)**2 / axis**2)
        scale2 = np.sqrt(1 - (n - central)**2 / axis**2)
        return np.array(u0) * scale1 * scale2

    #def elliptical_tapering(m, n, W, L, N, a, b, u0):
    #    central = (N+1)/2 - 1
    #    distance = np.sqrt(((n - central) / a)**2 + ((m - central) / b)**2)
    #    return np.array(u0) / (1+distance**2)
    
    #elliptical deformation
    def patch_slip_ellip(m,coords,look):
        """
        m      - model parameters
        coords - x/y coordinates for model prediciton
        look   - array of look vector components 
    
    	But, the the displacement taper out from the center in a elliptical pattern, realized by subdividing the fault patch into N x N pieces.
		
		How to Calculate the attributes of subdivided fault patches:
		First of all, 'divide' the fault patch into a (N x N) matrix, N should be an odd number, and the central index should be central = (N+1)/2 - 1 (-1 because this is in python). Enforce that
		the along-strike end of the patch is the very beginning of the matrix (0,0). Let's assume that the index for the sub-patch is (m,n), and generalize the
		properties of each sub-patch.
		
		
		z coordinate: z[m,n] = Z + m * W/N * sin(dip)
		x coordinate: x[m,n] = X + (central-n) * L/N * sin(strike) + m * W/N * cos(dip) * cos(strike)
		y coordinate: y[m,n] = Y + (central-n) * L/N * cos(strike) - m * W/N * cos(dip) * sin(strike)  
		slip: slip[m,n] = elliptical_tapering(m, n, W, L, N, a, b, u0) ; Let a = 4 and b =2;
    
    
        """

    # Unpack input parameters
        X, Y, Z, L, W, strike, dip, strike_slip, dip_slip = m
        x, y = coords
        slip = [strike_slip, dip_slip, 0]

    # Generate fault patch
    #patch = pyffit.finite_fault.Patch()
    #patch.add_self_geometry((X, Y, Z), strike, dip, l, w, slip=slip)
    
    # Define the size of matrix (NxN) and the central value, and create an empty matrix
        N = 7
        u0 = slip
        central = (N+1)/2 - 1
        matrix = np.zeros((N,N))

    # Define the semi-major and semi-minor axes of the ellipse
        #a = 4
        #b = 2
	
    # Divide the patch into sub-patches, and sum up all the surface displacement
        disp_summed = np.zeros((np.size(x),3))
        for m in range (N):
            for n in range(N):
                patch_sub = pyffit.finite_fault.Patch()
                x_sub = X + (central-n) * L/N * np.sin(np.deg2rad(strike)) + m * W/N * np.cos(np.deg2rad(dip)) * np.cos(np.deg2rad(strike))
                y_sub = Y + (central-n) * L/N * np.cos(np.deg2rad(strike)) - m * W/N * np.cos(np.deg2rad(dip)) * np.sin(np.deg2rad(strike))
                z_sub = Z + m * W/N * np.sin(np.deg2rad(dip))
                l_sub = L/N
                w_sub = W/N
                #slip_sub = elliptical_tapering(m, n, W, L, N, a, b, u0)
                slip_sub = elliptical_tapering(m,n,N,u0)
                patch_sub.add_self_geometry((x_sub,y_sub,z_sub),strike,dip,l_sub,w_sub,slip=slip_sub)
                disp = patch_sub.disp(x,y,0,alpha,slip=slip_sub)
                disp_summed += disp
        disp_summed = disp_summed.reshape(x.size,3,1,1)
        disp_LOS = pyffit.finite_fault.proj_greens_functions(disp_summed, look)[:, :, 0].reshape(-1)
        if -np.inf in disp_summed:
            return np.ones_like(disp_LOS) * np.inf * -1
        else:
            return disp_LOS




    # Define probability functions
    def patch_slip(m, coords, look):
        """

        m      - model parameters
        coords - x/y coordinates for model prediciton
        look   - array of look vector components 
        """

        # Unpack input parameters
        x_patch, y_patch, z,l, w, strike, dip, strike_slip, dip_slip = m
        x, y = coords
        slip = [strike_slip, dip_slip, 0]

        # Generate fault patch
        patch = pyffit.finite_fault.Patch()
        patch.add_self_geometry((x_patch, y_patch, z), strike, dip, l, w, slip=slip)

        # Complute full displacements
        disp     = patch.disp(x, y, 0, alpha, slip=slip).reshape(x.size, 3, 1, 1)
        disp_LOS = pyffit.finite_fault.proj_greens_functions(disp, look)[:, :, 0].reshape(-1)

        if -np.inf in disp:
            print('Error')
            return np.ones_like(disp_LOS) * np.inf * -1
        else:
            return disp_LOS

    # Calculate the displacement in x,y,z coordinates 
    def patch_slip_xyz(m,x,y):
        """

        m      - model parameters
        coords - x/y coordinates for model prediciton
        look   - array of look vector components
        """

        # Unpack input parameters
        x_patch, y_patch, z,l, w, strike, dip, strike_slip, dip_slip = m
        slip = [strike_slip, dip_slip, 0]

        # Generate fault patch
        patch = pyffit.finite_fault.Patch()
        patch.add_self_geometry((x_patch, y_patch, z), strike, dip, l, w, slip=slip)

        # Complute full displacements
        #disp     = patch.disp(x.flatten(), y.flatten(), 0, alpha, slip=slip).reshape(x.size, 3, 1, 1)
        disp     = patch.disp(x.flatten(), y.flatten(), 0, alpha, slip=slip)
        if -np.inf in disp:
            print('Error')
            return np.ones_like(disp_LOS) * np.inf * -1
        else:
            return disp

    def cost_function(m, coords, look, data, S_inv, model):
        """
        Cost function for initial optimization step (modified version of log. likelihood)

        INPUT:
        m      - model parameters
        coords - data coordinates
        d      - data values
        S_inv  - inverse covariance matrix
        model  - function handle for model

        OUTPUT:
        log[p(d|m)] - log likelihood of d given m 
        """

        # Make forward model calculation
        G_m = model(m, coords, look)
        r   = G_m - data

        return np.hstack((0.5**-0.5) * S_inv @ r)


    @numba.jit(nopython=True) # For a little speed boost
    def log_likelihood(G_m, data, S_inv, B):
        """
        Speedy version of log. likelihood function.
        Modified to accomodate     
    
        INPUT:
        G_m   (m,)   - model realization corresponding to each data point
        data  (m,)   - data values
        S_inv (m, m) - inverse data covariance matrix
        B     (m, m) - data weights

        OUTPUT
        """

        if -np.inf in G_m:
            return -np.inf

        else:
            r = data - G_m
            return -0.5 * r.T @ S_inv @ B @ r


    def log_prob_uniform(m, coords, look, data, S_inv, B, patch_slip, priors):
        """
        Determine log-probaility of model m using a uniform prior.
        """

        # Check prior
        if np.all([priors[key][0] <= m[i] <= priors[key][1] for i, key in enumerate(priors.keys())]):
            return log_likelihood(patch_slip(m, coords, look), data, S_inv, B) # Log. probability of sample is only the log. likelihood
        else:
            return -np.inf                                    # Exclude realizations outside of priors


    # ---------------------------------- CONFIGURATION ----------------------------------  #
    # Ingest InSAR data for inversion
    if sampling_mode ==0:
        datasets = pyffit.insar.prepare_datasets(insar_files, look_dirs, weights, ref_point,EPSG=EPSG, rms_min=rms_min, nan_frac_max=nan_frac_max, width_min=width_min, width_max=width_max)

    if sampling_mode ==1:
    	datasets = pyffit.insar.prepare_datasets_smooth(insar_files,smooth_files, look_dirs, weights, ref_point,EPSG=EPSG, mean_low=mean_low,rms_min2=rms_min2,rms_min=rms_min, nan_frac_max=nan_frac_max, width_min=width_min, width_max=width_max)
    
    if sampling_mode ==2:
    	datasets = pyffit.insar.prepare_datasets_model(insar_files,model_dirs, look_dirs, weights, ref_point,EPSG=EPSG, mean_low=mean_low,rms_min2=rms_min2,rms_min=rms_min, nan_frac_max=nan_frac_max, width_min=width_min, width_max=width_max)
    


    # # Plot downsampled data
    for dataset in datasets.keys():
        pyffit.figures.plot_quadtree(datasets[dataset]['data'], datasets[dataset]['extent'], (datasets[dataset]['x_samp'], datasets[dataset]['y_samp']), datasets[dataset]['data_samp'], vlim_disp=vlim_disp, file_name=f'{out_dir}/quadtree_init_{dataset}.png',cmap_disp='coolwarm',trace=[],original_data=[],cell_extents=[],show=False,dpi=500,figsize=(14,8.2))
        #full resolution coordinates
        x = datasets[dataset]['x']
        y = datasets[dataset]['y']
        look_full=datasets[dataset]['look']
        x_samp=datasets[dataset]['x_samp']
        y_samp=datasets[dataset]['y_samp']
        data_full=datasets[dataset]['data']
        


    # Aggregate NaN locations
    i_nans = np.concatenate([datasets[name]['i_nans'] for name in datasets.keys()])

    # Aggregate coordinates
    coords = (np.concatenate([datasets[name]['x_samp'] for name in datasets.keys()])[~i_nans],
              np.concatenate([datasets[name]['y_samp'] for name in datasets.keys()])[~i_nans],)

    # Aggregate look vectors
    look     = np.concatenate([datasets[name]['look_samp'] for name in datasets.keys()])[~i_nans]

    # Aggregate data, standard deviations, and weights
    data     = np.concatenate([datasets[name]['data_samp'] for name in datasets.keys()])[~i_nans]
    data_std = np.concatenate([datasets[name]['data_samp_std'] for name in datasets.keys()])[~i_nans]
    weights  = np.concatenate([np.ones_like(datasets[name]['data_samp']) * datasets[name]['weight'] for name in datasets.keys()])[~i_nans]
    B        = np.diag(weights)      # Weight matrix
    S_inv    = np.diag((data_std)**-2) # Covariance matrix



   #--------------------------------------FORWARD MODELING-------------------------------#
    m_forward     = [ 1.37,   5.41, 1.44, 22.72, 15.56, 27.21,  74.04, 2.57,     -0.8]
    disp_fit=patch_slip(m_forward,coords,look)

    #plot a forward model here
    if forward ==1:
        fig,axes=plt.subplots(1,1,figsize=(14, 8.2))
        sc=axes.scatter(coords[0], coords[1], c=disp_fit, cmap='coolwarm', vmin=vlim_disp[0], vmax=vlim_disp[1])
        cbar=fig.colorbar(sc,label='LOS Displacement (m)')
        axes.set_xlabel('X (km)',fontsize=20)
        axes.set_ylabel('Y (km)',fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=20)
        cbar.ax.tick_params(labelsize=20)
        plt.savefig(f'{out_dir}/forward_{dataset}_slip_square.png',dpi=500)
        plt.close()
        print('Quadtree subsampled forward model computed')



    #plot the full resolution vector field
    if vector ==1:
        disp_xyz=patch_slip_xyz(m_forward,x,y)
        m,n = x.shape
        skip = (slice(None, None, 15), slice(None, None, 15))
        fig,axes=plt.subplots(1,1,figsize=(14, 8.2))
        u_x=disp_xyz[:,0].reshape(m,n)
        u_y=disp_xyz[:,1].reshape(m,n)
        u_z=disp_xyz[:,2].reshape(m,n)
        axes.set_xlabel('X (km)',fontsize=20)
        axes.set_ylabel('Y (km)',fontsize=20)
        plt.quiver(x[skip],y[skip],u_x[skip],u_y[skip])
        plt.savefig(f'{out_dir}/forward_{dataset}_vec.png',dpi=500)
        plt.close()
        
        np.savetxt(f'{out_dir}/x_{dataset}.txt',x,delimiter=',')
        np.savetxt(f'{out_dir}/y_{dataset}.txt',y,delimiter=',')
        np.savetxt(f'{out_dir}/ux_{dataset}.txt',u_x,delimiter=',')
        np.savetxt(f'{out_dir}/uy_{dataset}.txt',u_y,delimiter=',')
        np.savetxt(f'{out_dir}/uz_{dataset}.txt',u_z,delimiter=',')
        
        print('Full resolution vector field computed')
        #exit()

    #plot the full resolution model misfit
    if mis ==1:
        coords_full=np.vstack((x.flatten(),y.flatten()))
        #disp_full=patch_slip(m_forward,coords_full,look_full)
        disp_full=patch_slip_ellip(m_forward,coords_full,look_full)
        misfit=data_full.flatten()-disp_full
        np.savetxt(f'{out_dir}/misfit_{dataset}.txt',misfit,delimiter=',')
        fig,axes=plt.subplots(1,1,figsize=(14, 8.2))
        axes.set_xlabel('East (km)',fontsize=20)
        axes.set_ylabel('North (km)',fontsize=20)
        sc=axes.scatter(coords_full[0], coords_full[1], c=misfit, cmap='coolwarm', vmin=vlim_disp[0], vmax=vlim_disp[1])
        cbar=fig.colorbar(sc,label='LOS Misfit (m)')
        cbar.ax.tick_params(labelsize=20)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.xlim(np.min(coords_full[0]),np.max(coords_full[0]))
        plt.ylim(np.min(coords_full[1]),np.max(coords_full[1]))
        #plt.savefig(f'{out_dir}/misfit_{dataset}_slip_square.png',dpi=500)
        plt.savefig(f'{out_dir}/misfit_{dataset}.png',dpi=500)
        print('full resolution model misfit computed')
        plt.close()

    # plot the full resolution forward model
    if forward_full ==1:
        coords_full=np.vstack((x.flatten(),y.flatten()))
        #dimension_list = [1,5,9,13,17,21]
        #for dimension in dimension_list:
        #    m_forward[3]=dimension
        #    m_forward[4]=dimension
        #    m_forward[2]= 10.5 - dimension/2
        #disp_full=patch_slip(m_forward,coords_full,look_full)
        disp_full=patch_slip_ellip(m_forward,coords_full,look_full)
        fig,axes=plt.subplots(1,1,figsize=(14, 8.2))
        axes.set_xlabel('East (km)',fontsize=20)
        axes.set_ylabel('North (km)',fontsize=20)
            #vlim_disp[0]=np.nanmin(disp_full)*0.8
            #vlim_disp[1]=np.nanmax(disp_full)*0.8
        sc=axes.scatter(coords_full[0], coords_full[1], c=disp_full, cmap='coolwarm', vmin=vlim_disp[0], vmax=vlim_disp[1])
        cbar=fig.colorbar(sc,label='LOS Misfit (m)')
        cbar.ax.tick_params(labelsize=20)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.xlim(np.min(coords_full[0]),np.max(coords_full[0]))
        plt.ylim(np.min(coords_full[1]),np.max(coords_full[1]))
        #    plt.savefig(f'{out_dir}/fullforward_{dataset}_{m_forward[3]}.png',dpi=500)
        plt.savefig(f'{out_dir}/fullforward_{dataset}.png',dpi=500)
        plt.close()
        print('full resolution forward model computed')
    
    if full_forward_data==1:
        np.savetxt(f'{out_dir}/x_{dataset}.txt',x,delimiter=',')
        np.savetxt(f'{out_dir}/y_{dataset}.txt',y,delimiter=',')
        coords_full=np.vstack((x.flatten(),y.flatten()))
        disp_full=patch_slip_ellip(m_forward,coords_full,look_full)
        np.savetxt(f'{out_dir}/disp_{dataset}.txt',disp_full,delimiter=',')
        
    if subsampled_data==1:
        np.savetxt(f'{out_dir}/coords_{dataset}.txt',coords,delimiter=',')
        np.savetxt(f'{out_dir}/look_{dataset}.txt',look,delimiter=',')
        np.savetxt(f'{out_dir}/data_{dataset}.txt',data,delimiter=',')
    
    exit()


    # ---------------------------------- INVERSION ----------------------------------  #
    # # Solve for initial state using non-linear least squares
    # print('Finding starting model...')

    # # Set up helper function for optimization
    # nll = lambda *args: -cost_function(*args)    

    # # Perform initial optimization for starting model parameters
    m0 = np.array([np.mean(priors[prior]) for prior in priors.keys()])
    # m0 = least_squares(nll, initial, args=(coords, look, data, S_inv, patch_slip), 
    #                    verbose=2,
    #                    bounds=([priors[prior][0] for prior in priors.keys()], [priors[prior][1] for prior in priors.keys()])).x

    # #           x    y    l     w  strike  dip  strike-slip dip-slip 
    # m0     = [  0,   0, 100, 12.5,  269.9,  90,        -2.5,     0.5]
    # m_true = [-10, -10,  80,   15,    317,  70,          -3,       0]

    # # Check initial model...
    # disp_test = patch_slip(m0, coords, look)
    # disp_true = patch_slip(m_true, coords, look)

    # fig, axes = plt.subplots(1, 3, figsize=(14, 8.2))
    # # axes[0].scatter(coords[0], coords[1], c=disp_true[:, 0], cmap='coolwarm',      vmin=vlim_disp[0], vmax=vlim_disp[1])
    # # axes[1].scatter(coords[0], coords[1], c=disp_true[:, 1], cmap='coolwarm', vmin=vlim_disp[0], vmax=vlim_disp[1])
    # # axes[2].scatter(coords[0], coords[1], c=disp_true[:, 2], cmap='coolwarm', vmin=vlim_disp[0], vmax=vlim_disp[1])
    # axes[0].scatter(coords[0], coords[1], c=data, cmap='coolwarm',      vmin=vlim_disp[0], vmax=vlim_disp[1])
    # axes[1].scatter(coords[0], coords[1], c=disp_test, cmap='coolwarm', vmin=vlim_disp[0], vmax=vlim_disp[1])
    # axes[2].scatter(coords[0], coords[1], c=disp_true, cmap='coolwarm', vmin=vlim_disp[0], vmax=vlim_disp[1])
    # plt.show()
    
    # return




    # Run inversion or reload previous results
    if inversion_mode == 'run':
        # Set up backend for storing MCMC results
        backend = pyffit.inversion.config_backend(f'{out_dir}/results.h5', n_walkers, len(priors)) 
        log_prob_args = (coords, look, data, S_inv, B, patch_slip, priors)
        samples, samp_prob, autocorr, discard, thin = pyffit.inversion.run_hammer(log_prob_args, priors, log_prob_uniform, n_walkers, n_step, m0, backend, moves, 
                                                                                  progress=True, init_mode=init_mode, parallel=False, processes=n_process)
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
    samples         = samples[:, abs(mean_chain_prob - mean_prob) <= std_prob]   
    samp_prob       = samp_prob[:, abs(mean_chain_prob - mean_prob) <= std_prob] 
    flat_samples    = samples[discard::thin, :, :].reshape(len(samples[discard::thin, 0, 0])*len(samples[0, :, 0]), len(samples[0, 0, :]))
    discard_walkers = n_walkers - samples.shape[1]

    print(f'Average log(p(m|d)) = {mean_prob} +/- {std_prob}')
    print(f'Chain  log(p(m|d))  = {mean_chain_prob} +/- {std_chain_prob}')
    print(f'Number of discarded ensemble members = {discard_walkers}')
    print(f'Number of effective samples = {len(flat_samples)}')
    
    # Compute mean and standard deviation of flat samples
    m_avg = np.mean(flat_samples,           axis=0)
    m_std = np.std(flat_samples,            axis=0)
    m_q1  = np.quantile(flat_samples, 0.16, axis=0)
    m_q2  = np.quantile(flat_samples, 0.50, axis=0)
    m_q3  = np.quantile(flat_samples, 0.84, axis=0)

    # # Compute RMSE for representative models
    # m_rms_avg  = wRMSE(model(m_avg, x), d, S_inv, B)
    # m_rms_q2   = wRMSE(model(m_q2, x),  d, S_inv, B)

    # Plot Markov chains for each parameter
    pyffit.figures.plot_chains(samples, samp_prob, discard, labels, units, scales, out_dir)

    # Plot parameter marginals and correlations
    pyffit.figures.plot_triangle(flat_samples, priors, labels, units, scales, out_dir)    

    # Plot the best-fitting model (tested by Xiaoyu)
    disp_fit=patch_slip(m_avg,coords,look)
    fig,axes=plt.subplots(1,1,figsize=(14, 8.2))
    sc=axes.scatter(coords[0], coords[1], c=disp_fit, cmap='coolwarm', vmin=vlim_disp[0], vmax=vlim_disp[1])
    cbar=fig.colorbar(sc,label='LOS Displacement (m)')
    axes.set_xlabel('X (km)',fontsize=20)
    axes.set_ylabel('Y (km)',fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    cbar.ax.tick_params(labelsize=20)
    plt.savefig(f'{out_dir}/best_fit.png',dpi=500)
    plt.close()


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
