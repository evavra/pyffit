import sys

##############append the path to your pyffit functions directory if necessary####################

#sys.path.append('') 


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
vector=0 #vector field (0: no I don't want it ; 1: yes I want)
mis=0 # full resolution misfit
full_forward_data=0 #save the forward model data (full resolution)
forward=0 #forward model
forward_full=0 #full resolution forward model
#------------------------------------------------------------------------------------------#



def main():
    try:
        forward()
    except ValueError:
        print('Value Error occurred.Try it again')
        exit()
    return

def forward():
	#Setup your forward modeling here!
    # Files
    insar_files = [ # Paths to InSAR datasets, enter as many as you want (required)

                   ]

    smooth_files = [ # Paths to smoothed input data for enhanced sampling, enter as many as you want (optional)

               ]

    look_dirs   = [ # Paths to InSAR look vectors, enter as many as you want (required)

                   ]
                   
    weights     = [ # Relative weights for datasets (required)

                   ]

    smooth = 0 # 0: no smoothing; 1: yes, smooth it
    # Geographic parameters
    ref_point   = [73.1603, 38.1025] # Cartesian coordinate reference point
    EPSG        = '32643'            # EPSG code for relevant UTM zone (zone 43 for the Pamir event)

    # Constituitive parameters
    poisson_ratio  = 0.25
    shear_modulus  = 30 * 10**9 # From Turcotte & Schubert
    lmda           = 2 * shear_modulus * poisson_ratio / (1 - 2 * poisson_ratio)
    alpha          = (lmda + shear_modulus) / (lmda + 2 * shear_modulus)    

    # Quadtree parameters
    rms_min      = 0.004  # RMS threshold (data units), default 0.1
    nan_frac_max = 0.7  # Fraction of NaN values allowed per cell, default 0.7
    width_min    = 0.05  # Minimum cell width (km), default 0.1
    width_max    = 24    # Maximum cell width (km), default 4
    mean_low     = 0.3  #lower Threshold for the mean data value
    mean_up      = 1  #upper threshold for the mean data value
    


    # Define your Output Directory
    out_dir         = 'forward_results'
    inversion_mode  = 'run' # 'run' to run inversion, 'reload' to load previous results and prepare output products

    # NOTE: Data coordinates and fault coordinates should be the same units (km or m) and 
    #       LOS displacement units and slip units should be same units (m, cm, or km), but
    #       spatial and slip units do not need to be the same.

    
    
    #                   x       y     z      l     w     strike  dip  strike-slip   dip-slip  
    m_forward     = [ 3.53,   7.03, 1.70, 26.39, 16.23, 26.03,  78.98, 1.92,     -0.10] # enter your forward modeling parameters here!


    # Plotting parameters
    vlim_disp = [-0.2,0.2]

    # Define the elliptical tapering function

    def elliptical_tapering(m,n,N,u0):
        central = (N+1)/2 -1
        axis = N/2
        scale1 = np.sqrt(1 - (m - central)**2 / axis**2)
        scale2 = np.sqrt(1 - (n - central)**2 / axis**2)
        return np.array(u0) * scale1 * scale2

    
    #elliptical deformation
    def patch_slip_ellip(m,coords,look):
        """
        m      - model parameters
        coords - x/y coordinates for model prediciton
        look   - array of look vector components 
    
    	Make the displacement taper out from the center in an elliptical pattern, realized by subdividing the fault patch into N x N pieces.
		
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

    
    # Define the size of matrix (NxN) and the central value, and create an empty matrix
        N = 3 
        u0 = slip
        central = (N+1)/2 - 1
        matrix = np.zeros((N,N))

	
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



 
    # Ingest InSAR data 
    if smooth ==0:
        datasets = pyffit.insar.prepare_datasets(insar_files, look_dirs, weights, ref_point,EPSG=EPSG, rms_min=rms_min, nan_frac_max=nan_frac_max, width_min=width_min, width_max=width_max)

    if smooth ==1:
        datasets = pyffit.insar.prepare_datasets_smooth(insar_files,smooth_files, look_dirs, weights, ref_point,EPSG=EPSG, mean_low=mean_low,mean_up=mean_up,rms_min=rms_min, nan_frac_max=nan_frac_max, width_min=width_min, width_max=width_max)



    for dataset in datasets.keys(): 
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
    disp_fit=patch_slip_ellip(m_forward,coords,look)

    #plot a downsampled forward model 
    if forward ==1:
        fig,axes=plt.subplots(1,1,figsize=(14, 8.2))
        sc=axes.scatter(coords[0], coords[1], c=disp_fit, cmap='coolwarm', vmin=vlim_disp[0], vmax=vlim_disp[1])
        cbar=fig.colorbar(sc,label='LOS Displacement (m)')
        axes.set_xlabel('X (km)',fontsize=20)
        axes.set_ylabel('Y (km)',fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=20)
        cbar.ax.tick_params(labelsize=20)
        plt.savefig(f'{out_dir}/forward_{dataset}_ds.png',dpi=500)
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
        print('Full resolution vector field computed')


    #plot the full resolution model misfit
    if mis ==1:
        coords_full=np.vstack((x.flatten(),y.flatten()))
        disp_full=patch_slip_ellip(m_forward,coords_full,look_full)
        misfit=data_full.flatten()-disp_full
        np.savetxt(f'{out_dir}/misfit_full_{dataset}.txt',misfit,delimiter=',')
        fig,axes=plt.subplots(1,1,figsize=(14, 8.2))
        axes.set_xlabel('X (km)',fontsize=20)
        axes.set_ylabel('Y (km)',fontsize=20)
        sc=axes.scatter(coords_full[0], coords_full[1], c=misfit, cmap='coolwarm', vmin=vlim_disp[0], vmax=vlim_disp[1])
        cbar=fig.colorbar(sc,label='LOS Misfit (m)')
        cbar.ax.tick_params(labelsize=20)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.savefig(f'{out_dir}/misfit_{dataset}_full.png',dpi=500)
        print('full resolution model misfit computed')
        plt.close()

    # plot the full resolution forward model
    if forward_full ==1:
        coords_full=np.vstack((x.flatten(),y.flatten()))
        fig,axes=plt.subplots(1,1,figsize=(14, 8.2))
        axes.set_xlabel('X (km)',fontsize=20)
        axes.set_ylabel('Y (km)',fontsize=20)
        cbar=fig.colorbar(sc,label='LOS Misfit (m)')
        cbar.ax.tick_params(labelsize=20)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.savefig(f'{out_dir}/forward_{dataset}_full.png',dpi=500)
        plt.close()
        print('full resolution forward model computed')
    
    if full_forward_data==1:
        np.savetxt(f'{out_dir}/x_{dataset}.txt',x,delimiter=',')
        np.savetxt(f'{out_dir}/y_{dataset}.txt',y,delimiter=',')
        coords_full=np.vstack((x.flatten(),y.flatten()))
        disp_full=patch_slip_ellip(m_forward,coords_full,look_full)
        np.savetxt(f'{out_dir}/disp_{dataset}.txt',disp_full,delimiter=',')
    
    exit()


    return


if __name__ == '__main__':
    main()
