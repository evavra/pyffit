import sys
# Set the path to the functions
sys.path.append('/raid/class239/xiaoyu/Pamir/pyffit-main')

import pyffit
import numba
import numpy as np
# For inverting the results of iterative sampling based on best-fitting models
# This current iterative sampling is based on R-based sample results, as recommended. But you can feel free to alter it to quad-tree iterative sampling.


# Files
insar_files = [ # Paths to InSAR datasets
               'real_data/SEN/asc/sen_asc_los_ll.grd',
               'real_data/SEN/des/sen_des_los_ll.grd',
               ]

smooth_files = [ # Paths to smoothed input data for enhanced sampling
               'real_data/SEN/asc/smooth.grd',
               'real_data/SEN/des/smooth.grd',
               ]

look_dirs   = [ # Paths to InSAR look vectors
               'real_data/SEN/asc/look',
                'real_data/SEN/des/look'
               ]

tri_dirs    = [ # Paths to triangular R-based sample directories
                'real_data/SEN/asc/tri_output_BF4',
                'real_data/SEN/des/tri_output_BF4'
               ]


weights     = [ # Relative inversion weights for datasets
               1, 
               1,
               ]



# Geographic parameters
ref_point   = [73.1603, 38.1025] # Cartesian coordinate reference point
EPSG        = '32643'            # EPSG code for relevant UTM zone

# Constituitive parameters
poisson_ratio  = 0.25
shear_modulus  = 30 * 10**9 # From Turcotte & Schubert
lmda           = 2 * shear_modulus * poisson_ratio / (1 - 2 * poisson_ratio)
alpha          = (lmda + shear_modulus) / (lmda + 2 * shear_modulus)    

# Quadtree parameters
#rms_min      = 0.004  # RMS threshold (data units), default 0.1
#nan_frac_max = 0.7  # Fraction of NaN values allowed per cell, default 0.7
#width_min    = 0.05  # Minimum cell width (km), default 0.1
#width_max    = 24    # Maximum cell width (km), default 4
#mean_low     = 0.3  #lower Threshold for the mean data value
#mean_up      = 1  #upper threshold for the mean data value

# Define uniform prior 
out_dir         = 'real_results_BF4'
inversion_mode  = 'run' # 'check_downsampling' to only  make quadtree downsampling plots, 'run' to run inversion, 'reload' to load previous results and prepare output products

# NOTE: Data coordinates and fault coordinates should be the same units (km or m) and 
#       LOS displacement units and slip units should be same units (m, cm, or km), but
#       spatial and slip units do not need to be the same.

# Prior limits
x_lim           = [-10, 10]     # (km) 
y_lim           = [-10, 10]     # (km)
z_lim           = [1,  12]   # (km)
l_lim           = [ 5, 35]     # (km)
w_lim           = [ 5,  35]     # (km)
strike_lim      = [ 0, 60] # (deg)
dip_lim         = [  70, 110]     # (deg)
strike_slip_lim = [  -30,  30]     # (m)
dip_slip_lim    = [  -5,  5]     # (m)

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

labels = ['x',   'y',  'z', 'l',  'w', 'strike', 'dip', 'strike_slip', 'dip_slip',] # Labels for plotting                        
units  = ['km', 'km', 'km', 'km', 'km',    'deg', 'deg',           'm',        'm',] # Unit labels for plotting
scales = [1   , 1   , 1   ,  1,    1,        1,     1,             1,          1,] # Unit scaling factors for plotting

# Plotting parameters
vlim_disp = [-0.2, 0.2]


# def mcmc_setup(insar_files, look_dirs, weights, ref_point, EPSG, rms_min, nan_frac_max, width_min, width_max, vlim_disp, inversion_mode='run'):
# ---------------------------------- CONFIGURATION ----------------------------------  #
# Ingest InSAR data for inversion

#if smooth ==0:
#    datasets = pyffit.insar.prepare_datasets(insar_files, look_dirs, weights, ref_point,EPSG=EPSG, rms_min=rms_min, nan_frac_max=nan_frac_max, width_min=width_min, width_max=width_max)

#if smooth ==1:
#    datasets = pyffit.insar.prepare_datasets_smooth(insar_files,smooth_files, look_dirs, weights, ref_point,EPSG=EPSG, mean_low=mean_low,mean_up=mean_up,rms_min=rms_min, nan_frac_max=nan_frac_max, width_min=width_min, width_max=width_max)
    
datasets = pyffit.insar.prepare_datasets_tri(insar_files, look_dirs, tri_dirs, weights, ref_point,EPSG=EPSG)



# Plot downsampled data
for dataset in datasets.keys():
    pyffit.figures.plot_quadtree(datasets[dataset]['data'], datasets[dataset]['extent'], (datasets[dataset]['x_samp'], datasets[dataset]['y_samp']), datasets[dataset]['data_samp'], vlim_disp=vlim_disp, file_name=f'{out_dir}/tri_{dataset}.png',)

if inversion_mode == 'check_downsampling':
    # Quit if only checking downsampling parameters
    exit()

# Aggregate NaN locations
i_nans = np.concatenate([datasets[name]['i_nans'] for name in datasets.keys()])

# Aggregate coordinates
coords = (np.concatenate([datasets[name]['x_samp'] for name in datasets.keys()])[~i_nans],
          np.concatenate([datasets[name]['y_samp'] for name in datasets.keys()])[~i_nans],)
#coords = (np.concatenate([datasets[name]['x_samp'] for name in datasets.keys()]),
#          np.concatenate([datasets[name]['y_samp'] for name in datasets.keys()]),)

# Aggregate look vectors
look     = np.concatenate([datasets[name]['look_samp'] for name in datasets.keys()])[~i_nans]

# Aggregate data, standard deviations, and weights
data     = np.concatenate([datasets[name]['data_samp'] for name in datasets.keys()])[~i_nans]
data_std = np.concatenate([datasets[name]['data_samp_std'] for name in datasets.keys()])[~i_nans]
weights  = np.concatenate([np.ones_like(datasets[name]['data_samp']) * datasets[name]['weight'] for name in datasets.keys()])[~i_nans]
B        = np.diag(weights)      # Weight matrix
S_inv    = np.diag((data_std)**-2) # Covariance matrix

# NOTE: we set up the model/probability functions here in order to define them WITH the data 
#       vectors baked-in. This way, we accrue less compuptational overhead in pickling the 
#       method arguments for parallelization 
#       (see emcee page for more info: https://emcee.readthedocs.io/en/stable/tutorials/parallel/)

# Define probability functions
def patch_slip(m):
    """
    m      - model parameters
    coords - x/y coordinates for model prediciton
    look   - array of look vector components 
    """

    # Unpack input parameters
    x_patch, y_patch, z, l, w, strike, dip, strike_slip, dip_slip = m
    x, y = coords
    slip = [strike_slip, dip_slip, 0]

    # Generate fault patch
    patch = pyffit.finite_fault.Patch()
    patch.add_self_geometry((x_patch, y_patch, z), strike, dip, l, w, slip=slip)

    # Complute full displacements
    disp     = patch.disp(x, y, 0, alpha, slip=slip).reshape(x.size, 3, 1, 1)
    disp_LOS = pyffit.finite_fault.proj_greens_functions(disp, look)[:, :, 0].reshape(-1)

    if -np.inf in disp:
        return np.ones_like(disp_LOS) * np.inf * -1
    else:
        return disp_LOS


#def elliptical_tapering(m, n, W, L, N, a, b, u0):
#    central = (N+1)/2 - 1
#    distance = np.sqrt(((n - central) * (L / N) / a)**2 + ((m - central) * (W / N) / b)**2)
#    return np.array(u0) * np.exp(-distance**2)

def elliptical_tapering(m,n,N,u0):
    central = (N+1)/2 - 1
    axis = N/2
    scale1 = np.sqrt(1 - (m - central)**2 / axis**2)
    scale2 = np.sqrt(1 - (n - central)**2 / axis**2)
    return np.array(u0) * scale1 * scale2

#elliptical deformation
def patch_slip_ellip(m):
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
    N = 3
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
            slip_sub = elliptical_tapering(m, n, N, u0)
            patch_sub.add_self_geometry((x_sub,y_sub,z_sub),strike,dip,l_sub,w_sub,slip=slip_sub)
            disp = patch_sub.disp(x,y,0,alpha,slip=slip_sub)
            disp_summed += disp
    disp_summed = disp_summed.reshape(x.size,3,1,1)
    disp_LOS = pyffit.finite_fault.proj_greens_functions(disp_summed, look)[:, :, 0].reshape(-1)
    if -np.inf in disp_summed:
        return np.ones_like(disp_LOS) * np.inf * -1
    else:
        return disp_LOS


@numba.jit(nopython=True) # For a little speed boost
def log_likelihood(G_m):
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

    if (np.inf in np.abs(G_m)) | (np.nan in G_m):
        # print('inf in G_m')
        return -np.inf

    # elif np.nan in G_m:
    #     # print('NaN in G_m')
    #     return -np.inf

    else:
        r = data - G_m
        likelihood = -0.5 * r.T @ S_inv @ B @ r

        if np.isnan(likelihood):
            print('WARNING: probability function returned NaN')

            print(f'G_m nan: {np.sum((np.abs(G_m) == np.inf))}')
            print(f'G_m inf: {np.sum(np.isnan(G_m))}')
            print(f'r nan: {np.sum(np.isnan(r))}')
            print(f'r inf: {np.sum((np.abs(r) == np.inf))}')
            likelihood = -np.inf

        return likelihood

def log_prob(m):
    """
    Determine log-probaility of model m using a uniform prior.
    """

    # Check prior
    if np.all([priors[key][0] <= m[i] <= priors[key][1] for i, key in enumerate(priors.keys())]):
        #return log_likelihood(patch_slip(m)) # Log. probability of sample is only the log. likelihood
        return log_likelihood(patch_slip_ellip(m)) # Use the elliptic deformation
    else:
        return -np.inf                                    # Exclude realizations outside of priors

