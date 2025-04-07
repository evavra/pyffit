#Test: inversion for GPS data
import pyffit
import numba
import numpy as np

# Files
gps_files = [ # Paths to InSAR datasets
               ]

               
weights     = [ # Relative inversion weights for datasets
               # 1, 
               1,
               ]

# Geographic parameters
ref_point   = [-115.929, 33.171] # Cartesian coordinate reference point
EPSG        = '32611'            # EPSG code for relevant UTM zone

# 3d or 2d GPS data?
data_type='2d'

# Constituitive parameters
poisson_ratio  = 0.25
shear_modulus  = 30 * 10**9 # From Turcotte & Schubert
lmda           = 2 * shear_modulus * poisson_ratio / (1 - 2 * poisson_ratio)
alpha          = (lmda + shear_modulus) / (lmda + 2 * shear_modulus)    


# Define uniform prior 
out_dir         = 'results' # directory that store the outputs
inversion_mode  = 'run' # 'run' to run inversion, 'reload' to load previous results and prepare output products

# NOTE: Data coordinates and fault coordinates should be the same units (km or m) and 
#       LOS displacement units and slip units should be same units (m, cm, or km), but
#       spatial and slip units do not need to be the same.

# Prior limits
x_lim           = [-50,  50]     # (km) 
y_lim           = [-60,  60]     # (km)
l_lim           = [ 50, 150]     # (km)
w_lim           = [  5,  20]     # (km)
strike_lim      = [300, 360] # (deg)
dip_lim         = [ 30, 150]     # (deg)
strike_slip_lim = [  -5,  0]     # (m)
dip_slip_lim    = [  -1,  1]     # (m)

# Construct prior
priors  = {
           'x':           x_lim,
           'y':           y_lim,
           'l':           l_lim,
           'w':           w_lim,
           'strike':      strike_lim,
           'dip':         dip_lim,
           'strike_slip': strike_slip_lim,
           'dip_slip':    dip_slip_lim,
           }

labels = ['x',   'y',  'l',  'w', 'strike', 'dip', 'strike_slip', 'dip_slip',] # Labels for plotting                        
units  = ['km', 'km', 'km', 'km',    'deg', 'deg',           'm',        'm',] # Unit labels for plotting
scales = [1   , 1   , 1   , 1   ,        1,     1,             1,          1,] # Unit scaling factors for plotting

# Plotting parameters
vlim_disp = [-1, 1]


# def mcmc_setup(insar_files, look_dirs, weights, ref_point, EPSG, rms_min, nan_frac_max, width_min, width_max, vlim_disp, inversion_mode='run'):
# ---------------------------------- CONFIGURATION ----------------------------------  #
# Ingest InSAR data for inversion
datasets = pyffit.gps.prepare_datasets_gps(gps_files,weights,ref_point,EPSG=EPSG,data_type=data_type)

# Aggregate coordinates
coords = (np.concatenate([datasets[name]['x_samp'] for name in datasets.keys()])[~i_nans],
          np.concatenate([datasets[name]['y_samp'] for name in datasets.keys()])[~i_nans],)


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
def patch_slip_ellip_gps(m,data_type=data_type):
    """
    m      - model parameters
    data_type: 2d or 3d
    
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
            slip_sub = elliptical_tapering(m, n, N, u0)
            patch_sub.add_self_geometry((x_sub,y_sub,z_sub),strike,dip,l_sub,w_sub,slip=slip_sub)
            disp = patch_sub.disp(x,y,0,alpha,slip=slip_sub)
            disp_summed += disp
    disp_summed = disp_summed.reshape(x.size,3,1,1)
    if data_type='2d':
    	disp_summed=disp_summed[0:1,:]
    	
    return disp_summed

@numba.jit(nopython=True) # For a little speed boost
def log_likelihood_gps(G_m):

	if (np.inf in np.abs(G_m)) | (np.nan in G_m):
    # print('inf in G_m')
        return -np.inf

    # Compute the residuals elementwise
    # This subtracts the observed data from the model output and normalizes by the standard deviation
    residuals = (G_m - data) / data_std
    
    # Compute the misfit: sum of squared normalized residuals over all m points and 3 directions
    misfit = np.sum(residuals**2)
    
    # Option 1: If you don't need the normalization constant, you can use:
    log_like = -0.5 * misfit

    # Option 2: If you want the full Gaussian log-likelihood, including the normalization constants:
    # Note: Here m is the number of data points.
    # normalization = sum over all points of log(sqrt(2*pi)*data_std) computed elementwise.
    # Uncomment the lines below if you wish to include these terms.
    #
    # normalization = np.sum(np.log(np.sqrt(2 * np.pi) * data_std))
    # log_like = -0.5 * misfit - normalization
    
    return log_like

def log_prob(m):
    """
    Determine log-probaility of model m using a uniform prior.
    """

    # Check prior
    if np.all([priors[key][0] <= m[i] <= priors[key][1] for i, key in enumerate(priors.keys())]):
        return log_likelihood_gps(patch_slip_ellip_gps(m,data_type=data_type)) # change your data type here
    else:
        return -np.inf                                    # Exclude realizations outside of priors

