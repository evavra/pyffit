import pyffit
import numba
import numpy as np

# Files
insar_files = [ # Paths to InSAR datasets
               # 'data/synthetic_data_1.grd',
               'data/synthetic_data_2.grd',
               ]
look_dirs   = [ # Paths to InSAR look vectors
               # 'data/look',
               'data/look',
               ]
weights     = [ # Relative inversion weights for datasets
               # 1, 
               1,
               ]

# Geographic parameters
ref_point   = [-115.929, 33.171] # Cartesian coordinate reference point
EPSG        = '32611'            # EPSG code for relevant UTM zone

# Constituitive parameters
poisson_ratio  = 0.25
shear_modulus  = 30 * 10**9 # From Turcotte & Schubert
lmda           = 2 * shear_modulus * poisson_ratio / (1 - 2 * poisson_ratio)
alpha          = (lmda + shear_modulus) / (lmda + 2 * shear_modulus)    

# Quadtree parameters
rms_min      = 0.1  # RMS threshold (data units)
nan_frac_max = 0.9  # Fraction of NaN values allowed per cell
width_min    = 0.1  # Minimum cell width (km)
width_max    = 4    # Maximum cell width (km)

# Define uniform prior 
out_dir         = 'results'
inversion_mode  = 'run' # 'check_downsampling' to only  make quadtree downsampling plots, 'run' to run inversion, 'reload' to load previous results and prepare output products

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
datasets = pyffit.insar.prepare_datasets(insar_files, look_dirs, weights, ref_point, 
                                              EPSG=EPSG, rms_min=rms_min, nan_frac_max=nan_frac_max, width_min=width_min, width_max=width_max)

# Plot downsampled data
for dataset in datasets.keys():
    pyffit.figures.plot_quadtree(datasets[dataset]['data'], datasets[dataset]['extent'], (datasets[dataset]['x_samp'], datasets[dataset]['y_samp']), datasets[dataset]['data_samp'], vlim_disp=vlim_disp, file_name=f'quadtree_{dataset}.png',)

if inversion_mode == 'check_downsampling':
    # Quit if only checking downsampling parameters
    exit()

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
    x_patch, y_patch, l, w, strike, dip, strike_slip, dip_slip = m
    x, y = coords
    slip = [strike_slip, dip_slip, 0]

    # Generate fault patch
    patch = pyffit.finite_fault.Patch()
    patch.add_self_geometry((x_patch, y_patch, 0), strike, dip, l, w, slip=slip)

    # Complute full displacements
    disp     = patch.disp(x, y, 0, alpha, slip=slip).reshape(x.size, 3, 1, 1)
    disp_LOS = pyffit.finite_fault.proj_greens_functions(disp, look)[:, :, 0].reshape(-1)

    if -np.inf in disp:
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
        return log_likelihood(patch_slip(m)) # Log. probability of sample is only the log. likelihood
    else:
        return -np.inf                                    # Exclude realizations outside of priors

