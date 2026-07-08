# -------------------------- NIF parameter file --------------------------
# The variables in this script will be imported to 'nif.py', so they should be defined as Python objects
# This allows for easier specification of variables with float, array-like, and other non-string types (i.e. colormaps)

# IMPORTANT: the order of the parameters matters! The required parameters are positional arguments to the relevant
# driver method in nif.py that corresponds to 'mode'. Thus, they should always be specified first and in the same 
# order as listed in the driver. However, the order of the remaining named parameters can be arbitrary.

# -------------------------- Imports --------------------------
import cmcrameri.cm as cmc

# -------------------------- Required parameters --------------------------
# Run mode(s)
# mode = ['testing'] # 'NIF' to run inversion, 'analyze' to make figures, or both             
# mode = ['downsample'] # 'NIF' to run inversion, 'analyze' to make figures, or both             
# mode = ['static'] # 'NIF' to run inversion, 'analyze' to make figures, or both             
# mode = ['filter'] # 'NIF' to run inversion, 'analyze' to make figures, or both             
# mode = ['covariances'] # 'NIF' to run inversion, 'analyze' to make figures, or both             
# mode = ['NIF', 'analyze_model'] # 'NIF' to run inversion, 'analyze' to make figures, or both             
# mode = ['analyze_model'] # 'NIF' to run inversion, 'analyze' to make figures, or both                    
# mode = ['NIF'] # 'NIF' to run inversion, 'analyze' to make figures, or both             

# Files and directories
# mesh_file           = '/Users/evavra/Projects/SSAF/Analysis/Finite_Fault_Modeling/Mesh/Geometry/mesh_points_simple.txt'
# triangle_file       = '/Users/evavra/Projects/SSAF/Analysis/Finite_Fault_Modeling/Mesh/Geometry/mesh_connectivity_simple.txt'
# downsampled_dir     = f'/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/data/low_resolution/downsampled_data'
# out_dir             = f'/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/data/low_resolution'
mesh_file           = '/Users/evavra/Projects/SSAF/Analysis/Finite_Fault_Modeling/Mesh/Geometry/mesh_points_529.txt'
triangle_file       = '/Users/evavra/Projects/SSAF/Analysis/Finite_Fault_Modeling/Mesh/Geometry/mesh_connectivity_529.txt'
# downsampled_dir     = f'/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/data/updated_mesh/testing_covariances/downsampled_data'
# out_dir             = f'/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/data/updated_mesh/testing_covariances'
# data_dir            = '/Users/evavra/Projects/SSAF/Data/InSAR/Sentinel_1/timeseries/decomposed/filt/corrections'
# file_format         = 'u_para_*_filt_10km_deramped.grd'
# date_index_range      = [-33, -23] # Indices of scene dates or IDs to be used for timestamping 
# dataset_name          = 'sentinel_fault_parallel'

downsampled_dir     = f'/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/data/mesh_529/low_pass_filt/downsampled_data'
# out_dir             = f'/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/data/mesh_529/low_pass_filt/full_covariance/sigma-1'
# out_dir             = f'/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/data/mesh_529/low_pass_filt/full_covariance/constrained'
out_dir             = f'/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/data/mesh_529/low_pass_filt/dense'
data_dir            = '/Users/evavra/Projects/SSAF/Data/InSAR/Sentinel_1/timeseries/decomposed/filt/corrections/low_pass_filt'
file_format         = 'u_para_*_hp-filt-10km_deramped_lp-filt-2.7km.grd'
date_index_range      = [-50, -40] # Indices of scene dates or IDs to be used for timestamping 
dataset_name          = 'low_pass_filt'

# -------------------------- 'Optional' parameters --------------------------
# The following parameters have default values in nif.py, but will almost certainly want to be chosen for each
# specific model run.

# Data info
xkey                  = 'lon' # Label of x-coordinates in dataset (usually 'x' or 'lon')
coord_type            = 'geographic'
check_lon             = False # For geographic coordinates, check to confirm longitude is [-180, 180]
reference_time_series = True  # Reference whole time series to the observation at t = 0
use_dates             = True  # True formatted dates ()'YYYY-MM-DD' or DateTime objects), False to use raw str specified in date_index_range.
use_datetime          = True  # Convert dates from str to DateTime
dt                    = 12    # Epoch length (days)
data_factor           = 1     # Convert from mm to desired units

# Covariance estimation parameters
estimate_covariance = True
mask_dists          = [5]
n_samp              = 1*10**7
r_inc               = 0.1
r_max               = 30
m0                  = [0.9, 1.5, 5]
c_max               = 10
sv_max              = 10

# Geographic parameters
ref_point   = [-116, 33.5]
avg_strike  = 315.8
trace_inc   = 0.01

# Fault parameters
poisson_ratio   = 0.25      # Poisson ratio
shear_modulus   = 6 * 10**9 # Shear modulus (Pa)
disp_components = [1]       # displacement components to use [0 for fault-perpendicular, 1 for fault-parallel, 2 for vertical]
slip_components = [0]       # slip components to use [0 for strike-slip, 1 for dip-slip, 2 for opening]

# Resolution based resampling
resolution_threshold = 0.99 # cutoff value for resolution matrix (lower values = more points)
width_min            = 0.1 # Min. allowed cell size (km)
width_max            = 2.0 # Max. allowed cell size (km)
max_intersect_width  = 0.1 # Max. allowed size for fault-intersecting cells (km)
min_fault_dist       = 0.1 # Buffer distance from fault to enforce max_intersect width
max_iter             = 10  # Max. allowed sampling iterations
smoothing_samp       = False
edge_slip_samp       = False

# NIF parameters
omega           = 10**2   # temporal smoothing hyperparameter
kappa           = 10**1   # spatial smoothing hyperparameter
sigma           = 10**1   # data covariance scaling hyperparameter  
rho             = 1e0   # ramp covariance scaling hyperparameter 

mu              = kappa # spatial smoothing hyperparameter
eta             = kappa # zero-edge-slip hyperparameter
  
# State vector settings
steady_slip     = False  # Include constant slip rate in state vector
ramp_type       = 'None' # Include ramp in state vector ('none', 'linear', or 'quadtratic', 'saved)
constrain       = False   # Perform nonlinear solve for constrained state vector

# State vector initial uncertainties
v_sigma         = 1e0    # initial uncertainty on interseimic slip rate (mm/yr) 
W_sigma         = 1e-9   # initial uncertainty on transient slip (mm) 
W_dot_sigma     = 1e-9   # initial uncertainty on transient slip rate (mm/yr) 
ramp_sigma      = 1e-9   # initial uncertainty on transient slip rate (mm/yr) 

# State vector constraints
v_lim           = (0, 0.1)    # min./max. bounds on steady slip rate values (mm/yr)
W_lim           = (0, 100)    # min./max. bounds on transient slip values (mm)
W_dot_lim       = (0, 1000)    # min./max. bounds on transient slip rate values (mm/yr)
ramp_lim        = (-100, 100) # min./max. bounds on ramp coefficient values

# Plot parameters
xlim       = [-35.77475071, 26.75029172]
ylim       = [-26.75029172, 55.08597388]
vlim_slip  = [0, 40]
vlim_disp  = [[-20, 20],
              [-20, 20],
              [-20, 20],
              [-20, 20]] 
cmap_disp  = cmc.vik
cmap_slip  = cmc.lajolla_r
figsize    = (10, 7)
dpi        = 150
markersize = 40

# Get file path for loading correct parameters in the future
# Note: automatically extracts this file's path -- do not edit!
param_file = __file__
# ----------------------------------------------------



