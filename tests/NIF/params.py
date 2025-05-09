# -------------------------- NIF parameter file --------------------------
# The variables in this script will be imported to 'nif.py', so they should be defined as Python objects
# This allows for easier specification of variables with float, array-like, and other non-string types (i.e. colormaps)

# IMPORTANT: the order of the parameters matters! Do not add or delete parameters without appropriately modifying
# the relevant driver method in nif.py first.

# -------------------------- Parameters --------------------------
# Imports
import cmcrameri.cm as cmc

# Run mode
mode = 'NIF' # NIF for standard NIF run              

# Files
mesh_file           = '/Users/evavra/Projects/SSAF/Analysis/Finite_Fault_Modeling/Mesh/Geometry/mesh_points.txt'
triangle_file       = '/Users/evavra/Projects/SSAF/Analysis/Finite_Fault_Modeling/Mesh/Geometry/mesh_connectivity.txt'
model_file          = '/Users/evavra/Software/pyffit/tests/NIF/synthetic_data_full/slip_model.h5'
mask_file           = '/Users/evavra/Software/pyffit/tests/NIF/synthetic_data_full/decorr_mask.grd'
# insar_dir           = '/Users/evavra/Software/pyffit/tests/NIF/synthetic_data/time_series'
# file_format         = 'time_series_*.grd'
# date_index_range    = [12, 15]

insar_dir           = '/Users/evavra/Software/pyffit/tests/NIF/synthetic_data_full/signal'
file_format         = 'signal_*.grd'
date_index_range    = [-7, -4]
xkey                = 'x'
ykey                = 'y'

look_dir            = f'/Users/evavra/Projects/SSAF/Data/InSAR/Sentinel_1/timeseries/combined/'
asc_velo_model_file = f'/Users/evavra/Projects/SSAF/Data/InSAR/Sentinel_1/timeseries/combined/v_LOS_asc.grd'
des_velo_model_file = f'/Users/evavra/Projects/SSAF/Data/InSAR/Sentinel_1/timeseries/combined/v_LOS_des.grd'
downsampled_dir     = f'/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/high_resolution/downsampled_data'
out_dir             = f'/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/high_resolution'

# Data parameters:
# version           = 1
# remove_mean       = True
# wavelength        = 55.465763
# proj_los          = False
# velo_model_factor = -0.1 # Scaling factor for velocity model
# data_factor       = 10 # Scaling factor for data
# dataset_names     = ['asc', 'des']

# Geographic parameters
EPSG        = '32611' 
ref_point   = [-116, 33.5]
data_region = [-116.4, -115.7, 33.25, 34.0]
avg_strike  = 315.8
trace_inc   = 0.01

# Fault parameters
poisson_ratio = 0.25      # Poisson ratio
shear_modulus = 6 * 10**9 # Shear modulus (Pa)
disp_components = [1]     # displacement components to use [0 for fault-perpendicular, 1 for fault-parallel, 2 for vertical]
slip_components = [0]     # slip components to use [0 for strike-slip, 1 for dip-slip, 2 for opening]

# Resolution based resampling
# resolution_threshold = 2.3e-1 # cutoff value for resolution matrix (lower values = more points)
resolution_threshold = 1 # cutoff value for resolution matrix (lower values = more points)
width_min            = 0.1    # Min. allowed cell size (km)
width_max            = 10     # Max. allowed cell size (km)
max_intersect_width  = 100    # Max. allowed size for fault-intersecting cells (km)
min_fault_dist       = 1      # Buffer distance from fault to enforce max_intersect width
max_iter             = 10     # Max. allowed sampling iterations
smoothing_samp       = False
edge_slip_samp       = False
smoothing_inv        = True
edge_slip_inv        = False

# NIF parameters
omega           = 5e1   # temporal smoothing hyperparameter
sigma           = 1e1   # data covariance scaling hyperparameter (Note: for single dataset, and single kappa value for steady-state velocity, transient slip, and transient velocity, sigma becomes reduntant)
# kappa           = 1e2   # spatial smoothing hyperparameter
kappa           = 2e1   # spatial smoothing hyperparameter
mu              = kappa # spatial smoothing hyperparameter
eta             = kappa # zero-edge-slip hyperparameter


# Uncertainties and limits
v_sigma         = 1e-9    # initial uncertainty on interseimic slip rate (mm/yr) 
W_sigma         = 1       # initial uncertainty on transient slip (mm) 
W_dot_sigma     = 1       # initial uncertainty on transient slip rate (mm/yr) \

v_lim           = (0, 3)  # min./max. bounds on steady slip rate values (mm/yr)
W_lim           = (0, 30) # min./max. bounds on transient slip values (mm)
W_dot_lim       = (0, 50) # min./max. bounds on transient rate slip values (mm/yr)

# Plot parameters
# xlim      = [-16, 30]
# ylim      = [-21, 30]
# xlim_r     = [-21, 26] 
# ylim_r     = [-5, 5] 
xlim       = [-35.77475071, 26.75029172]
ylim       = [-26.75029172, 55.08597388]
vlim_slip  = [0,   20]
vlim_disp  = [[-10, 10],
              [-10, 10],
              [-1, 1]] 
cmap_disp  = cmc.vik
figsize    = (10, 10)
dpi        = 75
markersize = 40

# Get file path
param_file = __file__

# ----------------------------------------------------

