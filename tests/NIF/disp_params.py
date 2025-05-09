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
mode = ['analyze_disp'] # 'NIF' to run inversion, 'analyze' to make figures, or both             

# Files and directories
# mesh_file           = '/Users/evavra/Projects/SSAF/Analysis/Finite_Fault_Modeling/Mesh/Geometry/mesh_points_simple.txt'
# triangle_file       = '/Users/evavra/Projects/SSAF/Analysis/Finite_Fault_Modeling/Mesh/Geometry/mesh_connectivity_simple.txt'
# downsampled_dir     = f'/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/data/low_resolution/downsampled_data'
# out_dir             = f'/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/data/low_resolution'
mesh_file           = '/Users/evavra/Projects/SSAF/Analysis/Finite_Fault_Modeling/Mesh/Geometry/mesh_points_updated.txt'
triangle_file       = '/Users/evavra/Projects/SSAF/Analysis/Finite_Fault_Modeling/Mesh/Geometry/mesh_connectivity_updated.txt'
data_dir            = '/Users/evavra/Projects/SSAF/Data/InSAR/Sentinel_1/timeseries/decomposed/filt/corrections'
file_format         = 'u_para_*_filt_10km_deramped.grd'
run_dir             = '/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/data/updated_mesh/omega_1.0e+03__kappa_1.0e+00__sigma_1.0e+00'
# run_dir             = '/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/firkin/data/high_resolution/omega_1.0e+03__kappa_1.0e+01__sigma_1.0e+01'
# samp_file           = '/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/firkin/data/high_resolution/downsampled_data/cutoff=9.9e-01_wmin=0.10_wmax=10.00_max_int_w=100.0_min_fdist=1.00_max_it=10.000000.pkl'
samp_file           = '/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/data/updated_mesh/downsampled_data/cutoff=1.0e+00_wmin=0.10_wmax=10.00_max_int_w=0.1_min_fdist=0.20_max_it=10.000000.h5'
site_file           = '/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/sites.txt'

# -------------------------- 'Optional' parameters --------------------------
# The following parameters have default values in nif.py, but will almost certainly want to be chosen for each
# specific model run.

# # Data info
# date_index_range      = [-22, -14] # Indices of scene dates or IDs to be used for timestamping 
# xkey                  = 'lon' # Label of x-coordinates in dataset (usually 'x' or 'lon')
# coord_type            = 'geographic'
# dataset_name          = 'sentinel_fault_parallel'
# check_lon             = False # For geographic coordinates, check to confirm longitude is [-180, 180]
# reference_time_series = True  # Reference whole time series to the observation at t = 0
# use_dates             = True  # True formatted dates ()'YYYY-MM-DD' or DateTime objects), False to use raw str specified in date_index_range.
# use_datetime          = True  # Convert dates from str to DateTime
# dt                    = 12    # Epoch length (days)
# data_factor           = 10    # Convert from cm to desired units

# # Files
# # model_file          = '/Users/evavra/Software/pyffit/tests/NIF/synthetic_data_full/slip_model.h5'
# # mask_file           = '/Users/evavra/Software/pyffit/tests/NIF/synthetic_data_full/decorr_mask.grd'
# model_file = ''
# mask_file  = ''
# mask_dir   = '/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/masks/'
# cov_dir    = '/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/covariance/'

# # Covariance estimation parameters
# estimate_covariance = False
# mask_dists          = [3]
# n_samp              = 2*10**7
# r_inc               = 0.2
# r_max               = 80
# m0                  = [0.9, 1.5, 5]
# c_max               = 2
# sv_max              = 2

# Geographic parameter
# ref_point   = [-116, 33.5]
# avg_strike  = 315.8
# trace_inc   = 0.01

# # Fault parameters
# poisson_ratio   = 0.25      # Poisson ratio
# shear_modulus   = 6 * 10**9 # Shear modulus (Pa)
# disp_components = [1]       # displacement components to use [0 for fault-perpendicular, 1 for fault-parallel, 2 for vertical]
# slip_components = [0]       # slip components to use [0 for strike-slip, 1 for dip-slip, 2 for opening]

# # Resolution based resampling
# # resolution_threshold = 2.3e-1 # cutoff value for resolution matrix (lower values = more points)
# resolution_threshold = 1.0 # cutoff value for resolution matrix (lower values = more points)
# width_min            = 0.1 # Min. allowed cell size (km)
# width_max            = 10  # Max. allowed cell size (km)
# max_intersect_width  = 100 # Max. allowed size for fault-intersecting cells (km)
# min_fault_dist       = 1   # Buffer distance from fault to enforce max_intersect width
# max_iter             = 10  # Max. allowed sampling iterations
# smoothing_samp       = False
# edge_slip_samp       = False

# # Plot parameters
# xlim       = [-35.77475071, 26.75029172]
# ylim       = [-26.75029172, 55.08597388]
# vlim_slip  = [0, 40]
# vlim_disp  = [[-20, 20],
#               [-20, 20],
#               [-20, 20]] 
# cmap_disp  = cmc.vik
# cmap_slip  = cmc.lajolla_r
# figsize    = (10, 7)
# dpi        = 75
# markersize = 40

# Get file path
# param_file = __file__
# ----------------------------------------------------

