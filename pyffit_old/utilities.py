import os
import numpy as np
import pandas as pd
from pyproj import CRS, Proj
from scipy.interpolate import interp1d


def proj_ll2utm(x, y, crs_epsg, inverse=False):
    """
    Project WGS84 lon/lat coordinates to UTM coordinates.
    Specify inverse=True to project to lon/lat.

    INPUT:
    x, y     - lon/lat coordinates (or UTM east/north for inverse=True)
    crs_epsg - EPSG code for UTM zone desired
    inverse  - project UTM to lat/lon

    Optional:
    hsphere - specify hemisphere. Default is north, specify for south.
    """
    # Get CRS
    # Define projection
    # myProj = Proj(f'+proj=utm +zone={zone}, +{hsphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs')
    myProj = Proj(CRS.from_epsg(crs_epsg))

    # Determine projection direction
    if inverse:
        x_proj, y_proj = myProj(x, y, inverse=inverse)
    else:
        x_proj, y_proj = myProj(x, y)

    return x_proj, y_proj


def add_utm_coords(df, crs_epsg):
    """
    Add UTM coordinates to DataFrame with WGS84 coordinates
    
    INPUT:
    df - DataFrame including columns for longitude and latitude. 
    
    OUTPUT:
    df - same as input but with ['UTMx', 'UTMy'] aadded
    """
    
    # Get lon/lat column names
    cols        = df.columns
    coords      = []
    coord_types = [['Longitude', 'Latitude'], ['longitude', 'latitude'], ['Lon', 'Lat'], ['lon', 'lat']]
    
    for coord_type in coord_types:
        if coord_type[0] in cols and coord_type[0] in cols:
            coords = coord_type
            
    if len(coords) == 0:
        print('Coordinate names not recognized!')
        return
    
    # Define UTM projection
    myProj = Proj(CRS.from_epsg(crs_epsg))

    # Project coordinates
    UTMx, UTMy = myProj(df[coords[0]].values, df[coords[1]].values)
    
    return df.assign(UTMx=UTMx, UTMy=UTMy)  


def add_ll_coords(df, crs_epsg):
    """
    Add  WGS84 coordinates to DataFrame with TM coordinates
    
    INPUT:
    df - DataFrame including columns for UTM x and y. 
    
    OUTPUT:
    df - same as input but with ['Longitude', 'Latitude'] aadded
    """
    
    # Get lon/lat column names
    cols        = df.columns
    coords      = []
    coord_types = [['UTMx', 'UTMy'], ['x', 'y']]
    
    for coord_type in coord_types:
        if coord_type[0] in cols and coord_type[0] in cols:
            coords = coord_type
            
    if len(coords) == 0:
        print('Coordinate names not recognized!')
        return
    
    # Define UTM projection
    myProj = Proj(crs_epsg)

    # Project coordinates
    Longitude, Latitude = myProj(df[coords[0]].values, df[coords[1]].values, inverse=True)
    
    return df.assign(Longitude=Longitude, Latitude=Latitude) 


def get_local_xy_coords(lon, lat, origin, EPSG='32611', unit='km'):
    """
    Convert lon/lat coordinates to local x/y metric coordiantes.

    INPUT:
    lon    - longitude values
    lat    - latitude values
    origin - lon/lat of reference point
    EPSG   - local UTM EPSG code
    unit   - meters (m) or kilometers (km)

    OUTPUT:
    x, y - projected x/y coordinates in local reference frame
    """

    if unit == 'km':
        scale = 1e-3
    else:
        scale = 1

    # Convert origin to UTM
    origin_utm = proj_ll2utm(origin[0], origin[1], EPSG)

    # Convert coordinates to UTM
    x, y = proj_ll2utm(lon, lat, EPSG)

    # Reference and scale accordingly
    x = (x - origin_utm[0]) * scale
    y = (y - origin_utm[1]) * scale

    return x, y
        

# @numba.jit(nopython=True)
def rotate(x, y, theta): 
    """
    Perform 2D rotation about origin.
    """

    x_r = np.cos(theta) * x - np.sin(theta) * y
    y_r = np.sin(theta) * x + np.cos(theta) * y

    return x_r, y_r


def fit_fault_spline(fault, bounds, seg_inc, EPSG='32611', key_str='F'):
    """
    Fit linear spline with segments of approximately seg_inc length (km) to fault trace 
    """

    # Define approx. increment for upsampling fault trade
    upsamp_inc = 10

    # Select nodes within specified bounds
    fault_sel = fault[(fault['Longitude'] >= bounds[0]) & (fault['Longitude'] <= bounds[1]) & (fault['Latitude'] >= bounds[2]) & (fault['Latitude'] <= bounds[3])]

    # Estimate average strike from selected portion of fault 
    p          = np.polyfit(fault_sel['UTMx'], fault_sel['UTMy'], 1)
    strike_avg = np.arctan(p[0])

    # Get approximate spline segment length from average fault strike
    # upsamp_deg   = abs(upsamp_inc * np.sin(strike_avg)/111.19)
    
    # Make spline function
    spl_upsamp    = interp1d(fault_sel['UTMx'], fault_sel['UTMy'])
    
    # Define upsampled longitude coordinates and interpolate latitude coordinates
    lon_upsamp   = np.arange(fault_sel['UTMx'].min(), fault_sel['UTMx'].max(), upsamp_inc)
    lat_upsamp   = spl_upsamp(lon_upsamp)
    fault_upsamp = pd.DataFrame({'UTMx': lon_upsamp[::-1], 'UTMy': lat_upsamp[::-1]})
    fault_upsamp = add_ll_coords(fault_upsamp, EPSG)

    # Calulate strikes from upsampled fault trace
    nodes_upsamp = get_nodes(fault_upsamp)

    # Calculate along strike distance
    nodes_upsamp = calc_fault_dist(nodes_upsamp)    

    # Get approx. fault length 
    f_length = nodes_upsamp['dist'].max()

    # Get desired node locations
    d_target = np.arange(0, f_length, seg_inc)
    # d_ind = np.empty(len(d_target))

    # Get index of best-approximation of d
    keys = [nodes_upsamp.iloc[np.argmin(abs(nodes_upsamp['dist'].values - d))].name for i, d in enumerate(d_target)]

    # nodes_spl = nodes_upsamp.iloc[i]
    
    nodes_spl = nodes_upsamp.loc[keys, nodes_upsamp.columns]
    nodes_spl['new_index'] = [f'{key_str}{i}' for i in range(len(nodes_spl))]
    nodes_spl.set_index('new_index', drop=True, inplace=True)

    # Check
    # plt.plot(fault['Longitude'], fault['Latitude'], nodes_upsamp['Longitude'], nodes_upsamp['Latitude'], nodes_spl['Longitude'], nodes_spl['Latitude'], '-o')
    # plt.show()

    return nodes_spl


def get_nodes(fault, EPSG='32611'):
    """
    Given verticies of fault trace, extract nodes for profile inversions.
    """
    nodes = {}
    UTMx      = fault['UTMx'].values
    UTMy      = fault['UTMy'].values

    # Initialize node arrays
    UTMx_n   = np.empty(len(fault) - 1)
    UTMy_n   = np.empty(len(fault) - 1)
    strike_n = np.empty(len(fault) - 1)
    keys_n   = []

    for i in range(len(fault) - 1):
        UTMx_n[i]    = (UTMx[i + 1] + UTMx[i])/2
        UTMy_n[i]    = (UTMy[i + 1] + UTMy[i])/2
        strike_n[i] = 360 + np.arctan((UTMx[i + 1] - UTMx[i])/(UTMy[i + 1] - UTMy[i]))*180/np.pi
        keys_n.append(f'F{i}')

    # Write to dataframe
    nodes = pd.DataFrame({'UTMx': UTMx_n, 'UTMy': UTMy_n, 'Strike': strike_n}, index=keys_n)
    nodes = add_ll_coords(nodes, EPSG)

    # nodes = {}
    # lon      = fault['Longitude'].values
    # lat      = fault['Latitude'].values

    # # Initialize node arrays
    # lon_n    = np.empty(len(fault) - 1)
    # lat_n    = np.empty(len(fault) - 1)
    # strike_n = np.empty(len(fault) - 1)
    # keys_n   = []

    # for i in range(len(fault) - 1):
    #     lon_n[i]    = (lon[i + 1] + lon[i])/2
    #     lat_n[i]    = (lat[i + 1] + lat[i])/2
    #     strike_n[i] = 360 + np.arctan((lon[i + 1] - lon[i])/(lat[i + 1] - lat[i]))*180/np.pi
    #     keys_n.append(f'F{i}')

    # # Write to dataframe
    # nodes = pd.DataFrame({'Longitude': lon_n, 'Latitude': lat_n, 'Strike': strike_n}, index=keys_n)

    return nodes


def calc_fault_dist(nodes):

    dist = np.empty(len(nodes))

    for i in range(len(nodes)):
        if i == 0:
            dist[i] = 0
        else:
            # dx = (nodes.iloc[i]['Longitude'] - nodes.iloc[i - 1]['Longitude'])*111.19
            # dy = (nodes.iloc[i]['Latitude'] - nodes.iloc[i - 1]['Latitude'])*111.19
            # dist[i] = (dx**2 + dy**2)**0.5 + dist[i - 1]
            dx = (nodes['UTMx'].iloc[i] - nodes['UTMx'].iloc[i - 1])
            dy = (nodes['UTMy'].iloc[i] - nodes['UTMy'].iloc[i - 1])
            dist[i] = (dx**2 + dy**2)**0.5 + dist[i - 1]

    nodes['dist'] = dist/1000

    return nodes


def get_swath(x, y, data, node, strike, l=1500, w=500, return_indicies=False):
    """ 
    INPUT:
    x (n,) - list of x coordinates of data
    y (n,) - list of y coordinates of data
    zdata (n,) - list of deformation data
    node   - location of profile center (x, y)
    strike - local strike at profile center (orthogonal to profile)
    
    Optional: (keyword args)
    l - maximum profile half-length (fault-perpendicular direction)
    w - maximum profile width (along-strike)
    return_indicies - output indicies of data within swath

    OUTPUT:
    x_r, y_r, data_r - subset of above x, y, data which correspond to the selected profile
    """

    # Rotate data by fault strike
    alpha    = strike * np.pi/180
    node_r   = rotate(node[0], node[1], alpha)
    x_r, y_r = rotate(x, y, alpha)

    # Calculate distances
    dl = np.sqrt((x_r - node_r[0])**2)
    dw = np.sqrt((y_r - node_r[1])**2)

    # Get pixels inside of swath
    i_swath = (dl <= l) & (dw <= w)

    if return_indicies:
        return x_r[i_swath], y_r[i_swath], data[i_swath], node_r, i_swath
    else:
        return x_r[i_swath], y_r[i_swath], data[i_swath], node_r


def clip_grid(x, y, grid, region, extent=False):
    """
    Clip gridded dataset based off of specified region.
    """

    i_region  = np.where((y[:, 0] >= region[2]) & (y[:, 0] <= region[3]))[0]
    j_region  = np.where((x[0, :] >= region[0]) & (x[0, :] <= region[1]))[0]
    x_clip    = x[i_region][:, j_region]
    y_clip    = y[i_region][:, j_region]
    grid_clip = grid[i_region][:, j_region]

    if extent:
        return x_clip, y_clip, grid_clip, [x_clip.min(), x_clip.max(), y_clip.max(), y_clip.min()]
    else:
        return x_clip, y_clip, grid_clip


def check_dir_tree(dir_path):
    """
    Check if directory tree exists and create directories if not.
    """

    dirs = dir_path.split('/')

    for l in range(len(dirs)):
        if len(dirs[l]) == 0:
            continue
        else:
            sub_dir = '/'.join(dirs[:l + 1])
            if os.path.isdir(sub_dir) is not True:
                os.mkdir(sub_dir)
    return
