import pandas as pd
from pyproj import CRS, Proj


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


# @numba.jit(nopython=True)
def rotate(x, y, theta): 
    """
    Perform 2D rotation about origin.
    """

    x_r = np.cos(theta) * x - np.sin(theta) * y
    y_r = np.sin(theta) * x + np.cos(theta) * y

    return x_r, y_r


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

