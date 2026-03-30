from pyproj import CRS, Proj


def proj_ll2utm(x, y, crs_epsg, inverse=False):
    """
    Project WGS84 lon/lat coordinates to UTM coordinates, or inverse.
    Accepts either an EPSG code (int/numeric string) or a proj4 string.
    """
    if isinstance(crs_epsg, str) and crs_epsg.strip().startswith('+'):
        myProj = Proj(crs_epsg)
    else:
        myProj = Proj(CRS.from_epsg(crs_epsg))

    if inverse:
        x_proj, y_proj = myProj(x, y, inverse=True)
    else:
        x_proj, y_proj = myProj(x, y)

    return x_proj, y_proj


def get_local_lonlat_coords(x, y, origin, unit='km'):
    """
    Convert local Cartesian x/y coordinates back to WGS84 lon/lat.
    Exact inverse of get_local_xy_coords in utilities.py.

    INPUT:
    x, y   - local Cartesian coordinates (must match the unit used in forward projection)
    origin - [lon, lat] reference point — must be the same ref_point used in get_local_xy_coords
    unit   - 'km' or 'm', must match what was used in get_local_xy_coords (default: 'km')

    OUTPUT:
    lon, lat - WGS84 geographic coordinates (degrees)

    EXAMPLE:
    >>> from coord_inverse import get_local_lonlat_coords
    >>> ref_point = [73.1603, 38.1025]
    >>> lon, lat = get_local_lonlat_coords(x_samp, y_samp, ref_point, unit='km')
    """

    if unit == 'km':
        scale = 1e3   # convert km back to metres before unprojecting
    else:
        scale = 1

    # Reconstruct the identical projection used in get_local_xy_coords —
    # centred on origin[0] so it is valid across any UTM zone boundary
    proj_str = (f'+proj=tmerc +lat_0=0 +lon_0={origin[0]} +k=0.9996 '
                f'+x_0=500000 +y_0=0 +datum=WGS84 +units=m +no_defs')

    # Recover the projected coordinates of the origin (same offset applied in forward step)
    origin_utm = proj_ll2utm(origin[0], origin[1], proj_str)

    # Undo referencing and unit scaling to get back to raw projected metres
    x_utm = x * scale + origin_utm[0]
    y_utm = y * scale + origin_utm[1]

    # Invert projection to lon/lat
    lon, lat = proj_ll2utm(x_utm, y_utm, proj_str, inverse=True)

    return lon, lat
