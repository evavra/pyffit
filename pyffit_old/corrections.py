import numpy as np
import matplotlib.pyplot as plt


def fit_ramp(x, y, z, x_ramp=[], y_ramp=[], deg=1, region=[-360, 360, -90, 90], grid=True):
    """
    Fit 2D polynomnial ramp to data.

    INPUT:
    x - x-coordinates of data [(n,), (m, n), or (m*n)]
    y - y-coordinates of data [(m,), (m, n), or (m*n)]
    z - data values [(m, n), or (m*n)]
    
    Optional:
    grid - True to return ramp as (m, n) if z is (m, n)
    deg  - 2D polynomial degree.

    OUTPUT:
    ramp - modeled ramp (m, n) or (m*n_)
    """

    # Handle coordinates    
    if (x.shape == z.shape) & (y.shape == z.shape) & (len(z.shape) == 1): # Pre-flattened
        pass
    elif (x.shape == y.shape) & (len(x.shape) == 2): # Fully gridded 
        x = x.flatten()
        y = y.flatten()
    elif (x.shape != y.shape) & len(x.shape) == 1: # Grid axes only
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
    else:
        print('Warning! Coordinates do not match')
        print(f'x: {x.shape}')
        print(f'y: {y.shape}')
        print(f'z: {z.shape}')

    d = z.flatten()

    # Handle region
    if len(region) == 4:
        region_mask = (x >= region[0]) & (x <= region[1]) & (y >= region[2]) & (y <= region[3])

    # Normalize to void numerical errors
    x  = x - x.min()
    x /= x.max()
    y  = y - y.min()
    y /= y.max()

    if (len(x_ramp) == 0) & (len(y_ramp) == 0):
        x_ramp = x
        y_ramp = y

    # Set up linear equations
    if deg == 1:
        get_G = lambda x, y: np.vstack((x, y, np.ones_like(x))).T

    elif deg == 2:
        get_G = lambda x, y: np.array([x**2, np.ones_like(x), x, y, (x**2)*y, (x**2)*(y**2), y**2, x*(y**2), x*y]).T

    # Get design matrices
    G_fit  = get_G(x, y)
    G_ramp = get_G(x_ramp, y_ramp)

    # Omit nans for least-squares solve
    nans = np.isnan(d)
    G    = G_fit[(~nans) & region_mask, :]
    d    = d[(~nans) & region_mask]
    # print(f'Percent NaN: {100*sum(nans)/len(nans):.1f} %')

    # Solve for ramp coefficients
    m, residuals, rank, s = np.linalg.lstsq(G, d, rcond=None)

    # Return ramp at specified points
    if grid & (len(z.shape) == 2):
        d_grid = G_ramp @ m
        return d_grid.reshape(z.shape)

    # Return grid at data locations only
    else:
        return G @ m



