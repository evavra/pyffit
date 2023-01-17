import numpy as np
import matplotlib.pyplot as plt


def fit_ramp(x, y, z, grid=True):
    """
    Fit linear ramp to data points.
        z = ax + by + c
    
    INPUT:
    x - x-coordinates of data [(n,), (m, n), or (m*n)]
    y - y-coordinates of data [(m,), (m, n), or (m*n)]
    z - data values [(m, n), or (m*n)]
    
    Optional:
    grid - True to return ramp as (m, n) if z is (m, n)

    OUTPUT:
    ramp - modeled ramp (m, n) or (m*n_)
    """

    # Handle coordinates
    if (x.shape == z.shape) & (y.shape == z.shape): # Pre-flattened
        pass
    elif (x.shape == y.shape) & len(x.shape) == 2: # Fully gridded 
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

    # Set up linear equations
    G = np.vstack((np.vstack((x, y)), np.ones_like(x))).T
    d = z.flatten()

    # Omit nans for least-squares solve
    nans = np.isnan(d)
    G = G[~nans] 
    d = d[~nans]

    # Solve for ramp coefficients
    m, residuals, rank, s = np.linalg.lstsq(G, d, rcond=None)

    # Return ramp at all grid points
    if grid & (len(z.shape) == 2):
        G_grid = np.vstack((np.vstack((x, y)), np.ones_like(x))).T
        d_grid = G_grid @ m
        return d_grid.reshape(z.shape)

    # Return grid at data locations only
    else:
        return G @ m