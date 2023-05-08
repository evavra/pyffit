import numpy as np
import matplotlib.pyplot as plt


def fit_ramp(x, y, z, deg=1, region=[], grid=True):
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

    d = z.flatten()

    # Handle region
    if len(region) == 4:
        region_mask = (x >= region[0]) & (x <= region[1]) & (y >= region[2]) & (y <= region[3])

    # Normalize 
    x  = x - x.min()
    x /= x.max()
    y  = y - y.min()
    y /= y.max()

    # Set up linear equations
    if deg == 1:
        G_grid = np.vstack((x, y, np.ones_like(x))).T
        # G_grid = np.polynomial.polynomial.polyvander2d(x, y, [deg, deg]) # Full vandemond matrix

    elif deg == 2:
        # G_grid = np.vstack((x**2, x*y, y**2, x, y, x**0)).T
        # G_grid = np.vstack((x**2, y**2, x*y, x, y, np.ones_like(x))).T
        # G_grid = np.vstack((x**2, y**2, x, y, np.ones_like(x))).T
        G_grid = np.array([x**2, np.ones_like(x), x, y, (x**2)*y, (x**2)*(y**2), y**2, x*(y**2), x*y]).T

        # G_grid = np.polynomial.polynomial.polyvander2d(x, y, [deg, deg]) # Full vandemond matrix

    # Omit nans for least-squares solve
    nans = np.isnan(d)
    G    = G_grid[(~nans) & region_mask, :]
    d    = d[(~nans) & region_mask]

    print(G_grid.shape, d.shape)
    print(G.shape, d.shape)
    print('Percent NaN', 100*sum(nans)/len(nans))

    # Solve for ramp coefficients
    m, residuals, rank, s = np.linalg.lstsq(G, d, rcond=None)

    # Return ramp at all grid points
    if grid & (len(z.shape) == 2):
        d_grid = G_grid @ m
        return d_grid.reshape(z.shape)

    # Return grid at data locations only
    else:
        return G @ m



