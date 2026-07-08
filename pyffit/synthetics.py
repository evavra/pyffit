import os
import h5py
import random
import numpy as np
from numba import jit


def get_aps_params(klim=[-3, 5], kinc=0.1, amp_ref=5.0, nexp_ref=2, Lc_ref=2, randomize=False):
    """
    Get instance of APS spectral parameters based on input reference values. 

    INPUT
    klim      - wavenumber range (log-scale) 
    kinc      - wavenumber increment (log-scale)
    amp_ref   - reference amplitude
    nexp_ref  - reference power law exponent 
    Lc_ref    - reference decay wavelength  
    randomize - scale amp, nexp, and Lc by random value (0-1); default is True 

    OUTPUT:
    amp - APS amplitude
    k   - wavenumber grid
    Pk  - power spectrum

    Old defaults: klim=[-3, 5], kinc=0.1, amp_ref=5.0, nexp_ref=1.5, Lc_ref=5
    """
    if randomize:
        perturb = np.random.uniform(size=3)
    else:
        perturb = np.array([1, 0, 1])

    k = np.concatenate(([0], 10**np.arange(klim[0], klim[1] + kinc, kinc))) # wavenumber grid
    # amp  = amp_ref  * np.random.uniform()                                    # amplitude
    # nexp = nexp_ref + np.random.uniform()                                    # power law exponent
    # Lc   = Lc_ref   * np.random.uniform()                                    # decay wavelength
    amp  = amp_ref  * perturb[0]                                               # amplitude
    nexp = nexp_ref + perturb[1]                                               # power law exponent
    Lc   = Lc_ref   * perturb[2]                                               # decay wavelength
    Pk   = 1/(k + 2*np.pi/Lc)**nexp                                          # power spectrum
    # Pk = 1/(1 + (2 * np.pi * k * Lc)**nexp)                                  # power spectrum for exponential decay
    # Pk = 1/(1 + (2 * np.pi * k * Lc)**nexp)                                    # power spectrum for exponential decay

    print(f'k = {k.min():.1e} - {k.max():.1e} | L_c = {Lc:.1f} | n_exp = {nexp:.1e}')
    return amp, k, Pk


def make_synthetic_aps(x, y, aps_amp=5, L_c=1, randomize=False):
    
    if randomize:
        aps_amp *= np.random.uniform()
        L_c *= L_c

    # Get grid information
    Nx = x.size
    Ny = y.size
    dx = np.diff(x)[0]
    dy = np.diff(y)[0]

    # Get wavenumber grids in x and y directions
    kx     = np.fft.fftfreq(Nx, d=dx)
    ky     = np.fft.fftfreq(Ny, d=dy)
    kx, ky = np.meshgrid(kx, ky, indexing='xy')

    # Isotropic wavenumber amplitude
    k = np.sqrt(kx**2 + ky**2)

    # Power spectrum of exponential kernel
    S = 1 / (1.0 + (2 * np.pi * L_c * k)**2)

    # Sample complex white noise in frequency domain
    white_noise = np.random.normal(size=(Ny, Nx)) + 1j * np.random.normal(size=(Ny, Nx))

    # Apply spectral filter
    filtered_noise_fft = np.sqrt(S) * white_noise

    # Inverse FFT to return to spatial domain
    aps = np.fft.ifft2(filtered_noise_fft).real

    # if aps_amp > 0:
    aps_mean = np.mean(aps)                                          # Get mean
    aps -= aps_mean                                                  # Remove mean
    aps  = (aps - np.min(aps))/(np.max(aps) - np.min(aps)) # Mormalize 0-1
    aps  = 2 * aps_amp * aps - aps_amp                     # Rescale to specified amplitude
    aps += aps_mean                                                  # Restore mean

    return aps


def make_synthetic_aps_old(X0, Z0, klim=[-5, 5], kinc=0.01, manual_amp=0, amp_ref=5.0, nexp_ref=1, Lc_ref=1, randomize=False):
    """
    Generate "correlated noise" Ur(X,  Z) with a power spectrum Pk(k) starting with white noise,    
    then multipling its fft by Pk(k), going back in spce domain (ifft2)
    
    INPUT:
    X0         - along strike coordinates (km)
    Z0         - along dip coordinates (km)  
    manual_amp - manually specify max/min value of noise
    amp_ref    - reference APS value
    nexp_ref, lc_ref - power spectrum parameters 

    OUTPUT:
    Ursub      - simulated atmospheric noise 
    """

    # Use larger grid dimension to start (will crop later)
    if len(X0) > len(Z0):
        X = X0
        Z = X0
    else:
        X = Z0
        Z = Z0

    # Get spectral parameters
    # amp, k, Pk = get_aps_params(klim=klim, kinc=kinc, amp_ref=amp_ref, nexp_ref=nexp_ref, Lc_ref=Lc_ref)
    if randomize:
        perturb = np.random.uniform(size=3)
    else:
        perturb = np.array([1, 0, 1])

    k = np.concatenate(([0], 10**np.arange(klim[0], klim[1] + kinc, kinc))) # wavenumber grid
    amp  = amp_ref  * perturb[0]                                               # amplitude
    nexp = nexp_ref + perturb[1]                                               # power law exponent
    Lc   = Lc_ref   * perturb[2]                                               # decay wavelength
    Pk   = (1/(k + (2*np.pi/Lc)))**nexp                                          # power spectrum
    # Pk = 1/(1 + (2 * np.pi * k * Lc)**nexp)                                  # power spectrum for exponential decay
    # Pk = 1/(1 + (2 * np.pi * k * Lc)**nexp)                                    # power spectrum for exponential decay

    print(f'k = {k.min():.1e} - {k.max():.1e} | L_c = {Lc:.1f} | n_exp = {nexp:.1e}')

    # Get grid information
    NX0   = len(X0)                   # get desired x grid dimesion
    NZ0   = len(Z0)                   # get desired z grid dimesion
    dx    = np.diff(X[:2])[0]            # grid size (km)
    LX    = np.max(X) - np.min(X)     # x range
    LZ    = np.max(Z) - np.min(Z)     # y range
    L     = np.sqrt(LX * LZ) * 1000   # len in m
    N_max = int(2*np.round(len(X)/2)) # even numbers

    # # start with white noise with a unif. pdf:
    U = np.random.uniform(size=(N_max, N_max))

    # Wavenumber matrix (with 0 frequency at the center, even number of points)
    dxm      = dx*1000                                               # grid size (m)
    Kny      = (1/dxm/2)*(2*np.pi)                                   # nyquist frequency * 2pi  =  max wavenumber
    # K        = np.arange(-Kny, Kny - (Kny*2)/N_max, (Kny*2)/N_max) # Get wavenumber range
    K        = np.arange(-Kny, Kny, (Kny*2)/N_max)                   # Get wavenumber range
    KXX, KZZ = np.meshgrid(K, K)                                     # KXX and KZZ are nz*nz matrix
    K        = np.sqrt(KXX**2 + KZZ**2)                              # K in m^-1 
    # print('K', K.shape)
    # print('K2', K2.shape)

    # adjust the power-spectrum of the image Pk(k)
    aps_K  = np.fft.fftshift(np.fft.fft2(U))                                # direct Fourier transform
    aks_Kv = np.interp(K.flatten() * L, k, Pk)                                # fix |U(k)| = Pk(Kv)
    aps_K  = aks_Kv.reshape((N_max, N_max)) * np.exp(1j * np.angle(aps_K))  #
    Ur     = np.real(np.fft.ifft2(np.fft.ifftshift(aps_K)))                 # inverse Fourier transform
    # The imag part may be non-zero due to numerical error (should be ~1e-16)

    # Crop to desired dimentions if not square
    # if NX0 > NZ0:
    #     Ursub = Ur[:, :NX0] # original size
    #     Ursub = Ur[:NZ0, :]
    # else:
    #     Ursub = Ur[:NZ0, :] # original size
    #     Ursub = Ur[:, :NX0]
    Ursub = Ur[:NZ0, :NX0]

    # Re-scale APS amplitude to have specified average
    # Ursub = Ursub/np.nanmax(np.abs(Ursub)) * amp_ref

    # Rescale APS amplutude to have xpeficied max
    if manual_amp > 0:
        U_mean = np.mean(Ursub)                                          # Get mean
        Ursub -= U_mean                                                  # Remove mean
        Ursub  = (Ursub - np.min(Ursub))/(np.max(Ursub) - np.min(Ursub)) # Mormalize 0-1
        Ursub  = 2 * manual_amp * Ursub - manual_amp                     # Rescale to specified amplitude
        Ursub -= U_mean                                                  # Restore mean
        # Ursub = Ursub/np.max(np.abs(Ursub)) * manual_amp

    # print(U_mean, np.nanmin(Ursub), np.nanmax(Ursub))
    return Ursub


def get_decorr_mask(data, nan_frac):
    """
    Introduce NaN values to data using uniform random distribution.

    INPUT:
    data (m, n) - data array
    nan_frac    - fraction of Nan values to introduce

    OUTPUT:
    data (m, n) - updated data array with Nan values
    """

    mask = np.ones_like(data)

    if nan_frac > 0:
        # Select NaN indices
        nans = random.sample(range(0, data.size), int(data.size*nan_frac))

        # Apply NaN values
        for i in nans:
            j = i//data.shape[1]
            k = i % data.shape[1]
            mask[j, k] = np.nan

    return mask


def get_fault_mask(X, Y, trace, rupture_nan_frac=0.9, width=1):
    """
    Make synthetic mask for fault trace to simulate coseismic rupture decorrelation.
    Assumes quasi-linear fault trace

    INPUT:
    X (m, n)     - x-coordinate array
    Y (m, n)     - y-coordinate array
    trace (p, 2) - x/y coordinates of fault trace
    l_mask       - distance (km) 
    """ 

    # Calculate distances
    dists = np.empty_like(X)

    for i in range(len(X)):
        for j in range(len(X[0])):
            dists[i, j] = get_line_segment_distance(trace, (X[i, j], Y[i, j]))


    # Apply Gaussian kernel to get Nan probabilities
    gauss = (1/(width * np.sqrt(2*np.pi))) * np.exp(-0.5 * (dists/width)**2)
    gauss *= rupture_nan_frac/gauss.max()

    # Evaluate probabilities
    mask = np.ones_like(X)
    probs = np.random.uniform(low=0, high=1, size=X.shape)
    mask[probs < gauss] = np.nan

    return mask


def get_line_segment_distance(segment, point):
    """
    Calculate min. distance ebtween point and line segment.

    INPUT:
    segment (m, 2) - 
    point   (2,)   - 

    OUTPUT:
    dist           - 
    """

    x_min = segment[0, 0]
    x_max = segment[-1, 0]
    y_min = segment[0, 1]
    y_max = segment[-1, 1]

    run  = x_max - x_min
    rise = y_max - y_min

    u =  ((point[0] - x_min) * run + (point[1] - y_min) * rise) / (run**2 + rise**2)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x_min + u * run
    y = y_min + u * rise

    dx = x - point[0]
    dy = y - point[1]

    dist = np.sqrt(dx**2 + dy**2)

    return dist


def get_synthetic_slip(mesh, triangles, plane='east', max_slip=1, strike_scale=0.5, dip_scale=0.5, x_shift=0):
    
    """
    Get Gaussian slip distribution
    """
    
    if plane == 'east':
        i = 0
    elif plane == 'north':
        i = 1

    X = 2*(mesh[:, i] - mesh[:, i].min())/(mesh[:, i].max() - mesh[:, i].min()) - 1

    # Z = (mesh[:, 2] - mesh[:, 2].min() )/(mesh[:, 2].max() - mesh[:, 2].min())
    Z = mesh[:, 2]

    sigma       = np.array([strike_scale, dip_scale])              # widths for Gaussian slip distribution
    slip_pts    = max_slip * np.array([np.exp(-0.5*np.array([[x0 + x_shift, z0]]) @ np.diag(1/sigma**2) @ np.array([[x0 + x_shift, z0]]).T) for x0, z0 in zip(X.flatten(), Z.flatten())]) # make gridded slip distribution, corresponding to triangle vertices
    slip_tri    = np.mean(slip_pts[triangles], axis=1).reshape(-1) # average slip at verticies to get triangle slip
    slip        = np.zeros((len(triangles), 3))                    # (strike-slip, dip-slip, opening)
    slip[:, 0]  = slip_tri

    return slip


@jit(nopython=True)
def dist(x0, x1, y0, y1):
    return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
