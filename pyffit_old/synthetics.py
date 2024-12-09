import numpy as np
import random


def get_aps_params(klim=[-3, 5], kinc=0.1, amp_ref=5.0, nexp_ref=1.5, lc_ref=5):
    """
    Get instance of APS spectral parameters based on input reference values. 
    """

    amp  = amp_ref * np.random.uniform() # amplitude
    k    = np.concatenate(([0], 10**np.arange(klim[0], klim[1] + kinc, kinc)))
    nexp = nexp_ref + np.random.uniform()
    lc   = lc_ref * np.random.uniform()
    Pk   = 1/(k + 2*np.pi/lc)**nexp

    return amp, k, Pk


def make_synthetic_aps(X0, Z0, klim=[-3, 5], kinc=0.1, manual_amp=0, amp_ref=5.0, nexp_ref=1.5, lc_ref=5):
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
    amp, k, Pk = get_aps_params(klim=klim, kinc=kinc, amp_ref=amp_ref, nexp_ref=nexp_ref, lc_ref=lc_ref)

    # Get grid information
    NX0   = len(X0)        # get desired x grid dimesion
    NZ0   = len(Z0)        # get desired z grid dimesion
    dx    = np.diff(X[:2]) # grid size(km)
    LX    = np.max(X) - np.min(X)
    LZ    = np.max(Z) - np.min(Z)
    L     = np.sqrt(LX*LZ)*1000 # len in m
    N_max = int(2*np.round(len(X)/2)) # even numbers

    # # start with white noise with a unif. pdf:
    U = np.random.uniform(size=(N_max, N_max))

    # Wavenumber matrix (with 0 frequency at the center, even number of points)
    dxm      = dx*1000                                             # grid size,   m
    Kny      = (1/dxm/2)*(2*np.pi)                                 # nyquist frequency * 2pi  =  max wavenumber
    # K        = np.arange(-Kny, Kny - (Kny*2)/N_max, (Kny*2)/N_max) # Get wavemenumber range
    K        = np.arange(-Kny, Kny, (Kny*2)/N_max) # Get wavemenumber range
    KXX, KZZ = np.meshgrid(K, K)                                   # KXX and KZZ are nz*nz matrix
    K        = np.sqrt(KXX**2 + KZZ**2)                            # K in m^-1 

    # adjust the power-spectrum of the image Pk(k)
    aps_K  = np.fft.fftshift(np.fft.fft2(U))                         # direct Fourier transform
    aks_Kv = np.interp(K.flatten()*L, k,  Pk)                                 # fix |U(k)| = Pk(Kv)
    aps_K  = aks_Kv.reshape((N_max, N_max)) * np.exp(1j * np.angle(aps_K)) 
    Ur  = np.real(np.fft.ifft2(np.fft.ifftshift(aps_K)))             # inverse Fourier transform
    # The imag part may be non-zero due to numerical error (should be ~1e-16)

    # Crop to desired dimentions if not square
    # if NX0 > NZ0:
    #     Ursub = Ur[:, :NX0] # original size
    #     Ursub = Ur[:NZ0, :]
    # else:
    #     Ursub = Ur[:NZ0, :] # original size
    #     Ursub = Ur[:, :NX0]

    Ursub = Ur[:NZ0, :NX0]

    # Re-scale APS amplitude 
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
