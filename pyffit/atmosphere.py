import numpy as np

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


def make_synthetic_aps(X0, Z0, klim=[-3, 5], kinc=0.1, amp_ref=5.0, nexp_ref=1.5, lc_ref=5):
    """
    Generate "correlated noise" Ur(X,  Z) with a power spectrum Pk(k) starting with white noise,    
    then multipling its fft by Pk(k), going back in spce domain (ifft2)
    
    INPUT:
    X0      - along strike coordinates (km)
    Z0      - along dip coordinates (km)  
    amp_ref - reference APS value
    nexp_ref, lc_ref - power spectrum parameters

    OUTPUT:
    Ursub
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
    K        = np.arange(-Kny, Kny - (Kny*2)/N_max, (Kny*2)/N_max) # Get wavemenumber range
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
    Ursub = Ursub/np.max(np.abs(Ursub)) * amp

    return Ursub