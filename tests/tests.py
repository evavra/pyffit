import os
import time
import h5py
import scipy
import pyffit
import numpy as np
import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
from guided_filter.filter import GuidedFilter
import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import cutde.halfspace as HS

def main():
    test_gaussian_filter()
    return


def test_gaussian_filter():
    # -------------------- Parameters --------------------
    # Fault parameters
    origin        = (0, 0, 0)
    strike        = 0
    l             = 15
    w             = 3
    slip          = 40
    poisson_ratio = 0.25      # Poisson ratio
    mu            = 6 * 10**9 # Shear modulus (Pa)
    
    # Data parameters
    n_x = 500
    n_y = 500
    xlim = [-20, 20]
    ylim = [-20, 20]

    # Noise parameters
    n_aps       = 5
    aps_amp     = 30
    noise_amp   = 20
    noise_width = 0.5
    radius      = 1
    eps         = 0.4**2
    nan_frac    = 0.1

    # Filter parameters
    sigma    = 8.0               # standard deviation for Gaussian kernel
    truncate = 4.0               # truncate filter at this many sigmas
    n_iter   = 5
    n_swath  = 10

    # -------------------- Analysis --------------------
    # Get coordinates
    x_rng = np.linspace(xlim[0], xlim[1], n_x)
    y_rng = np.linspace(ylim[0], ylim[1], n_y)
    x, y  = np.meshgrid(x_rng, y_rng)
    z     = np.zeros_like(x)
    dx    = x_rng[1] - x_rng[0]
    dy    = y_rng[1] - y_rng[0]
    pts   = np.array([x, y, z]).reshape((3, -1)).T.copy()
    extent = [xlim[0], xlim[1], ylim[1], ylim[0]]

    # Get NaNs
    n_nan      = int(x.size * nan_frac)
    i_nans     = np.random.randint(0, high=n_y, size=n_nan)
    j_nans     = np.random.randint(0, high=n_x, size=n_nan)

    # Get fault
    mesh = np.array([[0, -l, 0], [0, l, 0], [0, l, -w], [0, -l, -w]])
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    # Get fault masks
    mask_E = np.ones_like(x)
    mask_W = np.ones_like(x)
    mask_E[x < 0] = np.nan
    mask_W[x > 0] = np.nan

    # Get displacements
    slip_model = np.array([[-slip, 0, 0], 
                          [-slip, 0, 0]])
    disp_mat = HS.disp_matrix(obs_pts=pts, tris=mesh[tris], nu=poisson_ratio)
    disp     = disp_mat.reshape((-1, 6)).dot(slip_model.flatten())
    model    = disp.reshape((*x.shape, 3))[:, :, 1]

    for k in range(n_iter):
        # Get noise grids
        speckle = noise_amp * np.random.normal(loc=0, scale=noise_width, size=model.shape)
        aps = np.zeros_like(speckle)
        for i in range(n_aps):
            aps += pyffit.synthetics.make_synthetic_aps(x_rng, y_rng, manual_amp=aps_amp)
        aps /= n_aps
        aps_nan = pyffit.synthetics.make_synthetic_aps(x_rng, y_rng, manual_amp=aps_amp)
        aps_nan = np.abs(aps_nan)
        nans    = np.ones_like(aps)
        nans[aps_nan >= 2.5*aps_nan.std()] = np.nan
        nans[i_nans, j_nans] = np.nan
        noise = (aps + speckle) * nans

        # Apply noise
        data = (model + aps + speckle) * nans

        # Apply filter
        w_x = (2*int(truncate*sigma + 0.5) + 1) * dx
        w_y = (2*int(truncate*sigma + 0.5) + 1) * dy

        data_E, w = nan_gaussian_filter(data * mask_E, sigma=sigma, truncate=truncate, return_width=True, plot=False)
        data_W, w = nan_gaussian_filter(data * mask_W, sigma=sigma, truncate=truncate, return_width=True, plot=False)

        # Combine halves of observation
        data_filt = np.empty_like(data)
        data_filt[~np.isnan(mask_E)] = data_E[~np.isnan(mask_E)]
        data_filt[~np.isnan(mask_W)] = data_W[~np.isnan(mask_W)]
        data_filt -= np.nanmean(data_filt)

        # Apply filter with no masking
        data_unmasked, w = nan_gaussian_filter(data, sigma=sigma, truncate=truncate, return_width=True, plot=False)

        # Get RMS over a filter width of the fault
        masked_rms   = np.sqrt(np.nanmean((data -     data_filt)[:, n_x//2 - n_swath//2 - 1:n_x//2 + n_swath//2]**2))
        unmasked_rms = np.sqrt(np.nanmean((data - data_unmasked)[:, n_x//2 - n_swath//2 - 1:n_x//2 + n_swath//2]**2))
        
        # Get offsets
        offsets          = model[:, n_x//2 - 1] - model[:, n_x//2]
        offsets_masked   = data_filt[:, n_x//2 - 1] - data_filt[:, n_x//2]
        offsets_unmasked = data_unmasked[:, n_x//2 - 1] - data_unmasked[:, n_x//2]




        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(y_rng, offsets, c='k', label='Model')
        ax.plot(y_rng, offsets_unmasked, c='C0', label='Filtered')
        ax.plot(y_rng, offsets_masked, c='C3', label='Filtered w/mask')
        ax.set_xlabel('North (km)')
        ax.set_ylabel('Displacement (mm)')
        plt.savefig(f'filter_testing/{k}_offsets.png', dpi=300)
                
        
        # -------------------- Plotting --------------------
        vlim = slip + 1.5

        # Plot data
        width_ratios  = (1, 1, 1, 0.05)
        height_ratios = (1, 1)

        # Plot
        fig, axes, cax = pyffit.figures.get_gridspec(width_ratios, height_ratios, figsize=(14, 8), constrained_layout=False, gridspec_kwargs=dict(wspace=0.05, hspace=0.2))

        # Grids
        im = axes[0].imshow(data,          extent=extent, interpolation='none', cmap=cmc.vik, vmin=-vlim, vmax=vlim)
        im = axes[1].imshow(data_unmasked, extent=extent, interpolation='none', cmap=cmc.vik, vmin=-vlim, vmax=vlim)
        im = axes[2].imshow(data_filt,     extent=extent, interpolation='none', cmap=cmc.vik, vmin=-vlim, vmax=vlim)
        im = axes[3].imshow(model,         extent=extent, interpolation='none', cmap=cmc.vik, vmin=-vlim, vmax=vlim)
        im = axes[4].imshow(noise,         extent=extent, interpolation='none', cmap=cmc.vik, vmin=-vlim, vmax=vlim)
        # im = axes[5].imshow(blob,  interpolation='none', cmap=cmc.vik, vmin=-vlim, vmax=vlim)

        # Profiles
        for i in range(n_y//2 - n_swath//2, n_y//2 + n_swath//2 + 1):
            axes[-1].scatter(x_rng, data[i, :], c='gainsboro', marker='.', s=1)

        axes[-1].plot(x_rng, model[n_y//2, :], c='k')
        axes[-1].plot(x_rng, data_filt[n_y//2, :],     c='C3', label=f'RMS = {masked_rms:.1f}')
        axes[-1].plot(x_rng, data_unmasked[n_y//2, :], c='C0', label=f'RMS = {unmasked_rms:.1f}')
        axes[-1].yaxis.tick_right()
        axes[-1].yaxis.set_label_position("right")
        axes[-1].set_ylabel('Displacement (mm)')
        axes[-1].legend()

        # Labels
        axes[0].set_title(f'Data')
        axes[1].set_title(f'Filtered data (w = {max((w_x, w_y)):.1f} km)')
        axes[2].set_title(f'Filtered data w/mask')
        axes[3].set_title(f'Model')
        axes[4].set_title(f'Noise')
        axes[5].set_title(f'Swath at y = 0')

        for ax in axes:
            ax.set_xlim(xlim)

        for i in [1, 2, 4]:
            axes[i].set_yticks([])
            axes[i].set_yticklabels([])

        for i in [0, 1, 2]:
            axes[i].set_xticks([])
            axes[i].set_xticklabels([])
            # ax.set_anchor('C')  # Center anchor

        for ax in axes[:-1]:
            ax.set_aspect('equal', adjustable='box')  # Let data aspect be auto, but box stays fixed
            ax.set_ylim(ylim)

        axes[-1].set_aspect('auto', adjustable='box')  # Let data aspect be auto, but box stays fixed

        # Match ax2's height to that of ax1
        bbox1 = axes[4].get_position()
        bbox3 = axes[2].get_position()
        bbox2 = axes[-1].get_position()
        bbox2.y0 = bbox1.y0
        bbox2.y1 = bbox1.y1
        bbox2.x0 = bbox3.x0
        bbox2.x1 = bbox3.x1
        axes[-1].set_position(bbox2)

        plt.colorbar(im, cax=cax, label='Displacement (mm)', shrink=0.1)
        plt.savefig(f'filter_testing/{k}_panels.png', dpi=300)

    return
 
 
def anisodiff(img, niter=1, kappa=50, gamma=0.1, step=(1.,1.), option=1, ploton=False):
    """
    Anisotropic diffusion.
 
    Usage:
    imgout = anisodiff(im, niter, kappa, gamma, option)
 
    Arguments:
            img    - input image
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the image will be plotted on every iteration
 
    Returns:
            imgout   - diffused image.
 
    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.
 
    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)
 
    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x and y axes
 
    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.
 
    Reference: 
    P. Perona and J. Malik. 
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 
    12(7):629-639, July 1990.
 
    Original MATLAB code by Peter Kovesi  
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>
 
    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>
 
    June 2000  original version.       
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python

    # Original work: Copyright (c) 1995-2012 Peter Kovesi pk@peterkovesi.com
    # Modified work: Copyright (c) 2012 Alistair Muldal
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.
    #
    # The software is provided "as is", without warranty of any kind, express or
    # implied, including but not limited to the warranties of merchantability,
    # fitness for a particular purpose and noninfringement. In no event shall the
    # authors or copyright holders be liable for any claim, damages or other
    # liability, whether in an action of contract, tort or otherwise, arising from,
    # out of or in connection with the software or the use or other dealings in the
    # software.
    """
 
    # ...you could always diffuse each color channel independently if you
    # really want
    if img.ndim == 3:
        print("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)
 
    # initialize output array
    img    = img.astype('float32')
    imgout = img.copy()
 
    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()
 
    # create the plot figure, if requested
    if ploton: 
        fig = plt.figure(figsize=(20,5.5), num="Anisotropic diffusion")
        ax1, ax2 = fig.add_subplot(1,2,1), fig.add_subplot(1,2,2)

        ax1.imshow(img,interpolation='nearest')
        ih = ax2.imshow(imgout, interpolation='nearest',animated=True)
        ax1.set_title("Original image")
        ax2.set_title("Iteration 0")
        fig.canvas.draw()
        # plt.show()
 
    for ii in range(niter):
        # calculate the diffs
        deltaS[:-1, : ] = np.diff(imgout, axis=0)
        deltaE[: , :-1] = np.diff(imgout, axis=1)
 
        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaS/kappa)**2.)/step[0]
            gE = np.exp(-(deltaE/kappa)**2.)/step[1]

        elif option == 2:
            gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[1]
 
        # update matrices
        E = gE*deltaE
        S = gS*deltaS
 
        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:]     = S
        EW[:]     = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]
 
        # update the image
        imgout += gamma*(NS+EW)
 
        if ploton:
            iterstring = "Iteration %i" %(ii+1)
            ih.set_data(imgout)
            ax2.set_title(iterstring)


            # sleep(0.01)

 
    return imgout
 

def anisodiff3(stack,niter=1,kappa=50,gamma=0.1,step=(1.,1.,1.),option=1,ploton=False):
    """
    3D Anisotropic diffusion.
 
    Usage:
    stackout = anisodiff(stack, niter, kappa, gamma, option)
 
    Arguments:
            stack  - input stack
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (z,y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the middle z-plane will be plotted on every 
                 iteration
 
    Returns:
            stackout   - diffused stack.
 
    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.
 
    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)
 
    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x,y and/or z axes
 
    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.
 
    Reference: 
    P. Perona and J. Malik. 
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 
    12(7):629-639, July 1990.
 
    Original MATLAB code by Peter Kovesi  
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>
 
    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>
 
    June 2000  original version.       
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """
 
    # ...you could always diffuse each color channel independently if you
    # really want
    if stack.ndim == 4:
        warnings.warn("Only grayscale stacks allowed, converting to 3D matrix")
        stack = stack.mean(3)
 
    # initialize output array
    stack = stack.astype('float32')
    stackout = stack.copy()
 
    # initialize some internal variables
    deltaS = np.zeros_like(stackout)
    deltaE = deltaS.copy()
    deltaD = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    UD = deltaS.copy()
    gS = np.ones_like(stackout)
    gE = gS.copy()
    gD = gS.copy()
 
    # create the plot figure, if requested
    if ploton:
        import pylab as pl
        from time import sleep
 
        showplane = stack.shape[0]//2
 
        fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
        ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)
 
        ax1.imshow(stack[showplane,...].squeeze(),interpolation='nearest')
        ih = ax2.imshow(stackout[showplane,...].squeeze(),interpolation='nearest',animated=True)
        ax1.set_title("Original stack (Z = %i)" %showplane)
        ax2.set_title("Iteration 0")
 
        fig.canvas.draw()
 
    for ii in xrange(niter):
 
        # calculate the diffs
        deltaD[:-1,: ,:  ] = np.diff(stackout,axis=0)
        deltaS[:  ,:-1,: ] = np.diff(stackout,axis=1)
        deltaE[:  ,: ,:-1] = np.diff(stackout,axis=2)
 
        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gD = np.exp(-(deltaD/kappa)**2.)/step[0]
            gS = np.exp(-(deltaS/kappa)**2.)/step[1]
            gE = np.exp(-(deltaE/kappa)**2.)/step[2]
        elif option == 2:
            gD = 1./(1.+(deltaD/kappa)**2.)/step[0]
            gS = 1./(1.+(deltaS/kappa)**2.)/step[1]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[2]
 
        # update matrices
        D = gD*deltaD
        E = gE*deltaE
        S = gS*deltaS
 
        # subtract a copy that has been shifted 'Up/North/West' by one
        # pixel. don't as questions. just do it. trust me.
        UD[:] = D
        NS[:] = S
        EW[:] = E
        UD[1:,: ,: ] -= D[:-1,:  ,:  ]
        NS[: ,1:,: ] -= S[:  ,:-1,:  ]
        EW[: ,: ,1:] -= E[:  ,:  ,:-1]
 
        # update the image
        stackout += gamma*(UD+NS+EW)
 
        if ploton:
            iterstring = "Iteration %i" %(ii+1)
            ih.set_data(stackout[showplane,...].squeeze())
            ax2.set_title(iterstring)
            fig.canvas.draw()
            # sleep(0.01)
 
    return stackout


def test_filter():
    # Setttings
    signal_amp  = 10
    noise_amp   = 10
    noise_width = 0.1
    radius      = 1
    eps         = 0.4**2
    nan_frac    = 0.1
    n_chunk     = 9     # Number of chunks for full model computation

    # Geographic parameters
    ref_point   = [-116, 33.5]
    avg_strike  = 315.8
    trace_inc   = 0.01

    # Fault parameters
    mesh_file       = '/Users/evavra/Projects/SSAF/Analysis/Finite_Fault_Modeling/Mesh/Geometry/mesh_points_updated.txt'
    triangle_file   = '/Users/evavra/Projects/SSAF/Analysis/Finite_Fault_Modeling/Mesh/Geometry/mesh_connectivity_updated.txt'
    poisson_ratio   = 0.25      # Poisson ratio
    shear_modulus   = 6 * 10**9 # Shear modulus (Pa)
    disp_components = [1]       # displacement components to use [0 for fault-perpendicular, 1 for fault-parallel, 2 for vertical]
    slip_components = [0]       # slip components to use [0 for strike-slip, 1 for dip-slip, 2 for opening]
    mu              = 1 # spatial smoothing hyperparameter
    eta             = 1 # zero-edge-slip hyperparameter


    # Load fault model and regularization matrices
    fault = pyffit.finite_fault.TriFault(mesh_file, triangle_file, slip_components=slip_components, 
                                         poisson_ratio=poisson_ratio, shear_modulus=shear_modulus, 
                                         verbose=False, trace_inc=trace_inc, mu=mu, eta=eta,
                                         avg_strike=avg_strike, ref_point=ref_point)
    n_patch = len(fault.triangles)


    # Load data
    k = 204
    run_dir = '/Users/evavra/Projects/SSAF/Analysis/Time_Series/NIF/data/updated_mesh/testing_slip_rate/omega_1.0e+06__kappa_1.0e+00__sigma_1.0e+00'
    
    # Load data
    dataset = pyffit.data.load_insar_dataset(data_dir, file_format, dataset_name, ref_point, data_factor=data_factor, xkey=xkey, 
                                             coord_type=coord_type, date_index_range=date_index_range, 
                                             check_lon=check_lon, reference_time_series=reference_time_series, 
                                             incremental=False, use_dates=use_dates, use_datetime=use_datetime, 
                                             mask_file=mask_file, remove_mean=False)
    datasets = {dataset_name: dataset}
    n_obs    = len(dataset.date)
    x        = dataset.coords['x'].compute().data.flatten()
    y        = dataset.coords['y'].compute().data.flatten()

    # with h5py.File(full_model_file, 'r') as file:
    #     d_full = file[f'date_{k}'][()]

    # # data = d_full.reshape(orig.shape)[100:900, 200:]
    # data = d_full.reshape(orig.shape)

    # full_gf_file = f'{run_dir}/full_greens_functions.h5'

    # print(os.path.exists(full_gf_file))
    # print(full_gf_file)


    # G = -fault.greens_functions(x.flatten(), y.flatten(), disp_components=disp_components, slip_components=slip_components, rotation=avg_strike, squeeze=True)


    # if not os.path.exists(full_gf_file):
    #     print('yeet')
    #     n_data_full = x.size
    #     chunk_size  = n_data_full//n_chunk 
    #     remainder   = n_data_full % n_chunk

    #     for i in range(n_chunk - 1):
    #         print(f'Working on chunk {i + 1}...')
    #         start_chunk = time.time()
            
    #         # Get chunk indicies
    #         if i == n_chunk - 2:
    #             remainder   = n_data_full % n_chunk
    #         else:
    #             remainder = 0
    #         start  = chunk_size * i
    #         end    = chunk_size * (i + 1) + remainder

    #         # Get Greens functions and compute model prediction
    #         G = -fault.greens_functions(x.flatten()[start:end], y.flatten()[start:end], disp_components=disp_components, slip_components=slip_components, rotation=avg_strike, squeeze=True)

    #         end_chunk = time.time() - start_chunk
    #         print(f'Completed in {end_chunk:.1f} s')

    #         with h5py.File(full_gf_file, 'w') as file:
    #             file.create_dataset(f'chunk_{i}', data=G)

    return
    # Nans
    # n_nan      = int(data.size * nan_frac)
    # i_nans     = np.random.randint(0, high=ny, size=n_nan)
    # j_nans     = np.random.randint(0, high=nx, size=n_nan)
    
    # Noise
    noise      = noise_amp * np.random.normal(loc=0, scale=noise_width, size=data.shape)
    noise      = noise_amp * np.random.uniform(low=-1, high=1, size=data.shape)
    noisy_data = data + noise

    # Normalize
    rng        = [np.nanmin(noisy_data), np.nanmax(noisy_data)]
    data       = (data - rng[0])/(rng[1] - rng[0])
    noisy_data = (noisy_data - rng[0])/(rng[1] - rng[0])

    # ---------- Apply filter ----------
    # No nans
    GF = GuidedFilter(data, radius=radius, eps=eps)
    # GF = GuidedFilter(data * 20, radius=radius, eps=eps)
    filt_data = GF.filter(noisy_data)
    
    # Restore scale
    filt_data  = filt_data * (rng[1] - rng[0]) + rng[0] # rescale
    noisy_data = noisy_data * (rng[1] - rng[0]) + rng[0] # rescale
    data       = data * (rng[1] - rng[0]) + rng[0] # rescale


    # Plot 
    vlim = signal_amp + 1.5

    # Plot data
    width_ratios = (1, 1, 0.05)
    height_ratios = (1, 1)

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs  = fig.add_gridspec(2, 3, width_ratios=width_ratios, height_ratios=height_ratios,
                            hspace=0.0, wspace=0.0)

    axes = [fig.add_subplot(gs[i, j]) for i in range(len(height_ratios)) for j in range(len(width_ratios) - 1)]
    cax  = fig.add_subplot(gs[0, -1])

    im = axes[0].imshow(noisy_data,       cmap=cmc.vik, vmin=-vlim, vmax=vlim, interpolation='none')
    im = axes[1].imshow(filt_data,        cmap=cmc.vik, vmin=-vlim, vmax=vlim, interpolation='none')
    im = axes[2].imshow(data - filt_data, cmap=cmc.vik, vmin=-vlim, vmax=vlim, interpolation='none')
    im = axes[3].imshow(noisy_data - filt_data, cmap=cmc.vik, vmin=-vlim, vmax=vlim, interpolation='none')

    for ax in axes:
        ax.invert_yaxis()
    plt.colorbar(im, cax=cax, label='Displacement (mm)', shrink=0.5)
    plt.show()

    # Plot
    # pyffit.figures.plot_grid(x, y, d_full.reshape(data.shape), figsize=[8, 8],  show=True, cmap=cmc.vik, vlim=[-20, 20])



    return
    # Parameters
    nx          = 200
    ny          = 200
    signal_amp  = 10
    noise_amp   = 10
    noise_width = 0.5
    radius      = 1
    eps         = 0.2**2
    eps         = 1**2
    nan_frac    = 0.1

    # Signal
    data = np.zeros((ny, nx))
    data[:nx//2, :] += signal_amp/2
    data[nx//2:, :] -= signal_amp/2

    # Noise
    n_nan      = int(data.size * nan_frac)
    noise      = noise_amp * np.random.normal(loc=0, scale=noise_width, size=data.shape)
    i_nans     = np.random.randint(0, high=ny, size=n_nan)
    j_nans     = np.random.randint(0, high=nx, size=n_nan)
    noisy_data = data + noise

    # Normalize
    rng        = [np.nanmin(noisy_data), np.nanmax(noisy_data)]
    data       = (data - rng[0])/(rng[1] - rng[0])
    noisy_data = (noisy_data - rng[0])/(rng[1] - rng[0])

# ---------- Apply filter ----------
    # Nans
    # noisy_data[i_nans, j_nans] = np.nan
    # V = noisy_data.copy()
    # V[np.isnan(noisy_data)] = 0
    # GF = GuidedFilter(V, radius=radius, eps=eps)
    # VV = GF.filter(V)
    # W = 0 * noisy_data.copy() + 1
    # W[np.isnan(noisy_data)] = 0
    # GF = GuidedFilter(W, radius=radius, eps=eps)
    # WW = GF.filter(W)
    # filt_data = VV/WW

    # No nans
    GF = GuidedFilter(noisy_data, radius=radius, eps=eps)
    # GF = GuidedFilter(data * 20, radius=radius, eps=eps)
    filt_data = GF.filter(noisy_data)
    
    # Restore scale
    filt_data  = filt_data * (rng[1] - rng[0]) + rng[0] # rescale
    noisy_data = noisy_data * (rng[1] - rng[0]) + rng[0] # rescale
    data       = data * (rng[1] - rng[0]) + rng[0] # rescale


    # Plot 
    vlim = signal_amp + 1.5

    # Plot data
    width_ratios = (1, 1, 1, 0.05)
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs  = fig.add_gridspec(1, 4, width_ratios=width_ratios,
                            hspace=0.0, wspace=0.05)

    axes = [fig.add_subplot(gs[0, i]) for i in range(len(width_ratios) - 1)]
    cax  = fig.add_subplot(gs[0, -1])

    im = axes[0].imshow(noisy_data,       cmap=cmc.vik, vmin=-vlim, vmax=vlim)
    im = axes[1].imshow(filt_data,        cmap=cmc.vik, vmin=-vlim, vmax=vlim)
    im = axes[2].imshow(data - filt_data, cmap=cmc.vik, vmin=-vlim, vmax=vlim)

    plt.colorbar(im, cax=cax, label='Displacement (mm)')
    plt.show()


    return


def test_intf():
    path = '/Users/evavra/Projects/Taiwan/ALOS2/A139/F4/intf/20220807_20220918'
    intf = Interferogram(path, verbose=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.imshow(intf.data['phase'], extent=intf.extent)
    ax.set_aspect(0.25)

    if intf.track == 'A':
        ax.invert_yaxis()


    plt.show()
    
    return


def test_quadtree():
    # Data file
    data_file   = 'maduo/S1_LOS_D106.grd'

    # Quadtree parameters
    rms_min      = 1   # RMS threshold (data units)
    nan_frac_max = 0.4 # Fraction of NaN values allowed per cell
    width_min    = 2   # Minimum cell width (# of pixels)
    width_max    = 300 # Maximum cell width (# of pixels)
    
    # Plot parameters
    cmap = 'coolwarm'
    
    # ---------- LOAD DATA ----------
    x, y, data  = read_grd(data_file)
    X, Y   = np.meshgrid(x, y)
    extent = [x.min(), x.max(), y.max(), y.min()]

    # ---------- QUADTREE ALGORITHM ----------
    print('Performing quadtree downsampling...')
    x_samp, y_samp, z_samp, pixel_count, nan_frac = quadtree(X, Y, data, 0, rms_min=rms_min, nan_frac_max=nan_frac_max, width_min=width_min, width_max=width_max,)
    
    x_samp      = np.array(x_samp)[~np.isnan(x_samp)] 
    y_samp      = np.array(y_samp)[~np.isnan(y_samp)]
    z_samp      = np.array(z_samp)[~np.isnan(z_samp)]
    pixel_count = np.array(pixel_count)[~np.isnan(pixel_count)]
    nan_frac    = np.array(nan_frac)[~np.isnan(nan_frac)]

    print('Number of initial data points:', data[~np.isnan(data)].size)
    print('Number of downsampled data points:', len(z_samp))
    print('Percent reduction: {:.2f}%'.format(100*(data[~np.isnan(data)].size - len(z_samp))/data[~np.isnan(data)].size))
    print()

    # ---------- PLOT ----------
    fig = plt.figure(figsize=(14, 8.2))
    projection = ccrs.PlateCarree()
    ax  = fig.add_subplot(1, 1, 1, projection=projection)
    ax.add_feature(cfeature.LAKES.with_scale('10m'))
    ax.scatter(x_samp, y_samp, c=z_samp, cmap=cmap, transform=projection, zorder=100)
    plt.show()
    
    return


if __name__ == '__main__':
    main()