import os
import gc
import sys
import h5py
import time
import numpy as np
import pandas as pd
import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fmin_cobyla
from scipy.linalg import cholesky, qr, solve_triangular, cho_factor, cho_solve, sqrtm
from pyffit.covariance import exp_covariance
from pyffit.utilities import check_dir_tree
# from pyffit.figures import plot_matrix


# ------------------ NIF ------------------
def network_inversion_filter(fault, G, d, std, dt, omega, sigma, kappa, state_sigmas, cov_file='covariance.h5', steady_slip=False, rho=1, ramp_matrix=[], constrain=False, state_lim=[], result_dir='.', dist_file='dist.h5', cost_function='state'):
    """
    Invert time series of surface displacements for fault slip.
    Based on the method of Bekaert et al. (2016)

    INPUT:
    x_init (n_dim,)       - intial state vector (n_dim - # of model parameters) 
    P_init (n_dim, n_dim) - intial state covariance matrix 
    d (n_obs, n_data)     - observations (n_obs - # of time points, n_data - # of data points)

    H (n_data, n_dim)     - observation matrix: maps model output x to observations d
    R (n_data, n_data)    - data covariance matrix
    T (n_dim, n_dim)      - state transition matrix: maps state x_k to next state x_k+1
    Q (n_data, n_data)    - process noise covariance matrix: prescribes covariances (i.e. tuning parameters) related to temporal smoothing
    """

    # -------------------- Prepare NIF --------------------
    n_patch = len(fault.triangles)

    # Get fault parameter dimensions
    if steady_slip:
        n_dim   = 3 * n_patch
    else:
        n_dim   = 2 * n_patch

    # Determine number of ramp coefficients
    n_ramp = 0
    if len(ramp_matrix)> 0:
        n_ramp += ramp_matrix.shape[1]
        n_dim    += n_ramp

    # Form smoothing matrix S
    L = fault.smoothing_matrix

    # Form transition matrix T
    T = make_transition_matrix(n_patch, dt, steady_slip=steady_slip, ramp_matrix=ramp_matrix)

    # Form process covariance matrix Q
    Q = make_process_covariance_matrix(n_patch, dt, omega, steady_slip=steady_slip, ramp_matrix=ramp_matrix, rho=rho)

    # Form data covariance matrix R
    # R = make_data_covariance_matrix(C, n_patch, sigma, kappa, steady_slip=steady_slip)

    # Define initial state vector
    x_init = np.zeros(n_dim)

    # Form initial state prediction covariance matrix P_init
    P_init = make_prediction_covariance_matrix(n_patch, state_sigmas, steady_slip=steady_slip, ramp_matrix=ramp_matrix)

    # -------------------- Run NIF --------------------
    # # 1) Perform forward Kalman filtering
    model, resid, rms = kalman_filter(x_init, P_init, d, std, dt, G, L, T, Q, sigma=sigma, kappa=kappa, cov_file=cov_file, result_dir=result_dir, steady_slip=steady_slip, 
                                      ramp_matrix=ramp_matrix, constrain=constrain, state_lim=state_lim, cost_function=cost_function, dist_file=dist_file)
    # model, resid, rms = sqrt_filter(x_init, P_init, d, std, dt, G, L, T, Q, sigma=sigma, kappa=kappa, cov_file=cov_file, steady_slip=steady_slip, ramp_matrix=ramp_matrix, constrain=constrain, 
    # state_lim=state_lim, cost_function=cost_function, file_name=f'{result_dir}/results_forward.h5')
    with h5py.File(f'{result_dir}/results_forward.h5', 'r') as file:
        x_model_forward = file['x_a'][()]
    gc.collect()
    
    # Perform backward smoothing
    model, resid, rms, x_s, P_s = backward_smoothing(f'{result_dir}/results_forward.h5', d, dt, G, L, T, steady_slip=steady_slip, ramp_matrix=ramp_matrix, constrain=constrain, state_lim=state_lim, cost_function=cost_function)
    # model, resid, rms, x_s, P_s = sqrt_smoothing(f'{result_dir}/results_forward.h5', d, dt, G, L, T, steady_slip=steady_slip, ramp_matrix=ramp_matrix, constrain=constrain, state_lim=state_lim, cost_function=cost_function)
    write_kalman_filter_results(model, resid, rms, x_s=x_s, P_s=P_s, backward_smoothing=True, file_name=f'{result_dir}/results_smoothing.h5',)
    x_model_smoothing = x_s
    gc.collect()

    # results_smoothing = backward_smoothing(results_forward.x_f, results_forward.x_a, results_forward.P_f, results_forward.P_a, d, dt, G, L, T, 
                                        #    state_lim=state_lim, cost_function='state')

    return x_model_forward, x_model_smoothing


def sqrt_filter(x_init, P_init, d, std, dt, G, L, T, Q, sigma=1, kappa=1, state_lim=[], ramp_matrix=[], steady_slip=False, cov_file='covariance.h5', constrain=False, cost_function='state', file_name='sqrt_kf_forward.h5'):

    start_total = time.time()

    # ---------- Square-Root Kalman Filter ---------- 
    # Determine number of ramp coefficients
    n_ramp = 0
    if len(ramp_matrix) > 0:
        n_ramp += ramp_matrix.shape[1]

    # Get number of fault patches
    if steady_slip:
        n_patch = (x_init.size - n_ramp)//3
    else:
        n_patch = (x_init.size - n_ramp)//2

    # Get other dimensions
    n_dim      = x_init.size                 # model state vector
    n_obs      = d.shape[0]                  # number of temporal observations
    n_data     = d.shape[1] - n_dim + n_ramp # number of spatial data points
    slip_start = steady_slip * n_patch       # index for start of transient slip
    
    x_f     = np.empty((n_obs, n_dim)) # forecasted states
    x_a     = np.empty((n_obs, n_dim)) # analyzed states
    # P_f     = np.empty((n_dim, n_dim)) # forecasted covariances
    # P_a     = np.empty((n_dim, n_dim)) # analyzed covariances

    model   = np.empty((n_obs, n_data)) # forward model prediction
    resid   = np.empty((n_obs, n_data)) # data - model residual
    rms     = np.empty((n_obs,))        # root mean square 

    # covariance = h5py.File(cov_file, 'r')
    file       = h5py.File(f'{file_name}', 'w')
    covariance = h5py.File(f'{cov_file}', 'r')

    print('############### Running square-root Kalman filter ###############')

    # # Compute variances and et outliers to zero
    # var     = std**2
    # var_max = 3*np.nanstd(var)
    # var[var > var_max] = var_max
    # var[var <= 0] = np.min(var[var > 0])

    # Initialize state and covaraince forecasts
    P_f       = P_init
    x_f[0, :] = x_init
    P_f_sqrt = cholesky(P_f, lower=False)
    # P_f_sqrt = sqrtm(P_f)

    for k in range(0, n_obs):
        start_step = time.time()
        print(f'\n##### Working on Step {k} #####')

        # Update observation matrix H        
        H = make_observation_matrix(G, L, t=dt*k, steady_slip=steady_slip, ramp_matrix=ramp_matrix)

        # Make data covariance matrix
        if k > 0:
            kk = k
        else:
            kk = 1

        # Get estimated covariance matrix 
        # C = exp_covariance(0, 20, 4, dist)
        C  = covariance[f'covariance/{49}'][()]
        C -= C.min() # Enforce tapering to zero
        C /= C.max()
        # C[C < 1e-1] = 0
        # print(np.sum(C == 0)/C.size)
        C  = np.eye(len(C))
        R  = make_data_covariance_matrix(C, n_patch, sigma, kappa, steady_slip=steady_slip)

        # ---------- Analysis ----------
        start_update           = time.time()
        x_a[k, :], P_a_sqrt, z = state_update(H, R, d[k, :], x_f[k, :], P_f_sqrt)
        # x_a[k, :], P_a_sqrt, z = sqrt_update(H, R, d[k, :], x_f[k, :], P_f_sqrt)
        end_update             = time.time() - start_update

        P_a = P_a_sqrt @ P_a_sqrt.T
    
        print(f'P_f: full: min. {P_f.min():.2e} max. {P_f.max():.2e} | diag: min. {np.diag(P_f).min():.2e} max. {np.diag(P_f).max():.2e}')
        print(f'P_a: full: min. {P_a.min():.2e} max. {P_a.max():.2e} | diag: min. {np.diag(P_a).min():.2e} max. {np.diag(P_a).max():.2e}')
        print(f'P_a_sqrt is lower: {check_lower(P_a_sqrt)}')

        print()
        print(f'Update time: {end_update:.2f} s')
        print(f'Forecasted state range:  slip {x_f[k, :n_patch].min():.2f} - {x_f[k, :n_patch].max():.2f} | slip rate {x_f[k, n_patch:].min():.2f} - {x_f[k, n_patch:].max():.2f}')
        print(f'Updated state range:     slip {x_a[k, :n_patch].min():.2f} - {x_a[k, :n_patch].max():.2f} | slip rate {x_a[k, n_patch:].min():.2f} - {x_a[k, n_patch:].max():.2f}')

        # # Constrain state estimate 
        if k > 0:
            if constrain:
                # start_opt = time.time()
                x_a[k, :] = constrain_state(x_a[k, :], x_a[k - 1, :], state_lim, state_range=[slip_start, slip_start + n_patch], direction='increasing', y=d[k, :], H=H)
                print(f'Constrained state range: slip {x_a[k, :n_patch].min():.2f} - {x_a[k, :n_patch].max():.2f} | slip rate {x_a[k, n_patch:].min():.2f} - {x_a[k, n_patch:].max():.2f}')
                # end_opt = time.time() - start_opt
                # print(f'Constraint time: {end_update:.2f} s')

        # Store objects for later
        # file.create_dataset(f'{k}/P_f', data=P_f)
        # file.create_dataset(f'{k}/P_a', data=P_a)

        # ---------- Forecast ----------
        # Use current analysis initialize next forecast
        if k + 1 < n_obs:
            start_forecast = time.time()

            # Make forecast 
            x_f[k + 1, :] = T @ x_a[k, :]
            P_f_sqrt, S, Delta_a_sqrt, delta_a = time_update(x_f[k + 1, :], x_a[k, :], P_a_sqrt.T, T, Q)

            # print(f'P_f_sqrt is upper-triangular: {np.allclose(P_f_sqrt, np.triu(P_f_sqrt))}')

            P_f = P_f_sqrt.T @ P_f_sqrt
            # # P_f[P_f < 0] = 0

            # print(f'P_f: {P_f.min()}, {P_f.max()}')

            # print('Symmetric?', np.allclose(P_f, P_f.T, atol=1e-9, rtol=1e-9))
            # P_f_sqrt = cholesky(P_f, lower=False)

            file.create_dataset(f'{k}/S',            data=S)
            file.create_dataset(f'{k}/Delta_a_sqrt', data=Delta_a_sqrt)
            file.create_dataset(f'{k}/delta_a',      data=delta_a)

            end_forecast = time.time() - start_forecast
            print(f'Forecast time: {end_forecast:.2f} s')
        else:
            P_a = P_a_sqrt @ P_a_sqrt.T
            file.create_dataset(f'{k}/P_a', data=P_a)
            file.create_dataset(f'{k}/P_a_sqrt', data=P_a_sqrt)
            print('Forward filtering complete')

        # Compute error terms
        # z = d[k, :] - H @ x_f
        resid[k, :] = z[:n_data]
        rms[k]      = np.sqrt(np.mean(resid[k, :]**2))  
        model[k, :] = (d[k, :] - z)[:n_data]

        end_step = time.time() - start_step
        print(f'RMS:  {rms[k]:.2f}')
        print(f'Step {k} time: {end_step:.2f} s')

    file.create_dataset('model', data=model)
    file.create_dataset('resid', data=resid)
    file.create_dataset('rms',   data=rms)
    file.create_dataset('x_f', data=x_f)
    file.create_dataset('x_a', data=x_a)
    file.close()
    
    end_total = time.time() - start_total
    print(f'Elapsed time: {end_total/60:.1f} min')

    return model, resid, rms


def sqrt_smoothing(result_file, d, dt, G, L, T, steady_slip=False, ramp_matrix=[], constrain=False, state_lim=[], cost_function='state', rcond=1e-15,):
    """
    Square-root implementation of Rauch-Tung-Streibel smoothing.
    """

    start_total = time.time()
    file = h5py.File(f'{result_file}', 'r')

    # Load state estimates from forward filtering
    x_f  = file['x_f'][()]
    x_a  = file['x_a'][()]

    # Get dimensions
    n_obs  = x_f.shape[0]
    n_dim  = x_f.shape[1]

    # Determine number of ramp coefficients
    n_ramp = 0
    if len(ramp_matrix) > 0:
        n_ramp += ramp_matrix.shape[1]

    # Determine number of fault patches
    if steady_slip:
        n_patch = (n_dim - n_ramp)//3
    else:
        n_patch = (n_dim - n_ramp)//2
    
    n_data = d.shape[1] - n_dim + n_ramp 

    # Get start of transient slip
    slip_start = steady_slip * n_patch 
    
    # Initialize
    # P_s     = np.empty((n_obs, n_dim, n_dim))   # analyzed covariances
    x_s     = np.empty((n_obs, n_dim))            # analyzed states
    S       = np.empty((n_obs - 1, n_dim, n_dim)) # analyzed covariances
    model   = np.empty((n_obs, n_data))
    resid   = np.empty((n_obs, n_data))
    rms     = np.empty((n_obs,))

    print(f'\n##### Running backward smoothing ##### ')

    # Set last epoch to be equal to forward filtering result
    x_s[-1, :] = x_a[-1, :]
    # P_s        = file[f'{n_obs - 1}/P_a']
    # P_s_sqrt = cholesky(P_s, lower=True) 
    P_s_sqrt        = file[f'{n_obs - 1}/P_a_sqrt']

    for k in reversed(range(0, n_obs - 1)):
        start_step = time.time()
        print(f'\n##### Working on Step {k} #####')
        
        # Load objects
        S            = file[f'{k}/S'][()]
        Delta_a_sqrt = file[f'{k}/Delta_a_sqrt'][()]
        delta_a      = file[f'{k}/delta_a'][()]

        # Form block matrix A.T (P_s_sqrt is for k + 1)
        A_t = np.block([[S @ P_s_sqrt, Delta_a_sqrt]])

        # Perform QR decomposition on A to obtain B and extract relevant sub-matrices
        B_t      = qr(A_t.T, mode='r')[0].T
        P_s_sqrt = B_t[:n_dim, :n_dim] # sqrt(P_s) for k

        # Update state
        x_s[k, :] = S @ x_s[k + 1, :] + delta_a
        print(f'Smoothed range:  slip {x_s[k, :n_patch].min():.2f} - {x_s[k, :n_patch].max():.2f} | slip rate {x_s[k, n_patch:].min():.2f} {x_s[k, n_patch:].max():.2f}')

        # # Get smoothed states and covariances
        # x_s[k, :] = x_a[k, :] + S[k, :, :] @ (x_s[k + 1, :] - x_f[k + 1, :])
        # P_s       = P_a       + S[k, :, :] @ (P_s - P_f) @ S[k, :, :].T
        # x_smooth  = x_s[k, :].copy()
        
        # Constrain state estimate 
        if constrain:
            x_s[k, :] = constrain_state(x_s[k, :], x_s[k + 1, :], state_lim, state_range=[slip_start, slip_start + n_patch], direction='decreasing')
            print(f'Constrained range:  slip {x_s[k, :n_patch].min():.2f} - {x_s[k, :n_patch].max():.2f} | slip rate {x_s[k, n_patch:].min():.2f} {x_s[k, n_patch:].max():.2f}')

        end_step = time.time() - start_step
        print(f'Smoothed range:  slip {x_s[k, :n_patch].min():.2f} - {x_s[k, :n_patch].max():.2f} | slip rate {x_s[k, n_patch:].min():.2f} {x_s[k, n_patch:].max():.2f}')
        print(f'Step {k} time: {end_step:.2f} s')

    # Compute misfit statistics
    for k in range(n_obs):
        # Make observation matrix
        H = make_observation_matrix(G, L, t=dt*k, steady_slip=steady_slip, ramp_matrix=ramp_matrix)

        # Compute error terms
        pred        = H @ x_s[k, :]
        model[k, :] = pred[:n_data]
        resid[k, :] = (d[k, :] - pred)[:n_data]
        rms[k]      = np.sqrt(np.mean(resid[k, :]**2))  

    P_s = P_s_sqrt @ P_s_sqrt.T
    file.close()
    gc.collect()
    end_total = time.time() - start_total
    print('##### Backward smoothing complete #####')
    print(f'Elapsed time: {end_total/60:.1f} min')
    print()

    return model, resid, rms, x_s, P_s


def time_update(x_f, x_a, P_a_sqrt, T, Q, verbose=False):
    """
    Square-root Kalman filter time update step.
    """

    m = T.shape[0] # model dimension

    # Perform Cholesky factorizations
    Q_sqrt   = cholesky(Q, lower=True)
    # Q_sqrt   = sqrtm(Q)
    # P_a_sqrt = cholesky(P_a, lower=False)
    zeros    = np.zeros((m, m))

    # Form block matrix A.T
    A_t = np.block([[Q_sqrt, T @ P_a_sqrt], 
                    [zeros,      P_a_sqrt]])

    # Perform QR decomposition on A to get B
    B_t = qr(A_t.T, mode='r')[0].T

    # Extract sub-matrices from B.T
    P_f_sqrt     = B_t[:m, :m] # sqrt(P_f)
    S_P_f_sqrt   = B_t[m:, :m] # S @ sqrt(P_f)
    Delta_a_sqrt = B_t[m:, m:] # sqrt(delta_a)

    # mask           = np.diag(P_f_sqrt < 0)
    # P_f_sqrt[mask, :] = -P_f_sqrt[mask, :]

    if verbose:
        print(f'A.T          is upper-triangular: {np.allclose(A_t, np.triu(A_t))}')
        print(f'B_t          is lower-triangular: {np.allclose(B_t, np.tril(B_t))}')
        print(f'P_f_sqrt     is lower-triangular: {np.allclose(P_f_sqrt, np.tril(P_f_sqrt))}')
        print(f'Delta_a_sqrt is lower-triangular: {np.allclose(Delta_a_sqrt, np.tril(Delta_a_sqrt))}')

    # Compute sqrt(P_f)^-1 via back-substitution
    P_f_sqrt_inv = solve_triangular(P_f_sqrt.T, np.eye(m), lower=False)

    # Compute smoothing matrix S
    S = S_P_f_sqrt @ P_f_sqrt_inv

    # Compute boomerang residual
    delta_a = x_a - (S @ x_f)

    # P_f = P_f_sqrt @ P_f_sqrt.T
    # P_f[P_f < 0] = 0

    # print()
    # P_f_sqrt = cholesky(P_f, lower=False)
    P_f_sqrt = P_f_sqrt.T

    return P_f_sqrt, S, Delta_a_sqrt, delta_a


def state_update(H, R, y, x_f, P_f_sqrt, verbose=False):
    """
    Based on Chin (2023)
    """

    n, m = H.shape

    # Perform Cholesky factorizations
    R_sqrt   = cholesky(R, lower=False)
    # P_f_sqrt = cholesky(P_f, lower=False)
    zeros    = np.zeros((m, n))

    # Form block matrix A
    A_t = np.block([[R_sqrt, H @ P_f_sqrt], 
                    [zeros,      P_f_sqrt]])

    # Perform QR decomposition on A to obtain B and extract relevant sub-matrices
    B_t               = qr(A_t.T, mode='r')[0].T
    Y_sqrt            = B_t[:n, :n] # sqrt(H @ P_f @ H.T + R)
    K_Y_sqrt          = B_t[n:, :n] # K @ sqrt(H @ P_f @ H.T + R)
    P_a_sqrt          = B_t[n:, n:] # sqrt(P_a)
    # mask              = np.diag(P_a_sqrt < 0)
    # P_a_sqrt[mask, :] = -P_a_sqrt[mask, :]
    # mask              = np.diag(Y_sqrt < 0)
    # Y_sqrt[mask, :]   = -Y_sqrt[mask, :]

    if verbose:
        print(f'A.T      is upper-triangular: {np.allclose(A_t, np.triu(A_t))}')
        print(f'B_t      is lower-triangular: {np.allclose(B_t, np.tril(B_t))}')
        print(f'Y_sqrt   is lower-triangular: {np.allclose(Y_sqrt, np.tril(Y_sqrt))}')
        print(f'P_a_sqrt is lower-triangular: {np.allclose(P_a_sqrt, np.tril(P_a_sqrt))}')
        print(f'x_f shape                     {x_f.shape}')

    # Get inverse of sqrt(Y)(y - H @ x_f) via back-substitution
    z = y - (H @ x_f)
    Y_sqrt_inv_z = solve_triangular(Y_sqrt.T, z, lower=False)

    # test = (Y_sqrt.T @ Y_sqrt_inv_z) - z
    
    # Y = Y_sqrt @ Y_sqrt.T
    # print(f'Y:    {Y.min()} {Y.max()}')
    # print(f'test: {test.min()} {test.max()}')

    # Get updated state 
    update = K_Y_sqrt @ Y_sqrt_inv_z
    x_a    = x_f + update

    # Get updated covariance
    # P_a = P_a_sqrt @ P_a_sqrt.T

    return x_a, P_a_sqrt, z
    

def kalman_filter(x_init, P_init, d, std, dt, G, L, T, Q, sigma=1, kappa=1, mask_dist=5, cov_file='covariance.h5', result_dir='.', state_lim=[], ramp_matrix=[], steady_slip=False, constrain=False, cost_function='state', 
                  backward_smoothing=False, overwrite=True, dist_file='dist.h5', filter_file_name='results_forward.h5', likelihood_file_name='likelihood.h5'):
    """
    INPUT:
    x_init (n_dim,)     - intial state (n_dim - # of model parameters) 
    d (n_obs, n_data)   - observations (n_obs - # of time points, n_data - # of data points)

    G - greens funcions matrix
    L - fault smoothing matrix

    H (n_data, n_dim)   - observation matrix: maps model output x to observations d
    R (n_data, n_data)  - data covariance matrix
    T (n_dim, n_dim)    - state transition matrix: maps state x_k to next state x_k+1
    Q (n_data, n_data)  - process noise covariance matrix: prescribes covariances (i.e. tuning parameters) related to temporal smoothing
    """

    # ---------- Kalman Filter ---------- 
    n_dim = x_init.size

    # Determine number of ramp coefficients
    n_ramp = 0
    if len(ramp_matrix) > 0:
        n_ramp += ramp_matrix.shape[1]

    if steady_slip:
        n_patch = (x_init.size - n_ramp)//3
    else:
        n_patch = (x_init.size - n_ramp)//2

    n_obs      = d.shape[0]
    n_data     = d.shape[1] - n_dim + n_ramp
    slip_start = steady_slip * n_patch  # Get start of transient slip
    
    x_f     = np.empty((n_obs, n_dim)) # forecasted states
    x_a     = np.empty((n_obs, n_dim)) # analyzed states
    P_f     = np.empty((n_dim, n_dim)) # forecasted covariances
    P_a     = np.empty((n_dim, n_dim)) # analyzed covariances

    model   = np.empty((n_obs, n_data))
    resid   = np.empty((n_obs, n_data))
    rms     = np.empty((n_obs,))

    r     = np.empty((n_obs,)) # data covariance scaling terms
    det_Y = np.empty((n_obs,)) # determinants of the innovation covariance matrix


    with h5py.File(dist_file, 'r') as file:                
        dist = file['dist'][()]

    if backward_smoothing:
        print('############### Running Kalman filter in backward mode ###############')
    else:
        print('############### Running Kalman filter ###############')

    check_dir_tree(result_dir + '/std')

    start_total     = time.time()
    covariance      = h5py.File(cov_file, 'r')
    file            = h5py.File(f'{result_dir}/{filter_file_name}', 'w')
    likelihood_file = h5py.File(f'{result_dir}/{likelihood_file_name}', 'w')
    dates           = [key for key in covariance[f'mask_dist_{mask_dist}_km'].keys()]

    for k in range(0, n_obs):
        start_step = time.time()
        print(f'\n##### Working on Step {k} #####')

        # Update observation matrix H        
        H = make_observation_matrix(G, L, t=dt*k, steady_slip=steady_slip, ramp_matrix=ramp_matrix)

        # Get quadtree data variances
        if k > 0:
            kk = k
        else:
            kk = 1
        
        # Get estimated covariance matrix 
        key       = f'mask_dist_{mask_dist}_km/{dates[k]}'
        sv_params = covariance[key][()]

        C  = exp_covariance(sv_params[0], sv_params[1], sv_params[2], dist)
        # C_max = C.max()
        # C -= C.min() # Enforce tapering to zero
        # C /= C.max() # Scale
        C = (C - C.min())/(C.max() - C.min()) * C.max()
        R  = make_data_covariance_matrix(C, n_patch, sigma, kappa, steady_slip=steady_slip)

        # 1) ---------- Forecast ----------
        # Make forecast 
        start_forecast = time.time()

        # Compute covariance forecast
        if k == 0:
            x_f[k, :] = x_init
            P_f       = P_init

            # Print object information
            print(f'[G] Greens function matrix:     shape = {G.shape} | {G.size:.1e} elements | {100 * np.sum(G == 0)/G.size:.2f}% sparsity) | mean = {np.mean(G)} | std = {np.std(G)}')
            print(f'[H] Observation matrix:         shape = {H.shape} | {H.size:.1e} elements | {100 * np.sum(H == 0)/H.size:.2f}% sparsity) | mean = {np.mean(H)} | std = {np.std(H)}')
            print(f'[R] Data covariance matrix:     shape = {R.shape} | {R.size:.1e} elements | {100 * np.sum(R == 0)/R.size:.2f}% sparsity) | mean = {np.mean(R)} | std = {np.std(R)}')
            print(f'[T] Transition matrix:          shape = {T.shape} | {T.size:.1e} elements | {100 * np.sum(T == 0)/T.size:.2f}% sparsity) | mean = {np.mean(T)} | std = {np.std(T)}')
            print(f'[Q] Process covariance matrix:  shape = {Q.shape} | {Q.size:.1e} elements | {100 * np.sum(Q == 0)/Q.size:.2f}% sparsity) | mean = {np.mean(Q)} | std = {np.std(Q)}')

        else:
            P_f = T @ P_a @ T.T + Q 
            x_f[k, :] = T @ x_init
        
        end_forecast = time.time() - start_forecast
    
        # 2) ---------- Analysis ----------
        start_analysis = time.time()

        # # Update state and covariance        
        x_a[k, :], P_a, r[k], det_Y[k] = sqrt_update(H, R, d[k, :], x_f[k, :], P_f)
        x_update = x_a[k, :].copy()
        end_analysis = time.time() - start_analysis

        # # Constrain state estimate 
        start_opt = time.time()
        if constrain:
            x_a[k, :] = constrain_state(x_update, x_a[k - 1, :], state_lim, state_range=[slip_start, slip_start + n_patch], direction='increasing')
        end_opt = time.time() - start_opt

        print(f'Forecasted state range:  slip {x_f[k, :n_patch].min():.2f} - {x_f[k, :n_patch].max():.2f} | slip rate {x_f[k, n_patch:].min():.2f} - {x_f[k, n_patch:].max():.2f}')
        print(f'Updated state range:     slip {x_update[:n_patch].min():.2f} - {x_update[:n_patch].max():.2f} | slip rate {x_update[n_patch:].min():.2f} - {x_update[n_patch:].max():.2f}')
        print(f'Constrained state range: slip {x_a[k, :n_patch].min():.2f} - {x_a[k, :n_patch].max():.2f} | slip rate {x_a[k, n_patch:].min():.2f} - {x_a[k, n_patch:].max():.2f}')

        # Save covariance matrices
        file.create_dataset(f'{k}/P_f', data=P_f)
        file.create_dataset(f'{k}/P_a', data=P_a)

        # Use current analysis initialize next forecast
        x_init = x_a[k, :] 

        # Compute error terms
        pred        = H @ x_a[k, :]
        model[k, :] = pred[:n_data]
        resid[k, :] = (d[k, :] - pred)[:n_data]
        rms[k]      = np.sqrt(np.mean(resid[k, :]**2))  

        end_step = time.time() - start_step
        print(f'Step {k} time: {end_step:.2f} s (update: {end_analysis/end_step * 100:.1f} %, opt: {end_opt/end_step * 100:.1f} %, other: {(end_step - end_analysis - end_opt)/end_step * 100:.1f} %)')

    # Compute likelihood and estimate data covariance scaling
    N_d = n_data * n_obs

    L         = -0.5 * (N_d - N_d * np.log(N_d)) - 0.5 * np.sum(det_Y) - 0.5 * N_d * np.log(np.sum(r))
    sigma_hat = np.sqrt(np.sum(r)/N_d) 

    print(f'N_d        = {N_d}')
    print(f'sum(det_Y) = {np.sum(det_Y):.2e}')
    print(f'sum(r)     = {np.sum(r):.2e}')
    print(f'-2L        = {-2*L:.2e}')
    print(f'sigma_hat  = {sigma_hat:.2f}')

    end_total = time.time() - start_total

    if backward_smoothing:
        print('\n##### Backward smoothing complete #####')
    else:
        print('\n##### Kalman filter complete #####')
    print(f'Elapsed time: {end_total/60:.1f} min')
    print()

    file.create_dataset('model', data=model)
    file.create_dataset('resid', data=resid)
    file.create_dataset('rms',   data=rms)
    file.create_dataset('x_f',   data=x_f)
    file.create_dataset('x_a',   data=x_a)
    file.create_dataset('P_a',   data=P_a)
    likelihood_file.create_dataset('model', data=model)
    likelihood_file.create_dataset('resid', data=resid)
    likelihood_file.create_dataset('rms',   data=rms)
    likelihood_file.create_dataset('x_f',   data=x_f)
    likelihood_file.create_dataset('x_a',   data=x_a)
    likelihood_file.create_dataset('P_a',   data=P_a)
    likelihood_file.create_dataset('r',     data=r)
    likelihood_file.create_dataset('det_Y', data=det_Y)
    likelihood_file.create_dataset('L', data=L)
    likelihood_file.create_dataset('sigma_hat', data=sigma_hat)

    file.close()
    likelihood_file.close()
    return model, resid, rms


def backward_smoothing(result_file, d, dt, G, L, T, steady_slip=False, ramp_matrix=[], constrain=False, state_lim=[], cost_function='state', rcond=1e-15,):
    """
    Dimensions:
    n_obs  - # of time points
    n_dim  - # of model parameters
    n_data - # of data points

    INPUT:
    x_f, x_a (n_obs, n_dim)         - forecasted and analyzed state vectors
    P_f, P_a (n_obs, n_dim, n_dim)  - forecasted and analyzed covariance matrices 
    T        (n_dim, n_dim)         - state transition matrix: maps state x_k to next state x_k-1

    OUTPUT:
    """

    start_total = time.time()
    file = h5py.File(f'{result_file}', 'r')
    x_f  = file['x_f'][()]
    x_a  = file['x_a'][()]

    # ---------- Kalman Filter ---------- 
    # Get lengths
    n_obs   = x_f.shape[0]
    n_dim   = x_f.shape[1]

    # Determine number of ramp coefficients
    n_ramp = 0
    if len(ramp_matrix) > 0:
        n_ramp += ramp_matrix.shape[1]

    if steady_slip:
        n_patch = (n_dim - n_ramp)//3
    else:
        n_patch = (n_dim - n_ramp)//2

    n_data  = d.shape[1] - n_dim + n_ramp 

    slip_start = steady_slip * n_patch # Get start of transient slip
    
    # Initialize
    x_s     = np.empty((n_obs, n_dim))            # analyzed states
    # P_s     = np.empty((n_obs, n_dim, n_dim))   # analyzed covariances
    S       = np.empty((n_obs - 1, n_dim, n_dim)) # analyzed covariances
    model   = np.empty((n_obs, n_data))
    resid   = np.empty((n_obs, n_data))
    rms     = np.empty((n_obs,))

    print(f'\n##### Running backward smoothing ##### ')

    # Set last epoch to be equal to forward filtering result
    x_s[-1, :] = x_a[-1, :]
    P_s        = file[f'{n_obs - 1}/P_a']

    for k in reversed(range(0, n_obs - 1)):
        print(f'\n##### Working on Step {k} #####')

        # From Jackson impleentation
        # P_f_c = cholesky(file['P_f'][k + 1, :, :], lower=False)
        # y = x_s[k + 1, :] - x_f[k + 1, :]
        # tmp = solve_triangular(P_f_c, y.T, lower=True)
        # tmp = solve_triangular(P_f_c.T, tmp, lower=False)
        # x_s[k, :] = x_a[k, :] + file['P_a'][k, :, :] @ T.T @ tmp
        # C = (P_s[k + 1, :, :] - file['P_f'][k + 1, :, :])
        # BFt = file['P_a'][k, :] @ T.T
        # tmp = solve_triangular(P_f_c,   C,     lower=True)
        # tmp = solve_triangular(P_f_c.T, tmp.T, lower=False).T
        # tmp = solve_triangular(P_f_c,   tmp,   lower=True)
        # tmp = solve_triangular(P_f_c.T, tmp.T, lower=False).T
        # P_s[k, :, :] = file['P_a'][k, :] + BFt @ tmp @ BFt.T
        
        # Compute smoothing matrix S
        # P_f_inv = np.linalg.pinv(file[f'{k + 1}/P_f'], hermitian=True, rcond=rcond)
        # S[k, :, :] = file[f'{k + 1}/P_a'] @ T.T @ P_f_inv
        P_f = file[f'{k + 1}/P_f'][()]
        P_a = file[f'{k}/P_a'][()] 

        P_f_c      = cho_factor(P_f) # get cholesky factor
        S[k, :, :] = cho_solve(P_f_c, T @ P_a).T

        # Get smoothed states and covariances
        x_s[k, :] = x_a[k, :] + S[k, :, :] @ (x_s[k + 1, :] - x_f[k + 1, :])
        P_s       = P_a       + S[k, :, :] @ (P_s - P_f) @ S[k, :, :].T
        x_smooth  = x_s[k, :].copy()
        
        # Constrain state estimate 
        if constrain:
            x_s[k, :] = constrain_state(x_smooth, x_s[k + 1, :], state_lim, state_range=[slip_start, slip_start + n_patch], direction='decreasing')
            
            print(f'Smoothed range:     slip {x_smooth[:n_patch].min():.2f} - {x_smooth[:n_patch].max():.2f} | slip rate {x_smooth[n_patch:].min():.2f} {x_smooth[n_patch:].max():.2f}')
            print(f'Constrained range:  slip {x_s[k, :n_patch].min():.2f} - {x_s[k, :n_patch].max():.2f} | slip rate {x_s[k, n_patch:].min():.2f} {x_s[k, n_patch:].max():.2f}')

    # Compute misfit statistics
    for k in range(n_obs):
        # Make observation matrix
        H = make_observation_matrix(G, L, t=dt*k, steady_slip=steady_slip, ramp_matrix=ramp_matrix)

        # Compute error terms
        pred        = H @ x_s[k, :]
        model[k, :] = pred[:n_data]
        resid[k, :] = (d[k, :] - pred)[:n_data]
        rms[k]      = np.sqrt(np.mean(resid[k, :]**2))  

    file.close()
    gc.collect()
    end_total = time.time() - start_total
    print('##### Backward smoothing complete #####')
    print(f'Elapsed time: {end_total/60:.1f} min')
    print()

    return model, resid, rms, x_s, P_s


def sqrt_update(H, R, y, x_f, P_f, get_Y_sqrt=False):
    """
    DESCRIPTION:
    Perform Kalman filter analysis step using factored formulation.
    Based on NIF implementation available from Noel Jackson (KU) and the Stanford group.

    REFERENCES:
    Bartlow, N. M., S. Miyazaki, A. M. Bradley, and P. Segall (2011), Space-time correlation of
        slip and tremor during the 2009 Cascadia slow slip event, Geophys. Res. Lett., 38, L18309, 
        doi:10.1029/2011GL048714.

    Miyazaki, S., P. Segall, J. J. McGuire, T. Kato, and Y. Hatanaka (2006), Spatial and tmporal 
        evolution of stress and slip rate during the 2000 Tokai slow earthquake, J. Geophys. Res., 
        111, B03409, doi:10.1029/2004JB003426

    Segall, P., and M. Matthews (1997), Time dependent inversion of geodetic data, J. Geophys. 
        Res., 102, 22,391-22,409.

    INPUT:
    H     = the state -> observation matrix.
    R_sqrt   = chol(R), where R is the observation covariance matrix.
    y     = the vector of observations.
    x_f   = x_k|k-1 (forecasted state)
    P_f   = chol(P_k|k-1) (forecasted covariance)

    OUTPUT:
    x_a   = x_k|k (analyzed state)
    P_a   = P_k|k (analyzed covariance)
    z     = y - H*x_f
    Y_sqrt   = chol(H P_k|k-1 H' + R) (if get_Y_sqrt=True).

    NOTES (from Paul Segall):
    z and Y_sqrt can be used to calculate the likelihood
        p(y_k | Y_k-1),

    where Y_k-1(:,i) is measurement vector i, as follows. First,
        p(y_k | Y_k-1) = N(y_k; H x, R + H P_k|k-1 H')
                       = N(y_k - Hx; 0, Y_sqrt*Y_sqrt')

    where N(x; mu, Sigma) is the density for a normal distribution having mean mu and covariance Sigma. 
    Second, if R = chol(A), then
        log(det(A)) = 2*sum(log(diag(R))).

    Hence,
        log p(y_k | Y_k-1) = -n/2 log(2 pi) - sum(log(diag(Y_sqrt))) - 1/2 q q',

    where q = z' / Y_sqrt (or q = (Y_sqrt' \ z)'), n = length(y_k), and we have just taken the logarithm of the 
    expression for the normal probability density function.

    For
        A = [chol(R)           0
             chol(P_k|k-1) H'  chol(P_k|k-1)],

    the Schur complement of (A'A)(1:ny, 1:ny) in A'A is
       P_k|k = P_k|k-1 - P_k|k-1 H' inv(S) H P_k|k-1,

    where
        S = H P_k|k-1 H' + R.

    The key idea in this square-root filter is that
       [~, R] = qr(A)

    is a safe operation in the presence of numerical error and so assures a
    factorization of the filtered covariance matrix P_k|k.
    """
    
    n, m = H.shape

    # Perform Cholesky factorizations
    R_sqrt   = cholesky(R,   lower=False)
    P_f_sqrt = cholesky(P_f, lower=False)
    zeros    = np.zeros((n, m))

    # Form block matrix A
    A = np.block([[R_sqrt,           zeros], 
                  [P_f_sqrt @ H.T,   P_f_sqrt]])

    # Perform QR decomposition on A
    B = qr(A, mode='r')[0]

    # Extract the part of A that is sqrt(P_a),
    # I.e., the factorization of the Schur complement of interest.
    P_a_sqrt       = B[n:n+m, n:n+m]
    mask           = np.diag(P_a_sqrt < 0)
    P_a_sqrt[mask, :] = -P_a_sqrt[mask, :]

    # Extract sqrt(Y).
    Y_sqrt          = B[:n, :n]
    mask            = np.diag(Y_sqrt < 0)
    Y_sqrt[mask, :] = -Y_sqrt[mask, :]


    # xf = xp + Pfc'*(Pfc*(H'*(Rc\(Rc'\z))));

    # Innovation
    v = y - H @ x_f

    # Filtered state
    tmp1 = solve_triangular(R_sqrt.T, v, lower=True)
    tmp2 = solve_triangular(R_sqrt, tmp1, lower=False)
    tmp3 = H.T @ tmp2
    tmp4 = P_a_sqrt @ tmp3
    x_a  = x_f + P_a_sqrt.T @ tmp4
    # x_a  = x_f + P_a_sqrt.T @ P_a_sqrt @ H.T @ tmp2

    # Get full P_a
    P_a = P_a_sqrt.T @ P_a_sqrt

    # Compute likelihood terms
    # Solve for inv(Y) @ v
    tmp1 = solve_triangular(Y_sqrt, v, lower=True)       # Solve sqrt(Y) @ tmp1 = v where tmp1 = L.T @ inv(Y) @ v
    tmp2 = solve_triangular(Y_sqrt.T, tmp1, lower=False) # Solve sqrt(Y).T @ tmp2 = Y where tmp2 = inv(Y) @ v
    
    # Get k-th contribution to log-likelihood
    r     = v.T @ tmp2                      # compute v.T @ inv(Y) @ v 
    det_Y = np.sum(np.log(np.diag(Y_sqrt))) # compute det(Y) (sum of diagonal elements since Y is lower triangular)
    
    return x_a, P_a, r, det_Y


def read_kalman_filter_results(file_name):
    """
    Read Kalman filter results file and add to KalmanFilterResults object
    """

    with h5py.File(f'{file_name}', 'r') as file:

        for key in file.keys():
            print(key)
        backward_smoothing = file['backward_smoothing'][()]

        if backward_smoothing:
            # model = file['model'][()][::-1]  
            # resid = file['resid'][()][::-1]  
            # rms   = file['rms'][()][::-1]    
            # x_s = file['x_s'][()][::-1] 
            # P_s = file['P_s'][()][::-1] 
            model = file['model'][()]  
            resid = file['resid'][()]  
            rms   = file['rms'][()]    
            x_s = file['x_s'][()] 
            P_s = file['P_s'][()] 
            return KalmanFilterResults(model, resid, rms, x_s=x_s, P_s=P_s, backward_smoothing=backward_smoothing)

        else:
            model = file['model'][()] 
            resid = file['resid'][()] 
            rms   = file['rms'][()]   
            x_f = file['x_f'][()] 
            x_a = file['x_a'][()] 
            P_f = file['P_f'][()] 
            P_a = file['P_a'][()] 
            return KalmanFilterResults(model, resid, rms, x_f=x_f, x_a=x_a, P_f=P_f, P_a=P_a, backward_smoothing=backward_smoothing)


def write_kalman_filter_results(model, resid, rms, x_f=[], x_a=[], P_f=[], P_a=[], x_s=[], P_s=[], backward_smoothing=False, file_name='results.h5', overwrite=True):
    """
    Write Kalman filter results to HDF5 file.
    """

    if os.path.exists(file_name):
        if overwrite:
            os.remove(file_name)
        else:
            print(f'Error! {file_name} already exists.')
            print(f'Set "overwrite=True" to replace')
            sys.exit(1)

    with h5py.File(f'{file_name}', 'w') as file:

        file.create_dataset('model', data=model)
        file.create_dataset('resid', data=resid)
        file.create_dataset('rms',   data=rms)
        file.create_dataset('backward_smoothing', data=backward_smoothing)

        if backward_smoothing:
            file.create_dataset('x_s', data=x_s)
            file.create_dataset('P_s', data=P_s)
        else:
            print('Saving forecast...')
            file.create_dataset('x_f', data=x_f)
            file.create_dataset('P_f', data=P_f)
            del P_f
            gc.collect()

    if not backward_smoothing:
        with h5py.File(f'{file_name}', 'a') as file:
            print('Saving analysis...')
            file.create_dataset('x_a', data=x_a)
            file.create_dataset('P_a', data=P_a)

    return


def write_kalman_filter_results_old(results, file_name='results.h5', overwrite=True, backward_smoothing=False):
    """
    Write Kalman filter results to HDF5 file.
    """

    if os.path.exists(file_name):
        if overwrite:
            os.remove(file_name)
        else:
            print(f'Error! {file_name} already exists.')
            print(f'Set "overwrite=True" to replace')
            sys.exit(1)

    with h5py.File(f'{file_name}', 'w') as file:

        file.create_dataset('model', data=results.model)
        file.create_dataset('resid', data=results.resid)
        file.create_dataset('rms',   data=results.rms)
        file.create_dataset('backward_smoothing', data=results.backward_smoothing)

        if backward_smoothing:
            file.create_dataset('x_s', data=results.x_s)
            file.create_dataset('P_s', data=results.P_s)
        else:
            file.create_dataset('x_f', data=results.x_f)
            file.create_dataset('P_f', data=results.P_f)

    if not backward_smoothing:
        with h5py.File(f'{file_name}', 'a') as file:
            file.create_dataset('x_a', data=results.x_a)
            file.create_dataset('P_a', data=results.P_a)

    return


def backward_smoothing_old(x_f, x_a, P_f, P_a, d, dt, G, L, T, constrain=False, state_lim=[], cost_function='state'):
    """
    Dimensions:
    n_obs  - # of time points
    n_dim  - # of model parameters
    n_data - # of data points

    INPUT:
    x_f, x_a (n_obs, n_dim)         - forecasted and analyzed state vectors
    P_f, P_a (n_obs, n_dim, n_dim)  - forecasted and analyzed covariance matrices 
    T        (n_dim, n_dim)         - state transition matrix: maps state x_k to next state x_k-1

    OUTPUT:
    """

    start_total = time.time()

    # ---------- Kalman Filter ---------- 
    # Get lengths
    n_obs   = x_f.shape[0]
    n_dim   = x_f.shape[1]
    n_patch = n_dim//3
    n_data  = d.shape[1] - n_dim

    # Initialize
    x_s     = np.empty((n_obs, n_dim))        # analyzed states
    P_s     = np.empty((n_obs, n_dim, n_dim)) # analyzed covariances
    S       = np.empty((n_obs - 1, n_dim, n_dim)) # analyzed covariances
    model   = np.empty((n_obs, n_data))
    resid   = np.empty((n_obs, n_data))
    rms     = np.empty((n_obs,))

    # # Define cost function for optimization
    # if cost_function == 'joint':
    #     # cost_function = lambda x: np.linalg.norm(d[k, :] - H @ x, ord=2) + np.linalg.norm(P_a[k, :, :]**-1 * (x - x_a[k, :]), ord=2)
    #     cost_function = lambda x: np.linalg.norm(d[k, :] - H @ x, ord=2) + np.linalg.norm(x - x_a[k, :], ord=2)
    # else:

    print(f'\n##### Running backward smoothing ##### ')

    x_s[-1, :]    = x_a[-1, :]
    P_s[-1, :, :] = P_a[-1, :, :]

    for k in reversed(range(0, n_obs - 1)):
        
        # Compute smoothing matrix S
        P_f_inv = np.linalg.pinv(P_f[k + 1, :, :], hermitian=True)
        S[k, :, :] = P_a[k, :, :] @ T.T @ P_f_inv

        # Get smoothed states and covariances
        x_s[k, :]    = x_a[k, :] + S[k, :, :] @ (x_s[k + 1, :] - x_f[k + 1, :])
        P_s[k, :, :] = P_a[k, :] + S[k, :, :] @ (P_s[k + 1, :, :] - P_f[k + 1, :, :]) @ S[k, :, :].T
        
         # Constrain state estimate 
        start_opt = time.time()
        print(f'State range: {x_a[k, :].min():.2f} - {x_a[k, :].max():.2f}')
        if len(state_lim) == n_dim:

            bounds_flag = False
            
            # Define initial guess as unconstrained state vector with bounds enforced on parameters outside of range
            x_0 = np.zeros_like(x_s[k, :])

            for i in range(len(x_0)):
                # Enforce parameter bounds
                if x_s[k, i] < state_lim[i][0]:
                    x_0[i] = state_lim[i][0]
                    bounds_flag = True

                elif x_s[k, i] > state_lim[i][1]:
                    x_0[i] = state_lim[i][1]
                    bounds_flag = True
                
                else:
                    x_0[i] = x_s[k, i]
                
            # If value is less than the previous one, set to be previous one
            for i in range(len(x_0)):
                if k < len(n_obs):
                    if backward_smoothing:
                        if x_a[k, i] > x_s[k - 1, i]:
                            x_0[i] = x_s[k - 1, i]
                    elif x_a[k, i] < x_s[k - 1, i]:
                            x_0[i] = x_s[k - 1, i]

            # If at least one value falls outside of the supplied bounds, perform optimization
            if bounds_flag:
                print('Constraining...')
                # Define monotonic increase constraint (or decrease if in backward smoothing mode)
                if k != 0:
                    if backward_smoothing:
                        # State at k should be less than at state k - 1
                        constraints = (
                                        {'type': 'ineq', 'fun': lambda x: x_s[k - 1, n_patch:2*n_patch] - x[n_patch:2*n_patch]}, # W_k-1 - W_k >= 0
                                        )
                    else:
                        # State at k should be greater than at state k - 1
                        constraints = (
                                        {'type': 'ineq', 'fun': lambda x: x[n_patch:2*n_patch] - x_s[k - 1, n_patch:2*n_patch]}, # W_k - W_k-1 >= 0
                                        )
                else:
                    # There is no reference state if at k = 0
                    constraints = ()

                # Define cost function for optimization
                # if cost_function == 'joint':
                #     # cost_function = lambda x: np.linalg.norm(d[k, :] - H @ x, ord=2) + np.linalg.norm(P_a[k, :, :]**-1 * (x - x_a[k, :]), ord=2)
                #     cost_function = lambda x: np.linalg.norm(d[k, :] - H @ x, ord=2) + np.linalg.norm(x - x_a[k, :], ord=2)
                # else:
                cost_function = lambda x: np.linalg.norm(x - x_s[k, :], ord=2)
                # cost_function = lambda x: np.linalg.norm(d[k, :] - H @ x, ord=2) + np.linalg.norm(x - x_s[k, :], ord=2)

                # # Perform minimization
                if constrain:
                    results = minimize(cost_function, x_0, bounds=state_lim, method='COBYLA', constraints=constraints, tol=1e-6)
                    end_opt = time.time() - start_opt

                    x_s[k, :] = results.x

                    print(f'Constraint time: {end_opt:.2f} s')

                else:
                    end_opt = 0
                    x_s[k, :] = x_0

                print(f'Updated state range: {x_s[k, :].min():.2f} - {x_s[k, :].max():.2f}')
                
            else:
                print(f'Step {k} state is within bounds')
                end_opt = 0


    for k in range(n_obs):
        H = make_observation_matrix(G, L, t=dt*k)

        # Compute error terms
        pred        = H @ x_s[k, :]
        model[k, :] = pred[:n_data]
        resid[k, :] = (d[k, :] - pred)[:n_data]
        rms[k]      = np.sqrt(np.mean(resid[k, :]**2))  


    end_total = time.time() - start_total
    print('##### Backward smoothing complete #####')
    print(f'Elapsed time: {end_total/60:.1f} min')
    print()

    return KalmanFilterResults(model, resid, rms, x_f=x_f, x_a=x_a, P_f=P_f, P_a=P_a, x_s=x_a, P_s=P_a, backward_smoothing=True)
    # return   


def make_observation_matrix(G, R, t=1, steady_slip=False, ramp_matrix=[]):
    """
    Form observation from Greens function matrix G and smoothing matrix R.

    INPUT:
    G (n_obs, n_patch)   - Green's function matrix
    R (n_patch, n_patch) - smoothing matrix (i.e. nearest-neighbor or Laplacian)

    Optional:
    t                           - time step increment
    steady_slip                 - True include steady-slip rate in inversion
    ramp_matrix (n_obs, n_ramp) - design matrix for linear (n_ramp = 3) or quadtratic (n_ramp = 9) ramp function.
    """

    zeros_G = np.zeros_like(G)
    zeros_R = np.zeros_like(R)

    if steady_slip:
        # Form rows
        data         = np.hstack((  G * t,       G, zeros_G))
        v_smooth     = np.hstack((      R, zeros_R, zeros_R))
        W_smooth     = np.hstack((zeros_R,       R, zeros_R))
        W_dot_smooth = np.hstack((zeros_R, zeros_R,       R))
        H            = np.vstack((data, v_smooth, W_smooth, W_dot_smooth))
   
    else:
        # Form rows
        # data         = np.hstack((      G, zeros_G))
        # W_smooth     = np.hstack((      R, zeros_R))
        # W_dot_smooth = np.hstack((zeros_R,       R))
        # H            = np.vstack((data, W_smooth, W_dot_smooth))

        H = np.block([[G, zeros_G], 
                      [R, zeros_R], 
                      [zeros_R, R]])
    
        # H = np.block([[G], 
                    #   [R],])
        
    # Add columns for ramp, if specified 
    if len(ramp_matrix) > 0:
        A = np.zeros((H.shape[0], ramp_matrix.shape[1]))
        A[:ramp_matrix.shape[0], :ramp_matrix.shape[1]] = ramp_matrix
        H = np.hstack((H, A))

    return H


def make_transition_matrix(n_patch, dt, steady_slip=False, ramp_matrix=[], plot=False):
    """
    Form transtion matrix from Greens function matrix G and smoothing matrix R.
    """

    # Determine number of physical parameters
    if steady_slip:
        n_dim = 3 * n_patch
    else:
        n_dim = 2 * n_patch

    # Determine number of ramp coefficients
    n_ramp = 0
    if len(ramp_matrix) > 0:
        n_ramp += ramp_matrix.shape[1]
        n_dim  += n_ramp

    # Form T matrix
    T = np.eye(n_dim)
    T[n_dim - 2*n_patch - n_ramp:n_dim - n_patch - n_ramp, n_dim - n_patch - n_ramp:n_dim - n_ramp] = np.eye(n_patch) * dt # add slip rate term
    # print(f'({n_dim - 2*n_patch - n_ramp}:{n_dim - n_patch - n_ramp} {n_dim - n_patch - n_ramp}:{n_dim - n_ramp})')

    # Set bottom right block to zero if including ramp (i.e. no correlation between ramp at k and k+1)
    if len(ramp_matrix) > 0:
        T[-n_ramp:, -n_ramp] = 0

    # # Form base matrices
    # I     = np.eye(n_patch)
    # zeros = np.zeros((n_patch, n_patch))
    # # Form rows
    # v     = np.hstack((    I,  zeros,  zeros))
    # W     = np.hstack((zeros,      I, I * dt))
    # W_dot = np.hstack((zeros,  zeros,      I))
    # T     = np.vstack((v, W, W_dot))

    if plot:
        fig, ax = plt.subplots(figsize=(14, 8.2))
        ax.imshow(T, cmap=cmc.lajolla_r, vmin=0, vmax=0.06)
        plt.show()
        sys.exit()
    return T


def make_prediction_covariance_matrix(n_patch, state_sigmas, steady_slip=False, ramp_matrix=[]):
    """
    Form initial prediction covariance matrix P_0 based on uncertainties on model parameters.
    state_sigmas have length n_dim and should correspond to uncertainties on v and/or W and W_dot.
    v_sigma, W_sigma, W_dot_sigma should have units of v, W, W_dot (mm/yr and mm, typically).
    """
    if len(ramp_matrix) > 0:
        n_ramp = ramp_matrix.shape[1]
    else:
        n_ramp = 0

    # n_param = len(state_sigmas)
    n_param = 2 + steady_slip
    P_0     = np.eye(n_param * n_patch + n_ramp)
    
    # Add fault parameter constraints
    for i in range(n_patch):
        for j in range(n_param):
            P_0[i + j*n_patch, i + j*n_patch] *= state_sigmas[j]

    if n_ramp > 0:
        P_0[-n_ramp:, -n_ramp:] *= state_sigmas[-1]

    # # Form base matrices
    # I     = np.eye(n_patch)
    # zeros = np.zeros((n_patch, n_patch))
    # # Form rows
    # v     = np.hstack((v_sigma * I,  zeros,                 zeros))
    # W     = np.hstack((zeros,        W_sigma * I,           zeros))
    # W_dot = np.hstack((zeros,        zeros,       W_dot_sigma * I))
    # P_0     = np.vstack((v, W, W_dot))

    return P_0


def make_process_covariance_matrix(n_patch, dt, omega, steady_slip=True, slip_rate=True, ramp_matrix=[], rho=1):
    """
    Form process covariance matrix Q from number of fault elements n_patch, time-step dt, and temporal smoothing parameter omega.
    """

    # Form base matrices
    I     = np.eye(n_patch)
    # zeros = np.zeros((n_patch, n_patch))

    # # Form rows 
    # v     = np.hstack((zeros,                      zeros,                      zeros))
    # W     = np.hstack((zeros, (omega**2) * (dt**3)/3 * I, (omega**2) * (dt**2)/2 * I))
    # W_dot = np.hstack((zeros, (omega**2) * (dt**2)/2 * I,        (omega**2) * dt * I))
    # Q     = np.vstack((v, W, W_dot))

    # Form W/W_dot submatrix
    # W     = np.hstack(((omega**2) * (dt**3)/3 * I, (omega**2) * (dt**2)/2 * I))
    # W_dot = np.hstack(((omega**2) * (dt**2)/2 * I,        (omega**2) * dt * I))
    # Q_sub = np.vstack((W, W_dot))

    # W     = np.hstack(((omega**2) * (dt**3)/3 * I, (omega**2) * (dt**2)/2 * I))
    # W_dot = np.hstack(((omega**2) * (dt**2)/2 * I,        (omega**2) * dt * I))
    Q_sub = np.block([[(omega**2) * (dt**3)/3 * I, (omega**2) * (dt**2)/2 * I],
                      [(omega**2) * (dt**2)/2 * I,        (omega**2) * dt * I]])

    # Form complete Q matrix
    if steady_slip:
        Q = np.zeros((3*n_patch, 3*n_patch))
        Q[-2*n_patch:, -2*n_patch:] = Q_sub

    else:
        Q = Q_sub
    
    # Add ramp term
    if len(ramp_matrix) > 0:
        n_ramp = ramp_matrix.shape[1]
        Q_ramp = np.zeros((Q.shape[0] + n_ramp, Q.shape[1] + n_ramp))
        Q_ramp[:-n_ramp, :-n_ramp] = Q
        Q_ramp[-n_ramp:, -n_ramp:] = rho**2 * np.eye(n_ramp)

        Q = Q_ramp

    return Q


def make_data_covariance_matrix(C, n_patch, sigma, kappa, steady_slip=False, tol=1e-20, plot=False):
    """
    Form data covariance matrix R from observation covariance matrix C, data covariance weight sigma, and spatial smoothing weight kappa.
    """

    if steady_slip:
        n_dim = 3 * n_patch
    else:
        n_dim = 2 * n_patch

    n_data = len(C)

    # Form base matrices
    I = np.eye(n_patch)

    # Form R matrix
    R = np.eye((n_data + n_dim))
    R[:n_data, :n_data]      = sigma**2 * C
    R[n_data:, n_data:]     *= kappa**2
    # R[-n_patch:, -n_patch:] *= 100
    
    # Force small values to zero to ensure positive definite
    R[np.abs(R) < tol] = 0

    if plot:
        fig, ax = plt.subplots(figsize=(14, 8.2))
        ax.imshow(R, cmap=cmc.lajolla_r, vmax=1)
        plt.show()

    # zeros_I = np.zeros_like(I)
    # zeros_CI = np.zeros((C.shape[0], n_patch))
    # zeros_IC = np.zeros((n_patch, C.shape[1]))
    # # Form rows
    # d     = np.hstack((sigma**2 * C,     zeros_CI,     zeros_CI,     zeros_CI))
    # v     = np.hstack((    zeros_IC, kappa**2 * I,      zeros_I,      zeros_I))
    # W     = np.hstack((    zeros_IC,      zeros_I, kappa**2 * I,      zeros_I))
    # W_dot = np.hstack((    zeros_IC,      zeros_I,      zeros_I, kappa**2 * I))
    # R     = np.vstack((d, v, W, W_dot))
    return R


def exponential_covariance(h, a, b, file_name='', show=False, units='km'):

    """
    Compute exponential covariance function as described by Wackernagel (2003).

    INPUT:
    h - distance
    a - drop-off scaling parameter
    b - amplitude scaling parameter
    """
    
    C_exp = lambda h, a, b: b * np.exp(-np.abs(h)/a)

    # Plot
    if len(file_name) > 0:
        if h.size > 1:

            h_rng = np.linspace(0, np.max(h), 100)
            C_rng = C_exp(h_rng, a, b)

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.vlines(x=a, ymin=0, ymax=b, linestyle='--')
            ax.plot(h_rng, C_rng, linewidth=2, c='k')
            ax.set_xlabel('Distance (km)')
            ax.set_ylabel('Covariance')

            fig.savefig(file_name, dpi=300)

            if show:
                plt.show()

            plt.close()

    return C_exp(h, a, b)


def get_distances(x, y):
    """
    Compute Euclidian distance between points defined by (x, y)
    """
    r = np.empty((x.size, x.size))
    for i in range(x.size):
        for j in range(x.size):
            r[i, j] = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
    return r
    

def get_state_constraints(fault, bounds, ramp_matrix=[]):
    """
    Map supplied parameter bounds to the dimensions of the state vector.
    Typically bounds will consist of [v_lim, W_lim, W_dot_lim].
    Note that len(fault.patches) * len(bounds) should equal len(x) (or, n_dim).
    """
    # Get number of ramp parameters
    n_ramp = 0 
    use_ramp = len(ramp_matrix) > 0

    if use_ramp:
        n_ramp += ramp_matrix.shape[1]

    state_lim = []

    for i in range(len(bounds) - use_ramp):
        for j in range(len(fault.patches)):
            state_lim.append(bounds[i])
    
    for k in range(n_ramp):
        state_lim.append(bounds[-1])
    return state_lim


def constrain_state(x, x_ref, state_lim, y=[], H=[], state_range=[], direction='increasing', constrain=True,):
    """
    Constrain state vector x top be within bounds specified in state_lim and mononotically 
    increasing/decreasing with respect to reference state x_ref,

    INPUT
    x           (n,)   - present state
    x_ref       (n,)   - reference state for monotonic increasing/decreasing constraint
    state_lim   (n, 2) - list of (x_min, x_max) values corresponding to state parameters,
                         must be (n, 2) to use
    state_range (2,)   - index range over which to apply constraints
    direction   (str)  - (x_ref - x) >= 0 if 'decreasing' or (x - x_ref) >= 0 if 'increasing'
    constrain   (bool) - True to enforce constraints via optimization; False to return manually  
                         constrained state x_0, which is used as the initial guess in the 
                         optimization.
    """

    # Have to be careful about loop scoping, so we define a wrapper function to generate constraint
    # functions that are properly indexed to correspond to each individual parameter
    
    def make_monotonic(j):
        if direction == 'decreasing':
            def increasing(x_new):
                return x_ref[j] - x_new[j]
            return increasing
        else:
            def increasing(x_new):
                return x_new[j] - x_ref[j]
        return increasing
    
    def make_lower_bound(j):
        def lower_bound(x_new):
            return x_new[j] - state_lim[j][0]
        return lower_bound
    
    def make_upper_bound(j):
        def upper_bound(x_new):
            return state_lim[j][1] - x_new[j]
        return upper_bound
    
    # Define cost function for optimization
    cost_function = lambda x_new: np.linalg.norm(x_new - x, ord=2)
    # cost_function = lambda x_new: np.linalg.norm(x_new - x, ord=2) + np.linalg.norm(y - H @ x, ord=2)

    # Begin optimization
    start_opt = time.time()
    n_dim = len(x)

    # Determine parameters to enforce monotonic constraints
    if len(state_range) != 2:
        state_range = [0, n_dim - 1]

    # Define initial guess as unconstrained state vector with bounds enforced on parameters outside of range
    x_0 = np.zeros_like(x)

    # Set flag for optimization -- if state is within bounds and increases monotonically, then dont perform optimization
    flag = False

    # Enforce parameter bounds on initial guess x_0
    for i in range(len(x_0)):
        if x[i] < state_lim[i][0]:
            x_0[i] = state_lim[i][0]
            flag = True
        elif x[i] > state_lim[i][1]:
            x_0[i] = state_lim[i][1]
            flag = True
        else:
            x_0[i] = x[i]

    # Enforce monotonically increasing/decreasing on initial guess x_0
    # if value is less than the previous one, set to be previous one
    for i in range(state_range[0], state_range[1]):
        if direction =='decreasing':
            if x_0[i] > x_ref[i]: 
                x_0[i] = x_ref[i]
                flag = True
        else:
            if x_0[i] < x_ref[i]: 
                x_0[i] = x_ref[i]
                flag = True

    # If at least one value falls outside of the supplied bounds, perform optimization
    if flag:
        print(f'Constraining parameters {state_range}...')
        constraints = []
        
        # Define hard bounds on state parameters
        # for j in range(n_dim):
        for j in range(len(x_0)):
            # constraints.append(make_lower_bound(j)) # x_new[j] >= a
            # constraints.append(make_upper_bound(j)) # x_new[j] <= b
            constraints.append(({'type': 'ineq', 'fun': make_lower_bound(j)})) 
            constraints.append(({'type': 'ineq', 'fun': make_upper_bound(j)})) 

        # Define monotonic increase constraint (or decrease if in backward smoothing mode)
        # if constrain:
        #     for j in range(state_range[0], state_range[1]):
        #         # (x_ref[j] - x[j]) >= 0 if decreasing
        #         # (x[j] - x_ref[j]) >= 0 if increasing
        #         # constraints.append(make_monotonic(j)) 
        #         constraints.append(({'type': 'ineq', 'fun': make_monotonic(j)})) 

        # Perform optimiztion
        print(f'Number of constraints: {len(constraints)}')
        print(f'x_0: {x_0.min():.1f} - {x_0.max():.1f}')

        results = minimize(cost_function, x_0, constraints=constraints, bounds=state_lim, method='SLSQP').x
        # results = fmin_cobyla(cost_function, x_0, constraints).x
        end_opt = time.time() - start_opt
        print(f'Constraint time: {end_opt:.2f} s')

        x_new = results

    else:
        print(f'State is within bounds and increases monotonically')
        x_new = x

    return x_new


def check_upper(A):
    return np.allclose(A, np.triu(A))


def check_lower(A):
    return np.allclose(A, np.tril(A))



# ------------------ Classes ------------------
class KalmanFilterResults:

    """
    Object containing results of a Kalman filtering procedure.

    ATTRIBUTES:
    x_f, x_a, x_s (n_obs, n_dim)        - forecasted, analyzed, and smoothed state vectors
    P_f, P_a, x_s (n_obs, n_dim, n_dim) - forecasted, analyzed, and smoothed covariance matrices
    resid (n_obs, n_data)          - model misfits at each time step
    rms (n_obs,)                   - RMS error after analysis at each time step
    """

    def __init__(self, model, resid, rms, x_f=[], x_a=[], P_f=[], P_a=[], x_s=[], P_s=[], backward_smoothing=False):
        """
        Set attributes.
        """
        
        self.backward_smoothing = backward_smoothing

        if backward_smoothing:
            # self.x_s   = x_s[::-1, :]
            # self.P_s   = P_s[::-1, :]
            # self.model = model[::-1, :]
            # self.resid = resid[::-1]
            # self.rms   = rms[::-1]
            self.x_s   = x_s
            self.P_s   = P_s
            self.model = model
            self.resid = resid
            self.rms   = rms
        else:
            self.x_f   = x_f
            self.x_a   = x_a
            self.P_f   = P_f
            self.P_a   = P_a

            self.model = model
            self.resid = resid
            self.rms   = rms

