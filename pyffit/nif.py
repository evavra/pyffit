import os
import gc
import sys
import h5py
import time
import numpy as np
import pandas as pd
import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
from scipy.optimize import minimize 
from scipy.linalg import cholesky, qr, solve_triangular

# ------------------ NIF ------------------
def network_inversion_filter(fault, G, d, std, dt, omega, sigma, kappa, state_sigmas, cov_file='covariance.h5', steady_slip=False, rho=1, ramp_matrix=[], constrain=False, state_lim=[], result_dir='.', cost_function='state'):
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
    dim_ramp = 0
    if len(ramp_matrix)> 0:
        dim_ramp += ramp_matrix.shape[1]
        n_dim    += dim_ramp

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
    model, resid, rms = kalman_filter(x_init, P_init, d, std, dt, G, L, T, Q, sigma=sigma, kappa=kappa, cov_file=cov_file, result_dir=result_dir, steady_slip=steady_slip, ramp_matrix=ramp_matrix, constrain=constrain, state_lim=state_lim, cost_function=cost_function, file_name=f'{result_dir}/results_forward.h5')
    with h5py.File(f'{result_dir}/results_forward.h5', 'r') as file:
        x_model_forward = file['x_a'][()]

    # Perform backward smoothing
    model, resid, rms, x_s, P_s = backward_smoothing(f'{result_dir}/results_forward.h5', d, dt, G, L, T, steady_slip=steady_slip, ramp_matrix=ramp_matrix, constrain=constrain, state_lim=state_lim, cost_function=cost_function)
    write_kalman_filter_results(model, resid, rms, x_s=x_s, P_s=P_s, backward_smoothing=True, file_name=f'{result_dir}/results_smoothing.h5',)
    x_model_smoothing = x_s
    gc.collect()

    # results_smoothing = backward_smoothing(results_forward.x_f, results_forward.x_a, results_forward.P_f, results_forward.P_a, d, dt, G, L, T, 
                                        #    state_lim=state_lim, cost_function='state')

    return x_model_forward, x_model_smoothing


def kalman_filter(x_init, P_init, d, std, dt, G, L, T, Q, sigma=1, kappa=1, cov_file='covariance.h5', result_dir='.', state_lim=[], ramp_matrix=[], steady_slip=False, constrain=False, cost_function='state', 
                  backward_smoothing=False, overwrite=True, file_name='results_forward.h5'):
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

    covariance = h5py.File(cov_file, 'r')

    # Determine number of ramp coefficients
    dim_ramp = 0
    if len(ramp_matrix) > 0:
        dim_ramp += ramp_matrix.shape[1]

    if steady_slip:
        n_patch = (x_init.size - dim_ramp)//3
    else:
        n_patch = (x_init.size - dim_ramp)//2

    n_obs   = d.shape[0]
    n_data  = d.shape[1] - n_dim + dim_ramp
    slip_start = steady_slip * n_patch  # Get start of transient slip
    
    x_f     = np.empty((n_obs, n_dim))        # forecasted states
    x_a     = np.empty((n_obs, n_dim))        # analyzed states
    P_f     = np.empty((n_obs, n_dim, n_dim)) # forecasted covariances
    P_a     = np.empty((n_obs, n_dim, n_dim)) # analyzed covariances
    model   = np.empty((n_obs, n_data))
    resid   = np.empty((n_obs, n_data))
    rms     = np.empty((n_obs,))

    if backward_smoothing:
        print('############### Running Kalman filter in backward mode ###############')
    else:
        print('############### Running Kalman filter ###############')

    start_total = time.time()

    for k in range(0, n_obs):
        start_step = time.time()
        print(f'\n##### Working on Step {k} #####')

        # Update observation matrix H        
        start_H = time.time()
        H       = make_observation_matrix(G, L, t=dt*k, steady_slip=steady_slip, ramp_matrix=ramp_matrix)
        end_H   = time.time() - start_H
        print(f'H-matrix time: {end_H:.2f} s')

        # Update data covariance matrix R
        # R = make_data_covariance_matrix(C, n_patch, sigma, kappa, steady_slip=steady_slip)
        
        # # Check for positive definiteness --  use next R if 
        # C = covariance[f'covariance/{k}'][()]

        # # NOrmalize?
        # C_trunc = C.max() * (C - C.min())/(C.max() - C.min())

        C = np.eye((n_data))
        var = np.mean(std, axis=0)**2
        var[var == 0] = 1
        C = np.diag(var)
        
        # fig, ax = plt.subplots(figsize=(6, 6))
        # # ax.set_title(r'$\sigma$' + f' = {sigma:.1e}, ' + r'$\kappa$' + f' = {kappa:.1e}')
        # im = ax.imshow(C, cmap=cmc.hawaii, interpolation='none')
        # plt.colorbar(im)
        # plt.savefig(f'{result_dir}/Results/Matrices/C_{k}.png')
        # plt.close()

        # sigma = 1e-4
        R = make_data_covariance_matrix(C, n_patch, sigma, kappa, steady_slip=steady_slip)

        if not np.all(np.linalg.eigvals(R) > 0):
            print('Using next R')
            # C = covariance[f'covariance/{k + 1}'][()]
            C = np.diag(std[k + 1, :]**2)

            # C = covariance[f'covariance/{k + 1}'][()]
            # C_trunc = C.max() * (C - C.min())/(C.max() - C.min())

            R = make_data_covariance_matrix(C, n_patch, sigma, kappa, steady_slip=steady_slip)

        # 1) ---------- Forecast ----------
        # Make forecast 
        start_forecast = time.time()

        x_f[k, :] = T @ x_init
        
        # Compute covariance forecast
        if k == 0:
            P_f[k, :, :] = P_init

            # Print object information
            print(f'[G] Greens function matrix:     shape = {G.shape} | {G.size:.1e} elements | {100 * np.sum(G == 0)/G.size:.2f}% sparsity) | mean = {np.mean(G)} | std = {np.std(G)}')
            print(f'[H] Observation matrix:         shape = {H.shape} | {H.size:.1e} elements | {100 * np.sum(H == 0)/H.size:.2f}% sparsity) | mean = {np.mean(H)} | std = {np.std(H)}')
            print(f'[R] Data covariance matrix:     shape = {R.shape} | {R.size:.1e} elements | {100 * np.sum(R == 0)/R.size:.2f}% sparsity) | mean = {np.mean(R)} | std = {np.std(R)}')
            print(f'[T] Transition matrix:          shape = {T.shape} | {T.size:.1e} elements | {100 * np.sum(T == 0)/T.size:.2f}% sparsity) | mean = {np.mean(T)} | std = {np.std(T)}')
            print(f'[Q] Process covariance matrix:  shape = {Q.shape} | {Q.size:.1e} elements | {100 * np.sum(Q == 0)/Q.size:.2f}% sparsity) | mean = {np.mean(Q)} | std = {np.std(Q)}')

        else:
            P_f[k, :, :] = T @ P_a[k - 1, :, :] @ T.T + Q 
        
        end_forecast = time.time() - start_forecast
        print(f'Forecast time: {end_forecast:.2f} s')


        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(r'$\sigma$' + f' = {sigma:.1e}, ' + r'$\kappa$' + f' = {kappa:.1e}')
        im = ax.imshow(R, cmap=cmc.lajolla, interpolation='none')
        plt.colorbar(im)
        plt.savefig(f'{result_dir}/Results/Matrices/R_{k}.png')
        plt.close()

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(r'$\sigma$' + f' = {sigma:.1e}, ' + r'$\kappa$' + f' = {kappa:.1e}')
        im = ax.imshow(H @ P_f[k, :, :] @ H.T, cmap=cmc.imola, interpolation='none')
        plt.colorbar(im)
        plt.savefig(f'{result_dir}/Results/Matrices/HPHt_{k}.png')
        plt.close()


        # 2) ---------- Analysis ----------
        start_analysis = time.time()

        # # Get Kalman gain
        # HPfH = H @ P_f[k, :, :] @ H.T 
        # HPfH_R = HPfH + R
        # HPfHT_inv = np.linalg.pinv(HPfH_R, hermitian=True, rcond=rcond)

        # # Get Kalman gain
        # K = P_f[k, :, :] @ H.T @ HPfHT_inv 

        # # Update state and covariance        
        # x_a[k, :]      = x_f[k, :] + K @ (d[k, :] - H @ x_f[k, :])
        # P_a[k, :, :]   = P_f[k, :, :] - K @ H @ P_f[k, :, :]
        x_a[k, :], P_a[k, :, :], z = sqrt_update(H, R, d[k, :], x_f[k, :], P_f[k, :, :])
        end_analysis = time.time() - start_analysis
        print(f'Analysis time: {end_forecast:.2f} s')
    
        start_plot = time.time()
        vlim = np.max(abs(P_a[k, :, :]))
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        im = axes[0].imshow(P_f[k, :, :], vmin=-vlim, vmax=vlim, cmap=cmc.vik, interpolation='none')
        im = axes[1].imshow(P_a[k, :, :], vmin=-vlim, vmax=vlim, cmap=cmc.vik, interpolation='none')
        axes[0].set_title(r'$P_f$')
        axes[1].set_title(r'$P_a$')
        fig.colorbar(im)
        plt.savefig(f'{result_dir}/Results/Matrices/P_{k}.png', dpi=200)
        plt.close()
        end_plot = time.time() - start_plot
        print(f'Ploting time: {end_plot:.2f}')

        # Constrain state estimate 
        start_opt = time.time()
        end_opt = 0

        print(f'Forecasted state range: {x_f[k, :].min():.2f} - {x_f[k, :].max():.2f}')
        print(f'Updated state range:    {x_a[k, :].min():.2f} - {x_a[k, :].max():.2f}')
        
        if len(state_lim) == n_dim:

            bounds_flag = False
            
            # Define initial guess as unconstrained state vector with bounds enforced on parameters outside of range
            x_0 = np.zeros_like(x_a[k, :])

            for i in range(len(x_0)):
                # Enforce parameter bounds
                if x_a[k, i] < state_lim[i][0]:
                    x_0[i] = state_lim[i][0]
                    bounds_flag = True

                elif x_a[k, i] > state_lim[i][1]:
                    x_0[i] = state_lim[i][1]
                    bounds_flag = True
                
                else:
                    x_0[i] = x_a[k, i]
                
            # If value is less than the previous one, set to be previous one
            for i in range(len(x_0)):
                if k > 0:
                    if backward_smoothing:
                        if x_a[k, i] > x_a[k - 1, i]:
                            x_0[i] = x_a[k - 1, i]
                    elif x_a[k, i] < x_a[k - 1, i]:
                            x_0[i] = x_a[k - 1, i]

            # If at least one value falls outside of the supplied bounds, perform optimization
            if bounds_flag:

                print('Constraining...')
                # Define monotonic increase constraint (or decrease if in backward smoothing mode)
                if k != 0:
                    if backward_smoothing:
                        # State at k should be less than at state k - 1
                        constraints = (
                                        {'type': 'ineq', 'fun': lambda x: x_a[k - 1, slip_start:slip_start + n_patch] - x[slip_start:slip_start + n_patch]}, # W_k-1 - W_k >= 0
                                        )
                    else:
                        # # State at k should be greater than at state k - 1
                        # constraints = (
                        #                 {'type': 'ineq', 'fun': lambda x: x[slip_start:slip_start + n_patch] - x_a[k - 1, slip_start:slip_start +n_patch]}, # W_k - W_k-1 >= 0
                        #                 )

                        # Supply element-wise constraints on transient slip
                        constraints = []

                        for j in range(slip_start, slip_start + n_patch):
                            def monotonic(x):
                                return x[j] - x_a[k - 1, j]

                        constraints.append({'type': 'ineq', 'fun': lambda x: monotonic(x)}) # W[k, j] - W[k - 1, j]  >= 0

                else:
                    # There is no reference state if at k = 0
                    constraints = ()

                # Define cost function for optimization
                # if cost_function == 'joint':
                #     # cost_function = lambda x: np.linalg.norm(d[k, :] - H @ x, ord=2) + np.linalg.norm(P_a[k, :, :]**-1 * (x - x_a[k, :]), ord=2)
                #     cost_function = lambda x: np.linalg.norm(d[k, :] - H @ x, ord=2) + np.linalg.norm(x - x_a[k, :], ord=2)
                # else:

                cost_function = lambda x: np.linalg.norm(x - x_a[k, :], ord=2)
                # cost_function = lambda x: np.linalg.norm(d[k, :] - H @ x, ord=2) + np.linalg.norm(x - x_a[k, :], ord=2)

                # # Perform minimization
                if constrain:
                    results = minimize(cost_function, x_0, bounds=state_lim, method='COBYLA', constraints=constraints, tol=1e-8)
                    end_opt = time.time() - start_opt

                    x_a[k, :] = results.x

                    print(f'Constraint time: {end_opt:.2f} s')

                else:
                    end_opt = 0
                    x_a[k, :] = x_0

                print(f'Updated state range: {x_a[k, :].min():.2f} - {x_a[k, :].max():.2f}')
                
            else:
                print(f'Step {k} state is within bounds')
                end_opt = 0

        # Use current analysis initialize next forecast
        x_init = x_a[k, :] 

        # Compute error terms
        pred        = H @ x_a[k, :]
        model[k, :] = pred[:n_data]
        resid[k, :] = (d[k, :] - pred)[:n_data]
        rms[k]      = np.sqrt(np.mean(resid[k, :]**2))  

        end_step = time.time() - start_step
        print(f'Step {k} time: {end_step:.2f} s (update: {end_analysis/end_step * 100:.1f} %, opt: {end_opt/end_step * 100:.1f} %, other: {(end_step - end_opt)/end_step * 100:.1f} %)')

    covariance.close()
    end_total = time.time() - start_total

    if backward_smoothing:
        print('\n##### Backward smoothing complete #####')
    else:
        print('\n##### Kalman filter complete #####')
    print(f'Elapsed time: {end_total/60:.1f} min')
    print()

    # return KalmanFilterResults(model, resid, rms, x_f=x_f, x_a=x_a, P_f=P_f, P_a=P_a, x_s=x_a, P_s=P_a, backward_smoothing=backward_smoothing)

    # Save results
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
            file.create_dataset('x_s', data=x_a)
            file.create_dataset('P_s', data=P_a)
            del P_a
            gc.collect()

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
            del P_a
            gc.collect()


    # if backward_smoothing:
    #     return model, resid, rms, x_a, P_a
    # else:
        # return model, resid, rms, x_f, x_a, P_f, P_a
    return model, resid, rms


def sqrt_update(H, R, y, x_f, P_f, get_S_c=False):
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
    R_c   = chol(R), where R is the observation covariance matrix.
    y     = the vector of observations.
    x_f   = x_k|k-1 (forecasted state)
    P_f   = chol(P_k|k-1) (forecasted covariance)

    OUTPUT:
    x_a   = x_k|k (analyzed state)
    P_a   = P_k|k (analyzed covariance)
    z     = y - H*x_f
    S_c   = chol(H P_k|k-1 H' + R) (if get_S_c=True).

    NOTES (from Paul Segall):
    z and S_c can be used to calculate the likelihood
        p(y_k | Y_k-1),

    where Y_k-1(:,i) is measurement vector i, as follows. First,
        p(y_k | Y_k-1) = N(y_k; H x, R + H P_k|k-1 H')
                       = N(y_k - Hx; 0, S_c*S_c')

    where N(x; mu, Sigma) is the density for a normal distribution having mean mu and covariance Sigma. 
    Second, if R = chol(A), then
        log(det(A)) = 2*sum(log(diag(R))).

    Hence,
        log p(y_k | Y_k-1) = -n/2 log(2 pi) - sum(log(diag(S_c))) - 1/2 q q',

    where q = z' / S_c (or q = (S_c' \ z)'), n = length(y_k), and we have just taken the logarithm of the 
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
    R_c   = cholesky(R, lower=False)
    P_f_c = cholesky(P_f, lower=False)
    zeros = np.zeros((n, m))

    # Form block matrix A
    A = np.block([[R_c, zeros], 
                  [P_f_c @ H.T ,   P_f_c]])

    # Perform QR decomposition on A
    P_a_c = qr(A, mode='r')[0]

    # Extract the part of chol(A) that is chol(P_k|k),
    # I.e., the factorization of the Schur complement of interest.
    P_a_c          = P_a_c[n:n+m, n:n+m]
    mask           = np.diag(P_a_c < 0)
    P_a_c[mask, :] = -P_a_c[mask, :]

    # Innovation
    z = y - H @ x_f

    # Filtered state
    tmp1 = solve_triangular(R_c, z, lower=True)
    tmp2 = solve_triangular(R_c.T, tmp1, lower=False)
    tmp3 = H.T @ tmp2
    tmp4 = P_a_c @ tmp3
    x_a  = x_f + P_a_c.T @ tmp4

    # Get full P_c
    P_a = P_a_c.T @ P_a_c

    if get_S_c:
        # Extract chol(S).
        S_c          = P_a_c[:n, :n]
        mask         = np.diag(S_c < 0)
        S_c[mask, :] = -S_c[mask, :]
        return x_a, P_a, z, S_c
    
    else:
        return x_a, P_a, z


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
    dim_ramp = 0
    if len(ramp_matrix) > 0:
        dim_ramp += ramp_matrix.shape[1]

    if steady_slip:
        n_patch = (n_dim - dim_ramp)//3
    else:
        n_patch = (n_dim - dim_ramp)//2

    n_data  = d.shape[1] - n_dim + dim_ramp 
    slip_start = steady_slip * n_patch # Get start of transient slip
    
    # Initialize
    x_s     = np.empty((n_obs, n_dim))        # analyzed states
    P_s     = np.empty((n_obs, n_dim, n_dim)) # analyzed covariances
    S       = np.empty((n_obs - 1, n_dim, n_dim)) # analyzed covariances
    model   = np.empty((n_obs, n_data))
    resid   = np.empty((n_obs, n_data))
    rms     = np.empty((n_obs,))

    print(f'\n##### Running backward smoothing ##### ')

    # Set last epoch to be equal to forward filtering result
    x_s[-1, :]    = x_a[-1, :]
    P_s[-1, :, :] = file['P_a'][-1, :, :]

    for k in reversed(range(0, n_obs - 1)):
        print(f'\n##### Working on Step {k} #####')
        
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
        P_f_inv = np.linalg.pinv(file['P_f'][k + 1, :, :], hermitian=True, rcond=rcond)
        S[k, :, :] = file['P_a'][k, :, :] @ T.T @ P_f_inv

        # Get smoothed states and covariances
        x_s[k, :]    =         x_a[k, :] + S[k, :, :] @ (x_s[k + 1, :]    - x_f[k + 1, :])
        P_s[k, :, :] = file['P_a'][k, :] + S[k, :, :] @ (P_s[k + 1, :, :] - file['P_f'][k + 1, :, :]) @ S[k, :, :].T
        
         # Constrain state estimate 
        start_opt = time.time()
        print(f'State range: {x_s[k, :].min():.2f} - {x_s[k, :].max():.2f}')
        
        if len(state_lim) == n_dim:
            
            H = make_observation_matrix(G, L, t=dt*k, steady_slip=steady_slip, ramp_matrix=ramp_matrix)

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
                if x_0[i] > x_s[k + 1, i]:
                    x_0[i] = x_s[k + 1, i]
    
            # If at least one value falls outside of the supplied bounds, perform optimization
            if bounds_flag:
                print('Constraining...')
                # Define monotonic increase constraint (or decrease if in backward smoothing mode)
                if (k < n_obs - 1):
                    # State at k should be less than at state k - 1
                    # constraints = (
                    #                 {'type': 'ineq', 'fun': lambda x: x_s[k + 1, n_patch:2*n_patch] - x[n_patch:2*n_patch]}, # W_k+1 - W_k >= 0
                    #                 )
                    
                    # Supply element-wise constraints on transient slip
                    constraints = []

                    for j in range(slip_start, slip_start + n_patch):
                        def monotonic(x):
                            return x_s[k + 1, j] - x[j]

                    constraints.append({'type': 'ineq', 'fun': lambda x: monotonic(x)}) # W[k+1, j] - W[k, j] >= 0

                else:
                    # There is no reference state if at k = n_obs - 1
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
        # print(k)
        
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
    
    # Add columns for ramp, if specified 
    if len(ramp_matrix) > 0:
        A = np.zeros((H.shape[0], ramp_matrix.shape[1]))
        A[:ramp_matrix.shape[0], :ramp_matrix.shape[1]] = ramp_matrix
        H = np.hstack((H, A))

    return H


def make_transition_matrix(n_patch, dt, steady_slip=False, ramp_matrix=[]):
    """
    Form transtion matrix from Greens function matrix G and smoothing matrix R.
    """
    # Determine number of physical parameters
    if steady_slip:
        n_dim = 3 * n_patch
    else:
        n_dim = 2 * n_patch

    # Determine number of ramp coefficients
    dim_ramp = 0
    if len(ramp_matrix)> 0:
        dim_ramp += ramp_matrix.shape[1]
        n_dim    += dim_ramp

    # Form T matrix
    T = np.eye(n_dim)
    T[-2*n_patch - dim_ramp:-n_patch - dim_ramp, -n_patch - dim_ramp: -dim_ramp] = np.eye(n_patch) * dt # add slip rate term

    # Set bottom right block to zero if including ramp (i.e. no correlation between ramp at k and k+1)
    if len(ramp_matrix) > 0:
        T[-dim_ramp:, -dim_ramp] = 0

    # # Form base matrices
    # I     = np.eye(n_patch)
    # zeros = np.zeros((n_patch, n_patch))
    # # Form rows
    # v     = np.hstack((    I,  zeros,  zeros))
    # W     = np.hstack((zeros,      I, I * dt))
    # W_dot = np.hstack((zeros,  zeros,      I))
    # T     = np.vstack((v, W, W_dot))

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


def make_process_covariance_matrix(n_patch, dt, omega, steady_slip=True, ramp_matrix=[], rho=1):
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


def make_data_covariance_matrix(C, n_patch, sigma, kappa, steady_slip=False, tol=1e-20):
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
    R[:n_data, :n_data]  = sigma**2 * C
    R[n_data:, n_data:] *= kappa**2

    # Force small values to zero to ensure positive definite
    R[np.abs(R) < tol] = 0

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
    dim_ramp = 0 
    use_ramp = len(ramp_matrix) > 0

    if use_ramp:
        dim_ramp += ramp_matrix.shape[1]

    state_lim = []

    for i in range(len(bounds) - use_ramp):
        for j in range(len(fault.patches)):
            state_lim.append(bounds[i])
    
    for k in range(dim_ramp):
        state_lim.append(bounds[-1])
    return state_lim

