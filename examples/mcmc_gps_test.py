import sys
# append your directory to the source code
sys.path.append('/home/cheng/work/pyffit-main')

import os
import time
import emcee
import numba
import pyffit
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.optimize import least_squares
from multiprocessing import Pool
from matplotlib import colors
from gps_setup_test import priors, labels, units, scales, log_prob


def main():
    # prepare_synthetic_data()
    inversion()
    return

def inversion():
    # ---------------------------------- PARAMETERS ----------------------------------
    # Inversion parameters
    progress  = True
    parallel  = False                                 # Parallelize sampling (NOTE: may NOT always result in performance increase)
    n_process = 1                                    # number of threads to use for parallelization
    n_walkers = 50                                   # number of walkers in ensemble (must be at least 2*n + 1 for n free parameters)
    n_step    = 50000                                # number of steps for each walker to take
    moves     = [
                 # (emcee.moves.RedBlueMove(), 1.0),        # Choice of walker moves 
                 (emcee.moves.DEMove(), 0.6),        # Choice of walker moves 
                 (emcee.moves.DESnookerMove(), 0.4),
                 # (emcee.moves.StretchMove(), 1.0),
                ] # emcee default is [emcee.moves.StretchMove(), 1.0]
    #init_mode = 'gaussian'
    init_mode = 'uniform'

    # Define uniform prior 
    out_dir         = 'result'
    inversion_mode  = 'run' # 'check_downsampling' to only  make quadtree downsampling plots, 'run' to run inversion, 'reload' to load previous results and prepare output products

    # ---------------------------------- INVERSION ----------------------------------  #
    # pyffit.inversion.mcmc(priors, log_prob, out_dir, init_mode=init_mode, moves=moves, n_walkers=n_walkers, 
    #                       n_step=n_step, parallel=parallel, n_process=n_process, progress=progress, inversion_mode=inversion_mode)

    # Get starting model parameters
    m0 = np.array([np.mean(priors[prior]) for prior in priors.keys()])
    #m0 = np.array([12.6, -40.9, 149.8, 5.0, 330.3, 107.9, -2.8, 0.6,])

    # Run inversion or reload previous results
    if inversion_mode == 'run':
        # Set up backend for storing MCMC results
        backend = pyffit.inversion.config_backend(f'{out_dir}/results.h5', n_walkers, len(priors)) 
        samples, samp_prob, autocorr, discard, thin = pyffit.inversion.run_hammer((), priors, log_prob, n_walkers, n_step, m0, backend, moves, 
                                                                                  progress=progress, init_mode=init_mode, parallel=parallel, processes=n_process)
    elif inversion_mode == 'reload':
        result_file = f'{out_dir}/results.h5'
        samples, samp_prob, autocorr, discard, thin = pyffit.inversion.reload_hammer(result_file)

    # Assess consistence of convergence amongst the ensemble members
    samp_prob[np.abs(samp_prob) == np.inf] = np.nan   # Correct for infinite values
    mean_chain_prob = np.nanmean(samp_prob, axis=0)   # Get average prob. for each walker
    std_chain_prob  = np.nanstd(mean_chain_prob)      # Get std of walker means
    mean_prob       = np.nanmean(samp_prob.flatten()) # Get total average
    std_prob        = np.nanstd(samp_prob.flatten())  # Get total STD

    # Discard "lost" walkers
    # samples         = samples[:, abs(mean_chain_prob - mean_prob) <= std_prob]   
    samp_prob       = samp_prob[:, abs(mean_chain_prob - mean_prob) <= std_prob] 
    flat_samples    = samples[discard::thin, :, :].reshape(len(samples[discard::thin, 0, 0])*len(samples[0, :, 0]), len(samples[0, 0, :]))
    # discard_walkers = n_walkers - samples.shape[1]

    print(f'Average log(p(m|d)) = {mean_prob} +/- {std_prob}')
    print(f'Chain  log(p(m|d))  = {mean_chain_prob} +/- {std_chain_prob}')
    # print(f'Number of discarded ensemble members = {discard_walkers}')
    print(f'Number of effective samples = {len(flat_samples)}')

    # Compute mean and standard deviation of flat samples
    m_avg = np.mean(flat_samples,           axis=0)
    m_std = np.std(flat_samples,            axis=0)
    m_q1  = np.quantile(flat_samples, 0.16, axis=0)
    m_q2  = np.quantile(flat_samples, 0.50, axis=0)
    m_q3  = np.quantile(flat_samples, 0.84, axis=0)

    print(priors.keys())
    print(m_avg)
    print(m_std)

    # # Compute RMSE for representative models
    # m_rms_avg  = wRMSE(model(m_avg, x), d, S_inv, B)
    # m_rms_q2   = wRMSE(model(m_q2, x),  d, S_inv, B)

    # Plot Markov chains for each parameter
    #pyffit.figures.plot_chains(samples, samp_prob, priors, discard, labels, units, scales, out_dir)

    # Plot parameter marginals and correlations
    if inversion_mode == 'run':
    	pyffit.figures.plot_triangle(flat_samples, priors, labels, units, scales, out_dir,limits=[])    
    
    if inversion_mode == 'reload':
        limits=[[2.79,2.85],[5.18,5.28],[5.74,5.94],[24.74,24.9],[10.48,11.18],[26.95,27.05],[83,83.5],[3.23,3.41],[-0.25,-0.22]]
        pyffit.figures.plot_triangle(flat_samples, priors, labels, units, scales, out_dir,limits=limits,figsize=(12,12))
        print('triangle plot replotted')


    return  

if __name__ == '__main__':
    main()
