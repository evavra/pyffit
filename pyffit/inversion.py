import time
import numba
import emcee
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import matplotlib
import corner
from multiprocessing import Pool

# -------------------------- Probability methods --------------------------
def cost_function(m, coords, d, S_inv, model):
    """
    Cost function for initial optimization step (modified version of log. likelihood)

    INPUT:
    m     - model parameters
    x     - data coordinates
    d     - data values
    S_inv - inverse covariance matrix
    model - function handle for model

    OUTPUT:
    log[p(d|m)] - log likelihood of d given m 
    """

    # Make forward model calculation
    G_m = model(m, coords)
    r = G_m - d
    return np.hstack((0.5**-0.5) * S_inv @ r)


def log_prob_uniform(m, x, d, S_inv, model, priors):
    """
    Determine log-probaility of model m using a uniform prior
    """

    # Check prior
    if np.all([priors[key][0] <= m[i] <= priors[key][1] for i, key in enumerate(priors.keys())]):
        return log_likelihood(model(m, x), d, S_inv) # Log. probability of sample is only the log. likelihood
    else:
        return -np.inf                                    # Exclude realizations outside of priors


def log_prob_gaussian(m, x, d, S_inv, model, priors):
    """
    Determine log-probability of model m using a gaussian prior
    """

    # Evaluate prior
    mu    = np.array([priors[key][0] for key in priors.keys()])
    sigma = np.diag(np.array([priors[key][1] for key in priors.keys()])**-2)

    return log_likelihood(model(m, x), d, S_inv, B) + log_prior_gaussian(m, mu, sigma) # Log. probability is sum of log. likelihood and log. prior
    

@numba.jit(nopython=True)
def log_likelihood(G_m, d, S_inv, B):
    """
    Speedy version of log. likelihood function

    INPUT:
    G_m   - model realization
    d     - data values
    S_inv - inverse covariance matrix
    B     - data weights
    """

    r = d - G_m

    return -0.5 * r.T @ S_inv @ B @ r

# -------------------------- Sampling methods -------------------------- 
def run_hammer(log_prob_args, priors, log_prob, n_walkers, n_step, m0, backend, moves, 
               progress=False, init_mode='uniform', run_name='Sampling', parallel=False, processes=8):
    """
    Perform ensemble sampling using the MCMC Hammer.

    """

    print(f'Performing ensemble sampling...')

    n_dim = len(priors)

    # Initialize walkers around MLE solution
    # a) Gaussian ball
    if init_mode == 'gaussian':
        b_size = 1e-1
        pos = m0 + b_size * m0 * np.random.randn(n_walkers, n_dim)

    # b) Uniform over priors
    elif init_mode == 'uniform':
        pos   = np.empty((n_walkers, n_dim))
        for i in range(n_walkers):
            for j, prior in enumerate(priors.keys()):
                pos[i, j]  = np.random.uniform(low=priors[prior][0], high=priors[prior][1])

    # Run ensemble sampler
    s_start = time.time()

    if parallel:
        os.environ["OMP_NUM_THREADS"] = "1"
        print('Parallelizing sampling...')

        with Pool(processes=processes) as pool:
            sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob, args=log_prob_args, backend=backend, pool=pool, moves=moves)
            sampler.run_mcmc(pos, n_step, progress=progress)
    else:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob, args=log_prob_args, backend=backend, moves=moves)
        sampler.run_mcmc(pos, n_step, progress=progress)

    s_end = time.time() - s_start

    if s_end > 120:
        print(f'Time elapsed: {s_end/60:.2f} min')
    else:
        print(f'Time elapsed: {s_end:.2f} s')

    # Get autocorrelation times
    autocorr = sampler.get_autocorr_time(tol=0)
    print(f'Autocorrelation times: {autocorr}')

    # Flatten chain and thin based off of autocorrelation times
    discard = int(5 * np.nanmax(autocorr))
    thin    = int(0.5 * np.nanmin(autocorr))

    print(f'Burn-in:  {discard} samples')
    print(f'Thinning: {thin} samples')
    print()

    # Get samples
    samples      = sampler.get_chain()
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    samp_prob    = sampler.get_log_prob()

    return samples, samp_prob, autocorr, discard, thin


def run_inversion(params):
    """
    Perform Bayesian inversion using the MCMC Hammer


    INPUT:
    
    
    OUTPUT:
    
    """

    mode, top_dir, model, x, d, S_inv, B, priors, log_prob, n_walkers, n_step, m0, init_mode, labels, units, scales, X0, Y0, Z0, D0, dims, node_r, vlim, l, w, key, out_dir, parallel, moves = params

    # Perform ensemble sampling with Emcee or load most recent results
    if mode == 'reload':
        print(f'Reloading {key}...')
        result_file = out_dir + f'/results/{key}_Results.h5'
        samples, samp_prob, autocorr, discard, thin = reload_hammer(result_file)
    else:
        # Set up backend for storing MCMC results
        backend = config_backend(out_dir, key, n_walkers, len(priors))
        samples, samp_prob, autocorr, discard, thin = run_hammer(x, d, S_inv, B, model, priors, log_prob, n_walkers, n_step, m0, init_mode, labels, units, scales, key, parallel, backend, moves)

    # Assess consistence of convergence amongst the ensemble members
    samp_prob[np.abs(samp_prob) == np.inf] = np.nan   # Correct for infinite values
    mean_chain_prob = np.nanmean(samp_prob, axis=0)   # Get average prob. for each walker
    std_chain_prob  = np.nanstd(mean_chain_prob)      # Get std of walker means
    mean_prob       = np.nanmean(samp_prob.flatten()) # Get total average
    std_prob        = np.nanstd(samp_prob.flatten())  # Get total STD

    # Discard "lost" walkers
    samples         = samples[:, abs(mean_chain_prob - mean_prob) <= std_prob]   
    samp_prob       = samp_prob[:, abs(mean_chain_prob - mean_prob) <= std_prob] 
    flat_samples    = samples[discard::thin, :, :].reshape(len(samples[discard::thin, 0, 0])*len(samples[0, :, 0]), len(samples[0, 0, :]))
    discard_walkers = n_walkers - samples.shape[1]

    print(f'({key}) Average log(p(m|d)) = {mean_prob} +/- {std_prob}')
    print(f'({key}) Chain  log(p(m|d))  = {mean_chain_prob} +/- {std_chain_prob}')
    print(f'({key}) Number of discarded ensemble members = {discard_walkers}')
    print(f'({key}) Number of effective samples = {len(flat_samples)}')
    
    # Compute mean and standard deviation of flat samples
    m_avg = np.mean(flat_samples,           axis=0)
    m_std = np.std(flat_samples,            axis=0)
    m_q1  = np.quantile(flat_samples, 0.16, axis=0)
    m_q2  = np.quantile(flat_samples, 0.50, axis=0)
    m_q3  = np.quantile(flat_samples, 0.84, axis=0)

    # Compute RMSE for representative models
    m_rms_avg  = wRMSE(model(m_avg, x), d, S_inv, B)
    m_rms_q2   = wRMSE(model(m_q2, x),  d, S_inv, B)

    # Plot Markov chains for each parameter
    plot_chains(samples, samp_prob, discard, labels, units, scales, key, out_dir)

    # Plot parameter marginals and correlations
    plot_triangle(flat_samples, priors, labels, units, scales, key, out_dir)

    return key, m_avg, m_std, m_q1, m_q2, m_q3, m_rms_avg, m_rms_q2


def get_starting_model(coords, d, S_inv, model, cost_function, priors, prior_mode='uniform'):
    """ 
    Solve for initial state using non-linear least squares
    """

    print('Finding starting model...')

    # Set up helper function for optimization
    nll = lambda *args: -cost_function(*args)

    # Choose initial guess (means of priors)
    if prior_mode =='Gaussian':
        initial = np.array([priors[prior][0] for prior in priors.keys()])
    else:
        initial = np.array([np.mean(priors[prior]) for prior in priors.keys()])

    # Optimize function & parse results
    bounds = [priors[prior][0] for prior in priors.keys()], [priors[prior][1] for prior in priors.keys()]
    m0     = least_squares(nll, initial, args=(coords, d, S_inv, model), bounds=(bounds), jac='3-point').x
    
    return m0


def reload_hammer(result_file):
    """
    Load previous inversion results from output HDF5 file.
    """

    # Get mean and standard deviation of posterior distr
    # Load samples
    sampler = emcee.backends.HDFBackend(result_file)

    # Get autocorrelation times
    autocorr = sampler.get_autocorr_time(tol=0)

    # Flatten chain based off of autocorrelation times (rounded to nearest power of 10 * 3)
    discard = int(2 * np.nanmax(autocorr))
    thin    = int(0.5 * np.nanmin(autocorr))

    # Get samples
    samples       = sampler.get_chain()
    flat_samples  = sampler.get_chain(discard=discard, thin=thin, flat=True)
    samp_prob     = sampler.get_log_prob()


    return samples, samp_prob, autocorr, discard, thin, flat_samples


# -------------------------- Utility methods --------------------------
def config_backend(file_name, n_walkers, n_dim):
    """
    Set up backend for outputting results to HDF5 file.
    """

    backend   = emcee.backends.HDFBackend(file_name)
    backend.reset(n_walkers, n_dim)

    return backend


@numba.jit(nopython=True)
def RMSE(G_m, d, S_inv):
    """
    Compute the root-mean-square error
    G_m   (n,)  - model vector
    d     (n,)  - data vector
    S_inv (n,n) - covariance matrix (diagonal)
    """

    N = len(d)  

    return (((d - G_m).T @ S_inv @ (d - G_m))/N)**0.5


@numba.jit(nopython=True)
def wRMSE(G_m, d, S_inv, B):
    """
    Compute the weighted root-mean-square error
    G_m   (n,)  - model vector
    d     (n,)  - data vector
    S_inv (n,n) - covariance matrix (diagonal)
    B     (n,n) - matrix of data weights (diagonal)
    """

    r = G_m - d # Residuals

    return ((r.T @ S_inv @ B @ r)/np.sum(B))**0.5


@numba.jit(nopython=True)
def log_likelihood(G_m, d, S_inv):
    """
    Speedy version of log. likelihood function

    INPUT:
    G_m   - model realization
    d     - data values
    S_inv - inverse covariance matrix
    B     - data weights
    """

    r = d - G_m

    return -0.5 * r.T @ S_inv @ r


# -------------------------- Plotting methods --------------------------
def plot_chains(samples, samp_prob, priors, discard, labels, units, scales, key, out_dir, dpi=200, figsize=(20, 15)):
    """
    Plot Markov chains
    """
    n_dim = len(labels)

    prior_vals = [[prior * scales[i] for prior in priors[key]] for i, key in enumerate(priors.keys())]


    fig, axes = plt.subplots(n_dim + 1, figsize=figsize, sharex=True)
    
    for i in range(n_dim):
        ax = axes[i]
        ax.plot(samples[discard:, :, i] * scales[i], "k", alpha=0.3, linewidth=0.5)
        ax.set_xlim(0, len(samples))
        ax.set_ylim(prior_vals[i][0], prior_vals[i][1])
        ax.set_ylabel(labels[i] + f'\n ({units[i]})')
        # ax.yaxis.set_label_coords(-0.1, 0.5)

    # Plot log-probability
    ax = axes[n_dim]
    ax.plot(samp_prob[discard:], "k", alpha=0.3, linewidth=0.5) # log(p(d|m))
    # ax.plot(samp_prob, "k", alpha=0.3, linewidth=0.5)
    # ax.plot(samp_prob[:discard], color='tomato', linewidth=0.5)
    ax.set_xlim(0, len(samples))
    # ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.set_ylabel(r'log(p(m|d))') # log(p(d|m))
    axes[-1].set_xlabel("Step");
    fig.tight_layout()
    fig.savefig(f'{out_dir}/{key}_chains.png', dpi=dpi)
    plt.close()
    
    return


def plot_triangle(samples, priors, labels, units, scales, key, out_dir, dpi=200, figsize=(20, 20)):
    # Make corner plot
    font = {'size' : 6}

    # matplotlib.rc('font', **font)

    prior_vals = [[prior * scales[i] for prior in priors[key]] for i, key in enumerate(priors.keys())]

    fig = plt.figure(figsize=figsize, tight_layout={'h_pad':0.1, 'w_pad': 0.1})
    fig.suptitle(key)
    fig = corner.corner(samples * scales, 
                        quantiles=[0.16, 0.5, 0.84], 
                        # range=prior_vals,
                        labels=[f'{label} ({unit})' for label, unit in zip(labels, units)], 
                        # label_kwargs={'fontsize': 8},
                        show_titles=True,
                        # title_kwargs={'fontsize': 8},
                        fig=fig, 
                        labelpad=0.1
                        )

    # fig.tight_layout(pad=1.5)
    fig.savefig(f'{out_dir}/{key}_triangle.png', dpi=dpi)
    plt.close()
    return


# -------------------------- For future re-implementation --------------------------
# def reload_hammer(result_file):
#     # Get mean and standard deviation of posterior distribution
#     # Load samples
#     sampler = emcee.backends.HDFBackend(result_file)

#     # Get autocorrelation times
#     autocorr = sampler.get_autocorr_time(tol=0)

#     # Flatten chain based off of autocorrelation times (rounded to nearest power of 10 * 3)
#     discard = int(10**np.ceil(np.log10(np.nanmax(autocorr)))) * 3
#     thin    = int(np.nanmax(autocorr)//2)

#     # Get samples
#     samples       = sampler.get_chain()
#     flat_samples  = sampler.get_chain(discard=discard, thin=thin, flat=True)
#     samp_prob     = sampler.get_log_prob()

#     return samples, samp_prob, autocorr, discard, thin



