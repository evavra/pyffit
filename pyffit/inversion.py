import os
import time
import numba
import emcee
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import matplotlib
import corner
from multiprocessing import Pool
from scipy.optimize import lsq_linear
import h5py
import time
import pickle
from scipy.integrate import solve_ivp
# from matplotlib.lines import Line2D
from pyffit.quadtree import ResQuadTree


# -------------------------- Classes --------------------------
class LinearInversion:
    """
    Class to set up and perform a linear finite slip inversion

    INITIALIZATION:
    fault        - TriFault object for finite fault model
    dataset_dict -  dictionary containing InversionDataset objects organized by specified 
                    dataset labels
    verbose      - display progress/log messages (default = True)

    METHODS:
    LinearInversion.run() - perform linear inversion
    """

    def __init__(self, fault, dataset_dict, smoothing=False, edge_slip=False, verbose=True):

        # Form inversion inputs
        self.datasets         = dataset_dict
        self.greens_functions = np.vstack([self.datasets[key].greens_functions for key in self.datasets.keys()])
        self.data             = np.concatenate([self.datasets[key].tree.data for key in self.datasets.keys()])
        self.verbose          = verbose

        # ------------------ Perform inversion ------------------
        # Form full design matrix
        self.G = self.greens_functions
        self.d = self.data

        if smoothing:
            if verbose:
                print(f'Using smoothing regularization value of {fault.mu:.1e}')
            self.G = np.vstack((self.G, fault.mu*fault.smoothing_matrix)) # Add regularization        
            self.d = np.hstack((self.d, np.zeros(fault.smoothing_matrix.shape[0]))) # Pad data vector with zeros

        if edge_slip:
            if verbose:
                print(f'Using edge slip regularization value of {fault.eta:.1e}')
            self.G = np.vstack((self.G, fault.eta*fault.edge_slip_matrix)) # Add regularization        
            self.d = np.hstack((self.data, np.zeros(fault.edge_slip_matrix.shape[0]))) # Pad data vector with zeros


    def run(self, slip_lim=(-np.inf, np.inf), lsq_kwargs={}):
        """
        Perform least-squares solve and update internal InversionDataset objects
        """

        # Solve using bounded least squares to enforce slip direction constraint
        start      = time.time()
        results    = lsq_linear(self.G, self.d, bounds=slip_lim, **lsq_kwargs).x
        # results, _, _, _    = np.linalg.lstsq(self.G, self.d, rcond=-1)
        slip_model = np.column_stack((results, np.zeros_like(results), np.zeros_like(results)))
        end        = time.time() - start

        # Print run time
        if self.verbose:
            print('\n' + f'Inversion time: {end:.2f} s')

        # Get forward model prediction and update object
        for key in self.datasets.keys():
            self.datasets[key].add_results(slip_model[:, 0])

        # Compute error terms
        self.resids = np.concatenate([self.datasets[key].resids for key in self.datasets.keys()])
        self.rms    = np.sqrt(np.sum(self.resids**2)/len(self.resids))

        return slip_model


class InversionDataset:
    """
    Class for organizing data and results for finite slip inversions.
    """

    def __init__(self, tree, greens_functions, verbose=False):
        """
        Add input data and Green's functions to input to inverion
        """
        self.tree = tree
        self.greens_functions = greens_functions
        self.verbose = verbose

    def add_results(self, slip_model):
        """
        Compute forward model prediction and get residuals and RMS
        """
        self.slip_model = slip_model
        start = time.time()
        self.model      = self.greens_functions.dot(slip_model)
        end = time.time() - start

        self.resids     = self.tree.data - self.model     
        self.rms        = np.sqrt(np.sum(self.resids**2)/len(self.resids))

        self.model_time = end
        
        if self.verbose:
            print(f'Model time: {self.model_time:.5f}')


# -------------------------- Main methods --------------------------
def get_inversion_inputs(fault, datasets, quadtree_params={}, date=-1, LOS=False, disp_components=[0, 1, 2], slip_components=[0, 1, 2], rotation=np.nan, squeeze=False,
                         run_dir='.', quadtree_file_stem='_quadtree.pkl', greenfunc_file_stem='_greens_functions.h5', verbose=False):
    """
    Prepare data to be ingested by inversion

    INPUT:
    fault            - TriFault object
    datasets         - dictionary containing Xarray multi-file dataset(s)
    quadtree_params  - dictionary of quadtree parameters
    rotation         - (np.nan)
    """

    start = time.time()

    inputs = {}

    # Loop over datasets to generate InversionDataset objects
    for key in datasets.keys():

        # Select current data grid
        data = datasets[key]['z'].isel(date=date).compute().data

        print('\n' + f'######### Working on {key}... #########')
        
        # ---------- Downsampling ----------
        # Load existing quadtree or perform downsampling and save to disk 
        quadtree_file = f'{run_dir}/{key}{quadtree_file_stem}'
        look          = np.array([datasets[key]['look_e'].compute().data.flatten(), datasets[key]['look_n'].compute().data.flatten(), datasets[key]['look_u'].compute().data.flatten()]).T

        if os.path.exists(quadtree_file):

            # Load existing structure
            with open(quadtree_file, 'rb') as f:
                tree = pickle.load(f)

                # Check to see if parameters are the same, if not redo
                if tree.check_parameters(quadtree_params) == False:
                    print('\nRedoing downsampling')
                    tree = ResQuadTree(datasets[key].coords['x'].compute().data.flatten(), datasets[key].coords['y'].compute().data.flatten(), data.flatten(), look, np.arange(0, datasets[key].coords['x'].size), fault, verbose=verbose, **quadtree_params, )
                    tree.write(quadtree_file)
                else:
                    print('\nLoading quadtree')

        else:
            tree = ResQuadTree(datasets[key].coords['x'].compute().data.flatten(), datasets[key].coords['y'].compute().data.flatten(), data.flatten(), look, np.arange(0, datasets[key].coords['x'].size), fault, verbose=verbose, **quadtree_params)
            tree.write(quadtree_file)

        # Print parameter values
        tree.display_parameters(quadtree_params)
        print('\n' + f'Quadtree points: {len(tree.data)}')

        # ---------- Green's functions ------------------
        if LOS != True:
            look = []

        # Load or generate Green's functions
        greens_function_file = f'{run_dir}/{key}{greenfunc_file_stem}'

        if os.path.exists(greens_function_file):
            # Load existing Green's Functions
            gfile = h5py.File(greens_function_file, 'r')
            GF = gfile['GF'][()]
            gfile.close()

            if GF.shape[0] != tree.x.shape[0]:
                print('\n' + "Redoing Green's functions")

                # Generate LOS Green's functions for dowmsampled coordinates
                # GF = fault.LOS_greens_functions(tree.x, tree.y, tree.look)
                GF = fault.greens_functions(tree.x, tree.y, look=look, disp_components=disp_components, slip_components=slip_components, rotation=rotation, squeeze=squeeze)

                # Save to disk
                gfile = h5py.File(greens_function_file, 'w')
                gfile.create_dataset('GF', data=GF)
                gfile.close()
                print('\n' + f"Green's functions saved to {greens_function_file}")
            else:
                print('\n' + "Loading Green's functions")

        else:
            # Generate LOS Green's functions for dowmsampled coordinates
            # GF = fault.LOS_greens_functions(tree.x, tree.y, tree.look)
            GF = fault.greens_functions(tree.x, tree.y, look=look, disp_components=disp_components, slip_components=slip_components, rotation=rotation, squeeze=squeeze)

            # Save to disk
            gfile = h5py.File(greens_function_file, 'w')
            gfile.create_dataset('GF', data=GF)
            gfile.close()
            print('\n' + f"Green's functions saved to {greens_function_file}")

        # Include only strike-slip contribution
        inputs[key] = InversionDataset(tree, GF)    

    end = time.time() - start 
    print('\n' + f'Data prep time: {end:.2f} s')

    return inputs


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
    # flat_samples  = sampler.get_chain(discard=discard, thin=thin, flat=True)
    samp_prob     = sampler.get_log_prob()


    return samples, samp_prob, autocorr, discard, thin


def mcmc(priors, log_prob, out_dir, m0=[], init_mode='uniform', moves=[emcee.moves.StretchMove(), 1.0], 
         n_walkers=100, n_step=10000, parallel=False, n_process=8, progress=True, inversion_mode='run'):
    
    print(len(priors))
    print(len(m0))
    
    # Get starting model parameters
    if len(m0) != len(priors):
        m0 = np.array([np.mean(priors[prior]) for prior in priors.keys()])

    # Run inversion or reload previous results
    if inversion_mode == 'run':
        # Set up backend for storing MCMC results
        backend = config_backend(f'{out_dir}/results.h5', n_walkers, len(priors)) 
        samples, samp_prob, autocorr, discard, thin = run_hammer((), priors, log_prob, n_walkers, n_step, m0, backend, moves, 
                                                                                  progress=progress, init_mode=init_mode, parallel=parallel, processes=n_process)
    elif inversion_mode == 'reload':
        result_file = f'{out_dir}/results.h5'
        samples, samp_prob, autocorr, discard, thin = reload_hammer(result_file)

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

    print(f'Average log(p(m|d)) = {mean_prob} +/- {std_prob}')
    print(f'Chain  log(p(m|d))  = {mean_chain_prob} +/- {std_chain_prob}')
    print(f'Number of discarded ensemble members = {discard_walkers}')
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
    pyffit.figures.plot_chains(samples, samp_prob, discard, labels, units, scales, out_dir)

    # Plot parameter marginals and correlations
    pyffit.figures.plot_triangle(flat_samples, priors, labels, units, scales, out_dir)    
    return


# -------------------------- Kalman Filter --------------------------
def EnKF_driver():
    # Load results from previous HWs
    (B, F, J, R, T, U, l, mu_0, n_x, n_y, t, t_B, t_T, t_T_hat, x, x_0, x_0_hat, x_B, x_T, x_T_hat, x_attractor, x_s, y) = load_results('HW_1')
    
    # ---------- Set parameters ---------- 
    # Model & observations
    n_dim    = 40                       # Get model dimension
    n_e      = 20                        # ensemble size
    dT       = 0.2                      # observation sampling interval
    dt       = 0.05                     # Model timestep interval
    grid_dim = 2
    alpha    = np.logspace(-3, 0, grid_dim)[::-1] # inflation factors
    l        = np.linspace(1, 10, grid_dim)      # localization factors
    # alpha   = np.array([0.08])    # localization factors
    # l       = np.array([3,])    # localization factors
    discard = 500                      # number of spin-up steps to remove from mean RMSE and spread calculations
    out_dir = 'HW_6/grid_search' # Output directory

    # ---------- Load observations ---------- 
    y_t      = np.loadtxt('HW_6/HW6_obs_n40.txt', delimiter=',').T   # "Observations"
    x_t      = np.loadtxt('HW_6/HW6_truth_n40.txt', delimiter=',').T # Truths
    samp_inc = int(dT/dt)
    x_t_samp = x_t[samp_inc::samp_inc]                               # Sampled truths
    n_obs    = y_t.shape[0] # number of observations
    n_samp   = y_t.shape[1] # number of samples for each observation
    H        = np.eye(n_dim)[::n_dim//n_samp, :] # Sampling matrix
    T        = np.arange(dT, dT*n_obs + dT, dT)  # Observation times
    t        = np.arange(0, dt*x_t.shape[0], dt) # Model times
    x_idx    = np.arange(0, n_dim, 1)
    y_idx    = np.arange(0, n_dim, 2)
    steps    = np.arange(0, n_obs)
    R        = np.eye(n_samp)
    
    # ---------- Run Kalman filter ----------
       # Run long simulation
    start    = time.time()
    sol_long = L96(np.random.uniform(low=0, high=1, size=n_dim), [0, 1000], F)
    end      = time.time() - start
    x_long   = sol_long.y.T
    t_long   = sol_long.t 
    print('Long simulation complete: {:.1f} s'.format(end))

    # Initialize ensemble with states from long simulation x_B
    i_samp = np.random.randint(low=0, high=len(x_long), size=n_e)
    x_init = x_long[i_samp]

    rmse_means = np.empty((len(alpha), len(l)))
    count = 0
    for i in range(len(alpha)):
        for j in range(len(l)):
            rmse_means[i, j] = complete_EnKF(x_init, y_t, M, R, H, alpha[i], l[j], x_t_samp, discard, out_dir)
            count += 1
            print(f'Finished {count}/{l.size*alpha.size}')


    if len(alpha) | len(l) > 1:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        extent = [l[0], l[-1], alpha[0], alpha[-1]]
        grd = ax.imshow(rmse_means, extent=extent, cmap='magma_r')
        ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2])))
        ax.set_ylabel(r'$\alpha$')
        ax.set_xlabel(r'$l$')
        fig.colorbar(grd, label='Mean RMSE')
        fig.savefig(f'{out_dir}/grid_search.png', dpi=500)
        plt.show()
          

def M(x_0): 
    return L96(x_0, [0, 0.2], 8).y.T[-1, :] # Wrapper for forward model
    

@numba.jit(nopython=True)
def dx_dt(t, x, F):
    n = len(x)
    return [-x[i] + (x[(i + 1) % n] - x[i - 2]) * x[i - 1] + F for i in range(n)]


def L96(x_0, t, F):
    """
    Integrate the Lorentz '96 model over time history t.
    
    INPUT:
    t   - integration time range
    x_0 - initial state vector (n,)
    F   - forcing factor
    
    OUTPUT:
    x - model state vector at time T
    """
    
    # Define time derivative operator
    # dx_dt = lambda t, x: [-x[i] + (x[(i + 1) % n] - x[i - 2]) * x[i - 1] + F for i in range(len(x_0))]
    
    # Integrate
    return solve_ivp(dx_dt, t, x_0, args=(F,), method='RK45')


def cost_function(x_0, y, M, R, l, U, mu_0):
    """
    Cost function F(x).
    
    INPUT:
    x_0  (n_x,)          - initial model state vector
    y    (n_y,)          - data vector
    M    (function)      - model function handle which takes x_0 as an  
                           argument and returns an (n_x,) vector
    R    (n_y, n_y)      - diagonal data covariance matrix
    B    (n_x, n_x)      - background covariance matrix
    mu_0 (n_y or scalar) - regularization value(s)
    
    OUTPUT:
    F(x_0)               - cost function evaluated for x_0
    """
    
    # Get dimensions
    n_x = len(x_0)
    n_y = len(y)
    
    # Compute misfit term (H is implemented via indexing)
    # a = np.diag(np.diag(R)**-0.5) @ (y - M(x_0)[::n_x//n_y])    
    a = (y - M(x_0)[::n_x//n_y])    
         
    # Compute regularization term 
    # l, U = np.linalg.eig(B) # Get eigenvalues and eigenvectors
    # b = U @ np.diag(l**-0.5) @ (x_0 - mu_0) # Compute product
    b = U @  (x_0 - mu_0)/(l**0.5) # Compute product
  
    # Compute the cost of x_0 using a and b
    # return 0.5 * np.linalg.norm(a, ord=2)**2 + 0.5 * np.linalg.norm(b, ord=2)**2

    return np.hstack((a, b)) * (0.5**-0.5)


def load_results(hw):
    # Load and return results
    file = h5py.File('results.hdf5', 'r')

    # Print list of variables
    print(f'Getting results from {hw}:')
    print('(' + ', '.join(file[hw].keys()) + ')')

    # Get variables
    output = []
    for var in file[hw].keys():
        output.append(file[f'{hw}/{var}'][()])

    return output


def RMSE(m, d):
    """
    Compute root-mean-square error of a model m with respect to data d.
    """
    # return ((1/len(d)) * np.sum((m - d)**2))**0.5
    return ((1/len(d)) * np.sum((m - d)**2))**0.5


def spread(P):
    """
    Compute root-mean-square error of a model m with respect to data d.
    """
    return (np.trace(P)/P.shape[0])**0.5


def EnKF(x_init, y, M, R, H, alpha=0, L=[]):
    """
    INPUT:
    x_init (n_e, n_dim) - intial state, also defines size of ensemble n_e and model dimesion n_dim
    y (n_obs, n_samp)  - observations
    M (function)       - function handle for model
    R (n_samp, n_samp) - data covariance matrix
    H (n_samp, n_dim)  - matrix mapping model output x to observations y
    """

    # ---------- Ensemble Kalman Filter ---------- 
    n_e,   n_dim   = x_init.shape
    n_obs, n_samp  = y.shape
    x_f_e  = np.empty((n_obs, n_e, n_dim))   # Ensemble forecasts
    x_a_e  = np.empty((n_obs, n_e, n_dim))   # Ensemble analyses
    mu_f   = np.empty((n_obs, n_dim))        # Forecast means
    mu_a   = np.empty((n_obs, n_dim))        # Analysis means
    P_f    = np.empty((n_obs, n_dim, n_dim)) # Forecast means
    P_a    = np.empty((n_obs, n_dim, n_dim)) # Analysis means

    print()
    print('##### Running ensemble Kalman filter ##### ')
    print(f'{n_dim} parameters')
    print(f'{n_obs} observations')
    print(f'{n_e} ensemble members')

    start = time.time()

    for i in range(0, n_obs):
        # 1) ---------- Forecast ----------
        # Make forecast ensemble
        for j in range(n_e):
            x_f_e[i, j, :] = M(x_init[j, :])

        # Compute mean and covariance
        mu_f[i, :]   = np.mean(x_f_e[i, :, :], axis=0)
        P_f[i, :, :] = np.cov(x_f_e[i, :, :].T)

        # Inflate
        if alpha:
            x_f_e[i, :, :] = mu_f[i, :] + np.sqrt(1 + alpha)*(x_f_e[i, :, :] - mu_f[i, :])

        # Localize
        if len(L):
            P_f[i, :, :] = np.multiply(L, P_f[i, :, :])

        # 2) ---------- Analysis ----------
        # Perturb observation
        y_p = np.random.multivariate_normal(y[i, :], R, size=(n_e))

        # Get Kalman gain
        HPfHT_inv = np.linalg.lstsq((H @ P_f[i, :, :] @ H.T + R), np.eye(n_samp), rcond=None)[0] # Do linear solve for matrix inverse
        K         = P_f[i, :, :] @ H.T @ HPfHT_inv 

        # Update ensemble, mean, and covariance
        for j in range(n_e):
            x_a_e[i, j, :] = x_f_e[i, j, :] + K @ (y_p[j, :] - H @ x_f_e[i, j, :])

        mu_a[i, :]   = np.mean(x_a_e[i, :, :], axis=0)
        P_a[i, :, :] = np.cov(x_a_e[i, :, :].T)

        # Use current analysis ensemble to initialize next forecast
        x_init = x_a_e[i, :, :] 

    end = time.time() - start
    print('##### Ensemble Kalman filter complete #####')
    print('Elapsed time: {:.1f} s'.format(end))
    print()

    return x_f_e, x_a_e, mu_f, mu_a, P_f, P_a


def analyze_KF_error(x_KF, x_true, P_a):
    """
    Check Kalman filter realized error (RMSE) vs. predicted error (spread).

    INPUT:
    x_KF   (n_obs, n_dim)          - Kalman filter state estimates
    x_true (n_obs, n_dim)          - true states
    P_a    (n_obs, n_samp, n_samp) - analysis covariance matrices corresponding to state estimates

    OUTPUT:
    rmse (n_obs) - root-mean-square-error at each observation time
    sprd (n_obs) - spread of the analysis covariance matrix at each observation time
    
    """

    n_obs = x_KF.shape[0]
    rmse = np.empty(n_obs) # Analysis RMSE
    sprd = np.empty(n_obs) # Analysis spread

    # Compute RMSE and spread at each observation point
    for i in range(n_obs):
        rmse[i] = RMSE(x_KF[i, :], x_true[i, :]) 
        sprd[i] = spread(P_a[i, :, :])

    return rmse, sprd

    # # Check observation sampling


def complete_EnKF(x_init, y, M, R, H, alpha, l, x_true, discard, out_dir):
    """
    Complete 
    """
    # Get dimensions
    n_e,   n_dim   = x_init.shape
    n_obs, n_samp  = y.shape

    # Form localization matrix L
    L = get_L(n_dim, l)
    plot_L(L, l, out_dir)

    # Run ensemble kalman filter
    x_f_e, x_a_e, mu_f, mu_a, P_f, P_a = EnKF(x_init, y, M, R, H, alpha=alpha, L=L)

    # Perform error analysis
    rmse, sprd    = analyze_KF_error(mu_a, x_true, P_a)
    rmse_cumm_avg = np.array([np.mean(rmse[:i]) for i in range(n_obs)])
    sprd_cumm_avg = np.array([np.mean(sprd[:i]) for i in range(n_obs)])
    rmse_mean     = np.nanmean(rmse_cumm_avg[discard:])
    sprd_mean     = np.nanmean(sprd_cumm_avg[discard:])

    plot_errors(rmse, sprd, rmse_cumm_avg, sprd_cumm_avg, rmse_mean, sprd_mean, discard, alpha, l, out_dir)

    return rmse_mean


def get_L(n_dim, l):
    """
    Get Localization matrix L.
    """

    L = np.empty((n_dim, n_dim))

    for i in range(n_dim):
        for j in range(n_dim):
            r = abs(i - j)
            d = min(r, n_dim - r)
            L[i, j] = np.exp(-0.5 * (d/l)**2)

    return L


def plot_L(L, l, out_dir):
    # Check Localization matrix
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    fig.suptitle(f'l = {l}')
    im = ax.imshow(L, extent=[0, len(L), len(L), 0], cmap='Reds')
    ax.set_xlabel(r'$j$')
    ax.set_ylabel(r'$i$')
    fig.colorbar(im, label=r'$L_{ij}$')
    plt.savefig(f'{out_dir}/Fig03_l_{l}.png', dpi=500)
    return fig


def plot_errors(rmse, sprd, rmse_cumm_avg, sprd_cumm_avg, rmse_mean, sprd_mean, discard, alpha, l, out_dir):
    """
    Plot EnKF RMSE and spread evolution.
    """
    n_obs = len(rmse)
    ymax = int(np.max([rmse.max(), sprd.max()]) + 1)
    ylim = [0, ymax]

    fig, ax = plt.subplots(2, 1)
    fig.suptitle('Mean RMSE = {:.2f}, Mean spread = {:.2f}'.format(rmse_mean, sprd_mean))
    ax[0].fill_between([0, discard], [0, 0], [ymax, ymax], edgecolor=None, facecolor='gray', alpha=0.4)
    ax[0].plot(rmse, c='tomato',)
    ax[0].plot(sprd, c='steelblue',)
    ax[0].set_xlabel('')
    ax[0].set_xticklabels([])
    ax[0].set_ylabel('Error')
    
    ax[1].fill_between([0, discard], [0, 0], [ymax, ymax], edgecolor=None, facecolor='gray', alpha=0.4)
    ax[1].plot(rmse_cumm_avg, c='tomato',    label='RMSE')
    ax[1].plot(sprd_cumm_avg, c='steelblue', label='Spread')
    ax[1].set_ylim(ylim)
    ax[1].set_xlabel('Step')
    ax[1].set_ylabel('Average Error')
    ax[1].legend()

    for ax0 in ax:
        ax0.set_xlim(0, n_obs)

    plt.savefig(f'{out_dir}/errors_alpha_{alpha}_l_{l}.png', dpi=500)
    return


# -------------------------- Utility methods --------------------------
def config_backend(file_name, n_walkers, n_dim):
    """
    Set up backend for outputting results to HDF5 file.
    """

    backend = emcee.backends.HDFBackend(file_name)
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
        ax.set_ylabel(labels[i] + '\n' + f' ({units[i]})')
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



