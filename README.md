## pyffit: Python Finite Fault Inversion Tools

## New: inversion with GPS data (this is in an early testing phase. Please post any problem you encounter in the discussion panel!!!)
1. Format of your GPS data: station, lon, lat, ux, uy, (uz), std_x, std_y, (std_z). If you have z displacements, set the data_type to 3d; otherwise, set to 2d. Displacement and standard deviation should be in m. z is downward positive.
1. Go to examples/gps_setup_test.py, and specify your data and output directory, weights, and other parameters you would like to tweak.
2. Go to eamples/mcmc_gps_test.py, set the path to the code directory, and run the code.




## What to expect next
1. Elliptical deformation: currently the code only generates box-car deformation on a fault patch, which is probably not the most realistic way. In the future, the code will be updated so that the slips will taper out in an elliptical pattern away from the center.
   
2. Inversion that isn't based on Quadtree sampling: Quadtree sampling is good at capturing near-field data. But what if the deformation signals are mostly located 'mid-field'? Resolution-based sampling may be useful (the R-based sampling codes will be released separately). The Inversion will support the product of R-based sampling in the future.

3. Iterative sampling: the inversion will support iterative sampling based on the forward modeling of best-fitting parameters.

4. Sample from the misfits: The inversion will have the option of sampling from the area of large misfits, and adding them back to the original sampled data, which mitigates the misfits caused by insufficient sampling.

5. Inversion with GPS data

## References
If using the codes, please cite the paper: Vavra, E. J., Qiu, H., Chi, B., Share, P.-E., Allam, A., Morzfeld, M., et al. (2023). Active dipping interface of the Southern San Andreas fault revealed by space geodetic and seismic imaging. Journal of Geophysical Research: Solid Earth, 128, e2023JB026811. https://doi.org/10.1029/2023JB026811 and this GitHub repository if possible.

## Requirements and Installation
In addition to default Python libraries, this code also uses [`matplotlib`](https://matplotlib.org/), [`numpy`](https://numpy.org/), [`pandas`](https://pandas.pydata.org/), [`scipy`](https://scipy.org/) (users familiar with Python will likely already have these installed). The implementations for the rectangular and triangular dislocation elements are from Ben Thompson's [`okada_wrapper`](https://github.com/tbenthompson/okada_wrapper) and [`cutde`](https://github.com/tbenthompson/cutde), respectively.

Several other packages you may need to install are:
- [`xarray`](https://docs.xarray.dev/en/stable/): reading NETCDF data files
- [`h5py`](https://docs.h5py.org/en/stable/): reading/writing HDF5 output files
- [`emcee`](https://emcee.readthedocs.io/en/stable/): performing MCMC sampling
- [`numba`](https://numba.pydata.org/): accelerating some basic numerical calculations (i.e. fault model predictions)
- [`corner`](https://corner.readthedocs.io/en/latest/): plotting MCMC sampling results
- [`pyproj`](https://pypi.org/project/pyproj/): handling geographic coordinate systems
- [`cartopy`](https://scitools.org.uk/cartopy/docs/latest/gallery/index.html): handling and plotting geopatial datasets
- [`cmcrameri`](https://pypi.org/project/cmcrameri/0.9/):  a basic Python wrapper around Fabio Crameri's perceptually uniform colour maps 
- [`tqdm`](https://pypi.org/project/tqdm/): showing the inversion progress   
- [`corner`](https://pypi.org/project/tqdm/): corner plots

I would recommend using [Conda](https://conda.io/projects/conda/en/latest/index.html) to create a new Python environment to install and manage these packages. 
Or simply try 'pip install'

## A. How to run the inversion with single rectangular dislocation

1. Create a 'result' directory where all the results will be stored.
2. Go to setup.py (should be at the same level as your mcmc.py and result directory) to configure the input data path, prior limits, steps, number of walkers, path to the output directory ('result'), etc.
3. Run the mcmc.py for the inversion.
4. optional: Run the forward.py for the forward modeling products of your best-fitting parameters. 


## Example: Bayesian inversion for single rectangular dislocation.
In circumstances where information regarding the geometry and/or orientation of a fault producing a large earthquake are are poorly known it may be useful to perform an inversion for a simplified fault model consisting of a single rectangular (i.e. ``Okada'') dislocation, where the position, dimension, orientation, and slip are simultaneously estimated. In this particular example, I have implemented this procedure using a Markov Chain Monte Carlo (MCMC) algorithm, which allows for uncertainty quantification on the estimated fault parameters. This simplified inversion and resulting parameter estimates may be used as inputs and/or constaints to deriving a more complex finite-fault model to analyze the event's slip distribution. 

To demonstrate the inversion, I have created a synthetic example for the case of a (very simple) $M_w$ 7.3 earthquake along the San Jacinto fault system in Southern California. I use a single rectangular disclotion with 3 m of dextral slip to generate surface displacements, which are then projected into the line-of-sight direction (LOS) for Sentinel-1's descending track 173 over the Salton Trough. To generate semi-realistic noise, I introduce observed radar decorrelation from real Sentinel-1 observations (Vavra et al., 2024), simulated rupture decorrelation along the fault trace, and spatially correlated noise to simulate troposphere delays (Emardson et al., 2003) the synthetic LOS data. The noisy data are then down-sampeld using a quadtree algorithm (Jonsson, 2002; Simons, 2002) in order to reduce the computational cost of the inversion. In this example, two realizations of synthetic data (same fault displacements, different troposphere noise) are used since multiple InSAR datasets are often available for inverting. The data are then inverted for a single-patch fault model where the origin coordinates `x` and `y`, fault `strike`, fault `dip`, along-strike length `l`, along-dip width `w`, and `strike_slip` and `dip_slip` amplitudes are estimated. 

### 1. Synthetic data
![alt text](https://github.com/evavra/pyffit/blob/main/examples/LOS_clean.png "Synthetic line-of-sight displacements")

![alt text](https://github.com/evavra/pyffit/blob/main/examples/LOS_noisy.png "Synthetic line-of-sight displacements with added noise")


### 2. Quadtree downsampling
![alt text](https://github.com/evavra/pyffit/blob/main/examples/quadtree_init_synthetic_data_1.png "Downsampled line-of-sight displacements")


### 3. MCMC Results
![alt text](https://github.com/evavra/pyffit/blob/main/examples/triangle.png "Triangle plot")

## B. How to iteratively run the inversion

After you have a model from initial sampling, sometimes the inversion and sampling can be improved by sampling the original data based on the forward modeling of best-fitting model parameters. Here we have codes to run the inversion using iterative [R-based Sampling](https://github.com/x3zou/RBSamping)

To do the iterative sampling, assuming you have the results of initial sampling, you will need to:

1. Run the forward.py based on your initial sampling result to get the full-resolution forward model.
2. Using a full-resolution forward model as an input for R-based sampling.
4. Run R-based sampling
5. Run mcmc_BF.py
6. Repeat steps 1-3 until you get a satisfying results.


