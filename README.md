More info to come soon...


## Requirements and Installation
In addition to default Python libraries, this code also uses [`matplotlib`](https://matplotlib.org/), [`numpy`](https://numpy.org/), [`pandas`](https://pandas.pydata.org/), [`scipy`](https://scipy.org/) (users familiar with Python will likely already have these installed). The implementations for the rectangular and triangular dislocation elements are from Ben Thompson's [`okada_wrapper`](https://github.com/tbenthompson/okada_wrapper) and [`cutde`](https://github.com/tbenthompson/cutde), respectively.

Several other packages you may need to install are:
- [`xarray`](https://docs.xarray.dev/en/stable/): reading NETCDF data files
- [`hyp5`](https://docs.h5py.org/en/stable/): reading/writing HDF5 output files
- [`emcee`](https://emcee.readthedocs.io/en/stable/): performing MCMC sampling
- [`numba`](https://numba.pydata.org/): accelerating some basic numerical calculations (i.e. fault model predictions)
- [`corner`](https://corner.readthedocs.io/en/latest/): plotting MCMC sampling results
- [`pyproj`](https://pypi.org/project/pyproj/): handling geographic coordinate systems
- [`cartopy`](https://scitools.org.uk/cartopy/docs/latest/gallery/index.html): handling and plotting geopatial datasets

I would recommend using [Conda](https://conda.io/projects/conda/en/latest/index.html) to create a new Python environment to install and manage these packages. 


## Example A: Bayesian inversion for single rectangular dislocation.
In circumstances where information regarding the geometry and/or orientation of a fault producing a large earthquake are are poorly known it may be useful to perform an inversion for a simplified fault model consisting of a single rectangular (i.e. ``Okada'') dislocation, where the position, dimension, orientation, and slip are simultaneously estimated. In this particular example, I have implemented this procedure using a Markov Chain Monte Carlo (MCMC) algorithm, which allows for uncertainty quantification on the estimated fault parameters. This simplified inversion and resulting parameter estimates may be used as inputs and/or constaints to deriving a more complex finite-fault model to analyze the event's slip distribution. 

To demonstrate the inversion, I have created a synthetic example for the case of a (very simple) $M_w$ 7.3 earthquake along the San Jacinto fault system in Southern California. I use a single rectangular disclotion with 3 m of dextral slip to generate surface displacements, which are then projected into the line-of-sight direction (LOS) for Sentinel-1's descending track 173 over the Salton Trough. To generate semi-realistic noise, I introduce observed radar decorrelation from real Sentinel-1 observations (Vavra et al., 2024), simulated rupture decorrelation along the fault trace, and spatially correlated noise to simulate troposphere delays (Emardson et al., 2003) the synthetic LOS data. The noisy data are then down-sampeld using a quadtree algorithm (Jonsson, 2002; Simons, 2002) in order to reduce the computational cost of the inversion. In this example, two realizations of synthetic data (same fault displacements, different troposphere noise) are used since multiple InSAR datasets are often available for inverting. The data are then inverted for a single-patch fault model where the origin coordinates `x` and `y`, fault `strike`, fault `dip`, along-strike length `l`, along-dip width `w`, and `strike_slip` and `dip_slip` amplitudes are estimated. 

### 1. Synthetic data
![alt text](https://github.com/evavra/pyffit/blob/main/examples/LOS_clean.png "Synthetic line-of-sight displacements")

![alt text](https://github.com/evavra/pyffit/blob/main/examples/LOS_noisy.png "Synthetic line-of-sight displacements with added noise")


### 2. Quadtree downsampling
![alt text](https://github.com/evavra/pyffit/blob/main/examples/quadtree_init_synthetic_data_1.png "Downsampled line-of-sight displacements")


### 3. MCMC Results
![alt text](https://github.com/evavra/pyffit/blob/main/examples/triangle.png "Triangle plot")


