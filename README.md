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



## Example 1: Bayesian inversion for single rectangular dislocation.
In circumstances where information regarding the geometry and/or orientation of a fault producing a large earthquake are are poorly known it may be useful to perform an inversion for a simplified fault model consisting of a single rectangular (i.e. ``Okada'') dislocation, where the position, dimension, orientation, and slip are simultaneously estimated. In this particular example, I have implemented this procedure using a Markov Chain Monte Carlo (MCMC) algorithm, which allows for uncertainty quantification on the estimated fault parameters. This simplified inversion and resulting parameter estimates may be used as inputs and/or constaints to deriving a more complex finite-fault model to analyze the event's slip distribution.

