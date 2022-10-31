import numpy as np
import xarray as xr


class Interferogram(intf_path):
    """
    Interferogram object designed around GMTSAR interferogram conventions.
    
    ATTRIBUTES:
    intf_path - path to location of interferogram data files
    
    Will search for: 
    phase     - normal interferogram phase
    phasefilt - filtered interferogram phase
    corr      - interferogram coherence
    mask      - unwrapping mask
    landmask  - land/water mask
    unwrap    - unwrapped phase

    """

    def __init__(self):
        # Initialize attributes
        self.path      = intf_path
        self.defaults  = ['phase',
                          'phasefilt',
                          'corr',
                          'mask',
                          'landmask'
                          'unwrap',]
                        
        self.x         = None
        self.y         = None
        self.dims      = None
        self.xmin      = None
        self.xmax      = None
        self.ymin      = None
        self.ymax      = None
        self.extent    = None

        self.phase     = None
        self.phasefilt = None
        self.corr      = None
        self.mask      = None
        self.unwrap    = None
        self.landmask  = None

        self.epoch     = None
        self.days      = None



    def load_default_data(self, path):
        """
        Attempt to load default interferogram attributes.
        If files do not exist, attribute value will remain None.
        """

        for file in self.defaults:
            try:
                with xr.open_dataset(f'{path}/{file}.grd') as grd:
                    try:
                        x = grd.lon.values
                        y = grd.lat.values
                        z = grd.z.values

                    except AttributeError:
                        x = grd.x.values
                        y = grd.y.values
                        z = grd.z.values

                # Set spatial attributes
                self.x      = x
                self.y      = y
                self.xmin   = np.nanmin(x)
                self.xmax   = np.nanmax(x)
                self.ymin   = np.nanmin(y)
                self.ymax   = np.nanmax(y)
                self.xlim   = (self.xmin, self.xmax)
                self.ylim   = (self.ymin, self.ymax)
                self.extent = [self.xmin,
                               self.xmax,
                               self.ymin,
                               self.ymax,]
                self.region = self.extent
                               
            except:
