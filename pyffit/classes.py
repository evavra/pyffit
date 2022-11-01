import numpy as np
import xarray as xr
import datetime as dt


class Interferogram:
    """
    Interferogram object designed around GMTSAR interferogram conventions.
    
    ATTRIBUTES:
    path - path to location of interferogram data files
           Should follow GMTSAR interferogram naming convention:
               pathYYYYMMDD_YYYYMMDD

    Will search for: 
    amp       - SAR amplitude
    phase     - normal interferogram phase
    phasefilt - filtered interferogram phase
    corr      - interferogram coherence
    mask      - unwrapping mask
    landmask  - land/water mask
    unwrap    - unwrapped phase
    """

    def __init__(self, path, verbose=True, **kwargs):
        # Initialize attributes
        self.path      = path
        self.defaults  = ['amp',
                          'phase',
                          'phasefilt',
                          'corr',
                          'mask',
                          'landmask',
                          'unwrap',]
        self.data      = {}
        
        # Spatial
        self.x         = None
        self.y         = None
        self.dims      = None
        self.xmin      = None
        self.xmax      = None
        self.ymin      = None
        self.ymax      = None
        self.xlim      = None
        self.ylim      = None
        self.extent    = None
        self.region    = None
        self.track     = None

        # Temporal
        self.date_pair = None
        self.dates     = None
        self.days      = None

        # Load default attributes
        self.load_track()
        self.load_temporal_info()
        self.load_data(files=self.defaults, verbose=verbose)

        # Load additional attributes
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])


    def load_data(self, files=[], verbose=True):
        """
        Attempt to load interferogram attributes.
        If files do not exist, attribute value will remain None.
        Non-default files may be speficied with files argument.
        """

        for data_type in self.defaults:
            try:
                file_path = f'{self.path}/{data_type}.grd'

                with xr.open_dataset(file_path) as grd:
                    try:
                        # Try geographic coordinates
                        x = grd.lon.values
                        y = grd.lat.values
                        z = grd.z.values

                    except AttributeError:
                        # Try radar coordinates
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
                self.dims   = z.shape
                
                if self.track == 'D':
                    self.extent = [self.xmin,
                                   self.xmax,
                                   self.ymin,
                                   self.ymax,]
                else:
                    self.extent = [self.xmin,
                                   self.xmax,
                                   self.ymax,
                                   self.ymin,]
                self.region = self.extent

                # Set data values
                self.data[data_type] = z

            except ValueError:
                if verbose:
                    print(f'Warning: {file_path} not found')
        
        return


    def load_temporal_info(self,):
        """
        Extract datetime info from path.
        """

        # Date pair (directory name)
        self.date_pair   = self.path.split('/')[-1]

        # Datetime objects for dates
        self.dates = [dt.datetime.strptime(date, '%Y%m%d') for date in self.date_pair.split('_')]

        # Epoch length
        self.days = (self.dates[1] - self.dates[0]).days

        return


    def load_track(self, verbose=True):
        """
        Attempt to identify satellite orbital track from filepath.
        """

        # Possible split directory strings, which track directory could be above
        splits = ['F1', 'F2', 'F3', 'F4', 'F5', 'merge']

        for split in splits:
            track_strs = self.path.split(split) #[0].split('/')[-4]

            if len(track_strs) > 1:
                track = track_strs[0].split('/')[-2]

                if any(key in track for key in ['A', 'ASC', ]):
                    self.track = 'A'
                    return

                elif any(key in track for key in ['D', 'DES', ]):
                    self.track = 'D'
                    return

                else:
                    print('No satellite track identified.')
                    return
        return