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

    def __init__(self, path, files=[], verbose=True, **kwargs):
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
        
        # Spatial (radar coordinates)
        self.rng       = None
        self.azi       = None
        self.xmin_ra   = None
        self.xmax_ra   = None
        self.ymin_ra   = None
        self.ymax_ra   = None
        self.xlim_ra   = None
        self.ylim_ra   = None
        self.extent_ra = None
        self.region_ra = None
        self.dims_ra   = None

        # Spatial (radar coordinates)
        self.lon       = None
        self.lat       = None
        self.xmin_ll   = None
        self.xmax_ll   = None
        self.ymin_ll   = None
        self.ymax_ll   = None
        self.xlim_ll   = None
        self.ylim_ll   = None
        self.extent_ll = None
        self.region_ll = None
        self.dims_ll   = None

        self.track     = None

        # Temporal
        self.date_pair = None
        self.dates     = None
        self.days      = None

        # Load default attributes
        self.load_track()
        # self.load_temporal_info()
        self.load_data(files=self.defaults, verbose=verbose)

        if len(files) > 0:
            self.load_data(files=files, verbose=verbose)

        # Load additional attributes
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])


    def load_data(self, files=[], verbose=True):
        """
        Attempt to load interferogram attributes.
        If files do not exist, attribute value will remain None.
        Non-default files may be speficied with files argument.
        """

        for data_type in files:
            try:
                file_path = f'{self.path}/{data_type}.grd'

                with xr.open_dataset(file_path) as grd:
                    try:
                        # Try geographic coordinates
                        x = grd.lon.values
                        y = grd.lat.values
                        z = grd.z.values

                        # Set spatial attributes
                        self.lon      = x
                        self.lat      = y
                        self.xmin_ll   = np.nanmin(x)
                        self.xmax_ll   = np.nanmax(x)
                        self.ymin_ll   = np.nanmin(y)
                        self.ymax_ll   = np.nanmax(y)
                        self.xlim_ll   = (self.xmin_ll, self.xmax_ll)
                        self.ylim_ll   = (self.ymin_ll, self.ymax_ll)
                        self.dims_ll   = z.shape
                        
                        if self.track == 'D':
                            self.extent_ll = [self.xmin_ll,
                                           self.xmax_ll,
                                           self.ymin_ll,
                                           self.ymax_ll,]
                        else:
                            self.extent_ll = [self.xmin_ll,
                                           self.xmax_ll,
                                           self.ymax_ll,
                                           self.ymin_ll,]
                        self.region_ll = self.extent_ll

                    except AttributeError:
                        # Try radar coordinates
                        x = grd.x.values
                        y = grd.y.values
                        z = grd.z.values

                        # Set spatial attributes
                        self.rng      = x
                        self.azi      = y
                        self.xmin_ra   = np.nanmin(x)
                        self.xmax_ra   = np.nanmax(x)
                        self.ymin_ra   = np.nanmin(y)
                        self.ymax_ra   = np.nanmax(y)
                        self.xlim_ra   = (self.xmin_ra, self.xmax_ra)
                        self.ylim_ra   = (self.ymin_ra, self.ymax_ra)
                        self.dims_ra   = z.shape
                        
                        if self.track == 'D':
                            self.extent_ra = [self.xmin_ra,
                                           self.xmax_ra,
                                           self.ymin_ra,
                                           self.ymax_ra,]
                        else:
                            self.extent_ra = [self.xmin_ra,
                                           self.xmax_ra,
                                           self.ymax_ra,
                                           self.ymin_ra,]
                        self.region_ra = self.extent_ra

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