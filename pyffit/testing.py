import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd 
from pyffit.data import read_grd
import glob
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.animation import FuncAnimation

file_stem = '/Users/evavra/Projects/SSAF/Data/InSAR/S1/timeseries/ts_20221210_unfilt_ll.grd'
file_stem = '/Users/evavra/Projects/SSAF/Data/InSAR/S1/timeseries/ts_*_unfilt_ll.grd'

files     = sorted(glob.glob(file_stem))

# Load InSAR data
x, y, z = read_grd(files[0])

if max(abs(x)) > 180:
    x -= 360
extent = [x.min(), x.max(), y.max(), y.min()]

# Load fault trace
ssaf = '/Users/evavra/Data/faults/SSAF_trace.dat'
fault = pd.read_csv(ssaf, header=None, delim_whitespace=True)
fault.columns = ['Longitude', 'Latitude']
fault['Longitude'] -= 360

print(np.nanmin(z), np.nanmax(z))

cmap       = 'coolwarm'
vlim       = (-10, 10)
cbar_label = 'LOS displacement (mm/yr)'
data_crs   = ccrs.PlateCarree()
projection = ccrs.RotatedPole(pole_longitude=10, pole_latitude=30)

# A rotated pole projection again...
fig = plt.figure(figsize=(16, 5))
ax  = fig.add_subplot(1, 1, 1, projection=projection)

cardinal_labels = {'north': '', 'east': '', 'south': '', 'west': ''}
lon_formatter = LongitudeFormatter(number_format='.1f',
                                   cardinal_labels=cardinal_labels
                                  )
lat_formatter = LatitudeFormatter(number_format='.1f',
                                  cardinal_labels=cardinal_labels
                                 )

ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='lightgray', edgecolor='black')

ax.set_xlim(42.65, 43.3)
ax.set_ylim(-8.55, -8.4)


# Get approximate bounds and set manually using axes coordinates
# ax.set_extent([-118, -115, 32, 33.5], crs=ccrs.PlateCarree())
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()

im = ax.imshow(z, cmap=cmap, extent=extent, interpolation='none', transform=data_crs, vmin=vlim[0], vmax=vlim[1])
ax.plot(fault['Longitude'], fault['Latitude'], c='k', transform=data_crs)
ax.add_feature(cfeature.LAKES.with_scale('10m'))

# Add gridlines
ax.gridlines(draw_labels={"top": "x", "left": "y"}, xformatter=lon_formatter, yformatter=lat_formatter, dms=False, linewidth=0.5, x_inline=False, y_inline=False,)
fig.colorbar(im, fraction=0.07, shrink=0.7, label=cbar_label, pad=0.01)
fig.tight_layout()
# plt.show()


n = len(files)
frames = np.arange(1, n)

def update_mesh(i):
    x, y, z = read_grd(files[i])
    # im.set_array(z)
    im = ax.imshow(z, cmap=cmap, extent=extent, interpolation='none', transform=data_crs, vmin=vlim[0], vmax=vlim[1])
    return im
    
    
frames = [i for i in range(1, n)] + [0]

# Go back to the start to make it a smooth repeat
ani = FuncAnimation(fig, update_mesh, frames=frames,
                    interval=1)
ani.save('animation.gif')

