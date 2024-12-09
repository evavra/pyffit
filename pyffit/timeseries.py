import data
import utilities
import corrections
import figures
import numpy as np
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc


def main():


    return

def correct_time_series():

    return


def get_point_time_series(dataset, pos, coords=['lon', 'lat'], EPSG='32611', verbose=True, radius=100):
        
    """
    Given an input lat/con coordinate, extract the pixel-wise time series from an InSAR dataset,
    
    INPUT:
    dataset - xarray Dataset containing full time series
    coords  - x/y or lon/lat coordinates for extracting time series

    OUTPUT:
    ts - timeseries
    """

    # Convert to UTM if in geographic coordinates
    if any([coord.lower() in ['lon', 'longitude', 'lat', 'latitude'] for coord in list(dataset.coords)]):
        print(f'Projecting geographic coordinates {coords} to UTM using EPSG: {EPSG}')

        lon, lat     = np.meshgrid(dataset[coords[0]].data, dataset[coords[1]].data)
        x, y         = utilities.proj_ll2utm(lon, lat, EPSG)
        x_pos, y_pos = utilities.proj_ll2utm(pos[0], pos[1], EPSG)

    else:
        x = dataset[coords[0]].data
        y = dataset[coords[1]].data
        x_pos, ypos = pos


    dist = np.sqrt((x - x_pos)**2 + (y - y_pos)**2)

    selection = dist <= radius

    expanded_selection = selection[np.newaxis, :, :]
    selected_data = dataset.where(expanded_selection)


    dates = selected_data['date']
    ts = selected_data['z'].mean(dim=['lon', 'lat'])

    fig, ax = plt.subplots(figsize=(14, 8.2))
    ax.plot(dates, ts)
    plt.show()
    
    # print(selected_data['z'].mean(dim='date'))
    # print(dataset['z'].data[:, i].shape)
    # print(selected_data)
    return




if __name__ == '__main__':
    main()