import pandas as pd
from map_utilities import proj_ll2utm

def read_traces(file, mode, EPSG='32611'):
    
    faults = {}

    with open(file, 'r') as f:
        # Load SCEC Community Fault Model file
        if mode == 'SCEC':
            for line in f:
                if line[0] == '#':
                    continue
                if line[0] == '>':
                    name = line.split()[-1][1:-1]
                    faults[name] = ({'name': name, 
                                      'lon': [], 
                                      'lat': []})
                else:
                    coords = [float(item) for item in line.split()]
                    faults[name]['lon'].append(coords[0])
                    faults[name]['lat'].append(coords[1])

        # Load digitized fault traces from QGIS .csv
        elif mode == 'QGIS':
            df = pd.read_csv(file, sep=',')
            for name in df['Name'].unique():
                df_fault = df[df['Name'] == name].sort_values('X')
                faults[name] = {'Name': name, 
                                'lon':  df_fault['X'].values, 
                                'lat':  df_fault['Y'].values,}
    

    # Sort so that DataFrame is in descending latitude 
    for name in faults:
        fault = faults[name]
        if fault['lat'][0] < fault['lat'][-1]:
            fault['lon'] = fault['lon'][::-1]
            fault['lat'] = fault['lat'][::-1]

    # Add UTM coordinates
    if EPSG:
        for name in faults.keys():
            faults[name]['UTMx'], faults[name]['UTMy'] = proj_ll2utm(faults[name]['lon'], faults[name]['lat'], EPSG)

    return faults
