import numpy as np
def prepare_datasets_gps(gps_files, weights, ref_point, EPSG='32611',data_type='2d'):
    """
    Ingest gps data and convert them to local Cartesian coordinate system
    Specify the dimension of your data (2d or 3d)


    INPUT:
    gps data (station, lon, lat, ux, uy, (uz), std_x, std_y, (std_z))
    uz should be positive downward.

    OUTPUT:
    datasets - dictonary containing the following attributes:
        x_samp (k,)        - x-coordinates (km)
        y_samp (k,)        - y-coordinates (km)
        data_samp (k,2)/ (k,3)     - data values (see file for original units)
        data_samp_std (k,2) / (k,3) - data standard deviations (see file for original units)

    """

    datasets = {}
    
    for i in range(len(gps_files)):
        # Get dataset name
        dataset = gps_files[i].split('/')[-1][:-4]

        # Get dataset weight
        weight  = weights[i]

        # Read data
        df = pd.read_csv(dataset,delim_whitespace=True, header=0)
        
        if data_type=='2d':
        	df.columns=['station','lon','lat','ue','un','std_x','std_y']
        	data_samp=np.array([df['ue'].values,df['un'].values])
        	data_samp_std=np.array([df['std_x'].values,df['std_y'].values])
        else:
        	df.columns=['station','lon','lat','ue','un','uz','std_x','std_y','std_z']
        	data_samp=np.array([df['ue'].values,df['un'].values,df['uz'].values])
        	data_samp_std=np.array([df['std_x'].values,df['std_y'].values,df['std_z'].values])
        	
        df['UTMx'],df['UTMy'] = proj_ll2utm(df['lon'].values,df['lat'].values,EPSG)
        xo,yo = proj_ll2utm(ref_point[0],ref_point[1],EPSG)
        
        x_samp=df['UTMx'].values - xo
        y_samp=df['UTMy'].values - yo
        
        print(f'GPS data points: {len(data)}')

        # Add datasets to dictionary
        datasets[dataset]                  = {}
        datasets[dataset]['x_samp']        = x_samp
        datasets[dataset]['y_samp']        = y_samp
        datasets[dataset]['data_samp']     = data_samp
        datasets[dataset]['data_samp_std'] = data_samp_std
        datasets[dataset]['weight']        = weight

return datasets
    
    
