#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:42:54 2022

@author: marchett
"""
import os, glob
import sys
import json
import numpy as np
import h5py
from scipy.io import netcdf
from tqdm import tqdm
from contextlib import closing
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import xarray as xr
from tqdm import tqdm

#sys.path.append('/Users/marchett/Documents/SUDS_AQ/dev/suds-air-quality/research/yuliya/produce_summaries')
import summary_plots as plots


#-----------------read in data
# root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
# if not os.path.exists(root_dir):
#     root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
    
    
# #set the directory for th60at month
# sub_dir = '/bias/local/8hr_median/v1/'
# models_dir = f'{root_dir}/models/{sub_dir}'
# #set plot directory
# summaries_dir = f'{root_dir}/summaries/{sub_dir}/combined_data/'
# plots_dir = f'{root_dir}/summaries/{sub_dir}/summary_plots/'


def main(sub_dir, months = 'all'):
    
    if months == 'all':
        months = plots.MONTHS
    
    #-----------------read in data
    root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
    if not os.path.exists(root_dir):
        root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
        
    #sub_dir = '/bias/local/8hr_median/v1/'
    models_dir = f'{root_dir}/models/{sub_dir}'

    #set output directory
    summaries_dir = f'{root_dir}/summaries/{sub_dir}/combined_data/'
    # create one if it's not there
    # if not os.path.exists(plots_dir):
    #     os.makedirs(plots_dir)
    
    for month in months:
        print(f'-----> making X and y for {month}')
        data_x = np.hstack(glob.glob(f'{models_dir}/{month}/*/test.data.nc'))
        data_y = glob.glob(f'{models_dir}/{month}/*/test.target.nc')
        data_y0 = glob.glob(f'{models_dir}/{month}/*/test.predict.nc')
        
        
        #train data
        ds_y = xr.open_mfdataset(data_y)
        # ds_y_flat = ds_y.stack(coord=("lon", "lat", 'time'))
        # ds_y_drop = ds_y_flat.dropna(dim='coord')
        ds_y_ = ds_y.to_array().stack({'loc': ["lon", "lat", 'time']})
        mask = ~np.isnan(ds_y_.values[0])
        
        y = ds_y_.values[0][mask]
        y_full = ds_y_.values[0]
        years = ds_y_['time.year'].values[mask]
        days = ds_y_['time.day'].values[mask]
        lons = ds_y_['lon'].values[mask]
        lats = ds_y_['lat'].values[mask]
        
        
        ds_x = xr.open_mfdataset(data_x)
        ds_x_ = ds_x.to_array().stack({'loc': ["lon", "lat", 'time']})
        var_names = ds_x_['variable'].values
        X = ds_x_.values[:, mask].T
        
        #original prediction
        ds_y0 = xr.open_mfdataset(data_y0)
        ds_y0_ = ds_y0.to_array().stack({'loc': ["lon", "lat", 'time']})
        y0 = ds_y0_.values[0][mask]
        y0_full = ds_y0_.values[0]
        
        #save ML ready data
        # xr.Dataset(
        #     {'X': ('coords', 'var'), X},
        #     coords = {'coords': np.arange(len(X), 
        #               'var': var_names,
        #               'lons': lons,
        #               'lats': lats} )
        
        save_file = f'{summaries_dir}/{month}/data.h5'
        with closing(h5py.File(save_file, 'w')) as f:
                     f['X'] = X
                     f['y'] = y
                     f['y0'] = y0
                     f['lons'] = lons
                     f['lats'] = lats
                     f['years'] = years
                     f['days'] = days
                     f['mask'] = mask
                     f['var_names'] = var_names
    
        #return X, y, y0, years, days, lons, lats, mask, var_names



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_dir', type=str, default = '/bias/local/8hr_median/v1/')
    parser.add_argument('--months', default = 'all', nargs = '*', type=str)
    #parser.add_argument('--parameter', type=str, default=None)

    args = parser.parse_args()
    main(**vars(args)) 

