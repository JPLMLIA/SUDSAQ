#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 11:05:24 2022

@author: marchett
"""
import os, glob
import sys
import numpy as np
import h5py
from tqdm import tqdm
from contextlib import closing
from joblib import load
#from config_all import REQUIRED_VARS
import pickle
import xarray as xr


def main(sub_dir):

    #sub_dir = '/bias/local/8hr_median/v4.1/'   
    #-----------------read in data
    root_dir = '/Volumes/MLIA_active_data-1/data_SUDSAQ/'
    if not os.path.exists(root_dir):
        root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
    
    summ_dir = f'{root_dir}/summaries/{sub_dir}'
    models_dir = f'{root_dir}/models/{sub_dir}'
    # if months == 'all':
    #     months = glob.glob(f'{models_dir}/*/')
    # else:
    #     if len(months) < 1:
    #         months = np.atleast_1d(months)
    #     months = [f'{models_dir}/{x}/' for x in months]
    
    years = ['2005', '2006', '2007', '2008', '2009', '2010', 
             '2011', '2012', '2013', '2014', '2015']
    
    #years = ['2005', '2006', '2007', '2008', '2009', '2010']
    #years = ['2011', '2012', '2013', '2014', '2015']
   
    for year in tqdm(years, desc = 'Saving data annually'):
        
        out_dir = f'{summ_dir}/annual_data/{year}/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        
        #save predicts
        print(f'merging -----> yhat/test.predict')
        files_y0 = glob.glob(f'{models_dir}/*/{year}/test.predict.nc')
        data_y0 = xr.open_mfdataset(files_y0, parallel=True, lock=False)
        lons = (data_y0.lon + 180) % 360 - 180
        data_y0['lon'] = lons
        data_y0 = data_y0.sortby('lon')
        
        filename = f'{out_dir}/test.predict.nc'
        if os.path.exists(filename):
            os.remove(filename)
        xr.save_mfdataset([data_y0], [filename], engine = 'netcdf4') 
        
        
        #to match predictions so that the size of data is the same
        match_dirs = ['/'.join(x.split('/')[:-1]) for x in files_y0]
        
        
        # print(f'merging -----> y/test.target')
        #files_y = glob.glob(f'{dirs}/*/test.target.nc')
        files_y = [x + '/test.target.nc' for x in match_dirs]
        #data_y = xr.open_mfdataset(files_y, parallel=True)
        data_y = []
        for f in files_y:
            data_y.append(xr.open_dataset(f))
        data_y = xr.merge(data_y)
        lons = (data_y.lon + 180) % 360 - 180
        data_y['lon'] = lons
        data_y = data_y.sortby('lon')
        xr.save_mfdataset([data_y], [f'{out_dir}/test.target.nc'])
        
        
        #save contributions
        print(f'merging + mean -----> contributions/test.contributions')
        files_c = [x + '/test.contributions.nc' for x in match_dirs]
        data = xr.open_mfdataset(files_c, parallel=True)
        lons = (data.lon + 180) % 360 - 180
        data['lon'] = lons
        data = data.sortby('lon')
        #data_mean = data.mean(dim='time', skipna= True)
        filename = f'{out_dir}/test.contributions.nc'
        if os.path.exists(filename):
            os.remove(filename)
        xr.save_mfdataset([data], [filename], engine = 'netcdf4')
        
        #save data
        print(f'merging + mean -----> X/test.data')
        files_x = [x + '/test.data.nc' for x in match_dirs]
        data = xr.open_mfdataset(files_x, parallel=True, lock = False)
        lons = (data.lon + 180) % 360 - 180
        data['lon'] = lons
        data = data.sortby('lon')
        filename = f'{out_dir}/test.data.nc'
        if os.path.exists(filename):
            os.remove(filename)
        xr.save_mfdataset([data], [filename], engine = 'netcdf4')
        
        

    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_dir', type=str, default = '/models/2011-2015/bias-8hour/')
    #parser.add_argument('--out_dir', type=str, default = '/summaries/2011-2015/bias-8hour/')
    #parser.add_argument('--months', default = 'all', nargs = '*', type=str)
    #parser.add_argument('--parameter', type=str, default=None)

    args = parser.parse_args()
    main(**vars(args)) 


    
    
    

    





