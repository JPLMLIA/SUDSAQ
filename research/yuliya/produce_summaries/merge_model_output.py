#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 11:05:24 2022

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
from datetime import datetime, timedelta, date
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import scipy as sp
from joblib import load
#from config_all import REQUIRED_VARS
import pickle
import xarray as xr


def main(months, models_dir):

    #-----------------read in data
    root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
    if not os.path.exists(root_dir):
        root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
    
    summ_dir = f'{root_dir}/summaries/{models_dir}'
    models_dir = f'{root_dir}/models/{models_dir}'
    if months == 'all':
        months = glob.glob(f'{models_dir}/*/')
    else:
        if len(months) < 1:
            months = np.atleast_1d(months)
        months = [f'{models_dir}/{x}/' for x in months]
    
    #save inputs
    for m, dirs in enumerate(months):
        m = dirs.split('/')[-2]
        print(f'merging -----> {dirs}, month {m}')

        out_dir = f'{summ_dir}/combined_data/{m}/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        print(f'merging + mean -----> X/test.data')
        files_momo = glob.glob(f'{dirs}/*/test.data.nc')
        #ds_momo = xr.open_mfdataset(files_momo, parallel=True)
        dat_momo = []
        for f in files_momo:
            dat_momo.append(xr.open_dataset(f))
        data = xr.merge(dat_momo)
        
        # filename = f'{out_dir}/test.data.nc'
        # if os.path.exists(filename):
        #     os.remove(filename)
        # xr.save_mfdataset([data], [filename], engine = 'netcdf4')
        
        data_mean = data.mean(dim='time', skipna= True)
        filename = f'{out_dir}/test.data.mean.nc'
        if os.path.exists(filename):
            os.remove(filename)
        xr.save_mfdataset([data_mean], [filename], engine = 'netcdf4')


        #save predicts
        print(f'merging -----> yhat/test.predict')
        files_momo = glob.glob(f'{dirs}/*/test.predict.nc')
        #ds_momo = xr.open_mfdataset(files_momo, parallel=True)
        dat_momo = []
        for f in files_momo:
            dat_momo.append(xr.open_dataset(f))    
        data = xr.merge(dat_momo)
        xr.save_mfdataset([data], [f'{out_dir}/test.predict.nc'], engine = 'netcdf4')
    
        #save truth
        print(f'merging -----> y/test.target')
        files_momo = glob.glob(f'{dirs}/*/test.target.nc')
        #ds_momo = xr.open_mfdataset(files_momo, parallel=True)
        dat_momo = []
        for f in files_momo:
            dat_momo.append(xr.open_dataset(f))    
        data = xr.merge(dat_momo)
        xr.save_mfdataset([data], [f'{out_dir}/test.target.nc'], engine = 'netcdf4')
        

        print(f'merging + mean -----> contributions/test.contributions')
        files_momo = glob.glob(f'{dirs}/*/test.contributions.nc')
        #ds_momo = xr.open_mfdataset(files_momo, parallel=True)
        dat_momo = []
        for f in files_momo:
            dat_momo.append(xr.open_dataset(f))
        data = xr.merge(dat_momo)
        # filename = f'{out_dir}/test.contributions.nc'
        # if os.path.exists(filename):
        #     os.remove(filename)
        # xr.save_mfdataset([data], [filename], engine = 'netcdf4')
        
        data_mean = data.mean(dim='time', skipna= True)
        filename = f'{out_dir}/test.contributions.mean.nc'
        if os.path.exists(filename):
            os.remove(filename)
        xr.save_mfdataset([data_mean], [filename], engine = 'netcdf4')

    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', type=str, default = '/models/2011-2015/bias-8hour/')
    #parser.add_argument('--out_dir', type=str, default = '/summaries/2011-2015/bias-8hour/')
    parser.add_argument('--months', default = 'all', nargs = '*', type=str)
    #parser.add_argument('--parameter', type=str, default=None)

    args = parser.parse_args()
    main(**vars(args)) 


    
    
    

    





