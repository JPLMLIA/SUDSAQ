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
from joblib import load
#from config_all import REQUIRED_VARS
import pickle
import xarray as xr
import pandas as pd

def main(months, sub_dir):

    #-----------------read in data
    root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
    if not os.path.exists(root_dir):
        root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
    
    summ_dir = f'{root_dir}/summaries/{sub_dir}'
    models_dir = f'{root_dir}/models/{sub_dir}'
    if months == 'all':
        months = glob.glob(f'{models_dir}/*/')
    else:
        if len(months) < 1:
            months = np.atleast_1d(months)
        months = [f'{models_dir}/{x}/' for x in months]
    
    #save inputs
    for dirs in tqdm(months, desc='Merging data by month'):
        m = dirs.split('/')[-2]
        #print(f'merging -----> {dirs}, month {m}')
        files_y0 = glob.glob(f'{dirs}/*/test.predict.nc')
        if len(files_y0) < 1:
            print(f'no predictions found')
            continue

        out_dir = f'{summ_dir}/combined_data/{m}/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        #save predicts
        #print(f'merging -----> yhat/test.predict')
        files_y0 = glob.glob(f'{dirs}/*/test.predict.nc')
        data_y0 = xr.open_mfdataset(files_y0, parallel=True)
        
        if data_y0.lon.max() > 180:
            lons = (data_y0.lon + 180) % 360 - 180
            data_y0['lon'] = lons
            data_y0 = data_y0.sortby('lon')
        xr.save_mfdataset([data_y0], [f'{out_dir}/test.predict.nc'])
        
        
        #to match predictions so that the size of data is the same
        match_dirs = ['/'.join(x.split('/')[:-1]) for x in files_y0]

        #print(f'merging + mean -----> X/test.data')
        #files_momo = glob.glob(f'{dirs}/*/test.data.nc')
        files_x = [x + '/test.data.nc' for x in match_dirs]
        data = xr.open_mfdataset(files_x, parallel=True)
        
        if data.lon.max() > 180:
            lons = (data.lon + 180) % 360 - 180
            data['lon'] = lons
            data = data.sortby('lon')
        # dat_momo = []
        # for f in files_x:
        #     dat_momo.append(xr.open_dataset(f))
        # data = xr.merge(dat_momo)
        
        # filename = f'{out_dir}/test.data.nc'
        # if os.path.exists(filename):
        #     os.remove(filename)
        # xr.save_mfdataset([data], [filename], engine = 'netcdf4')
        
        data_mean = data.mean(dim='time', skipna= True)
        filename = f'{out_dir}/test.data.mean.nc'
        if os.path.exists(filename):
            os.remove(filename)
        xr.save_mfdataset([data_mean], [filename], engine = 'netcdf4')
        
    
        #make to remove leap year day in feb
        if m == 'feb': 
            f = f'{models_dir}/feb/2012/test.target.nc'
            dat_year = xr.open_dataset(f, lock = False)
            #print(dat_year.time.shape[0])
            if dat_year.time.shape[0] == 29:
                os.rename(f, f'{models_dir}/feb/2012/test1.target.nc')
                dat = dat_year.drop_sel(time = pd.Timestamp('2012-02-29'))
                #dat = dat_year.where(dat_year.time != '2012-02-29', drop=True)
                xr.save_mfdataset([dat], [f], engine = 'netcdf4')
                
        #save truth, sometimes times are not sorted
        #print(f'merging -----> y/test.target')
        #files_y = glob.glob(f'{dirs}/*/test.target.nc')
        files_y = [x + '/test.target.nc' for x in match_dirs]
        #data_y = xr.open_mfdataset(files_y, parallel=True)
        data_y = []
        for f in files_y:
            data_y.append(xr.open_dataset(f))
        data_y = xr.merge(data_y)
        
        if data_y.lon.max() > 180:
            lons = (data_y.lon + 180) % 360 - 180
            data_y['lon'] = lons
            data_y = data_y.sortby('lon')
        # f = f'{models_dir}/feb/2012/test.target.nc'
        # dat_year = xr.open_dataset(f)
        # dat = dat_year.drop_sel(time = ('2012-02-29'))
        # xr.save_mfdataset([dat], [f])
        xr.save_mfdataset([data_y], [f'{out_dir}/test.target.nc'])
        
        
        
        #print(f'merging + mean -----> contributions/test.contributions')
        #files_momo = glob.glob(f'{dirs}/*/test.contributions.nc')
        files_c = [x + '/test.contributions.nc' for x in match_dirs]
        data = xr.open_mfdataset(files_c, parallel=True)
        
        if data.lon.max() > 180:
            lons = (data.lon + 180) % 360 - 180
            data['lon'] = lons
            data = data.sortby('lon')
        # dat_momo = []
        # for f in files_c:
        #     dat_momo.append(xr.open_dataset(f))
        # data = xr.merge(dat_momo)
        # filename = f'{out_dir}/test.contributions.nc'
        # if os.path.exists(filename):
        #     os.remove(filename)
        # xr.save_mfdataset([data], [filename], engine = 'netcdf4')
        
        data_mean = data.mean(dim='time', skipna= True)
        filename = f'{out_dir}/test.contributions.mean.nc'
        if os.path.exists(filename):
            os.remove(filename)
        xr.save_mfdataset([data_mean], [filename], engine = 'netcdf4')


        #print(f'merging -----> importances/test.importances')
        files_imp = glob.glob(f'{dirs}/*/test.importance.h5')
        mi = []; pi = []; mi_names = []; pi_names = []
        for file in files_imp:
            with closing(h5py.File(file, 'r')) as f:
                P = len(f['model']['block0_values'][0,:])
                mi.append(f['model']['block0_values'][0,:])
                mi_names.append(f['model']['axis0'][:].astype(str))
                
                if 'permutation' in f.keys():
                    pi.append(f['permutation']['block0_values'][0, :])
                    pi_names.append(f['permutation']['axis0'][:].astype(str))
                    #pi_months.append(file.split('/')[-3])
                else:
                    pi.append(np.repeat(np.nan, P))
                    pi_names.append(f['model']['axis0'][:].astype(str))
                    
        with closing(h5py.File(f'{out_dir}/test.importances.h5', 'w')) as f:
            f['model/values'] = np.column_stack(mi)
            f['model/names'] = np.column_stack(mi_names).astype(np.string_)
            f['permutation/values'] = np.column_stack(pi)
            f['permutation/names'] = np.column_stack(pi_names).astype(np.string_)
                

    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_dir', type=str, default = '/models/2011-2015/bias-8hour/')
    #parser.add_argument('--out_dir', type=str, default = '/summaries/2011-2015/bias-8hour/')
    parser.add_argument('--months', default = 'all', nargs = '*', type=str)
    #parser.add_argument('--parameter', type=str, default=None)

    args = parser.parse_args()
    main(**vars(args)) 


    
    
    

    





