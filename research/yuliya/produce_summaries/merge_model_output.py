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
    for m, dirs in enumerate(months):
        m = dirs.split('/')[-2]
        print(f'merging -----> {dirs}, month {m}')

        out_dir = f'{summ_dir}/combined_data/{m}/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        
        #save predicts
        print(f'merging -----> yhat/test.predict')
        files_y0 = glob.glob(f'{dirs}/*/test.predict.nc')
        data_y0 = xr.open_mfdataset(files_y0, parallel=True)
        # dat_momo = []
        # for f in files_y0:
        #     dat_momo.append(xr.open_dataset(f))    
        # data = xr.merge(dat_momo)
        xr.save_mfdataset([data_y0], [f'{out_dir}/test.predict.nc'], engine = 'netcdf4')
        
        #to match predictions so that the size of data is the same
        match_dirs = ['/'.join(x.split('/')[:-1]) for x in files_y0]

        print(f'merging + mean -----> X/test.data')
        #files_momo = glob.glob(f'{dirs}/*/test.data.nc')
        files_x = [x + '/test.data.nc' for x in match_dirs]
        data = xr.open_mfdataset(files_x, parallel=True)
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
        
    
        #save truth
        print(f'merging -----> y/test.target')
        #files_y = glob.glob(f'{dirs}/*/test.target.nc')
        files_y = [x + '/test.target.nc' for x in match_dirs]
        data_y = xr.open_mfdataset(files_y, parallel=True)
        # dat_momo = []
        # for f in files_y:
        #     dat_momo.append(xr.open_dataset(f))    
        # data = xr.merge(dat_momo)
        xr.save_mfdataset([data_y], [f'{out_dir}/test.target.nc'], engine = 'netcdf4')
        
        
        
        print(f'merging + mean -----> contributions/test.contributions')
        #files_momo = glob.glob(f'{dirs}/*/test.contributions.nc')
        files_c = [x + '/test.contributions.nc' for x in match_dirs]
        data = xr.open_mfdataset(files_c, parallel=True)
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


        print(f'merging -----> importances/test.importances')
        files_imp = glob.glob(f'{dirs}/*/test.importance.h5')
        mi = []; pi = []; mi_names = []; pi_names = []
        for file in files_imp:
            with closing(h5py.File(file, 'r')) as f:
                mi.append(f['model']['block0_values'][0,:])
                mi_names.append(f['model']['axis0'][:].astype(str))
                
                pi.append(f['permutation']['block0_values'][0, :])
                pi_names.append(f['permutation']['axis0'][:].astype(str))
                #pi_months.append(file.split('/')[-3])
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


    
    
    

    





