#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 13:00:10 2022

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

from scipy import stats
import read_data_momo


# data_root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
# momo_root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/MOMO/'
# toar_output = f'{data_root_dir}/processed/summary_dp/TOAR2/'
# momo_output = f'{data_root_dir}/processed/summary_dp/MOMO/'


def main(years, months, dtype, plotting = True):
    
    root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
    if not os.path.exists(root_dir):
        root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
    #momo_root_dir = f'{root_dir}/MOMO/'
    
    #dtype = 'PRCP'
    isd_output = f'{root_dir}/processed/summary_dp/ISD/{dtype}'
    momo_output = f'{root_dir}/processed/summary_dp/MOMO/' 
    
    if not os.path.exists(isd_output):
        print(f'no {dtype} variable found for matching...')
        return None
    
    # years = ['2012']
    # months = ['06', '07', '08']
    # names = ['o3', 't', 'q']
    summaries = ['mean', 'std', 'count']
    if len(months) == 0:
        months = [f'{x}'.zfill(2) for x in np.arange(1, 13)]
    
    years = np.atleast_1d(years) 
    months = np.atleast_1d(months)
    
 year in years:
        
        for month in tqdm(months):
            #save
            file = f'{isd_output}/ghcnd_{year}_{month}.h5'
            with closing(h5py.File(file, 'r')) as f:
                    lon = f['lon'][:]
                    lat = f['lat'][:]
                    data = f['data'][:]
                    date = f['date'][:].astype(str)
                    station =  f['station'][:].astype(str)
            
           
            momo_file = f'{momo_output}/momo_{year}_{month}.h5' 
            momo = {}
            with closing(h5py.File(momo_file, 'r')) as f:
                keys = list(f)
                for k in keys:
                    momo[k] = f[k][:]
            
            momo_lon = np.hstack([momo['lon'], 360.]) - 180.
            momo_lat = np.hstack([momo['lat'], 90.])    
            x, y = np.meshgrid(momo['lon'] - 180., momo['lat'])
            
            #mdaily daytime average
            days = np.unique(momo['date'][:,2].astype(str))
            #mean_hours = [f'{x}'.zfill(2) for x in np.arange(8, 17)]
            
            data_to_momo = {k: [] for k in summaries}
            for d in days:
                mask_data = np.in1d(date[:, 2], d)                
                for s in summaries:
                    ret = stats.binned_statistic_2d(lat[mask_data], lon[mask_data], 
                                                data[mask_data], s, 
                                                bins=[momo_lat, momo_lon],
                                                expand_binnumbers = True)
                    
                    #TO DO: need to make sure correctly formatted
                    temp = ret.statistic
                    temp[temp == 0] = np.nan
                    data_to_momo[s].append(temp)
                
            data_to_momo = {s: np.stack(data_to_momo[s]) for s in summaries} 
            
            if plotting:
                plot_dir = f'{root_dir}/processed/plots/inputs/isd_{dtype}/'
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                 
                #montly mean
                for s in ['mean', 'std']:
                    stat = np.nanmean(data_to_momo[s], axis = 0)
                    fig = plt.figure(figsize=(18, 9))
                    ax = plt.subplot(projection = ccrs.PlateCarree())
                    plt.pcolor(x, y, stat, cmap = 'coolwarm')
                    plt.clim(np.nanpercentile(stat, [1, 99]))
                    plt.colorbar()
                    ax.set_global()
                    ax.coastlines()
                    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
                    gl.xformatter = LONGITUDE_FORMATTER
                    gl.yformatter = LATITUDE_FORMATTER
                    ax.stock_img()
                    plt.title(f'monthly {s}, {dtype}, year = {year}, month = {month}')
                    ax.set_extent([-140, -50, 10, 80], crs=ccrs.PlateCarree())
                    plt.savefig(f'{plot_dir}/{dtype}_{year}_{month}_{s}.png', bbox_inches = 'tight')
                    plt.close()
                
            
            #save matched  
            output_dir = f'{root_dir}/processed/coregistered/inputs/{dtype}/' 
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            with closing(h5py.File(f'{output_dir}/{dtype}_matched_{year}_{month}.h5', 'w')) as f:
                for s in summaries:
                    f['data/' + str(s)] = data_to_momo[s]
                f['lon'] = momo['lon']
                f['lat'] = momo['lat']
                f['date'] = momo['date']
                    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    #parser.add_argument('data_root_dir', type=str)
    parser.add_argument('--years', default = ['2012'], nargs = '*', type=str)
    parser.add_argument('--months', default = [], nargs = '*', type=str)
    parser.add_argument('--dtype', type=str, default='PRCP')
    parser.add_argument('--plotting', type=bool, default=True)
    #parser.add_argument('out_file', type=str)

    args = parser.parse_args()
    main(**vars(args))


                    
                    