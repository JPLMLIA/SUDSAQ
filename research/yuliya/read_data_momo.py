#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 13:57:30 2021

@author: marchett
"""

import glob, os
import numpy as np
import h5py
import netCDF4 as nc4
from scipy.io import netcdf
import matplotlib as mp
from contextlib import closing
from datetime import datetime, timedelta, date
from matplotlib import pyplot as plt
from tqdm import tqdm
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER



def daterange(start_date, end_date, hours = 2):
    if hours is not None:
        delta = timedelta(hours = hours)
    else:
        delta = timedelta(days = 1) 
    while start_date < end_date:
        yield start_date
        start_date += delta


def daily_average(x, dates, hours):
    mean_hours = [f'{x}'.zfill(2) for x in np.arange(8, 17)]
    days = np.unique(dates[:, 2])
    y = []
    for d in days:
        mask_d = np.in1d(dates[:, 2], d)
        mask_h = np.in1d(hours[mask_d], mean_hours)
        temp = x[mask_d][mask_h]
        y.append(temp.mean(axis = 0))
    
    return y   



def main(outputs, inputs, years, months, mda8 = True):
    
    
    if len(months) == 0:
        months = [f'{x}'.zfill(2) for x in np.arange(1, 13)]
    
    
    years = np.atleast_1d(years)
    months = np.atleast_1d(months)
    
    root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
    if not os.path.exists(root_dir):
        root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
    momo_root_dir = f'{root_dir}/MOMO/'
    data_output_dir = f'{root_dir}/processed/summary_dp/MOMO/'
    
    if inputs == 'None':
        inputs = None
    if inputs == 'all':
        subdirs = glob.glob(momo_root_dir+ '/inputs/*')
        inputs = [x.split('/')[-1] for x in subdirs]
        #inputs = ['t', 'q', 'ps', 'u', 'v']
    print(f'getting data for: {inputs}')    
    #data_dict = {}
    
    for year in years:
        print(f'year -----> {year}')
        #output_file = glob.glob(f'{momo_root_dir}/outputs/2hr_{outputs}_{year}*')[0]
        output_file = glob.glob(f'{momo_root_dir}/mda8/{outputs}_mda8_*{year}*')[0]
        nc_out = netcdf.netcdf_file(output_file,'r')
        k = list(nc_out.variables.keys())[-1]
    
        # data_dict = {}
        # data_dict['lon'] = nc_out.variables['lon'][:]
        # data_dict['lat'] = nc_out.variables['lat'][:]
        momo_dat = nc_out.variables[k][:]
        
        if inputs is not None:
            inputs_dict = {}
            for v in inputs:
                input_file = glob.glob(f'{momo_root_dir}/inputs/{v}/2hr_{v}_{year}*')[0]
                nc_in = netcdf.netcdf_file(input_file,'r')
                k = list(nc_in.variables.keys())[-1]
                inputs_dict[v] = nc_in.variables[k][:]
        
        start_date = datetime(int(year), 1, 1)
        end_date = datetime(int(year) + 1, 1, 1)
        #for averaging inputs
        dates_dt = []
        for single_date in daterange(start_date, end_date, hours = 2):
            dates_dt.append(single_date.strftime("%Y-%m-%d %H:%M"))
        dates = np.row_stack([a.split(' ')[0].split('-') for a in dates_dt])
        hours = np.row_stack([a.split(' ')[1].split(':')[0] for a in dates_dt])
        
        
        #for subsetting the month if there's no hourly average
        dates_dt = []
        for single_date in daterange(start_date, end_date, hours = None):
            dates_dt.append(single_date.strftime("%Y-%m-%d %H:%M"))
        dates_output = np.row_stack([a.split(' ')[0].split('-') for a in dates_dt]) 
        hours_output = np.row_stack([a.split(' ')[1].split(':')[0] for a in dates_dt])
        
        #save monthly files
        for month in tqdm(months):
            
            data_dict = {}
            data_dict['lon'] = nc_out.variables['lon'][:]
            data_dict['lat'] = nc_out.variables['lat'][:]
            
            mask = dates_output[:, 1] == month
            if mda8:
                data_dict[outputs] = momo_dat[dates_output[:, 1] == month] 
            else:
                #TO DO: fix average for output too
                data_dict[outputs] = momo_dat[mask] 
            
            data_dict['date'] = dates_output[mask].astype(np.string_)
            data_dict['hour'] = hours_output[mask].astype(np.string_)
            
            if inputs is not None:
                for v in inputs:
                    mask_m = dates[:, 1] == month
                    input_daily = daily_average(inputs_dict[v][mask_m], 
                                                dates[mask_m], hours[mask_m])
                    data_dict[v] = np.stack(input_daily)
            
            [data_dict[k].shape for k in data_dict.keys()]
            #save
            ofile = f'momo_{year}_{month}.h5'
            with closing(h5py.File(data_output_dir + ofile, 'w')) as f:
                for k in data_dict.keys():
                    f[k] = data_dict[k]

    
    # outputs = get_momo(input_var = None, month = '07')
    # monthly_mean = outputs['o3'].mean(axis = 0)
    
    # x, y = np.array(np.meshgrid(outputs['lon'], outputs['lat']))
    # plt.figure(figsize=(18, 9))
    # ax = plt.subplot(projection = ccrs.PlateCarree())
    # #plt.contourf(lon-180, lat, (momo_dat - means), levels = 50, cmap = 'coolwarm')
    # plt.pcolor(x, y, monthly_mean, cmap = 'jet')
    # plt.colorbar()
    # ax.set_global()
    # ax.coastlines()
    # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
    #                   linewidth=1, color='gray', alpha=0.5, linestyle='--')
    # gl.xformatter = LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER
    # #ax.add_feature(cfeature.STATES)
    # ax.stock_img()
    # #north america
    # #ax.set_extent([-140, -50, 10, 80], crs=ccrs.PlateCarree())
    # #ax.set_extent([-125, -112, 30, 40], crs=ccrs.PlateCarree())
    # plt.title('Mean ozone estimate for 07-2012')



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    #parser.add_argument('data_root_dir', type=str)
    parser.add_argument('--years', default = ['2012'], nargs = '*', type=str)
    parser.add_argument('--months', default = [], type=str)
    parser.add_argument('--inputs', type=str, default='all')
    parser.add_argument('--outputs', type=str, default='o3')
    parser.add_argument('--mda8', type=bool, default=True)
    #parser.add_argument('out_file', type=str)

    args = parser.parse_args()
    main(**vars(args))
    
    
    
    




