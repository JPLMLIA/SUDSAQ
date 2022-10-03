#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 15:41:11 2022

@author: marchett
"""
import os, glob
import numpy as np
from datetime import datetime, timedelta, date
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import xarray as xr
import scipy.cluster.hierarchy as sch
from sklearn.metrics import pairwise_distances

#-----------------read in data
root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
if not os.path.exists(root_dir):
    root_dir = '/data/MLIA_active_data/data_SUDSAQ/'

#everything is in -180 to 180 lon
bbox_dict = {'globe':[-180, 180, -90, 90],
            'europe': [-20, 40, 25, 80],
            'asia': [110, 160, 10, 70],
            'australia': [130, 170, -50, -10],
            'north_america': [-140, -50, 10, 80],
            'west_europe': [-20, 10, 25, 80],
            'east_europe': [10, 40, 25, 80],
            'west_na': [-140, -95, 10, 80],
            'east_na': [-95, -50, 10, 80], }


#choose parameters
region = 'globe'
bbox = bbox_dict[region]
year = 2011
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

#set the directory for th60at month
models_dir = f'{root_dir}/model/new/model_data/'
#set plot directory
plots_dir = f'{root_dir}/model/new/summary_plots/'
# create one if it's not there
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    
#get and sort monthly files
monthly_files = glob.glob(f'{models_dir}/*/combined/test.data.nc')
months_list = [x.split('/')[8] for x in monthly_files]
idx = [np.where(np.hstack(months_list) == x)[0][0] for x in months]
monthly_files = [monthly_files[x] for x in idx] 


### -----------------
### this will plot momo variables

name = 'momo.t'
coord = [33, -84]
#data = xr.open_dataset(f'{models_dir}/test.data.nc')

vals = []
dates = []
for m in range(len(monthly_files)):
    data = xr.open_dataset(monthly_files[m])
    data.coords['lon'] = (data.coords['lon'] + 180) % 360 - 180
    data = data.sortby(data.lon)
    data_cropped = data.sel(lat=slice(bbox[2], bbox[3]), 
                         lon=slice(bbox[0], bbox[1]))   
    
    data_annual = data.sel(time =slice(f'{year}-01-01', f'{year}-12-31'))
    #lists all variables    
    var_names = list(data.keys())
    lat = data_cropped.lat.values
    lon = data_cropped.lon.values

    #choose a location
    xi = np.searchsorted(lon, coord[1])
    yi = np.searchsorted(lat, coord[0])
    
    vals.append(data_annual[name][yi, xi].values)
    dates.append(data_annual.time.values)
    
vals = np.hstack(vals)
dates = np.hstack(dates)

labels = np.hstack(['-'.join(str(x).split('T')[0].split('-')[1:]) for x in dates])
days = np.hstack([x.split('-')[-1] for x in labels])
tick_idx = list(np.where(np.in1d(days, '01'))[0])

plt.figure(figsize = (10, 5))
plt.plot(vals, '-', color = '0.8') 
plt.plot(vals, '.', color = '0.5', ms = 3) 
plt.xticks(tick_idx, labels[tick_idx])
plt.xlabel(f'time')
plt.ylabel(f'{name}')
plt.title(f'variable = {name}, year {year}, location = lon {coord[1]}, lat {coord[0]}')  
plt.grid(ls = ':')
plt.savefig(f'{plots_dir}/time_series_{name}_{year}_loc{coord}.png')
plt.close()


### -----------------
### this will plot bias mda8 and toar mda8, then will compute momo mda8

coord = [33, -84]
bias_files = glob.glob(f'{root_dir}/model/new/model_data/*/combined/test.target.nc')
toar_files = glob.glob(f'{root_dir}/model/toar/model_data/*/combined/test.target.nc')
bias_files = [bias_files[x] for x in idx] 
#toar_files = [toar_files[x] for x in idx] 

bias_vals = []
toar_vals = []
dates = []
for m in range(len(monthly_files)):
    
    data_bias = xr.open_dataset(bias_files[m])
    data_bias.coords['lon'] = (data_bias.coords['lon'] + 180) % 360 - 180
    data_bias = data_bias.sortby(data_bias.lon)
    data_cropped = data_bias.sel(lat=slice(bbox[2], bbox[3]), 
                         lon=slice(bbox[0], bbox[1])) 
    
    data_annual = data_bias.sel(time =slice(f'{year}-01-01', f'{year}-12-31'))
    #lists all variables    
    var_names = list(data_bias.keys())
    lat = data_annual.lat.values
    lon = data_annual.lon.values
    
    bias_vals.append(data_annual['target'][yi, xi, :].values)
    
    data_toar = xr.open_dataset(toar_files[m])
    data_toar.coords['lon'] = (data_toar.coords['lon'] + 180) % 360 - 180
    data_toar = data_toar.sortby(data_toar.lon)
    data_cropped = data_toar.sel(lat=slice(bbox[2], bbox[3]), 
                         lon=slice(bbox[0], bbox[1])) 
    
    data_annual = data_toar.sel(time =slice(f'{year}-01-01', f'{year}-12-31'))

    #choose a location
    toar_vals.append(data_annual['target'][yi, xi, :].values)
    dates.append(data_annual.time.values)
    


bias_vals = np.hstack(bias_vals)
toar_vals = np.hstack(toar_vals)
dates = np.hstack(dates)

labels = np.hstack(['-'.join(str(x).split('T')[0].split('-')[1:]) for x in dates])
days = np.hstack([x.split('-')[-1] for x in labels])
tick_idx = list(np.where(np.in1d(days, '01'))[0])

plt.figure(figsize = (10, 5))
plt.plot(toar_vals, '-', color = '0.8', label = 'toar ozone')
plt.plot(toar_vals, '.', color = '0.8', ms = 3)  
plt.plot(toar_vals + bias_vals, '-', color = 'r', alpha = 0.3, label = 'momo_ozone') 
plt.plot(toar_vals + bias_vals, '.', color = 'r', ms = 3) 
plt.xticks(tick_idx, labels[tick_idx])
plt.xlabel(f'time')
plt.ylabel(f'ppb')
plt.legend()
plt.title(f'toar vs momo ozone, year {year}, location = lon {coord[1]}, lat {coord[0]}')  
plt.grid(ls = ':')
plt.savefig(f'{plots_dir}/time_series_ozone_{year}_loc{coord}.png')
plt.close()

    
    
    
    
    
    
    





 

