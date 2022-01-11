#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 11:43:31 2022

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

import read_data_momo as momo


data_root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
momo_root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/MOMO_test/'

year = '2012'
month = '07'

#save
toar_output = f'{data_root_dir}/processed/TOAR2/summary_dp'
toar_file = f'{toar_output}/toar2_{year}_{month}.h5'
with closing(h5py.File(toar_file, 'r')) as f:
        lon = f['lon'][:]
        lat = f['lat'][:]
        toar_data = f['data'][:]
        date = f['date'][:].astype(str)
        station =  f['station'][:].astype(str)

        
        
momo = momo.get_momo(output_var = 'o3', input_var = ['t', 'q'], 
                year = year, month = month)
momo_lon = np.hstack([momo['lon'], 360.]) - 180
momo_lat = np.hstack([momo['lat'], 90.])    

#monthly average
days = np.unique(momo['date'][:,2])
mean_hours = [f'{x}'.zfill(2) for x in np.arange(8, 17)]
mask_h = np.in1d(np.hstack(momo['hour']), mean_hours)
momo_dat = momo['o3'][mask_h, :, :].mean(axis = 0)


   
counts = np.histogram2d(lon, lat, bins = [momo_lon, momo_lat])[0]
H = np.histogram2d(lon, lat, bins = [momo_lon, momo_lat],
                    weights = np.hstack(toar_data))[0]
means = H.T / counts.T
    
#plot monthly average bias maps
x, y = np.meshgrid(momo_lon, momo_lat)
plt.figure(figsize=(18, 9))
ax = plt.subplot(projection = ccrs.PlateCarree())
#plt.contourf(lon-180, lat, (momo_dat - means), levels = 50, cmap = 'coolwarm')
plt.pcolor(x, y, momo_dat - means, cmap = 'coolwarm')
plt.clim([-20, 20])
plt.colorbar()
ax.set_global()
ax.coastlines()
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
ax.add_feature(cfeature.STATES)
ax.stock_img()
#north america
#ax.set_extent([-140, -50, 10, 80], crs=ccrs.PlateCarree())
ax.set_extent([-125, -112, 30, 40], crs=ccrs.PlateCarree())
#ax.set_extent([-100, -90, 30, 40], crs=ccrs.PlateCarree())
plt.title('Ozone bias (momo - toar) for 07-2012')



    
  
        
   



        
        
        
        
        
            
    
    
    
        
        
        



