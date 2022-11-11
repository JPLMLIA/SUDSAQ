#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:13:25 2022

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
import subprocess
from scipy.ndimage.filters import gaussian_filter




def spatial_map(x, y, Z, name_params, bias = None, extent = [-140, -50, 10, 80], 
                subdir = None):
    
    root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
    if not os.path.exists(root_dir):
        root_dir = '/data/MLIA_active_data/data_SUDSAQ/'

    plotdir = f'{root_dir}/processed/plots/{subdir}'
    if not os.path.exists(f'{plotdir}'):
        os.makedirs(f'{plotdir}')
    
    year = name_params['year']
    month = name_params['month']
    name = name_params['name']
    #montly mean
    fig = plt.figure(figsize=(18, 9))
    ax = plt.subplot(projection = ccrs.PlateCarree())
    #plt.contourf(lon-180, lat, (momo_dat - means), levels = 50, cmap = 'coolwarm')
    #plt.pcolor(x, y, toar_to_momo.mean(axis = 2), cmap = 'coolwarm')
    plt.pcolor(x, y, Z, cmap = 'coolwarm')
    #plt.clim(cmin, cmax)
    plt.colorbar()
    ax.set_global()
    ax.coastlines()
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.stock_img()
    plt.title(f'{name}, year = {year}, month = {month}')
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    plt.savefig(f'{plotdir}/{name}_{year}_{month}.png', 
                bbox_inches = 'tight')
    plt.close()
  

    
    
    
    
    
    
    
    
    