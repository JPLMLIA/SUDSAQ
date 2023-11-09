#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:37:36 2022

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
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from scipy import stats
import read_data_momo
 

root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
if not os.path.exists(root_dir):
    root_dir = '/data/MLIA_active_data/data_SUDSAQ/'

data_output_dir = f'{root_dir}/processed/summary_dp/TOAR2/'
ofile = f'toar2_metadata.h5'
meta_dict = {}        
with closing(h5py.File(data_output_dir + ofile, 'r')) as f:
        for key in list(f):
            meta_dict[key] = f[key][:]

lon = meta_dict['station_lon']
lat = meta_dict['station_lat'] 

station_stats = ['station_alt', 'station_population_density', 'station_nightlight_1km',
                 'station_omi_no2_column', 'station_climatic_zone', 'station_htap_region',
                 'station_dominant_landcover', 'station_toar_category']


# station_numerical = ['station_alt', 'station_population_density', 'station_nightlight_1km',
#                      'station_omi_no2_column']
# station_categorical = ['station_climatic_zone', 'station_dominant_landcover', 
#                        'station_toar_category', 'station_htap_region']

station_summaries = ['mean', 'std']


new_file = f'{root_dir}/processed/coregistered/momo_matched_{2012}_{12}.h5' 
with closing(h5py.File(new_file, 'r')) as f:
    momo_lon = f['lon'][:]
    momo_lat = f['lat'][:]
bbox = [-140+180, -50+180, 10, 80]
mask_lon = (bbox[0] < momo_lon) & (bbox[1] > momo_lon)
mask_lat = (bbox[3] > momo_lat) & (bbox[2] < momo_lat) 

   
x, y = np.meshgrid(momo_lon -180, momo_lat)
matched_dict = {}
for stat in station_stats:
    
    matched_dict[stat] = {}
    
    for s in station_summaries:
        ret = stats.binned_statistic_2d(lat, lon, 
                                    meta_dict[stat],
                                    s,
                                    bins=[np.hstack([momo_lat, 90]), 
                                          np.hstack([momo_lon - 180, 180])],
                                    expand_binnumbers = True)
    
        values = ret.statistic
        values[values == 0] = np.nan
        matched_dict[stat][s] = values
    
    max_x = meta_dict[stat].max()
    if max_x > 30:
        max_cb = 256
 
    cmap = mp.colors.ListedColormap(sns.color_palette('Spectral_r', max_cb))
    plt.subplots(1,2, figsize=(18, 9))
    for j, s in enumerate(station_summaries):
        vmin = np.nanmin(matched_dict[stat][s][mask_lat, :][:, mask_lon])
        vmax = np.nanmax(matched_dict[stat][s][mask_lat, :][:, mask_lon])
        
        ax = plt.subplot(1, 2, j+1, projection = ccrs.PlateCarree())
        #plt.contourf(lon-180, lat, (momo_dat - means), levels = 50, cmap = 'coolwarm')
        plt.pcolor(x, y, matched_dict[stat][s], cmap = cmap)
        plt.clim((vmin, vmax))
        #plt.scatter(lon, lat, c = meta_dict['station_dominant_landcover'])
        #plt.colorbar(ax=ax, fraction=0.035, pad=0.08)
        ax.set_global()
        ax.coastlines()
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        #ax.add_feature(cfeature.STATES)
        ax.stock_img()
        #north america
        ax.set_extent([-140, -50, 10, 80], crs=ccrs.PlateCarree())
        vmin, vmax = plt.gci().get_clim()
        plt.colorbar(ax=ax, fraction=0.035, pad=0.08)
        #ax.set_extent([-125, -112, 30, 40], crs=ccrs.PlateCarree())
        plt.title(f'TOAR stations {stat}, {s}, max = {max_x}')

    plt.savefig(f'{root_dir}/processed/plots/toar_stats/toar_station_{stat}.png', 
                bbox_inches = 'tight')
    plt.close()


#save
data_output_dir = f'{root_dir}/processed/coregistered/'
ofile = f'momo_matched_metadata.h5'
with closing(h5py.File(data_output_dir + ofile, 'w')) as f:
    for key in matched_dict.keys():
        for s in matched_dict[key].keys():
            f[f'{key}/{s}'] = matched_dict[key][s]


# data_output_dir = f'{root_dir}/processed/coregistered/'
# ofile = f'momo_matched_metadata.h5'
# matched_dict1 = {}
# with closing(h5py.File(data_output_dir + ofile, 'r')) as f:
#     for key in list(f):
#         matched_dict1[key] = {}
#         for s in list(f[key]):
#             matched_dict1[key][s] = f[key][s][:]


def types_func(x):
    return len(np.unique(x))  


    