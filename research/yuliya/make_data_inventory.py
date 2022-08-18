#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 12:40:01 2022

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
from scipy import stats
import pywt, copy
import dtaidistance as dt
from sklearn.cluster import AgglomerativeClustering 


#bbox = [0, 360, -90, 90]
bbox = [-140+180, -50+180, 10, 80]


root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
if not os.path.exists(root_dir):
    root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
#momo_root_dir = f'{root_dir}/MOMO/'
toar_output = f'{root_dir}/processed/summary_dp/TOAR2/'
momo_output = f'{root_dir}/processed/summary_dp/MOMO/'

years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
years = ['2012']
months = [f'{x}'.zfill(2) for x in np.arange(1, 13)]



assignments = {k: [] for k in years}

#processed clean dataset
momo_lon = []
momo_lat = []
momo_dat = []
toar_dat = []
momo_ud = [] 
for year in years:
    for month in months:
        new_file = f'{root_dir}/processed/coregistered/momo_matched_{year}_{month}.h5' 
        with closing(h5py.File(new_file, 'r')) as f:
            momo_dat.append(f['o3'][:])
            toar_dat.append(f['toar']['mean'][:])
            momo_ud.append(f['date'][:])
            momo_lon.append(f['lon'][:])
            momo_lat.append(f['lat'][:])
    
    
#x, y = np.meshgrid(momo_lon[0], momo_lat[0])
momo_dat = np.row_stack(momo_dat)
momo_ud = np.row_stack(momo_ud).astype(str)
toar_dat = np.row_stack(toar_dat)

mask_lon = (bbox[0] < momo_lon[0]) & (bbox[1] > momo_lon[0])
mask_lat = (bbox[3] > momo_lat[0]) & (bbox[2] < momo_lat[0]) 
#region_mask = mask_lon & mask_lat

toar_dat_region = toar_dat[:, mask_lat, :][:, :, mask_lon]
momo_dat_region = momo_dat[:, mask_lat, :][:, :, mask_lon]
momo_lon_region = momo_lon[0][mask_lon]
momo_lat_region = momo_lat[0][mask_lat]

mask_all = np.isnan(toar_dat_region)
mask = mask_all.sum(axis = 0) < toar_dat_region.shape[0]
idx = np.where(mask)


#data inventory
for year in years:
    
    #locations with missing counts
    mask_y = momo_ud[:, 0] == year
    n = mask_y.sum()
    counts_ = mask_all[mask_y].sum(axis = 0).astype(float)
    counts_[counts_ == n] = np.nan
    
    n_locs = (~np.isnan(counts_)).sum()
    
    x, y = np.meshgrid(momo_lon_region-180, momo_lat_region)
    fig = plt.figure(figsize=(18, 9))
    ax = plt.subplot(projection = ccrs.PlateCarree())
    #plt.contourf(lon-180, lat, (momo_dat - means), levels = 50, cmap = 'coolwarm')
    #plt.pcolor(x, y, toar_to_momo.mean(axis = 2), cmap = 'coolwarm')
    plt.pcolor(x, y, counts_, cmap = 'BuPu')
    plt.clim((0,366))
    plt.colorbar()
    ax.set_global()
    ax.coastlines()
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.stock_img()
    plt.title(f'missing counts, {year}, n = {n_locs}')
    ax.set_extent([-140, -50, 10, 80], crs=ccrs.PlateCarree())
    plt.savefig(f'{root_dir}/processed/plots/data_inventory/data_inventory_{year}.png', 
             bbox_inches = 'tight')
    plt.close()
    
    
    plt.figure()
    for j, month in enumerate(months):
        mask_m = momo_ud[mask_y, 1] == month
        mask_locs = mask_all[mask_y][mask_m].sum(axis = 0)
        n_matches = (mask_locs < np.max(mask_locs)).sum()
        plt.bar(j, n_matches, color = '0.5', alpha = 0.5)
    plt.ylim((0,500))
    plt.xlabel('months')
    plt.ylabel('# of locs available')
    plt.grid(ls=':', alpha = 0.5)
    plt.title(f'{year}')
    plt.savefig(f'{root_dir}/processed/plots/data_inventory/data_inventory_monthly_{year}.png', 
             bbox_inches = 'tight')
    plt.close()




#-------------------INPUTS
#processed clean dataset
maindir = f'{root_dir}/processed/coregistered/'
inputs = ['aero_nh4','aero_no3', 'aero_sul', 'ch2o', 'co', 'hno3', 
          'oh', 'pan', 'ps', 'q', 'so2', 't', 'u', 'v']
months = [f'{x}'.zfill(2) for x in np.arange(1, 13)]

momo_lon = []
momo_lat = []
momo_dat = {k: [] for k in inputs}
momo_ud = [] 
momo_bias = []
for year in years:
    for month in months:
        new_file = f'{maindir}/momo_matched_{year}_{month}.h5' 
        with closing(h5py.File(new_file, 'r')) as f:
            keys = list(f)
            for k in inputs:
                momo_dat[k].append(f[k][:])
            momo_ud.append(f['date'][:])
            momo_lon.append(f['lon'][:])
            momo_lat.append(f['lat'][:])
            momo_bias.append(f['o3'][:] - f['toar']['mean'][:])
    
    
#extract regions
#x, y = np.meshgrid(momo_lon[0], momo_lat[0])
mask_lon = (bbox[0] < momo_lon[0]) & (bbox[1] > momo_lon[0])
mask_lat = (bbox[3] > momo_lat[0]) & (bbox[2] < momo_lat[0]) 

momo_dat_region = {}
for k in inputs:
    temp = np.row_stack(momo_dat[k])
    momo_dat_region[k] = temp[:, mask_lat, :][:, :, mask_lon]
momo_ud = np.row_stack(momo_ud).astype(str)
momo_lon_region = momo_lon[0][mask_lon]
momo_lat_region = momo_lat[0][mask_lat]
bias_region = np.row_stack(momo_bias)[:, mask_lat, :][:, :, mask_lon]

n1 = len(momo_lon_region)
n2 = len(momo_lat_region)

mask_all = np.isnan(bias_region)
mask = mask_all.sum(axis = 0) < momo_dat_region[k].shape[0]
idx = np.where(mask)
 
#data inventory
for year in years:
    for k in inputs:
        var = {}
        #mean
        var['mean'] = np.nanmean(momo_dat_region[k], axis = 0)
        var['mean'][~mask] = np.nan
        var['std'] = np.nanstd(momo_dat_region[k], axis = 0)
        var['std'][~mask] = np.nan
        
        for s in list(var):
            x, y = np.meshgrid(momo_lon_region-180, momo_lat_region)
            fig = plt.figure(figsize=(18, 9))
            ax = plt.subplot(projection = ccrs.PlateCarree())
            #plt.contourf(lon-180, lat, (momo_dat - means), levels = 50, cmap = 'coolwarm')
            #plt.pcolor(x, y, toar_to_momo.mean(axis = 2), cmap = 'coolwarm')
            plt.pcolor(x, y, var[s], cmap = 'viridis')
            #plt.clim((0,366))
            plt.colorbar()
            ax.set_global()
            ax.coastlines()
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=1, color='gray', alpha=0.5, linestyle='--')
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            ax.stock_img()
            plt.title(f'{s}, {k}, {year}')
            ax.set_extent([-140, -50, 10, 80], crs=ccrs.PlateCarree())
            plt.savefig(f'{root_dir}/processed/plots/inputs/momo/{s}_{k}_{year}.png', 
                     bbox_inches = 'tight')
            plt.close()
        
        corrs = np.zeros((n2,n1))
        corrs[:] = np.nan
        for i in range(len(idx[0])):
            r = idx[0][i]
            c = idx[1][i]
            x1 = bias_region[:, r, c]
            x2 = momo_dat_region[k][:,r,c]
            mask1 = ~np.isnan(x1)
            mask2 = ~np.isnan(x2)
            corrs[r, c] = np.corrcoef(x1[mask1&mask2], x2[mask1&mask2])[0,1]
        

        x, y = np.meshgrid(momo_lon_region-180, momo_lat_region)
        fig = plt.figure(figsize=(18, 9))
        ax = plt.subplot(projection = ccrs.PlateCarree())
        #plt.contourf(lon-180, lat, (momo_dat - means), levels = 50, cmap = 'coolwarm')
        #plt.pcolor(x, y, toar_to_momo.mean(axis = 2), cmap = 'coolwarm')
        plt.pcolor(x, y, corrs, cmap = 'bwr')
        plt.clim((-0.7, 0.7))
        plt.colorbar()
        ax.set_global()
        ax.coastlines()
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        ax.stock_img()
        plt.title(f'pearson correlation with bias, {k}, {year}')
        ax.set_extent([-140, -50, 10, 80], crs=ccrs.PlateCarree())
        plt.savefig(f'{root_dir}/processed/plots/inputs/momo/corr_{k}_{year}.png', 
                 bbox_inches = 'tight')
        plt.close()








