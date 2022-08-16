#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 18:15:02 2022

@author: marchett
"""
import os
import numpy as np
from datetime import datetime, timedelta, date
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import xarray as xr


#-----------------read in data
root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
if not os.path.exists(root_dir):
    root_dir = '/data/MLIA_active_data/data_SUDSAQ/'

bbox_dict = {'globe':[0+180, 360+180, -90, 90],
              'europe': [-20+360, 40+360, 25, 80],
              'asia': [110+360, 160+360, 10, 70],
              'australia': [130+360, 170+360, -50, -10],
              'north_america': [-140+360, -50+360, 10, 80]}


#choose parameters
bbox = bbox_dict['north_america']
month = 'jan'
years = [2011, 2012, 2013, 2014]

#set the directory for that month
models_dir = f'{root_dir}/model/new/model_data/{month}/combined/'
#set plot directory
plots_dir = f'{models_dir}/plots/'
# create one if it's not there
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)


# --------------- PLOT true vs predicted ozone bias 
#ds_momo = xr.open_mfdataset(files_momo, parallel=True)
bias_true = xr.open_dataset(f'{models_dir}/test.target.nc').load()
bias_pred = xr.open_dataset(f'{models_dir}/test.predict.nc').load()

#select years
bias_true = bias_true.sel(time = np.in1d(bias_true['time.year'], years))
lat = bias_true.lat.values
lon = bias_true.lon.values
#get true bias
bias_true = bias_true['target'].values
true_monthly_mean = np.nanmean(bias_true, axis = 2)

#get predicted bias
bias_pred = bias_pred.sel(time = np.in1d(bias_pred['time.year'], years))
bias_pred = bias_pred['predict'].values
pred_monthly_mean = np.nanmean(bias_pred, axis = 2)

#plot difference between true and predicted
x, y = np.meshgrid(lon, lat)
fig = plt.figure(figsize=(18, 9))
ax = plt.subplot(projection = ccrs.PlateCarree())
plt.pcolor(x, y, true_monthly_mean - pred_monthly_mean, cmap = 'jet')
ax.set_global()
ax.coastlines()
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
ax.set_extent([bbox[0], bbox[1], bbox[2], bbox[3]], crs=ccrs.PlateCarree())
ax.stock_img()
plt.colorbar()
plt.title(f'true - predicted bias residuals')
plt.savefig(f'{plots_dir}/difference_monthly_mean_map.png')
plt.close()




# --------------- PLOT an input variable
name = 'momo.osrc'
#ds_momo = xr.open_mfdataset(files_momo, parallel=True)
# f = f'{root_dir}/model/new/{month}/combined/test.data.nc'
# data = xr.open_dataset(f)
data = xr.open_dataset(f'{models_dir}/test.data.nc')

#lists all variables    
var_names = data.variable.values
lat = data.lat.values
lon = data.lon.values
# get all the variables
vals = data['stack-50e2f0b556989bc5a94867be166bc66f'].values

#get variable of interest and take monthly mean
name_mask = var_names == name
var_monthly_mean = np.nanmean(vals[name_mask, :, :, :], axis = 3)[0,:,:]

#plot the monthly mean
x, y = np.meshgrid(lon, lat)
fig = plt.figure(figsize=(18, 9))
ax = plt.subplot(projection = ccrs.PlateCarree())
plt.pcolor(x, y, var_monthly_mean, cmap = 'jet')
ax.set_global()
ax.coastlines()
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
ax.set_extent([bbox[0], bbox[1], bbox[2], bbox[3]], crs=ccrs.PlateCarree())
ax.stock_img()
plt.colorbar()
plt.title(f'momo variable {name}, monthly average')
plt.savefig(f'{plots_dir}/{name}_monthly_mean_map.png')
plt.close()




# --------------- correlation analysis

#read in data and subset a region from bbox
data = xr.open_dataset(f'{models_dir}/test.data.nc')
var_names = data.variable.values

#take mean over time
data_mean = data.mean(dim='time', skipna= True)

#crop the data for the region
data_cropped = data_mean.sel(lat=slice(bbox[2], bbox[3]), 
                        lon=slice(bbox[0], bbox[1]))
data_stacked = data_cropped.stack(z=('lon', 'lat'))

#remove locations with nan values
data_array = data_stacked['stack-50e2f0b556989bc5a94867be166bc66f'].values
counts_nan = np.isnan(data_array).sum(axis = 0)
mask_locs = counts_nan < len(var_names)

# some variables have all zeros, mask them
counts_zero = (data_array == 0).sum(axis = 1)
mask_zero =  counts_zero < mask_locs.sum()

# make the full correlation matrix
corr_mat = np.corrcoef(data_array[:, mask_locs][mask_zero, :])
#clean up variable names to make them shorter
names = np.hstack([x.split('.')[-1] for x in var_names])[mask_zero]

#plot full correlation matrix
plt.figure(figsize = (12, 10))
plt.pcolor(corr_mat)
plt.xticks(np.arange(len(names))+0.5, names, rotation = 90);
plt.yticks(np.arange(len(names))+0.5, names, rotation = 0);
plt.colorbar()
plt.title(f'momo variable correlations')
plt.savefig(f'{plots_dir}/variable_corr_matrix.png')
plt.close()


# only use variables with the largest sum of correlations, e.g. > 20
mask_large = np.nansum(corr_mat, axis = 1) > 20
plt.figure(figsize = (12, 10))
plt.pcolor(corr_mat[mask_large, :][:, mask_large])
plt.xticks(np.arange(len(names[mask_large]))+0.5, names[mask_large], rotation = 90);
plt.yticks(np.arange(len(names[mask_large]))+0.5, names[mask_large], rotation = 0);
plt.colorbar()
plt.title(f'momo variable correlations')
plt.savefig(f'{plots_dir}/variable_corr_largest_matrix.png')
plt.close()


# only variables with the strongest correlations
#choose one variable
name = 'osrc'
name_idx = np.where(names == name)[0][0]
corr_mat2 = corr_mat - np.eye(corr_mat.shape[0])
corr_var = corr_mat2[name_idx, :]
#find the strongest correlated
idx = np.argmax(corr_var)

#create scatter plot
plt.figure()
plt.plot(data_array[name_idx, mask_locs], data_array[idx,mask_locs], '.',
         label = f'corr = {corr_var[idx]}')
plt.xlabel(name)
plt.ylabel(names[idx])
plt.grid(ls = ':')
plt.legend()
plt.savefig(f'{plots_dir}/{name}_{names[idx]}_scatter.png')
plt.close()














