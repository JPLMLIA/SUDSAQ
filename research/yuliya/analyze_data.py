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
region = 'north_america'
bbox = bbox_dict[region]
month = 'jan'
years = [2011, 2012, 2013, 2014]

#set the directory for th60at month
models_dir = f'{root_dir}/model/new/model_data/{month}/combined/'
#set plot directory
plots_dir = f'{models_dir}/plots/'
# create one if it's not there
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)


# -----------------------------------
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
#bias_pred = bias_pred['predict'].values
bias_pred = bias_pred['stack-1fd40670e8e2b515c6221d4afa3a768a'].values
pred_monthly_mean = np.nanmean(bias_pred, axis = 2)

#plot difference between true and predicted
x, y = np.meshgrid(lon, lat)
fig = plt.figure(figsize=(18, 9))
ax = plt.subplot(projection = ccrs.PlateCarree())
plt.pcolor(x, y, pred_monthly_mean, cmap = 'jet')
#plt.clim((-1200, 0))
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



# -----------------------------------
# --------------- PLOT an input variable
name = 'momo.osrc'
data = xr.open_dataset(f'{models_dir}/test.data.nc')
data.coords['lon'] = (data.coords['lon'] + 180) % 360 - 180
data = data.sortby(data.lon)
data_cropped = data.sel(lat=slice(bbox[2], bbox[3]), 
                         lon=slice(bbox[0], bbox[1]))

#lists all variables    
var_names = list(data.keys())
lat = data_cropped.lat.values
lon = data_cropped.lon.values
# get all the variables
var_monthly_mean = data_cropped[name].mean(dim='time', skipna= True).values


#plot the monthly mean
x, y = np.meshgrid(lon, lat)
fig = plt.figure(figsize=(18, 9))
ax = plt.subplot(projection = ccrs.PlateCarree())
plt.pcolor(x, y, var_monthly_mean, cmap = 'jet')
plt.clim(-1200, 0)
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






# -----------------------------------
# --------------- correlation analysis

#option1: read in data and subset a region from bbox
data = xr.open_dataset(f'{models_dir}/test.data.mean.nc')
data.coords['lon'] = (data.coords['lon'] + 180) % 360 - 180
data = data.sortby(data.lon)

#option2: optionally can run on contributions
data = xr.open_dataset(f'{models_dir}/test.contributions.mean.nc')
data.coords['lon'] = (data.coords['lon'] + 180) % 360 - 180
data = data.sortby(data.lon)

#take mean over time
var_names = list(data.keys())
#data_mean = data.mean(dim='time', skipna= True)

#crop the data for the region
data_cropped = data.sel(lat=slice(bbox[2], bbox[3]), 
                        lon=slice(bbox[0], bbox[1]))
data_stacked = data_cropped.stack(z=('lon', 'lat'))


#remove locations with nan values
data_array = data_stacked.to_array().values
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
mask_large = np.nansum(np.abs(corr_mat), axis = 1) > 20
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



#cluster the correlation matrix to see groups better
X0 = corr_mat.copy()
D = pairwise_distances(X0)
H = sch.linkage(D, method='average')
d1 = sch.dendrogram(H, no_plot=True)
idx = d1['leaves']
X = X0[idx,:][:, idx]

plt.figure(figsize = (14, 12))
plt.pcolor(X, cmap = 'coolwarm')
plt.clim((-1,1))
plt.xticks(np.arange(len(names))+0.5, names, rotation = 90, fontsize = 8);
plt.yticks(np.arange(len(names))+0.5, names, rotation = 0, fontsize = 8);
plt.colorbar()
plt.title(f'momo variable correlations {region}, {month}, clustered')
plt.savefig(f'{plots_dir}/variable_corr_cluster_matrix_{region}.png')
plt.close()









