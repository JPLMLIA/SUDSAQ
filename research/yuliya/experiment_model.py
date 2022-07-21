#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 13:30:34 2022

@author: marchett
"""
import os
import sys
import h5py
import glob
import cv_analysis
import numpy as np
from joblib import dump
from utils import REQUIRED_VARS
from utils import ISD_FEATURES
from utils import TRAIN_FEATURES
from utils import format_data_v2
from utils import format_data_v3
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
#from treeinterpreter import treeinterpreter as ti
from contextlib import closing
from tqdm import tqdm
import matplotlib.pyplot as plt
import xarray as xr
import datetime as dt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from config_all import REQUIRED_VARS

#-----------------read in data
root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
if not os.path.exists(root_dir):
    root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
#momo_root_dir = f'{root_dir}/MOMO/'
data_dir = f'{root_dir}/processed/coregistered/'
toar_dir = f'{root_dir}/data/toar/matched/'
momo_dir = f'{root_dir}/data/momo/'


bbox_dict = {'globe':[0, 360, -90, 90],
              'europe': [-20+180, 40+180, 25, 80],
              'asia': [110+180, 160+180, 10, 70],
              'australia': [130+180, 170+180, -50, -10],
              'north_america': [-140+180, -50+180, 10, 80]}

bbox_dict = {'globe':[0, 360, -90, 90],
              'europe': [-20+360, 40+360, 25, 80],
              'asia': [110+360, 160+360, 10, 70],
              'australia': [130+360, 170+360, -50, -10],
              'north_america': [-140+360, -50+360, 10, 80]}


region = 'globe'
bbox = bbox_dict[region]


REQUIRED_VARS = ['t']
RESPONSE_VARS = 'mda8'
year_list = ['2012', '2013', '2014', '2015']
# REQUIRED_VARS = ['aerosol/nh4', 'aerosol/no3', 'aerosol/sul', 
#                  'ch2o', 'co', 'hno3', 'oh', 'pan', 'ps', 'q', 
#                  'so2', 't', 'u', 'v']
    

month = '07'
file_list = [f'{momo_dir}/{x}/{month}.nc' for x in year_list]
toar_flist = [f'{toar_dir}/{x}/{month}.nc' for x in year_list]
ds_momo = xr.open_mfdataset(file_list, engine='scipy', parallel=True)
ds_toar = xr.open_mfdataset(toar_flist, engine='scipy', parallel=True)


#read momo ozone
print('read data momo y ....')
nsy = ds_momo[REQUIRED_VARS]
nsy.load()
time = nsy.time.dt.time
time_mask = time == dt.time(1)
ssy = nsy.where(time_mask, drop=True)

lon = nsy['lon'].values
lat = nsy['lat'].values

nt = int(ssy.time.shape[0] / len(year_list))
mask_lon = (bbox[0] <= lon) & (bbox[1] >= lon)
mask_lat = (bbox[3] >=  lat) & (bbox[2] <= lat) 
nx = mask_lon.sum()
ny = mask_lat.sum()


#read momo variables
print('read data momo x ....')
nsx = ds_momo[REQUIRED_VARS]
nsx.load()
#daytime average
time = nsx.time.dt.time
time_mask = (dt.time(8) < time) & (time < dt.time(16))
ssx = nsx.where(time_mask, drop=True)
rsx = ssx.resample(time='1D').mean()
rsx = rsx.dropna('time')


#read TOAR
print('read data toar ....')
tlon = ds_toar['lon'].load()
ts = ds_toar['toar/o3/dma8epa/mean'].load()
obs = ts.values[:, mask_lat, :][:, :, mask_lon]
mask = np.isnan(obs)



#daytime average
# time = ts.time.dt.time
# time_mask = time == dt.time(0)
# sst = ts.where(time_mask, drop=True)    # Drop timestamps that don't match
# rst = sst.resample(time='1D').mean()
# rst = rst.dropna('time')

#gs = ss.groupby('time.day').mean()
#rs = xr.concat([ss.sel(time=x).groupby('time.day').mean() for x in year_list], dim = 'day')
groups = np.repeat(year_list, ny*nx*nt).reshape(mask.shape)[~mask].astype(int)

y_momo = ssy[RESPONSE_VARS].values[:, mask_lat, :][:, :, mask_lon][~mask]
    
X = []
for var in REQUIRED_VARS:
    flat_arr = rsx[var].values[:, mask_lat, :][:, :, mask_lon][~mask]
    X.append(flat_arr)


y_toar = obs[~mask]      
X = np.column_stack(X)

#local_dir = f'/Users/marchett/Documents/SUDS_AQ/data/modeling'
new_file = f'{root_dir}/model/research/all-vars/{month}/model_data_{region}_{month}.h5'
with closing(h5py.File(new_file, 'w')) as f:
    f['y_momo'] = y_momo
    f['y_toar'] = y_toar
    f['X'] = X
    f['groups'] = groups
    f['mask_toar'] = mask
    f['lon'] = lon
    f['lat'] = lat




#read
# month = '01'
# new_file = f'{root_dir}/model/research/all-vars/{month}/model_data_{region}_{month}.h5'
# with closing(h5py.File(new_file, 'r')) as f:
#     y_momo = f['y_momo'][:]
#     y_toar = f['y_toar'][:]
#     X = f['X'][:]
#     groups = f['groups'][:]
#     mask = f['mask_toar'][:]
#     lon = f['lon'][:]
#     lat = f['lat'][:]
    
 
    
# y_plot = np.zeros(mask.shape)
# y_plot[:] = np.nan
# y_plot[~mask] = y_momo  
# y_plot = np.nanmean(y_plot, axis = 0)  
# x, y = np.meshgrid(lon[mask_lon], lat[mask_lat])

# y_plot = rsx['q'].values.mean(axis = 0)
# y_plot = ssy[RESPONSE_VARS].values.mean(axis = 0)
# y_plot = np.nanmean(ts.values, axis = 0)
# x, y = np.meshgrid(lon, lat)

# y_plot = np.nanmean(obs, axis = 0)
# x, y = np.meshgrid(lon[mask_lon], lat[mask_lat])

# fig = plt.figure(figsize=(18, 9))
# ax = plt.subplot(projection = ccrs.PlateCarree())
# plt.pcolor(x, y, y_plot)
# ax.set_global()
# ax.coastlines()
# gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                   linewidth=1, color='gray', alpha=0.5, linestyle='--')
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER
# ax.stock_img()
 



#modeling

# y = y_momo-y_toar
# y = y_toar.copy()
# importances = []    
# pred_y = np.zeros((len(groups), ))
# group_kfold = GroupKFold(n_splits=len(year_list))
# for cv_index, (train_index, test_index) in enumerate(group_kfold.split(X, y, groups)):
#     print(cv_index)
#     train_x, test_x = X[train_index], X[test_index]
#     train_y, test_y = y[train_index], y[test_index]

#     rf_predictor = RandomForestRegressor()
#     rf_predictor.fit(train_x, train_y)
#     importances.append(rf_predictor.feature_importances_)
    
#     pred_y[test_index] = rf_predictor.predict(test_x)
    


# rmse = np.sqrt(np.mean((y - pred_y)**2)) 
# nrmse = rmse / (np.mean(y))
# plt.figure()
# plt.plot(y, pred_y, '.', ms = 0.8)
# plt.xlabel(f'truth') 
# plt.ylabel(f'predicted') 
# plt.grid(linestyle='dotted')
# plt.title(f'rmse = {rmse}, rmse/mean(y) = {nrmse}')
# plt.savefig(f'{root_dir}/model/research/rmse_{region}_{month}.png',
#               bbox_inches='tight')
# plt.close()   

# y_plot = np.zeros(mask.shape)
# y_plot[:] = np.nan
# y_plot[~mask] = pred_y  
# y_plot = np.nanmean(y_plot, axis = 0)  
# x, y = np.meshgrid(lon[mask_lon], lat[mask_lat])

# fig = plt.figure(figsize=(18, 9))
# ax = plt.subplot(projection = ccrs.PlateCarree())
# plt.pcolor(x, y, y_plot)
# ax.set_global()
# ax.coastlines()
# gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                   linewidth=1, color='gray', alpha=0.5, linestyle='--')
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER
# ax.stock_img()

 
# mean_imp = np.column_stack(importances).mean(axis = 1)
# feature_importance_arr = mean_imp / np.max(mean_imp)



# plt.figure(figsize = (5,12))
# plt.barh(np.hstack(REQUIRED_VARS), feature_importance_arr)
# plt.title('Random Forest Feature Importance')
# plt.ylabel('Features')
# plt.xlabel('Feature Importance')
# plt.yticks(np.arange(0, len(REQUIRED_VARS)), np.hstack(REQUIRED_VARS), fontsize = 6)
# plt.grid(linestyle='dotted')
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.savefig(f'{root_dir}/model/research/importance_bar_{region}_{month}.png',
#               bbox_inches='tight')
# plt.close()
  

# mask_imp = feature_importance_arr > 0.25
# P = mask_imp.sum()
# mask_imp = np.argsort(feature_importance_arr)[-10:]
# P = len(mask_imp)
   
# plt.figure()
# plt.barh(np.hstack(REQUIRED_VARS)[mask_imp][::-1], feature_importance_arr[ mask_imp][::-1])
# plt.title('Random Forest Feature Importance')
# plt.ylabel('Features')
# plt.xlabel('Feature Importance')
# plt.yticks(np.arange(0,P), np.hstack(REQUIRED_VARS)[mask_imp][::-1], fontsize = 10)
# plt.grid(linestyle='dotted')
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.savefig(f'{root_dir}/model/research/importance_bar_top_{region}_{month}.png',
#               bbox_inches='tight', dpi = 150)
# plt.close()

      

            



            
            
            
            
            
            
            