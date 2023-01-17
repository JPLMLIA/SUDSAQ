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
             'europe': [-20+180, 40+180, 25, 80],
             'asia': [110+180, 160+180, 10, 70],
             'australia': [130, 170, -50, -10],
             'north_america': [-140, -50, 10, 80]}

region = 'globe'
bbox = bbox_dict[region]

MONTHS_DICT = {
    'jan': '01',
    'feb': '02',
    'mar': '03',
    'apr': '04',
    'may': '05',
    'jun': '06',
    'jul': '07',
    'aug': '08',
    'sep': '09',
    'oct': '10',
    'nov': '11',
    'dec': '12'
}


REQUIRED_VARS = ['2dsfc/Br2',
 '2dsfc/BrCl',
 '2dsfc/BrONO2',
 '2dsfc/BrOX',
 '2dsfc/C10H16',
 '2dsfc/C2H5OOH',
 '2dsfc/C2H6',
 '2dsfc/C3H6',
 '2dsfc/C3H7OOH',
 '2dsfc/C5H8',
 '2dsfc/CCl4',
 '2dsfc/CFC11',
 '2dsfc/CFC113',
 '2dsfc/CFC12',
 '2dsfc/CH2O',
 '2dsfc/CH3Br',
 '2dsfc/CH3CCl3',
 '2dsfc/CH3CHO',
 '2dsfc/CH3COCH3',
 '2dsfc/CH3COO2',
 '2dsfc/CH3COOOH',
 '2dsfc/CH3Cl',
 '2dsfc/CH3O2',
 '2dsfc/CH3OH',
 '2dsfc/CH3OOH',
 '2dsfc/CHBr3',
 '2dsfc/Cl2',
 '2dsfc/ClONO2',
 '2dsfc/ClOX',
 '2dsfc/DCDT/HOX',
 '2dsfc/DCDT/OY',
 '2dsfc/DCDT/SO2',
 '2dsfc/DMS',
 '2dsfc/H1211',
 '2dsfc/H1301',
 '2dsfc/H2O2',
 '2dsfc/HACET',
 '2dsfc/HBr',
 '2dsfc/HCFC22',
 '2dsfc/HCl',
 '2dsfc/HNO3',
 '2dsfc/HNO4',
 '2dsfc/HO2',
 '2dsfc/HOBr',
 '2dsfc/HOCl',
 '2dsfc/HOROOH',
 '2dsfc/ISON',
 '2dsfc/ISOOH',
 '2dsfc/LR/HOX',
 '2dsfc/LR/OY',
 '2dsfc/LR/SO2',
 '2dsfc/MACR',
 '2dsfc/MACROOH',
 '2dsfc/MGLY',
 '2dsfc/MPAN',
 '2dsfc/N2O5',
 '2dsfc/NALD',
 '2dsfc/NH3',
 '2dsfc/NH4',
 '2dsfc/OCS',
 '2dsfc/OH',
 '2dsfc/ONMV',
 '2dsfc/PAN',
 '2dsfc/PROD/HOX',
 '2dsfc/PROD/OY',
 '2dsfc/SO2',
 '2dsfc/SO4',
 '2dsfc/dflx/bc',
 '2dsfc/dflx/dust',
 '2dsfc/dflx/hno3',
 '2dsfc/dflx/nh3',
 '2dsfc/dflx/nh4',
 '2dsfc/dflx/oc',
 '2dsfc/dflx/salt',
 '2dsfc/dms',
 '2dsfc/doxdyn',
 '2dsfc/doxphy',
 '2dsfc/mc/bc',
 '2dsfc/mc/dust',
 '2dsfc/mc/nh4',
 '2dsfc/mc/nitr',
 '2dsfc/mc/oc',
 '2dsfc/mc/pm25/dust',
 '2dsfc/mc/pm25/salt',
 '2dsfc/mc/salt',
 '2dsfc/mc/sulf',
 '2dsfc/taut',
 'T2',
 'ccover',
 'ccoverh',
 'ccoverl',
 'ccoverm',
 'cumf',
 'cumf0',
 'dqcum',
 'dqdad',
 'dqdyn',
 'dqlsc',
 'dqvdf',
 'dtcum',
 'dtdad',
 'dtdyn',
 'dtlsc',
 'dtradl',
 'dtrads',
 'dtvdf',
 'evap',
 'olr',
 'olrc',
 'osr',
 'osrc',
 'prcp',
 'prcpc',
 'prcpl',
 'precw',
 'q2',
 'sens',
 'slrc',
 'slrdc',
 'snow',
 'ssrc',
 'taugxs',
 'taugys',
 'taux',
 'tauy',
 'twpc',
 'u10',
 'uvabs',
 'v10',
 'aerosol/nh4',
 'aerosol/no3',
 'aerosol/sul',
 'ch2o',
 'co',
 'hno3',
 'oh',
 'pan',
 'ps',
 'q',
 'so2',
 't',
 'u',
 'v']

REQUIRED_VARS = ['t', 'q']
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
nsy = ds_momo[[RESPONSE_VARS]]
nsy.load()
time = nsy.time.dt.time
time_mask = time == dt.time(1)
ssy = nsy.where(time_mask, drop=True)

lon = nsy['lon'].values
lat = nsy['lat'].values
nx = len(lon)
ny = len(lat)
nt = int(ssy.time.shape[0] / len(year_list))
mask_lon = (bbox[0] <= lon) & (bbox[1] >= lon)
mask_lat = (bbox[3] >=  lat) & (bbox[2] <= lat) 


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


    
#data['date'] = np.row_stack([data['date'], h5_data['date'][:]])
# for var in REQUIRED_VARS:
#     dat = ds_momo[var].to_numpy()
#     dat = dat[:, mask_lat, :][:, :, mask_lon]
#     data[var] = np.hstack([ data[var], dat[~mask]])

y_toar = obs[~mask]
y = y_momo -  y_toar        
X = np.column_stack(X)

#local_dir = f'/Users/marchett/Documents/SUDS_AQ/data/modeling'
new_file = f'{root_dir}/model/model_data_{region}_{month}_v2.h5'
with closing(h5py.File(new_file, 'w')) as f:
    f['y_momo'] = y_momo
    f['y_toar'] = y_toar
    f['X'] = X
    f['groups'] = groups
    f['mask_toar'] = mask
    f['lon'] = lon
    f['lat'] = lat

#read
with closing(h5py.File(new_file, 'r')) as f:
    y_momo = f['y_momo'][:]
    y_toar = f['y_toar'][:]
    X = f['X'][:]
    groups = f['groups'][:]
    mask = f['mask_toar'][:]
    lon = f['lon'][:]
    lat = f['lat'][:]
    
 
    
y_plot = np.zeros(mask.shape)
y_plot[:] = np.nan
y_plot[~mask] = y_momo  
y_plot = np.nanmean(y_plot, axis = 0)  
x, y = np.meshgrid(lon[mask_lon], lat[mask_lat])

y_plot = rsx['q'].values.mean(axis = 0)
y_plot = ssy[RESPONSE_VARS].values.mean(axis = 0)
y_plot = np.nanmean(ts.values, axis = 0)

x, y = np.meshgrid(lon, lat)

fig = plt.figure(figsize=(18, 9))
ax = plt.subplot(projection = ccrs.PlateCarree())
plt.pcolor(x, y, y_plot)
ax.set_global()
ax.coastlines()
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
ax.stock_img()
 



#modeling

y = y_momo-y_toar
importances = []    
pred_y = np.zeros((len(groups), ))
group_kfold = GroupKFold(n_splits=len(year_list))
for cv_index, (train_index, test_index) in enumerate(group_kfold.split(X, y, groups)):
    print(cv_index)
    train_x, test_x = X[train_index], X[test_index]
    train_y, test_y = y[train_index], y[test_index]

    rf_predictor = RandomForestRegressor()
    rf_predictor.fit(train_x, train_y)
    importances.append(rf_predictor.feature_importances_)
    
    pred_y[test_index] = rf_predictor.predict(test_x)
    


# np.sqrt(np.mean((y - pred_y)**2))
      
plt.figure()
plt.plot(y, pred_y, '.')      

y_plot = np.zeros(mask.shape)
y_plot[:] = np.nan
y_plot[~mask] = pred_y  
y_plot = np.nanmean(y_plot, axis = 0)  
x, y = np.meshgrid(lon[mask_lon], lat[mask_lat])

fig = plt.figure(figsize=(18, 9))
ax = plt.subplot(projection = ccrs.PlateCarree())
plt.pcolor(x, y, y_plot)
ax.set_global()
ax.coastlines()
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
ax.stock_img()

 
mean_imp = np.column_stack(importances).mean(axis = 1)
feature_importance_arr = mean_imp / np.max(mean_imp)



plt.figure(figsize = (5,12))
plt.barh(np.hstack(REQUIRED_VARS), feature_importance_arr)
plt.title('Random Forest Feature Importance')
plt.ylabel('Features')
plt.xlabel('Feature Importance')
plt.yticks(np.arange(0, len(REQUIRED_VARS)), np.hstack(REQUIRED_VARS), fontsize = 6)
plt.grid(linestyle='dotted')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{root_dir}/model/importance_bar_{region}_{month}.png',
              bbox_inches='tight')
plt.close()
  

mask_imp = feature_importance_arr > 0.25
   
plt.figure()
plt.barh(np.hstack(REQUIRED_VARS)[mask_imp], feature_importance_arr[ mask_imp])
plt.title('Random Forest Feature Importance')
plt.ylabel('Features')
plt.xlabel('Feature Importance')
plt.yticks(np.arange(0, mask_imp.sum()), np.hstack(REQUIRED_VARS)[mask_imp], fontsize = 10)
plt.grid(linestyle='dotted')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{root_dir}/model/importance_bar_top_{region}_{month}.png',
              bbox_inches='tight', dpi = 150)
plt.close()

      

            



            
            
            
            
            
            
            