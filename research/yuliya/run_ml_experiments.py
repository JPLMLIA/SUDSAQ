#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 12:23:48 2022

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import scipy as sp
from scipy import stats
import xarray as xr
from tqdm import tqdm


#-----------------read in data
root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
if not os.path.exists(root_dir):
    root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
    

#month = 'nov'
years = [2011, 2012, 2013, 2014, 2015]
months = ['dec', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov']  

#set the directory for th60at month
models_dir = f'{root_dir}/models/2011-2015/bias-8hour/'
#set plot directory
summaries_dir = f'{root_dir}/summaries/2011-2015/bias-8hour/combined_data/'
plots_dir = f'{root_dir}/summaries/2011-2015/bias-8hour/plots/'
# create one if it's not there
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    


month = 'jul'
data_x = np.hstack(glob.glob(f'{models_dir}/{month}/rf/*/test.data.nc'))
data_y = glob.glob(f'{models_dir}/{month}/rf/*/test.target.nc')
data_y0 = glob.glob(f'{models_dir}/{month}/rf/*/test.predict.nc')


#train data
ds_y = xr.open_mfdataset(data_y)
# ds_y_flat = ds_y.stack(coord=("lon", "lat", 'time'))
# ds_y_drop = ds_y_flat.dropna(dim='coord')
ds_y_ = ds_y.to_array().stack({'loc': ["lon", "lat", 'time']})
mask = ~np.isnan(ds_y_.values[0])

y = ds_y_.values[0][mask]
years = ds_y_['time.year'].values[mask]
lons = ds_y_['lon'].values[mask]
lats = ds_y_['lat'].values[mask]


ds_x = xr.open_mfdataset(data_x)
ds_x_ = ds_x.to_array().stack({'loc': ["lon", "lat", 'time']})
var_names = ds_x_['variable'].values
X = ds_x_.values[:, mask].T

#original prediction
ds_y0 = xr.open_mfdataset(data_y0)
ds_y0_ = ds_y0.to_array().stack({'loc': ["lon", "lat", 'time']})
y0 = ds_y0_.values[0][mask]

#save ML ready data
save_file = f'{summaries_dir}/{month}/data.h5'
with closing(h5py.File(save_file, 'w')) as f:
             f['X'] = X
             f['y'] = y
             f['y0'] = y0
             f['lons'] = lons
             f['lats'] = lats
             f['years'] = years
             f['mask'] = mask
    
un_years = np.unique(years)
yhat = np.zeros_like(y)
rmse = []
for i in tqdm(range(len(un_years))):
    mask_test = years == un_years[i]
    X_train = X[~mask_test,:]
    y_train = y[~mask_test]
    
    X_test = X[mask_test, :]
    y_test = y[mask_test]
    
    rf = RandomForestRegressor(n_estimators = 20,
                               max_features = int(len(var_names)*0.33),
                               random_state=6789)
    rf.fit(X_train, y_train)
    yhat[mask_test] = rf.predict(X_test)
    
    rmse.append(np.sqrt(mean_squared_error(y_test, yhat[mask_test])))


rmse_total = np.sqrt(mean_squared_error(y, yhat))

from scipy import stats
kernel  = stats.gaussian_kde([y, yhat])
density = kernel([y, yhat])
plt.figure(figsize = (12,10))
plt.scatter(yhat, y, s = 3, c=density, alpha = 0.5, cmap = 'coolwarm')
plt.axvline(x=0, color = '0.5', alpha = 0.5, ls = '--')
plt.axhline(y=0, color = '0.5', alpha = 0.5, ls = '--')
plt.plot(yhat.mean(), y.mean(), '+', ms = 10, color = 'r', alpha = 0.7)
plt.plot([-100,100], [-100,100], color = 'r', alpha = 0.5, ls = '--')
plt.ylim((-75, 75))
plt.xlim((-75, 75))
plt.text(0.1, 0.9, f'rmse {np.round(rmse_total,2)}',transform=plt.gca().transAxes) 
plt.grid(ls = ':')
plt.xlabel(f'predicted ppb')
plt.ylabel(f'true ppb')

# H, xedges, yedges = np.histogram2d(y, yhat, bins = 100)
# H[H == 0] = np.nan
# plt.figure()
# plt.pcolor(xedges, yedges, H, cmap = 'coolwarm')
# plt.axvline(x=0, color = '0.5', alpha = 0.5, ls = '--')
# plt.axhline(y=0, color = '0.5', alpha = 0.5, ls = '--')
# plt.ylim((-75, 75))
# plt.xlim((-75, 75))
# plt.grid(ls = ':')
# plt.xlabel(f'true ppb')
# plt.ylabel(f'predicted ppb')
# plt.colorbar()
#plt.contour(H, levels = 5, cmap = 'coolwarm')
# plt.plot([-100,100], [-100,100], color = 'r', alpha = 0.2, ls = '--')
#plt.title(f'predicted vs true bias, all months, rmse total = {np.round(rmse_total, 2)}')
rmse0 = []
for i in range(len(un_years)):
    mask_test = years == un_years[i]
    rmse0.append(np.sqrt(mean_squared_error(y[mask_test], y0[mask_test])))


from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(yhat.reshape(-1,1), y)
reg.score(yhat.reshape(-1,1), y)
line = reg.predict(yhat.reshape(-1,1))

#fitted vs actual
plt.figure(figsize = (12,10))
plt.plot(yhat, y, 'o', ms = 5, alpha = 0.2, markerfacecolor = 'cornflowerblue', markeredgecolor = 'k')
plt.axvline(x=0, color = '0.5', alpha = 0.5, ls = '--')
plt.axhline(y=0, color = '0.5', alpha = 0.5, ls = '--')
plt.plot(yhat.mean(), y.mean(), 'x', ms = 10, color = 'r', alpha = 0.7)
# plt.axvline(x = yhat.mean(), color = '0.5')
# plt.axhline(y = y.mean(), ls='--')
plt.plot([-100,100], [-100,100], color = 'r', alpha = 0.5, ls = '--')
plt.ylim((-50, 50))
plt.xlim((-50, 50))
plt.grid(ls = ':')
plt.xlabel(f'predicted ppb')
plt.ylabel(f'true ppb')


#histrograms per year
plt.figure()
for i in range(len(un_years)):
   
    mask_test = years == un_years[i]
    plt.hist(y[mask_test], bins = 100, histtype='step', 
             density = True, color = '0.5', alpha = (i+1)/10);
    plt.hist(yhat[mask_test], bins = 100, histtype='step', 
             alpha = (i+1)/10, color = 'b', density = True, label = f'{un_years[i]}');
    plt.legend()
    #plt.axvline(x = y[mask_test].mean(), color = '0.5')
    #plt.axvline(x = yhat[i].mean(), ls='--')
    plt.title(f'{un_years[i]}')
plt.grid(ls=':', alpha = 0.5)


plt.figure()
plt.hist(y, bins = 100, histtype='step', 
         density = True, color = '0.5', label = f'true, mean {np.round(y.mean(),2)}');
plt.hist(yhat, bins = 100, histtype='step', 
         density = True, label = f'pred, mean {np.round(yhat.mean(),2)}');
plt.axvline(x = y.mean(), color = '0.5')
plt.axvline(x = yhat.mean(), ls='--')
plt.legend()
plt.grid(ls=':', alpha = 0.5)
#plt.text(0.1, 0.9, f'mean {np.round(y_.mean(),2)}',transform=plt.gca().transAxes)  


mask_big = np.abs(y) > 40
np.unique(lons[mask_big]).shape

un_lons, un_lats = np.unique([lons, lats], axis = 1)
un_y = np.zeros_like(un_lons)
un_yhat = np.zeros_like(un_lons)
for i in range(len(un_lons)):
    mask1 = np.in1d(lons, un_lons[i])
    mask2 = np.in1d(lats, un_lats[i])
    mask3 = mask1 & mask2
    un_y[i] = y[mask3].mean()
    un_yhat[i] = yhat[mask3].mean()


fig = plt.figure(figsize=(18, 9))
ax = plt.subplot(projection = ccrs.PlateCarree())
plt.scatter(un_lons, un_lats, s= 2, c=un_y - un_yhat, cmap = 'coolwarm')
plt.clim((-10, 10))
#plt.pcolor(H, cmap = 'Reds')
#plt.scatter(lons[mask_big], lats[mask_big], y[mask_big], cmap = 'jet')
#plt.clim(-1200, 0)
ax.set_global()
ax.coastlines()
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
# ax.set_extent([bbox[0], bbox[1], bbox[2], bbox[3]], crs=ccrs.PlateCarree())
#ax.stock_img()
plt.colorbar()
plt.title(f'ML model residuals, {month}')














    
    