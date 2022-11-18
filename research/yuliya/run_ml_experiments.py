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
from scipy import stats
from treeinterpreter import treeinterpreter as ti
import statsmodels.api as sm
from sklearn.ensemble import ExtraTreesRegressor
import summary_plots as plots
#import make_X as make
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

#-----------------read in data
root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
if not os.path.exists(root_dir):
    root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
if not os.path.exists(root_dir):
    root_dir = '/Users/marchett/Documents/SUDS_AQ/analysis_mount/'    

#month = 'nov'
years = [2011, 2012, 2013, 2014, 2015]
#months = ['dec', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov']  

#set the directory for th60at month
sub_dir = '/bias/local/8hr_median/v1/'
models_dir = f'{root_dir}/models/{sub_dir}'
#set plot directory
summaries_dir = f'{root_dir}/summaries/{sub_dir}/combined_data/'
# create one if it's not there


month = 'jul'
version = 'remove_dups_standard'

testing_dir = f'{summaries_dir}/{month}/experiments/'
if not os.path.exists(testing_dir):
    os.makedirs(testing_dir)

#--read pre-processed ml-ready data
save_file = f'{summaries_dir}/{month}/data.h5'
with closing(h5py.File(save_file, 'r')) as f:
    y = f['y'][:]
    y0 = f['y0'][:]
    X = f['X'][:]
    years = f['years'][:]
    days = f['days'][:]
    lons = f['lons'][:]
    lats = f['lats'][:]
    mask = f['mask'][:]
    var_names = f['var_names'][:].astype(str)
        

exclude = ['momo.hno3', 'momo.oh', 'momo.pan', 'momo.q2',
           'momo.sens', 'momo.so2', 'momo.T2', 'momo.taugxs',
           'momo.taugys', 'momo.taux', 'momo.tauy', 'momo.twpc',
           'momo.2dsfc.CFC11', 'momo.2dsfc.CFC113', 'momo.2dsfc.CFC12',
           'momo.ch2o', 'momo.cumf0', 'momo.2dsfc.dms']


mask_vars = ~np.in1d(var_names, exclude)
XX = X[:, mask_vars]
#scale
scaler = StandardScaler()
XX = scaler.fit(XX).transform(XX)



from scipy.stats import spearmanr
corr = spearmanr(XX).correlation



un_years = np.unique(years)
yhat = np.zeros_like(y)
rmse = []
importances = {'fi': [], 'ci': np.zeros_like(XX), 'pi': []}
for i in tqdm(range(len(un_years))):
    mask_test = years == un_years[i]
    X_train = XX[~mask_test,:]
    y_train = y[~mask_test]
    
    X_test = XX[mask_test, :]
    y_test = y[mask_test]
    
    rf = RandomForestRegressor(n_estimators = 20,
                                min_samples_leaf=20,
                                max_features = 0.33,
                                random_state=6789)
    rf.fit(X_train, y_train)
    #yhat[mask_test] = rf.predict(X_test)
    treeint = ti.predict(rf, X_test)
    yhat[mask_test] = np.hstack(treeint[0])
    importances['fi'].append(rf.feature_importances_)
    importances['ci'][mask_test] = treeint[2]
    
    pi_dict = permutation_importance(rf, X_test, y_test, 
                                n_repeats=15, 
                                max_samples = 0.8,
                                random_state=6789)
    importances['pi'].append(pi_dict['importances_mean'])
    
    rmse.append(np.sqrt(mean_squared_error(y_test, yhat[mask_test])))

rmse_total = np.sqrt(mean_squared_error(y, yhat))


var_names_reduced = var_names[mask_vars]
with closing(h5py.File(f'{testing_dir}/test.importances_{version}.h5', 'w')) as f:
    f['fi'] = np.column_stack(importances['fi'])
    f['pi'] = np.column_stack(importances['pi'])
    f['ci'] = importances['ci']

    
fi_stacked = np.column_stack(importances['fi'])
fi_norm = fi_stacked / np.nanmax(fi_stacked, axis = 0)[None,:]
fi_mean = fi_norm.mean(axis = 1)

pi_stacked = np.column_stack(importances['pi'])
pi_norm = pi_stacked / np.nanmax(pi_stacked, axis = 0)[None,:]
pi_mean = pi_norm.mean(axis = 1)    


plt.figure()
plt.hist(y, histtype = 'step', bins = 100, density = True);
plt.hist(yhat, histtype = 'step', bins = 100, density = True);
plt.grid(ls=':', alpha = 0.5)
plt.title(f'rmse {rmse_total}')
plt.savefig(f'{testing_dir}/hist_{version}.png', dpi = 150, bbox = 'tight')
plt.close()


mask_top = get_top_mask(30, 
                        var1 = fi_mean,
                        var2 = pi_mean)
plots.imp_barplotv(labels = var_names_reduced,
                 var1 = [fi_mean, np.nanstd(fi_norm, axis = 1)],
                 var2 = [pi_mean, np.nanstd(pi_norm, axis = 1)],
                 var3 = None, 
                 mask_top = mask_top,
                 plots_dir = None)
plt.savefig(f'{testing_dir}/imps_{version}.png', dpi = 150, bbox = 'tight')
plt.close()
















#subsample data
# y_denoise = np.zeros_like(y)
# for s in range(len(un_lons)):
#     mask1 = np.in1d(lons, un_lons[s])
#     mask2 = np.in1d(lats, un_lats[s])
#     mask3 = mask1 & mask2
    
#     mask_q = np.repeat(True, mask3.sum())
#     mask_q[::2] = False
#     y_temp = y[mask3].copy()
#     y_temp[mask_q] = np.nan
#     y_denoise[mask3] = y_temp

# mask_denoise = ~np.isnan(y_denoise)
# X_denoise = X[mask_denoise, :]
# y_denoise = y[mask_denoise]
# years_denoise = years[mask_denoise]


# un_years = np.unique(years_denoise)
# yhat = np.zeros_like(y)
# fi = []
# rmse_d = []
# for i in tqdm(range(len(un_years))):
#     mask_train = years_denoise == un_years[i]
#     mask_test = years == un_years[i]
#     X_train = X_denoise[~mask_train,:]
#     y_train = y_denoise[~mask_train]
    
#     X_test = X[mask_test, :]
#     y_test = y[mask_test]
 
#     rf = RandomForestRegressor(n_estimators = 20,
#                                 min_samples_leaf=20,
#                                 max_features = 0.33,
#                                 random_state=6789)
#     rf.fit(X_train, y_train)
#     fi.append(rf.feature_importances_)
#     yhat[mask_test] = rf.predict(X_test)
    
#     rmse_d.append(np.sqrt(mean_squared_error(y_test, yhat[mask_test])))


# rmse_total = np.sqrt(mean_squared_error(y, yhat))
# with closing(h5py.File(save_file, 'a')) as f:
#              f['yhat'] = yhat


# truth = y.copy()
# kernel  = stats.gaussian_kde([truth, yhat])
# density = kernel([truth, yhat])
# plt.figure(figsize = (12,10))
# plt.scatter(yhat, truth, s = 3, c=density, alpha = 0.5, cmap = 'coolwarm')
# plt.axvline(x=0, color = '0.5', alpha = 0.5, ls = '--')
# plt.axhline(y=0, color = '0.5', alpha = 0.5, ls = '--')
# plt.plot(yhat.mean(), y.mean(), '+', ms = 10, color = 'r', alpha = 0.7)
# plt.plot([-100,100], [-100,100], color = 'r', alpha = 0.5, ls = '--')
# plt.ylim((-75, 75))
# plt.xlim((-75, 75))
# plt.text(0.1, 0.9, f'rmse {np.round(rmse_total,2)}',transform=plt.gca().transAxes) 
# plt.grid(ls = ':')
# plt.xlabel(f'predicted ppb')
# plt.ylabel(f'true ppb')






# un_lons, un_lats = np.unique([lons, lats], axis = 1)
# y_denoise = np.zeros_like(y)
# y_denoise[:] = np.nan
# for s in range(len(un_lons)):
#     mask1 = np.in1d(lons, un_lons[s])
#     mask2 = np.in1d(lats, un_lats[s])
#     mask3 = mask1 & mask2
#     y_denoise[mask3] = y[mask3]

#smooth with lowess
# lowess = sm.nonparametric.lowess
# y_denoise = np.zeros_like(y)
# y_denoise[:] = np.nan
# mask_all = []
# for s in range(len(un_lons)):
#     mask1 = np.in1d(lons, un_lons[s])
#     mask2 = np.in1d(lats, un_lats[s])
#     mask3 = mask1 & mask2
#     #count = mask3.sum()
#     t = np.arange(0, len(y[mask3]))
#     z = lowess(y[mask3], t, frac = 0.1)
    
#     res = y[mask3] - z[:,1]
#     q1, q3 = np.percentile(res, [25,75])
#     w1 = q1 - (q3 - q1) * 10
#     w3 = q3 + (q3 - q1) * 10
#     mask_q = (y[mask3] < w1) | (y[mask3] > w3)
    
#     y_temp = y[mask3].copy()
#     y_temp[mask_q] = np.nan
#     y_denoise[mask3] = y_temp
    
#     mask_all.append(mask_q)
    
# mask_denoise = ~np.isnan(y_denoise)        
# X_denoise = X[mask_denoise, :]
# y_denoise = y_denoise[mask_denoise]
# years_denoise = years[mask_denoise]    
#y_smoo = y_smoo[mask_denoise]



# count = np.hstack([x.sum() for x in mask_all])
# np.where((count > 0) & (count< 10))[0]
# s = 123
# mask1 = np.in1d(lons, un_lons[s])
# mask2 = np.in1d(lats, un_lats[s])
# mask3 = mask1 & mask2
# lowess = sm.nonparametric.lowess
# t = np.arange(0, len(y[mask3]))
# z = lowess(y[mask3], t, frac = 0.1)

# res = y[mask3] - z[:,1]
# q1, q3 = np.percentile(res, [25,75])
# w1 = q1 - (q3 - q1) * 10
# w3 = q3 + (q3 - q1) * 10
# mask_q = (y[mask3] < w1) | (y[mask3] > w3)

# plt.figure()
# plt.plot(y[mask3], '-.')
# plt.plot(z[:,1], '-', color = 'r')
# plt.plot(np.arange(mask3.sum())[mask_q], y[mask3][mask_q], 'x')



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
plt.plot(yhat, y_denoise, 'o', ms = 5, alpha = 0.2, markerfacecolor = 'cornflowerblue', markeredgecolor = 'k')
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
         density = True, label = f'pred, mean {np.round(yhat.mean(), 2)}');
plt.axvline(x = y.mean(), color = '0.5')
plt.axvline(x = y0.mean(), ls='--')
plt.legend()
plt.grid(ls=':', alpha = 0.5)
#plt.text(0.1, 0.9, f'mean {np.round(y_.mean(),2)}',transform=plt.gca().transAxes)  
plt.title(f'{month}')




un_lons, un_lats = np.unique([lons, lats], axis = 1)
un_y = np.zeros_like(un_lons)
un_yhat = np.zeros_like(un_lons)
un_y0 = np.zeros_like(un_lons)
for i in range(len(un_lons)):
    mask1 = np.in1d(lons, un_lons[i])
    mask2 = np.in1d(lats, un_lats[i])
    mask3 = mask1 & mask2
    un_y[i] = y[mask3].mean()
    un_yhat[i] = yhat[mask3].mean()
    un_y0[i] = y0[mask3].mean()


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














    
    