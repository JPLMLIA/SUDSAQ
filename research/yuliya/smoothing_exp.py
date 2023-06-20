#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:12:05 2023

@author: marchett

run basic random forest (by month) on smoothed data:
    shows how smoothing affects prediction

--smooth response data by rolling mean and time window
--smooth input data the same way
--run random forest
--compare the results with predictions based on raw data

"""
import os, glob
import sys
import numpy as np
import h5py
from tqdm import tqdm
from contextlib import closing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import xarray as xr
from treeinterpreter import treeinterpreter as ti
sys.path.insert(0, '/Users/marchett/Documents/SUDS_AQ/analysis_mount/code/suds-air-quality/research/yuliya/produce_summaries')
import summary_plots as plots
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pandas as pd



sub_dir = '/bias/gattaca.v4.bias-median'
root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
if not os.path.exists(root_dir):
    root_dir = '/data/MLIA_active_data/data_SUDSAQ/'


#----- set experiment name
version = 'rolling'

#set plot directory
models_dir = f'{root_dir}/models/{sub_dir}'
summaries_dir = f'{root_dir}/summaries/{sub_dir}/combined_data/'
plots_dir = f'{root_dir}/summaries/{sub_dir}/summary_plots/'
research_dir = f'{root_dir}/summaries/research/{sub_dir}/{version}/'

#----- set experiment month
month = 'jul'

testing_dir = f'{research_dir}/{month}/'
save_file = f'{summaries_dir}/{month}/data.h5'

output = {'X': [], 'pred': [], 'truth': [], 'lon': [], 'lat': [], 'years': [], 
          'months': [], 'days': []}
with closing(h5py.File(save_file, 'r')) as f:
    output['X'].append(f['X'][:])
    output['truth'].append(f['y'][:])
    output['pred'].append(f['y0'][:])
    output['lon'].append(f['lons'][:])
    output['lat'].append(f['lats'][:])
    output['years'].append(f['years'][:])
    output['months'].append(f['months'][:])
    output['days'].append(f['days'][:])  
    var_names = f['var_names'][:].astype(str)

ds_momo = xr.open_dataset(f'{summaries_dir}/{month}/test.predict.nc')
mlon = ds_momo['lon'].values
mlat = ds_momo['lat'].values


#---- process data
scaler = StandardScaler()
XX = scaler.fit(output['X'][0]).transform(output['X'][0])

y = output['truth'][0]
years = output['years'][0]
months = np.hstack(output['months'][0]) 
days = np.hstack(output['days'][0])

lon = np.hstack(output['lon'][0])
lat = np.hstack(output['lat'][0])
lon = (lon + 180) % 360 - 180
un_lons, un_lats = np.unique([lon, lat], axis = 1)


#----- set time window size
window = 15
y_denoise = np.zeros_like(y)
y_denoise[:] = np.nan
sigma_gp = np.zeros_like(y)
sigma_gp[:] = np.nan

wk = np.zeros_like(un_lons)
gpmat_time = np.zeros((160*320, len(un_lons)))
for s in tqdm(range(len(un_lons))):
    
    mask, idx = plots.unique_loc(un_lons[s], un_lats[s], output)
    
    time = years[mask] * 10000 + months[mask] * 100 + days[mask]
    sidx = np.argsort(time)
    yt = y[mask][sidx]
    yt_pred = output['pred'][0][mask][sidx]
    #plots.time_series_loc(un_lons[s], un_lats[s], output)
    
    ds = pd.Series(yt, index = pd.to_datetime(time, format='%Y%m%d'))
    new_idx = pd.date_range(ds.index.min(), ds.index.max(), freq='1D')
    
    ds = ds.reindex(new_idx, fill_value = np.nan)
    ys = ds.rolling(window = window, min_periods = int(np.ceil((window*0.25))), 
                    center = True).mean()
    
    month_mask = ds.index.month == 7
    mask_nan = ~np.isnan(ds)
    y_denoise[mask] = ys.values[mask_nan] 

    #-----
    # plt.figure()
    # plt.plot(x_pred[train_idx], response[train_idx], '.-', 
    #          color = 'b', alpha = 0.6,
    #          label = 'data')
    # plt.plot(x_pred, np.hstack(y_pred), 
    #          color = '0.6', label = 'GP mean')
    # #ax = plt.gca()
    # #positions = plt.gca().get_xticks()
    # #labels = ds.index.year[month_mask]
    # plt.fill_between(x_pred.ravel(), y_pred.ravel()-1.96*sigma, 
    #                  y_pred.ravel()+1.96*sigma, 
    #                   color='0.9', alpha = 0.5)
    # #plt.xticks(positions, labels, rotation = 0);
    # plt.legend()
    # plt.grid(ls = ':', alpha = 0.5)
    # plt.title('July 2011-2015 actual vs GP mean (no gaps)')
    #-----
    
    

#---- smooth inputs X
XS = np.zeros_like(XX)
for s in tqdm(range(len(un_lons))):
    
    mask, idx = plots.unique_loc(un_lons[s], un_lats[s], output)
    time = years[mask] * 10000 + months[mask] * 100 + days[mask]
    sidx = np.argsort(time)
    
    for j in range(XX.shape[1]):
        XXt = XX[mask, j][sidx]
        ds_x = pd.Series(XXt, index = pd.to_datetime(time, format='%Y%m%d'))
        new_idx_x = pd.date_range(ds_x.index.min(), ds_x.index.max(), freq='1D')
        
        ds_x = ds_x.reindex(new_idx_x, fill_value = np.nan)
        xs = ds_x.rolling(window = window, min_periods = int(np.ceil((window*0.25))), 
                          center = True).mean()
        mask_nan = ~np.isnan(ds_x)
        XS[mask, j] = xs.values[mask_nan]        
        
        
#---- predict with smooth inputs + ouput
mask_1 = np.isnan(XS.sum(axis = 1))
mask_2 = np.isnan(y_denoise)
mask_ = mask_1 | mask_2
y_s = y_denoise[~mask_]
years_s = output['years'][0][~mask_]

#---- run basic random forest
yhat_s, imp = rf_function(XS[~mask_], y_s, years_s, days[~mask_], 
                          lon[~mask_], lat[~mask_])


yhat_0 = output['pred'][0][~mask_]  
rmse_w_s = np.round(np.sqrt(mean_squared_error(y_s, yhat_s)), 2)
#rmse_w_r = np.round(np.sqrt(mean_squared_error(y_s, yhat_0)), 2)
pve_w_s = np.round(r2_score(y_s, yhat_s), 2)

rmse_s = np.round(np.sqrt(mean_squared_error(y[~mask_], yhat_s)), 2)
rmse_r = np.round(np.sqrt(mean_squared_error(y[~mask_], yhat_0)), 2)
pve_s = np.round(r2_score(y[~mask_], yhat_s), 2)
pve_r = np.round(r2_score(y[~mask_], yhat_0), 2)


#---- histograms
plt.figure()
plt.hist(y, bins = 100, density = True, alpha = 0.2, color = '0.5', label = 'truth');
plt.hist(yhat_s, histtype = 'step', bins = 100, density = True, label = 'pred-s');
plt.hist(yhat_0, histtype = 'step', bins = 100, density = True, label = 'pred-r');
plt.grid(ls=':', alpha = 0.5)
plt.legend(loc=1)
plt.xlim((-50, 55))
plt.title(f'{month}, smooth vs raw, running window {window}')
plt.text(0.05, 0.9, s=f'rmse-r = {rmse_r}, pve-r = {pve_r}', 
         fontsize = 8, transform = plt.gca().transAxes)
plt.text(0.05, 0.86, s=f'rmse-s = {rmse_s}, pve-s = {pve_s}', 
         fontsize = 8, transform = plt.gca().transAxes)
plt.text(0.05, 0.82, s=f'rmse-w-s = {rmse_w_s}, pve-w-s = {pve_w_s}', 
         fontsize = 8, transform = plt.gca().transAxes)
plt.savefig(f'{testing_dir}/hist_comp_preds_w{window}.png')
plt.close()



output_test = {}
output_test['truth'] = y[~mask_]
output_test['pred'] = yhat_s
output_test['pred0'] = yhat_0
output_test['years'] = years_s
output_test['months'] = months[~mask_]
output_test['days'] = days[~mask_]
output_test['lon'] = lon[~mask_]
output_test['lat'] = lat[~mask_]
os.mkdir(f'{testing_dir}/time_plots/w{window}')


#---- time series plots
#s = 103
np.random.seed(79340)
sample = np.random.choice(np.arange(len(un_lons)), 100)
for s in sample:
    plots.time_series_loc_noGaps(un_lons[s], un_lats[s], 
                                 output_test, 
                                 plots_dir = f'{testing_dir}/time_plots/w{window}/')


res = plots.map_variable(un_lons, un_lats, yhat_s - yhat_0,
                         lon[~mask_], lat[~mask_], years_s, months[~mask_], 
                         days[~mask_])
plots.residual_scatter(un_lons, un_lats, res, 'residual', 
                       zlim = (-0.5,0,0.5), plots_dir = None)




def rf_function(XX, y, years, days, lons, lats):
    
    un_years = np.unique(years)
    yhat = np.zeros_like(y)
    #rmse = []
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
        yhat[mask_test] = rf.predict(X_test)
        # treeint = ti.predict(rf, X_test)
        # yhat[mask_test] = np.hstack(treeint[0])
        importances['fi'].append(rf.feature_importances_)
        # importances['ci'][mask_test] = treeint[2]
        
        # pi_dict = permutation_importance(rf, X_test, y_test, 
        #                             n_repeats=15, 
        #                             max_samples = 0.8,
        #                             random_state=6789)
        # importances['pi'].append(pi_dict['importances_mean'])
        #rmse.append(np.sqrt(mean_squared_error(y_test, yhat[mask_test])))


    return yhat, importances
        



