#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 12:23:48 2022

@author: marchett
"""
import os, glob, sys
import numpy as np
import h5py
from tqdm import tqdm
from contextlib import closing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy import stats
import xarray as xr
from scipy import stats
from treeinterpreter import treeinterpreter as ti
import statsmodels.api as sm
sys.path.append('/home/marchett/code/suds-air-quality/research/yuliya/produce_summaries')
import summary_plots as plots
import summarize_explanations as se
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

#-----------------read in data
root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
if not os.path.exists(root_dir):
    root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
if not os.path.exists(root_dir):
    root_dir = '/Users/marchett/Documents/SUDS_AQ/analysis_mount/'    

#month = 'nov'
#years = [2011, 2012, 2013, 2014, 2015]

#set the directory for th60at month
sub_dir = '/bias/local/8hr_median/v4.1/'
models_dir = f'{root_dir}/models/{sub_dir}'
#set plot directory
summaries_dir = f'{root_dir}/summaries/{sub_dir}/combined_data/'
#month = 'jul'
version = 'geo_reg_east_europe'
research_dir = f'{root_dir}/summaries/research/{sub_dir}/{version}/combined_data/'


exclude = ['momo.dtdad', 'momo.cumf', 'momo.ccoverh', 'momo.prcpc', 'momo.2dsfc.BrCl',
              'momo.2dsfc.Br2', 'momo.dtcum', 'momo.2dsfc.mc.sulf', 'momo.2dsfc.HOBr', 
              'momo.dtlsc','momo.2dsfc.Cl2', 'momo.2dsfc.CH3CCl3', 'momo.2dsfc.CH3Br', 
              'momo.2dsfc.ONMV', 'momo.2dsfc.MACROOH', 'momo.2dsfc.MACR',
              'momo.2dsfc.HBr']

months = ['jul']
months = plots.MONTHS

for month in months:
    #month = 'jul'
    print(f'making {month}')
    testing_dir = f'{research_dir}/{month}/'
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
    save_file = f'{summaries_dir}/{month}/test.contributions.mean.nc'
    with closing(h5py.File(save_file, 'r')) as f:
        lons_grid = f['lon'][:]
        lats_grid = f['lat'][:]
    
    mask_exclude = ~np.in1d(var_names, exclude)
    mask_zero = ~(np.std(X, axis = 0) == 0)
    mask_vars = mask_exclude & mask_zero
    XX = X[:, mask_vars].copy()
    var_names_XX = var_names[mask_vars].copy()
    
    #scale
    scaler = StandardScaler()
    XX = scaler.fit(XX).transform(XX)

    bbox = plots.bbox_dict['east_europe']
    lons = (lons + 180) % 360 - 180
    mask_lons = (lons > bbox[0]) & (lons < bbox[1])
    mask_lats = (lats > bbox[2]) & (lats < bbox[3])
    mask_reg = mask_lons & mask_lats 
    
    # un_lons, un_lats = np.unique([lons[mask_reg], lats[mask_reg]], axis = 1)
    # res = np.zeros_like(un_lons)
    # for s in range(len(un_lons)):
    #     mask1 = np.in1d(lons[mask_reg], un_lons[s])
    #     mask2 = np.in1d(lats[mask_reg], un_lats[s])
    #     mask3 = mask1 & mask2
        
    #     time = years[mask_reg][mask3] * 100 + days[mask_reg][mask3]
    #     sidx = np.argsort(time)
    #     ys = y[mask_reg][mask3][sidx]
        
    #     #t = np.arange(0, len(y[mask3]))
    #     res[s] = np.mean(ys)
        
    # plots.residual_scatter(un_lons, un_lats, res, 'y', zlim = (-5,20), 
    #                      cmap = 'bwr', plots_dir = None)

    XX = XX[mask_reg]
    y = y[mask_reg] 
    years = years[mask_reg] 
    
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
    pve_total = r2_score(y, yhat)
    
    save_file = f'{testing_dir}/data.h5'
    with closing(h5py.File(save_file, 'w')) as f:
                 f['X'] = XX
                 f['y'] = y
                 f['y0'] = yhat
                 f['lons'] = lons[mask_reg]
                 f['lats'] = lats[mask_reg]
                 f['years'] = years
                 f['days'] = days[mask_reg]
                 f['mask'] = mask
                 f['var_names'] = var_names[mask_vars].astype(np.string_)
                 
    #save importances
    var_names_stacked = var_names[mask_vars]
    var_names_list = [var_names_stacked.astype(np.string_)] * len(importances['fi'])
    with closing(h5py.File(f'{testing_dir}/test.importances.h5', 'w')) as f:
        #f['ci'] = importances['ci']
        f['model/values'] = np.column_stack(importances['fi'])
        f['model/names'] = np.column_stack(var_names_list)
        f['permutation/values'] = np.column_stack(importances['pi'])
        f['permutation/names'] = np.column_stack(var_names_list)
    
    
    #save contributions
    un_lons, un_lats = np.unique([lons[mask_reg], lats[mask_reg]], axis = 1)
    ds = xr.Dataset(coords = {'lon': (['lon'], lons_grid),
                              'lat': (['lat'], lats_grid)} )
    for n in range(len(var_names_stacked)):
        temp = plots.make_unique_locs(importances['ci'][:, n], 
                                lons[mask_reg], lats[mask_reg], years, 
                                days[mask_reg])
        
        test = np.zeros((len(lons_grid), len(lats_grid)))
        test[:] = np.nan
        for s in range(len(temp)):
            xi = np.where(np.in1d(lons_grid, un_lons[s]))[0]
            yi = np.where(np.in1d(lats_grid, un_lats[s]))[0]
            test[xi, yi] = temp[s]
        
        ds[var_names_stacked[n]] = (['lon', 'lat'], test)
    ds.to_netcdf(f'{testing_dir}/test.contributions.mean.nc')    
    
    
    fi_stacked = np.column_stack(importances['fi'])
    fi_norm = fi_stacked / np.nanmax(fi_stacked, axis = 0)[None,:]
    fi_mean = fi_norm.mean(axis = 1)
    
    pi_stacked = np.column_stack(importances['pi'])
    pi_stacked[pi_stacked < 0] = 0.
    pi_norm = pi_stacked / np.nanmax(pi_stacked, axis = 0)[None,:]
    pi_mean = pi_norm.mean(axis = 1)   
    
    ci_stacked = np.abs(importances['ci']).mean(axis = 0)
    ci_norm = ci_stacked / ci_stacked.max() 


    plt.figure()
    plt.hist(y, histtype = 'step', bins = 100, density = True);
    plt.hist(yhat, histtype = 'step', bins = 100, density = True);
    plt.grid(ls=':', alpha = 0.5)
    plt.title(f'{version}, rmse {np.round(rmse_total, 2)}, pve {np.round(pve_total, 2)}')
    plt.savefig(f'{testing_dir}/hist_{version}.png', dpi = 150, bbox_inches = 'tight')
    plt.close()
    
    
    mask_top = se.get_top_mask(20, 
                            var1 = fi_mean,
                            var2 = pi_mean,
                            var3 = ci_norm)
    plots.imp_barplotv(labels = var_names_XX,
                     var1 = [fi_mean, np.nanstd(fi_norm, axis = 1)],
                     var2 = [pi_mean, np.nanstd(pi_norm, axis = 1)],
                     var3 = [ci_norm, np.repeat(0, len(ci_norm))], 
                     mask_top = mask_top,
                     plots_dir = None)
    plt.savefig(f'{testing_dir}/imps_{version}_top.png', dpi = 150, bbox = 'tight')
    plt.close()

    res = plots.make_unique_locs(y - yhat, lons[mask_reg], lats[mask_reg], years, 
                     days[mask_reg])
    
    # un_lons, un_lats = np.unique([lons[mask_reg], lats[mask_reg]], axis = 1)
    # res = np.zeros_like(un_lons)
    # for s in range(len(un_lons)):
    #     mask1 = np.in1d(lons[mask_reg], un_lons[s])
    #     mask2 = np.in1d(lats[mask_reg], un_lats[s])
    #     mask3 = mask1 & mask2
        
    #     time = years[mask3] * 100 + days[mask_reg][mask3]
    #     sidx = np.argsort(time)
    #     ys = y[mask3][sidx]
    #     ys_hat = yhat[mask3][sidx]
        
    #     #t = np.arange(0, len(y[mask3]))
    #     res[s] = np.mean(ys - ys_hat)
        
    plots.residual_scatter(un_lons, un_lats, res, 'residual', zlim = (-10,10), 
                          cmap = 'bwr', 
                          plots_dir = f'{testing_dir}')



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















    
    