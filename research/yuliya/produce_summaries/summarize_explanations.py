#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:10:12 2022

@author: marchett
"""
import os, glob
import sys
import json
import numpy as np
import h5py
import xarray as xr
from scipy.io import netcdf
from tqdm import tqdm
from contextlib import closing
from datetime import datetime, timedelta, date
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pickle
import scipy.cluster.hierarchy as sch
from sklearn.metrics import pairwise_distances
import seaborn
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from scipy import stats
import summary_plots as plots
import read_output as read

#-----------------read in data
# root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
# if not os.path.exists(root_dir):
#     root_dir = '/data/MLIA_active_data/data_SUDSAQ/'


#choose parameters
#region = 'globe'
#month = 'jan'
#years = [2011, 2012, 2013, 2014, 2015]
#MONTHS = plots.MONTHS

#set the directory for th60at month
# sub_dir = '/bias/local/8hr_median/v1/'
# models_dir = f'{root_dir}/models/{sub_dir}'

# #set plot directory
# summaries_dir = f'{root_dir}/summaries/{sub_dir}/combined_data/'
# plots_dir = f'{root_dir}/summaries/{sub_dir}/summary_plots/'
# # create one if it's not there
# if not os.path.exists(plots_dir):
#     os.makedirs(plots_dir)



#get the mask for the top number a variables
def get_top_mask(a, var1, var2 = None, var3 = None):
    
    metrics_mean = np.stack([v for v in list([var1, var2, var3]) if v is not None])
    #metrics_mean = np.column_stack([var1, var2, var3])
    mean_metrics = metrics_mean.mean(axis = 0)
    mean_metrics[np.isnan(mean_metrics)] = 0. 
    idx_sort = np.argsort(mean_metrics)[::-1]
    mask_top = idx_sort[:a]
    
    return mask_top
    


def main(sub_dir):
    
    root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
    if not os.path.exists(root_dir):
        root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
    
    #set plot directory
    models_dir = f'{root_dir}/models/{sub_dir}'
    summaries_dir = f'{root_dir}/summaries/{sub_dir}/combined_data/'
    plots_dir = f'{root_dir}/summaries/{sub_dir}/summary_plots/'
    # create one if it's not there
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    
    ### ---------------- the simplest barplot and bubble plots for top X
    importances = read.load_importances(models_dir)
    contributions = read.load_contributions(summaries_dir, importances['reference'])
    
    
    #------------plots for TOP 20
    mask_top = get_top_mask(20, var1 = importances['model']['mean'],
                                var2 = importances['permutation']['mean'],
                                var3 = contributions['mean'])
    
    
    plots.imp_barplotv(labels = importances['labels'],
                     var1 = [importances['model']['mean'], importances['model']['std']],
                     var2 = [importances['permutation']['mean'], importances['permutation']['std']],
                     var3 = [contributions['mean'], contributions['std']], 
                     mask_top = mask_top,
                     plots_dir = plots_dir)
    
    
    plots.imp_bubble(labels = importances['labels'],
                    var1 = importances['model']['monthly_mean'],
                    key = 'model',
                    var2 = importances['permutation']['monthly_mean'],
                    var3 = contributions['norm'],
                    mask_top = mask_top,
                    plots_dir = plots_dir)
    
    
    #--------------plots for all
    mask_top = get_top_mask(len(importances['reference']), 
                            var1 = importances['model']['mean'],
                            var2 = importances['permutation']['mean'],
                            var3 = contributions['mean'])
    
    plots.imp_barplotv(labels = importances['labels'],
                     var1 = [importances['model']['mean'], importances['model']['std']],
                     var2 = [importances['permutation']['mean'], importances['permutation']['std']],
                     var3 = [contributions['mean'], contributions['std']], 
                     mask_top = mask_top,
                     plots_dir = plots_dir)
    
    keys = ['model_importance', 'permutation_impotrance', 'contributions']
    for i, metric in enumerate([importances['model']['norm'], 
                                importances['permutation']['norm'], 
                                contributions['norm']]):    
        plots.imp_fullbar(labels = importances['labels'],
                         var1 = metric,
                         key = keys[i],
                         mask_top = mask_top,
                         plots_dir = plots_dir) 
    
    
    for i, metric in enumerate([importances['model']['monthly_mean'], 
                                importances['permutation']['monthly_mean'], 
                                contributions['norm']]): 
        plots.imp_bubble(labels = importances['labels'],
                        var1 = metric,
                        key = keys[i],
                        mask_top = mask_top,
                        plots_dir = plots_dir)
    
    
    #--------------spatial plots contributions
    # for month in plots.MONTHS:
    #     plots.cont_map(month, ['momo.pan'], summaries_dir, plots_dir = None)
    



    #--------- prediction and RESIDUAL plots
    # output = read.load_predictions(summaries_dir)
    
    # boxes = []
    # rmse = []
    # nrmse = []
    # mape = []
    # mae = []
    # pve = []
    # for m in range(len(output['pred'])):
        
    #     if len(output['pred'][m]) > 1:
    #         diff = output['truth'][m] - output['pred'][m]
    #         boxes.append(diff)
            
    #         error = np.sqrt(mean_squared_error(output['truth'][m], output['pred'][m]))
    #         #q = np.percentile(truth_list[m], [25, 75])
    #         nrmse.append(error / np.std(output['truth'][m]))
    #         mae.append(mean_absolute_error(output['truth'][m], output['pred'][m]))
    #         pve.append(explained_variance_score(output['truth'][m], output['pred'][m]))
    #         rmse.append(error)
    #     else:
    #         rmse.append(np.nan)
    #         mae.append(np.nan)
    #         pve.append(np.nan)
    
    # rmse_total = np.sqrt(mean_squared_error(np.hstack(output['truth']), 
    #                                         np.hstack(output['pred'])))    
        
    # #------rmse vs mae
    # plt.figure()
    # plt.plot(np.arange(1, len(rmse)+1), rmse, 'x', color = 'blue', label = 'rmse');
    # plt.plot(np.arange(1, len(rmse)+1), rmse, ':', alpha = 0.5, color = 'blue');
    # plt.plot(np.arange(1, len(mae)+1), mae, 'x', color = 'green', label = 'mae');
    # plt.plot(np.arange(1, len(mae)+1), mae, ':', alpha = 0.5, color = 'green');
    # plt.xticks(np.arange(1, len(plots.MONTHS)+1), plots.MONTHS)
    # plt.grid(ls=':', alpha = 0.5)
    # plt.ylabel('ppb')
    # plt.title(f'model rmse/mae')
    # plt.legend(frameon = True)
    # plt.savefig(f'{plots_dir}/rmse_mae_all.png', bbox_inches='tight')
    # plt.close()
    
    
    # #------histrograms and kde
    # plots.predicted_kde(output, plots_dir = plots_dir)
    # plots.predicted_hist(output, plots_dir = plots_dir)



    # #-------- large residuals
    # lon = np.hstack(output['lon'])
    # lat = np.hstack(output['lat'])
    # un_lons, un_lats = np.unique([lon, lat], axis = 1)
    
    # y = np.hstack(output['truth'])
    # yhat = np.hstack(output['pred'])
    # keys = ['y', 'yhat', 'res', 'res_std']
    # res = {}
    # for k in keys:
    #     res[k] = np.zeros_like(un_lons)
    # for s in range(len(un_lons)):
    #     mask1 = np.in1d(lon, un_lons[s])
    #     mask2 = np.in1d(lat, un_lats[s])
    #     mask3 = mask1 & mask2
    #     #t = np.arange(0, len(y[mask3]))
    #     res['y'][s] = np.mean(y[mask3])
    #     res['yhat'][s] = np.mean(yhat[mask3])
    #     res['res'][s] = np.mean(y[mask3] - yhat[mask3])
    #     res['res_std'][s] = np.std(y[mask3] - yhat[mask3])

    # #---------- residuals on maps
    # plots.residual_scatter(un_lons, un_lats, res['y'], zlim = (-5, 20), 
    #                        key = keys[0], plots_dir = plots_dir)
    # plots.residual_scatter(un_lons, un_lats, res['yhat'], zlim = (-5, 20), 
    #                        key = keys[1], plots_dir = plots_dir)
    # plots.residual_scatter(un_lons, un_lats, res['res'], zlim = (-10, 10), 
    #                        key = keys[2], plots_dir = plots_dir)   
    # plots.residual_scatter(un_lons, un_lats, res['res_std'], zlim = (-30, 30), 
    #                        key = keys[3], plots_dir = plots_dir)

    # idx_res = np.where(res['res'] > 7.5)[0] 
    # #s = idx_res[1]
    
    # if not os.path.exists(f'{plots_dir}/large_residuals/'):
    #     os.makedirs(f'{plots_dir}/large_residuals/')
    # for s in idx_res:
    #     mask1 = np.in1d(lon, un_lons[s])
    #     mask2 = np.in1d(lat, un_lats[s])
    #     mask3 = mask1 & mask2
    #     mr = res['res'][s]
        
    #     plt.figure()
    #     plt.plot(yhat[mask3], label = 'pred') 
    #     plt.plot(y[mask3], color = '0.5', alpha = 0.5, label = 'true') 
    #     plt.legend()
    #     plt.grid(ls=':', alpha = 0.5)    
    #     plt.title(f'location: {un_lons[s]}, {un_lats[s]}, mean res: {np.round(mr,2)}')
    #     plt.savefig(f'{plots_dir}/large_residuals/signal_{un_lons[s]}_.{un_lats[s]}.png',
    #                 bbox_inches='tight')
    #     plt.close()


          







