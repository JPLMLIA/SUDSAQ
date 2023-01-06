#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:45:32 2022

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
from sklearn.metrics import pairwise_distances
import seaborn
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy import stats
import summary_plots as plots

#MONTHS = plots.MONTHS

#-----------------read in data
root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
if not os.path.exists(root_dir):
    root_dir = '/data/MLIA_active_data/data_SUDSAQ/'



#--------- LOAD importances
# def load_importances(models_dir):
#     output_files = glob.glob(f'{models_dir}/*/*/*importance.h5')
    
#     mi = []
#     pi = []
#     mi_names = []
#     pi_names = []
#     pi_months = []
#     for file in output_files:
#         try:
#             with closing(h5py.File(file, 'r')) as f:
#                 mi.append(f['model']['block0_values'][0,:])
#                 mi_names.append(f['model']['axis0'][:].astype(str))
                
#                 pi.append(f['permutation']['block0_values'][0, :])
#                 pi_names.append(f['permutation']['axis0'][:].astype(str))
#                 pi_months.append(file.split('/')[-3])
#         except:
#             mi_months[-1] = np.nan
#             pi_months[-1] = np.nan
#             continue
    
#     #sort everything to the first model importance order
#     reference_labels = mi_names[0]
#     tick_labels = np.hstack(['.'.join(x.split('.')[1:]) for x in reference_labels])
#     tick_labels = np.hstack([x.split('2dsfc')[-1].split('.')[-1] for x in tick_labels])
    
#     #sort each importance vector by name
#     mi_sorted = np.zeros((len(reference_labels), len(mi)))
#     pi_sorted = np.zeros((len(reference_labels), len(pi)))
#     for i in range(len(mi)):
#         mi_idx = np.hstack([np.where(mi_names[i] == label)[0] for label in reference_labels])
#         mi_sorted[:, i] = mi[i][mi_idx] / mi[i][mi_idx].max()
        
#         pi_idx = np.hstack([np.where(pi_names[i] == label)[0] for label in reference_labels])
#         pi_sorted[:, i] = pi[i][pi_idx] / pi[i][pi_idx].max()
              
#     months_list = np.hstack([x.split('/')[-3] for x in output_files])
#     mi_monthly_mean = np.zeros((len(reference_labels), len(plots.MONTHS)))
#     mi_monthly_std = np.zeros((len(reference_labels), len(plots.MONTHS)))
#     pi_monthly_mean = np.zeros((len(reference_labels), len(plots.MONTHS)))
#     pi_monthly_std = np.zeros((len(reference_labels), len(plots.MONTHS)))
#     for m in range(len(plots.MONTHS)):
#         mask_month = months_list == plots.MONTHS[m]
#         if mask_month.sum() > 0:
#             mi_monthly_mean[:, m] = mi_sorted[:, mask_month].mean(axis = 1)
#             mi_monthly_std[:, m] = mi_sorted[:, mask_month].std(axis = 1)
#             pi_monthly_mean[:, m] = pi_sorted[:, mask_month].mean(axis = 1)
#             pi_monthly_std[:, m] = pi_sorted[:, mask_month].std(axis = 1)
#         else:
#             mi_monthly_mean[:, m] = np.repeat(np.nan, len(reference_labels))
#             mi_monthly_std[:, m] = np.repeat(np.nan, len(reference_labels))
#             pi_monthly_mean[:, m] = np.repeat(np.nan, len(reference_labels))
#             pi_monthly_std[:, m] = np.repeat(np.nan, len(reference_labels))
    
   
#     importances = {'model': 
#                    {'norm': mi_sorted, 
#                     'monthly_mean': mi_monthly_mean,
#                     'mean': np.nanmean(mi_monthly_mean, axis = 1), 
#                     'monthly_std': mi_monthly_std,
#                     'std': np.nanmean(mi_monthly_std, axis = 1)}, 
#                    'permutation': 
#                        {'norm': pi_sorted, 
#                         'monthly_mean': pi_monthly_mean, 
#                         'mean': np.nanmean(pi_monthly_mean, axis = 1),
#                         'monthly_std': pi_monthly_std,
#                         'std': np.nanmean(pi_monthly_std, axis = 1)},
#                    'months_list': months_list,
#                    'labels': tick_labels,
#                    'reference': reference_labels}
    
#     return importances



#--------- LOAD importances
def load_importances(summaries_dir):
    
    output_files = glob.glob(f'{summaries_dir}/*/*importances*.h5')
    
    mi = []
    pi = []
    mi_names = []
    pi_names = []
    pi_months = []
    for file in output_files:
        with closing(h5py.File(file, 'r')) as f:
            mi.append(f['model']['values'][:])
            mi_names.append(f['model']['names'][:].astype(str))
            
            pi.append(f['permutation']['values'][:])
            pi_names.append(f['permutation']['names'][:].astype(str))
            
            p = pi[-1].shape[1]
            pi_months.append(np.repeat(file.split('/')[-2], p))
    
    #sort everything to the first model importance order
    reference_labels = mi_names[0][:, 0]
    tick_labels = np.hstack(['.'.join(x.split('.')[1:]) for x in reference_labels])
    tick_labels = np.hstack([x.split('2dsfc')[-1].split('.')[-1] for x in tick_labels])
    
    mi = np.column_stack(mi)
    pi = np.column_stack(pi)
    mi_names = np.column_stack(mi_names)
    pi_names = np.column_stack(pi_names)
    mi_sorted = mi.copy(); pi_sorted = pi.copy()
    for i in range(mi.shape[1]):
        mi_idx = np.hstack([np.where(mi_names[:, i] == label)[0] 
                            for label in reference_labels])
        mi_sorted[:, i] = mi[mi_idx, i] / mi[mi_idx, i].max()
        
        pi_idx = np.hstack([np.where(pi_names[:, i] == label)[0] 
                            for label in reference_labels])
        pi_sorted[:, i] = pi[pi_idx, i] / pi[pi_idx, i].max()
    
    
    #sort each importance vector by name
    # mi_sorted = []
    # pi_sorted = np.zeros((len(reference_labels), len(pi)))
    # for i in range(len(mi)):
        
    #     mi_idx = np.hstack([np.where(mi_names[i] == label)[0] for label in reference_labels])
    #     mi_sorted.append(mi[i][mi_idx] / mi[i][mi_idx].max(axis = 0))
        
    #     pi_idx = np.hstack([np.where(pi_names[i] == label)[0] for label in reference_labels])
    #     pi_sorted[:, i] = pi[i][pi_idx] / pi[i][pi_idx].max()
    
    months_list = np.hstack(pi_months)          
    #months_list = np.hstack([x.split('/')[-2] for x in output_files])
    mi_monthly_mean = np.zeros((len(reference_labels), len(plots.MONTHS)))
    mi_monthly_std = np.zeros((len(reference_labels), len(plots.MONTHS)))
    pi_monthly_mean = np.zeros((len(reference_labels), len(plots.MONTHS)))
    pi_monthly_std = np.zeros((len(reference_labels), len(plots.MONTHS)))
    for m in range(len(plots.MONTHS)):
        mask_month = months_list == plots.MONTHS[m]
        if mask_month.sum() > 0:
            mi_monthly_mean[:, m] = mi_sorted[:, mask_month].mean(axis = 1)
            mi_monthly_std[:, m] = mi_sorted[:, mask_month].std(axis = 1)
            pi_monthly_mean[:, m] = pi_sorted[:, mask_month].mean(axis = 1)
            pi_monthly_std[:, m] = pi_sorted[:, mask_month].std(axis = 1)
        else:
            mi_monthly_mean[:, m] = np.repeat(np.nan, len(reference_labels))
            mi_monthly_std[:, m] = np.repeat(np.nan, len(reference_labels))
            pi_monthly_mean[:, m] = np.repeat(np.nan, len(reference_labels))
            pi_monthly_std[:, m] = np.repeat(np.nan, len(reference_labels))
    
   
    importances = {'model': 
                   {'norm': mi_sorted, 
                    'monthly_mean': mi_monthly_mean,
                    'mean': np.nanmean(mi_monthly_mean, axis = 1), 
                    'monthly_std': mi_monthly_std,
                    'std': np.nanmean(mi_monthly_std, axis = 1)}, 
                   'permutation': 
                       {'norm': pi_sorted, 
                        'monthly_mean': pi_monthly_mean, 
                        'mean': np.nanmean(pi_monthly_mean, axis = 1),
                        'monthly_std': pi_monthly_std,
                        'std': np.nanmean(pi_monthly_std, axis = 1)},
                   'months_list': months_list,
                   'labels': tick_labels,
                   'reference': reference_labels}
    
    return importances







#--------- load CONTRIBUTIONS
def load_contributions(summaries_dir, reference_labels):
    
    models = np.hstack(glob.glob(f'{summaries_dir}/*/test.contributions.mean.nc'))
    months_list_sort = np.hstack([x.split('/')[-2] for x in models]) 
    # sort_idx = np.hstack([np.where(months_list_sort == x) for x in MONTHS])[0]
    # models = models[sort_idx]
    
    #read in contibutions and take absolute value
    cont_abs_mean = []
    cont_names = []
    for m in range(len(plots.MONTHS)):
        mask_month = months_list_sort == plots.MONTHS[m]
        if mask_month.sum() > 0:
            data = xr.open_dataset(models[mask_month][0])
            #mean of the absolute value
            temp = np.nanmean(np.abs((data.to_array().values)), axis = (1,2))
            cont_abs_mean.append(temp)
            #cont_abs_mean.append(np.abs((data.to_array().values)).mean(axis = (1,2)))
            cont_names.append(np.hstack(list(data.keys())))
        else:
            cont_abs_mean.append(np.repeat(np.nan, len(reference_labels)))
            cont_names.append(np.repeat(np.nan, len(reference_labels)))
    #cont_names = np.hstack(cont_names).astype(str)
    
    #process contributions
    cont_norm = [a / np.nanmax(a) for a in cont_abs_mean]
    cont_sorted = np.zeros((len(reference_labels), len(cont_norm)))
    for i in range(len(cont_norm)):
        cont_idx = np.hstack([np.where(cont_names[i] == label)[0] for label in reference_labels])
        if len(cont_idx) > 0:
            cont_sorted[:, i] = cont_norm[i][cont_idx] / cont_norm[i][cont_idx].max()
        else:
            cont_sorted[:, i] = np.nan 
    
    # cont_idx = [np.hstack([np.where(names == label)[0] for label in reference_labels]) 
    #           for names in cont_names]
    # cont_sorted = [a[b] / a[b].max() for a, b in zip(cont_norm, cont_idx)]
    # cont_sorted = np.column_stack(cont_sorted)
    
    contributions = {'norm': cont_sorted, 
                     'mean': np.nanmean(cont_sorted, axis = 1),
                     'std': np.nanstd(cont_sorted, axis = 1)}
    
    return contributions




#--------- load predictions and truth
def load_predictions(summaries_dir):
    ml_files = glob.glob(f'{summaries_dir}/*/data.h5')
    
    output = {'pred': [], 'truth': [], 'lon': [], 'lat': [], 'years': [], 'days': []}
    for m in range(len(ml_files)):
        with closing(h5py.File(ml_files[m], 'r')) as f:
            output['truth'].append(f['y'][:])
            output['pred'].append(f['y0'][:])
            output['lon'].append(f['lons'][:])
            output['lat'].append(f['lats'][:])
            output['years'].append(f['years'][:])
            output['days'].append(f['days'][:])
    
    return output

    

#--------- load predictions and truth
# def load_predictions(summaries_dir):
#     pred_files = glob.glob(f'{summaries_dir}/*/test.predict.nc')
#     true_files = glob.glob(f'{summaries_dir}/*/test.target.nc')
    
#     months_pred = [x.split('/')[-2] for x in pred_files]
#     idx = [np.where(np.hstack(months_pred) == x)[0] for x in plots.MONTHS]
#     pred_files = [pred_files[x[0]] if len(x) > 0 else [] for x in idx]
#     true_files = [true_files[x[0]] if len(x) > 0 else [] for x in idx]
    
#     output = {'pred': [], 'truth': [], 'lon': [], 'lat': [], 'years': [], 'days': []}
#     for m in tqdm(range(len(plots.MONTHS)), desc = 'Loading predicts/truth'):
#         is_month = (np.hstack(months_pred) == plots.MONTHS[m]).sum()
#         if is_month > 0:
#             data_pred = xr.open_dataset(pred_files[m])
#             dsp = data_pred.to_array().stack({'loc': ["lon", "lat", 'time']})
#             data_truth = xr.open_dataset(true_files[m])
#             dst = data_truth.to_array().stack({'loc': ["lon", "lat", 'time']})
            
#             pred = dsp.values[0]
#             truth = dst.values[0]
#             mask_truth = np.isnan(truth)
#             mask_pred = np.isnan(pred)
#             mask = mask_truth | mask_pred
        
#             output['pred'].append(pred[~mask])
#             output['truth'].append(truth[~mask])
            
#             output['years'].append(dst['time.year'][~mask])
#             output['days'].append(dst['time.day'].values[~mask])
#             output['lon'].append(dst['lon'].values[~mask])
#             output['lat'].append(dst['lat'].values[~mask])
                  
#         else:
#             for k in output.keys():
#                 output[k].append([])
                
   
            
#     return output




