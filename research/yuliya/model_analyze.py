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
from scipy.io import netcdf
from tqdm import tqdm
from contextlib import closing
from datetime import datetime, timedelta, date
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import scipy as sp
from scipy import stats
import pywt, copy
import dtaidistance as dt
from sklearn.cluster import AgglomerativeClustering 
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import adjusted_rand_score
from astropy.convolution import Gaussian1DKernel, convolve
from joblib import load
from utils import TRAIN_FEATURES
from utils import REQUIRED_VARS
import pickle

#-----------------read in data
root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
if not os.path.exists(root_dir):
    root_dir = '/data/MLIA_active_data/data_SUDSAQ/'

model_dir = f'{root_dir}/model/bias-modeling/rf-temporal-cv/'
ISD_FEATURES = ['PRCP/mean', 'TMAX/mean', 'TMIN/mean', 'TOBS/mean']




#-------------------------
#---------IMPORTANCE plots
months = [f'{x}'.zfill(2) for x in np.arange(1, 13)]
output_file = f'{model_dir}/models/importances.pickle' 
if os.path.isfile(output_file):
   with open(output_file, "rb") as f:
       importances = pickle.load(f)    
else:
    p = len(TRAIN_FEATURES)
    importances = {}
    models = glob.glob(model_dir + '/*/')
    months = [x.split('/')[-2] for x in models]
    for m in tqdm(range(len(models))):
        models_cv = glob.glob(f'{models[m]}/*/*.joblib', recursive=True)
        
        #importances[months[m]] = np.zeros((0,p))
        importances[months[m]] = []
        # Load Random Forest model
        for m0 in range(len(models_cv)):
            try:
                with open(models_cv[m0], 'rb') as f:
                    rf_model = load(f)
                    #rf_model = pickle.load(f)
            except:
                continue
            importance_norm = rf_model.feature_importances_ / np.max(rf_model.feature_importances_)
            importances[months[m]].append(importance_norm)
            #importances[months[m]] = np.row_stack([importances[months[m]], importance_norm])
            
        #importances[months[m]] = np.hstack(importances[months[m]]) 
    output_file = f'{model_dir}/importances.pickle' 
    with open(output_file, "wb") as f:
        pickle.dump(importances, f)
 
    
#load all importances 
# output_file = f'{model_dir}/importances.pickle' 
# with open(output_file, "rb") as f:
#     importances = pickle.load(f)    

p = len(TRAIN_FEATURES)
boxes = np.zeros((0,p))
by_month = []
for k in importances.keys():
    n = len(importances[k])
    temp = importances[k].reshape(int(n/p), p)
    boxes = np.row_stack([boxes, temp])
    by_month.append(temp.mean(axis =0))
    
#boxplot
feature_box = [x for x in boxes.T]
plt.figure()
plt.boxplot(feature_box);
plt.xticks(np.arange(1, p+1), TRAIN_FEATURES, rotation = 45);
plt.grid(ls=':', alpha = 0.5)
plt.title(f'feature importance distributions per month+cv')
plt.savefig(f'{model_dir}/plots-v2/importance_bar.png',
             bbox_inches='tight')
plt.close()

#heatmap 
plt.figure()
plt.pcolor(np.row_stack(by_month))
plt.yticks(np.arange(len(months))+0.5, months)
plt.ylabel('month')
plt.xticks(np.arange(0, p)+0.5, TRAIN_FEATURES, rotation = 45)
plt.xlabel('feature')
plt.colorbar()
plt.tight_layout()
plt.savefig(f'{model_dir}/plots-v2/importance_heat.png',
               bbox_inches='tight')
plt.close()     
 
        

#-------------------------
#---------CONTRIBUTION plots
pred_dir = f'{model_dir}/preds/'

TOTAL_FEATURES = TRAIN_FEATURES + ISD_FEATURES
p = len(TOTAL_FEATURES)

output_file = f'{model_dir}/models/contributions.pickle' 
if os.path.isfile(output_file):
   with open(output_file, "rb") as f:
       contributions = pickle.load(f)
   models = glob.glob(pred_dir + '/*/')
   with closing(h5py.File(models_cv[0], 'r')) as f:
         lon = f['lon'][:]
         lat = f['lat'][:]
         date = f['date'][:]
else:
    contributions = {}
    bias = {}
    models = glob.glob(pred_dir + '/*/')
    months = [x.split('/')[-2] for x in models]
    for m in tqdm(range(len(models))):
        models_cv = glob.glob(f'{models[m]}/*/*.h5', recursive=True)
        
        bias[months[m]] = []
        contributions[months[m]] = []
        for m0 in range(len(models_cv)):
            with closing(h5py.File(models_cv[m0], 'r')) as f:
                contrib_cv = f['contribution'][:]
                bias_cv = f['true_bias'][:]
                lon = f['lon'][:]
                lat = f['lat'][:]
                date = f['date'][:]
            contributions[months[m]].append(contrib_cv)
            bias[months[m]].append(bias_cv) 
    
    with open(output_file, "wb") as f:
        pickle.dump(contributions, f)    


#barplot with error bars
boxes = np.zeros((0,p))
by_month_std = []
by_month = []
for m in range(len(months)): 
    vals = np.row_stack(contributions[months[m]])
    boxes = np.row_stack([boxes, vals])
    #stat_cont = np.nanpercentile(vals, [99], axis = 0)
    std_cont = np.nanstd(vals, axis = 0)
    stat_cont = np.nanmean(np.abs(vals), axis = 0)
    by_month.append(stat_cont)
    by_month_std.append(std_cont)
    

#----boxplots
feature_box = [x for x in boxes.T]
plt.figure()
plt.boxplot(feature_box, flierprops={'marker': 'x', 'markersize': 0.5, 
                              'markerfacecolor': '0.5', 'markeredgecolor': '0.5',
                              'alpha': 0.7});
plt.xticks(np.arange(1, p+1), TRAIN_FEATURES, rotation = 45);
plt.grid(ls=':', alpha = 0.5)
plt.axhline(y=0, color = 'r', ls=':')
plt.title(f'contributions distributions, full year, all locations')
plt.savefig(f'{model_dir}/plots-v2/contributions_boxplots.png',
             bbox_inches='tight')
plt.close()

#----heatmap
plt.figure()
plt.barh(TRAIN_FEATURES, np.row_stack(by_month).mean(axis=0), 
         xerr = np.row_stack(by_month).mean(axis=0))
plt.title(f'Contributors to bias, full year, |mean|')
plt.ylabel('Features')
plt.xlabel('contribution (ppb)')
plt.grid(linestyle='dotted')
plt.title(f'absolute contributions, overall mean and std')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{model_dir}/plots-v2/contributions_mean_barplot.png',
            bbox_inches='tight')
plt.close()    

#----barplot with errors
plt.figure()
plt.pcolor(np.row_stack(by_month))
plt.yticks(np.arange(len(months))+0.5, months)
plt.ylabel('month')
plt.xticks(np.arange(0, p)+0.5, TRAIN_FEATURES, rotation = 45)
plt.xlabel('feature')
plt.colorbar()
plt.title(f'average contributions, all locations by month')
plt.tight_layout()
plt.savefig(f'{model_dir}/plots-v2/contributions_mean_heat.png',
               bbox_inches='tight')
plt.close()     
 


#total contribution over the whole year
for P in range(p):
    #toar_mask = np.isnan(np.array(bias))[0,:]
    monthly_cont = []
    for m in range(len(months)):
        for i in range(len(contributions[months[m]])):
            toar_mask = np.isnan(np.array(bias[months[m]][i]))
            N = contributions[months[m]][i].shape[0]
            T = int(N/(62*80))
            cont_day = contributions[months[m]][i][:,P].reshape(T, 62*80)
            cont_space = cont_day.reshape(T, 62, 80).copy()
            cont_space[toar_mask] = np.nan
            monthly_cont.append(np.nanmean(cont_space, axis = 0))
    
    stat_funcs = {'mean': np.nanmean, 'std': np.nanstd}
    for k in stat_funcs.keys():
        total_cont = stat_funcs[k](np.dstack(monthly_cont), axis = 2)
        x, y = np.meshgrid(lon, lat)
        fig, ax = plt.subplots(figsize=(18, 9),
                                   subplot_kw={'projection': ccrs.PlateCarree()})
        plt.pcolor(x, y, total_cont, cmap='coolwarm')
        if k == 'mean':
            plt.clim((-2, 2))
        else:
            plt.clim((0, 4))
        plt.colorbar()
        ax.coastlines()
        ax.stock_img()
        ax.set_extent([-140, -50, 10, 80], crs=ccrs.PlateCarree())  # NA region
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        plt.title(f'Contribution variability, {TRAIN_FEATURES[P]}')
        plt.savefig(f'{model_dir}/plots-v2/cont_{k}_{TRAIN_FEATURES[P]}.png',
                    bbox_inches='tight')
        plt.close()



# m = 0
# #N = contributions[months[m]][0].shape[0]
# stat_funcs = {'mean': np.nanmean, 'std': np.nanstd}
# for P in range(p):
#     cont_space_stat = []
#     for i in range(len(contributions[months[m]])):
#         N = contributions[months[m]][i].shape[0]
#         T = int(N/4960)
#         cont_day = contributions[months[m]][i][:,P].reshape(T, 4960)
#         cont_space = cont_day.reshape(T, 62, 80)
#         cont_space_stat.append(np.nanmean(cont_space, axis = 0))
    
#     for k in stat_funcs.keys():    
#         cont_std = stat_funcs[k](np.dstack(cont_space_stat), axis = 2)
        
#         x, y = np.meshgrid(lon, lat)
#         fig, ax = plt.subplots(figsize=(18, 9),
#                                    subplot_kw={'projection': ccrs.PlateCarree()})
#         plt.pcolor(x, y, cont_std, cmap='coolwarm')
#         if k == 'mean':
#             plt.clim((-8, 8))
#         else:
#             plt.clim((0, 4))
#         plt.colorbar()
#         ax.coastlines()
#         ax.stock_img()
#         ax.set_extent([-140, -50, 10, 80], crs=ccrs.PlateCarree())  # NA region
#         gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                           linewidth=1, color='gray', alpha=0.5, linestyle='--')
#         gl.xformatter = LONGITUDE_FORMATTER
#         gl.yformatter = LATITUDE_FORMATTER
#         plt.title(f'Contribution variability, {TOTAL_FEATURES[P]}, {months[m]}')
#         plt.savefig(f'{model_dir}/plots/cont_{k}_{P}_{months[m]}.png',
#                     bbox_inches='tight')
#         plt.close()








