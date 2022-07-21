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
from config_all import REQUIRED_VARS
import pickle
import scipy.cluster.hierarchy as sch
from sklearn.metrics import pairwise_distances
import seaborn

#-----------------read in data
root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
if not os.path.exists(root_dir):
    root_dir = '/data/MLIA_active_data/data_SUDSAQ/'

model_dir = f'{root_dir}/model/bias-modeling/rf-temporal-cv-nc/'
#ISD_FEATURES = ['PRCP/mean', 'TMAX/mean', 'TMIN/mean', 'TOBS/mean']


bbox_dict = {'globe':[0, 360, -90, 90],
              'europe': [-20+360, 40+360, 25, 80],
              'asia': [110+360, 160+360, 10, 70],
              'australia': [130+360, 170+360, -50, -10],
              'north_america': [-140+360, -50+360, 10, 80]}




#-------------------------
#---------IMPORTANCE plots
months = [f'{x}'.zfill(2) for x in np.arange(1, 13)]
output_file = f'{model_dir}/models/importances.pickle' 
if os.path.isfile(output_file):
   with open(output_file, "rb") as f:
       importances = pickle.load(f)    
else:
    print(f'run extract_importances script.py')
  



full_labels = np.hstack([x.split('/')[-1] for x in REQUIRED_VARS])
months = np.sort(np.hstack(importances).astype(int))
p = importances['1'][0].shape[0]
#p = len(TRAIN_FEATURES)
boxes = np.zeros((0, p))
by_month = []
for k in months:
    mi = str(k)
    n = len(importances[mi])
    temp = np.column_stack(importances[mi])
    #temp = importances[k].reshape(int(n/p), p)
    boxes = np.row_stack([boxes, temp.T])
    by_month.append(np.median(temp, axis = 1))
by_month = np.row_stack(by_month).T


#-----------boxplot
feature_box = [x for x in boxes.T]
box_mask = np.median(boxes.T, axis = 1) > 0.2
colors = np.repeat('k', len(feature_box))
colors[box_mask] = 'red'

plt.figure(figsize = (20, 5))
bx = plt.boxplot(feature_box, flierprops=dict(color='0.5', markeredgecolor='0.5'));
for patch, color in zip(bx['boxes'], colors):
    patch.set_color(color)
plt.xticks(np.arange(1, p+1), full_labels, rotation = 90, color = 'k'); 
[plt.gca().get_xticklabels()[x].set_color("red") for x in np.where(box_mask)[0]]
plt.grid(ls=':', alpha = 0.5)
plt.title(f'feature importance distributions per month+cv')
plt.tight_layout()
plt.savefig(f'{model_dir}/plots/importance_boxplots.png',
             bbox_inches='tight')
plt.close()


#---------heatmap
D = pairwise_distances(by_month)
H = sch.linkage(D, method='average')
d1 = sch.dendrogram(H, no_plot=True)
idx = d1['leaves']
X = by_month[idx,:]

a = 0.3
mask = X.max(axis = 1) < a
x_labels = np.hstack([x.split('/')[-1] for x in REQUIRED_VARS])[idx]
x_labels[mask] = ''

#full 
plt.figure(figsize = (20, 5))
plt.pcolor(X.T)
plt.yticks(np.arange(len(months))+0.5, months)
plt.ylabel('month')
plt.xticks(np.arange(0, p)+0.5, full_labels[idx], rotation = 90)
plt.xlabel('feature')
plt.colorbar()
plt.tight_layout()
plt.title('full set of driver importances, median')
plt.savefig(f'{model_dir}/plots/importance_heat_full.png',
               bbox_inches='tight')
plt.close() 

#partial    
plt.figure(figsize = (20, 5))
plt.pcolor(X[~mask,:].T)
plt.clim((0,1))
plt.yticks(np.arange(len(months))+0.5, months)
plt.ylabel('month')
plt.xticks(np.arange(0, (~mask).sum())+0.5, x_labels[~mask], rotation = 90)
plt.xlabel('feature')
plt.colorbar()
plt.tight_layout()
plt.title(f'subset of drivers with max importances > {a}')
plt.savefig(f'{model_dir}/plots/importance_heat_subset.png',
               bbox_inches='tight')
plt.close() 

 
        

#-------------------------
#---------CONTRIBUTION plots
pred_dir = f'{model_dir}/preds/'
output_file = f'{model_dir}/models/contributions.h5' 
models = glob.glob(pred_dir + '/*/*.h5')
with closing(h5py.File(models[0], 'r')) as f:
      lon = f['lon'][:]
      lat = f['lat'][:]

full_labels = np.hstack([x.split('/')[-1] for x in REQUIRED_VARS])

 #-------spatial maps
for k in range(len(full_labels)):
    name = full_labels[k]
    with closing(h5py.File(output_file, 'r')) as f:
        keys = list(f)
        contributions = []
        for m in keys:
            contributions.append(np.hstack(f[m][:, :, k]).reshape(160, 320))
            #mask = ~np.isnan(contributions)      
    

    #-------spatial maps
    vals = np.nanmean(np.dstack(contributions), axis = 2)
    x, y = np.meshgrid(lon, lat, indexing='xy')
    
    regions = list(bbox_dict)
    #globe and zoom in regions
    for r in ['north_america']:
        fig, ax = plt.subplots(figsize=(18, 9),
                                   subplot_kw={'projection': ccrs.PlateCarree()})
        plt.pcolor(x,y, vals, cmap='coolwarm')
        plt.clim((-1, 1))
        plt.colorbar()
        ax.coastlines()
        ax.stock_img()
        ax.set_extent(bbox_dict[r], crs=ccrs.PlateCarree())  # NA region
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        plt.tight_layout()
        plt.title(f'mean over all months, {name}')
        plt.savefig(f'{model_dir}/plots/contributions/{r}/contrib_map_{r}_{name}.png',
                    bbox_inches='tight')
        plt.close()


months = np.sort(np.hstack(keys).astype(int))  

#-------boxplots
with closing(h5py.File(output_file, 'r')) as f:
    keys = list(f)
    contributions = [[]] * len(full_labels)
    mean_array = np.zeros((len(keys), len(full_labels)))
    std_array = np.zeros((len(keys), len(full_labels)))
    for m in range(len(keys)):
        for k in range(len(full_labels)):
            vals = f[keys[m]][:, :, k]
            vals = vals[~np.isnan(vals)]
            #stat = np.mean(np.abs(vals))
            contributions[k] = np.hstack([contributions[k], vals])
            mean_array[m, k] = np.median(vals)
            std_array[m, k] = np.std(vals)



#----boxplots per driver
#feature_box = [x for x in boxes.T]
plt.figure(figsize = (20, 5))
plt.boxplot(contributions, flierprops={'marker': 'x', 'markersize': 0.5, 
                              'markerfacecolor': '0.5', 'markeredgecolor': '0.5',
                              'alpha': 0.7});
plt.xticks(np.arange(1, p+1), full_labels, rotation = 90);
plt.ylim((-5, 5))
plt.grid(ls=':', alpha = 0.5)
plt.axhline(y=0, color = 'r', ls=':')
plt.title(f'contributions distributions, full year, all locations')
plt.tight_layout()
plt.savefig(f'{model_dir}/plots/contributions_boxplots.png',
             bbox_inches='tight')
plt.close()


std_idx = np.argsort([np.std(x) for x in contributions])[::-1] 
a = 20
#----boxplots per driver
#feature_box = [x for x in boxes.T]
plt.figure(figsize = ((20/80)*20, 5))
plt.boxplot([contributions[x] for x in std_idx[:a]], flierprops={'marker': 'x', 'markersize': 0.8, 
                              'markerfacecolor': '0.5', 'markeredgecolor': '0.5',
                              'alpha': 0.7});
plt.xticks(np.arange(1, p+1)[:a], full_labels[std_idx[:a]], rotation = 90);
plt.ylim((-15, 15))
plt.grid(ls=':', alpha = 0.5)
plt.axhline(y=0, color = 'r', ls=':')
plt.title(f'contributions distributions, top 20 std , \n full year, all locations')
plt.tight_layout()
plt.savefig(f'{model_dir}/plots/contributions_boxplots_sorted.png',
             bbox_inches='tight')
plt.close()


stat_name = ['median', 'std']
stat_clim = [(-0.2, 0.2), (0, 3)]
for i, array in enumerate([mean_array, std_array]):
    D = pairwise_distances(array.T)
    H = sch.linkage(D, method='average')
    d1 = sch.dendrogram(H, no_plot=True)
    idx = d1['leaves']
    X = array.T[idx,:]
    
    mask = np.argsort(np.abs(X).sum(axis = 1))[::-1][:20]
    x_labels = np.hstack([x.split('/')[-1] for x in REQUIRED_VARS])[idx]
    #x_labels[mask] = ''
    
    plt.figure(figsize = (20, 5))
    plt.pcolor(X.T)
    plt.yticks(np.arange(len(months))+0.5, months)
    plt.ylabel('month')
    plt.xticks(np.arange(0, p)+0.5, full_labels[idx], rotation = 90)
    plt.xlabel('feature')
    plt.clim(stat_clim[i])
    plt.colorbar()
    plt.tight_layout()
    plt.title('full set of driver importances, {stat_name[i]}')
    plt.savefig(f'{model_dir}/plots/contributions_heat_full_{stat_name[i]}.png',
                   bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize = (0.5*20, 5))
    plt.pcolor(X[mask,:].T)
    plt.clim(stat_clim[i])
    plt.yticks(np.arange(len(months))+0.5, months)
    plt.ylabel('month')
    plt.xticks(np.arange(0, len(mask))+0.5, x_labels[mask], rotation = 90)
    plt.xlabel('feature')
    plt.colorbar()
    plt.tight_layout()
    plt.title('subset of drivers, with greatest cont sum')
    plt.savefig(f'{model_dir}/plots/contributions_heat_subset_{stat_name[i]}.png',
                   bbox_inches='tight')
    plt.close() 




#--------- RESIDUAL plots
pred_dir = f'{model_dir}/preds/'
models = glob.glob(pred_dir + '/*/*.h5')
months = np.sort(np.unique([x.split('/')[-2] for x in models]).astype(int))
with closing(h5py.File(models[0], 'r')) as f:
      lon = f['lon'][:]
      lat = f['lat'][:]
      
#-------spatial maps
bias_m = []
for m in months.astype(str):
    models_m = glob.glob(f'{pred_dir}/{m}/*.h5')
    bias = []
    for z in range(len(models_m)): 
        with closing(h5py.File(models_m[z], 'r')) as f:
            y_pred = f['prediction'][:]
            y_true = f['truth'][:]
            bias.append(y_true - y_pred)
        
    bias_m.append(np.row_stack(bias))    
        
      
rmse = []
boxes = [] 
no_outliers = []       
for m in range(len(bias_m)):
    vals = bias_m[m]
    vals = vals[~np.isnan(vals)]
    rmse.append(np.sqrt(np.mean(vals**2)))
    boxes.append(vals)
    q1, q2 = np.percentile(vals, [1, 99])
    mask = (vals > q1) & (vals < q2)
    no_outliers.append(vals[mask])

cmap = plt.get_cmap('coolwarm', 11)
colors = cmap(np.linspace(0,1, 11))
sidx = np.argsort(rmse)
clist = [[]] * len(colors)
for i in range(len(sidx)):
    clist[sidx[i]] = colors[i] 

plt.figure()
seaborn.violinplot(data=no_outliers, bw=.2, palette = clist)
plt.grid(ls=':', alpha = 0.5)
plt.axhline(y=0, color = 'r', ls=':')
plt.xticks(np.arange(0, len(months)), months);
plt.title(f'residuals by month (true - predicted)')
plt.savefig(f'{model_dir}/plots/residuals_violin_all.png',
               bbox_inches='tight')
plt.close() 

# plt.figure()
# plt.violinplot(boxes, showmedians=True, showmeans = True, showextrema=False);





#----heatmap
# plt.figure()
# plt.barh(REQUIRED_VARS, np.row_stack(by_month).mean(axis=0), 
#          xerr = np.row_stack(by_month).mean(axis=0))
# plt.title(f'Contributors to bias, full year, |mean|')
# plt.ylabel('Features')
# plt.xlabel('contribution (ppb)')
# plt.grid(linestyle='dotted')
# plt.title(f'absolute contributions, overall mean and std')
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.savefig(f'{model_dir}/plots-v2/contributions_mean_barplot.png',
#             bbox_inches='tight')
# plt.close()    

# #----barplot with errors
# plt.figure()
# plt.pcolor(np.row_stack(by_month))
# plt.yticks(np.arange(len(months))+0.5, months)
# plt.ylabel('month')
# plt.xticks(np.arange(0, p)+0.5, TRAIN_FEATURES, rotation = 45)
# plt.xlabel('feature')
# plt.colorbar()
# plt.title(f'average contributions, all locations by month')
# plt.tight_layout()
# plt.savefig(f'{model_dir}/plots-v2/contributions_mean_heat.png',
#                bbox_inches='tight')
# plt.close()     
 




#barplot with error bars
# p = contributions.shape[2]
# boxes = []
# for m in range(p): 
#     vals = contributions[:,:,m]
#     vals = vals[~np.isnan(vals)]
#     boxes.append(vals)
#     #stat_cont = np.nanpercentile(vals, [99], axis = 0)
#     # std_cont = np.nanstd(vals, axis = 0)
#     #stat_cont = np.nanmean(np.abs(vals), axis = 0)
#     # by_month.append(stat_cont)
#     # by_month_std.append(std_cont)

# #total contribution over the whole year
# for P in range(p):
#     #toar_mask = np.isnan(np.array(bias))[0,:]
#     monthly_cont = []
#     for m in range(len(months)):
#         for i in range(len(contributions[months[m]])):
#             toar_mask = np.isnan(np.array(bias[months[m]][i]))
#             N = contributions[months[m]][i].shape[0]
#             T = int(N/(62*80))
#             cont_day = contributions[months[m]][i][:,P].reshape(T, 62*80)
#             cont_space = cont_day.reshape(T, 62, 80).copy()
#             cont_space[toar_mask] = np.nan
#             monthly_cont.append(np.nanmean(cont_space, axis = 0))
    
#     stat_funcs = {'mean': np.nanmean, 'std': np.nanstd}
#     for k in stat_funcs.keys():
#         total_cont = stat_funcs[k](np.dstack(monthly_cont), axis = 2)
#         x, y = np.meshgrid(lon, lat)
#         fig, ax = plt.subplots(figsize=(18, 9),
#                                    subplot_kw={'projection': ccrs.PlateCarree()})
#         plt.pcolor(x, y, total_cont, cmap='coolwarm')
#         if k == 'mean':
#             plt.clim((-2, 2))
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
#         plt.title(f'Contribution variability, {TRAIN_FEATURES[P]}')
#         plt.savefig(f'{model_dir}/plots-v2/cont_{k}_{TRAIN_FEATURES[P]}.png',
#                     bbox_inches='tight')
#         plt.close()



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








