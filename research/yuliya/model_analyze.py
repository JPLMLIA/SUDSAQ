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

#-----------------read in data
root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
if not os.path.exists(root_dir):
    root_dir = '/data/MLIA_active_data/data_SUDSAQ/'

#everything is in -180 to 180 lon
bbox_dict = {'globe':[-180, 180, -90, 90],
            'europe': [-20, 40, 25, 80],
            'asia': [110, 160, 10, 70],
            'australia': [130, 170, -50, -10],
            'north_america': [-140, -50, 10, 80],
            'west_europe': [-20, 10, 25, 80],
            'east_europe': [10, 40, 25, 80],
            'west_na': [-140, -95, 10, 80],
            'east_na': [-95, -50, 10, 80], }


#choose parameters
region = 'globe'
bbox = bbox_dict[region]
month = 'jan'
years = [2011, 2012, 2013, 2014, 2015]

#set the directory for th60at month
models_dir = f'{root_dir}/model/new/model_data/{month}/combined/'
#set plot directory
plots_dir = f'{root_dir}/model/new/summary_plots/'
# create one if it's not there
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    
months = ['dec', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov']    
    

#-------------------------
#---------IMPORTANCE plots


#--------- LOAD importances
import_dir = f'{root_dir}/models/2011-2015/bias/'
output_files = glob.glob(f'{import_dir}/*/*/*/*importance.h5')
#output_files = glob.glob(f'{import_dir}/{month}/*/*/*importance.h5')

mi = []
pi = []
m_names = []
p_names = []
pi_months = []
for file in output_files:
    try:
        with closing(h5py.File(file, 'r')) as f:
            mi.append(f['model']['block0_values'][0,:])
            m_names.append(f['model']['axis0'][:].astype(str))
            pi.append(f['permutation']['block0_values'][0, :])
            p_names.append(f['permutation']['axis0'][:].astype(str))
            pi_months.append(file.split('/')[8])
    except:
        pi_months[-1] = np.nan
        continue

months_list = [x.split('/')[8] for x in output_files]    
mi = np.column_stack(mi)        
pi = np.column_stack(pi)
m_names =  np.column_stack(m_names)
p_names = np.column_stack(p_names)

m_labels = m_names[:,0]
#tick_labels = np.hstack([x.split('.')[-1] for x in m_labels])
tick_labels = np.hstack(['.'.join(x.split('.')[1:]) for x in m_labels])
mi_sorted = mi.copy()
pi_sorted = pi.copy()
for m in range(mi.shape[1]):
    m_idx = np.hstack([np.where(m_labels == x)[0] for x in m_names[:,m]])
    mi_sorted[:, m] = mi[m_idx, m] / mi[m_idx, m].max()

for m in range(pi.shape[1]):    
    p_idx = np.hstack([np.where(m_labels == x)[0] for x in p_names[:,m]])
    pi_sorted[:, m] = pi[p_idx, m] / pi[p_idx, m].max()
    

### all
p = len(m_labels)
mi_box = [x for x in mi_sorted]
colors = np.repeat('k', len(mi_box))
box_mask = np.median(mi, axis = 1) > 0.2

pi_box = [x for x in pi_sorted]







#--------- load CONTRIBUTIONS
pred_dir = f'{models_dir}/preds/'
cont_dir = f'{root_dir}/model/new/model_data/' 
models = np.hstack(glob.glob(f'{cont_dir}/*/*/test.contributions.mean.nc'))
months_list_sort = np.hstack([x.split('/')[8] for x in models]) 
sort_idx = [np.where(months_list_sort == x)[0][0] for x in months]
models = models[sort_idx]

cont_data = []
for m in range(len(models)):
    data = xr.open_dataset(models[m])
    data_mean = data.mean()
    var_names = list(data_mean.keys()).astype(str)
    cont_data.append(data_mean.to_array().values)

var_names = np.hstack(var_names).astype(str)

cont = np.abs(np.column_stack(cont_data))
cont_norm = cont / np.max(cont, axis = 0, keepdims=True)
cont_mean = np.mean(cont_norm, axis = 1) 

sort_idx = np.argsort(cont_mean)[::-1]
cont_box = [x for x in np.abs(cont_norm)[sort_idx, :]]
labels = np.hstack(var_names)[sort_idx]
labels = np.hstack(['.'.join(x.split('.')[1:]) for x in labels])

idx = np.hstack([np.where(tick_labels == x)[0] for x in labels])
cont_box = [cont_box[x] for x in idx]


var_sort_idx = np.hstack([np.where(var_names == x)[0] for x in m_labels])
cont_sorted = cont[var_sort_idx, :]

#--------- load SHAP
#XXX




# the simplest overlay
mi_mean_box = np.hstack([np.mean(x) for x in mi_box])
mi_std_box = np.hstack([np.std(x) for x in mi_box]) 
pi_mean_box = np.hstack([np.mean(x) for x in pi_box])
pi_std_box = np.hstack([np.std(x) for x in pi_box])
cont_mean_box = np.hstack([np.mean(x) for x in cont_box])
cont_std_box = np.hstack([np.std(x) for x in cont_box])


a = None
alpha = 75
if alpha is not None:
    q_mi = np.percentile(mi_mean_box, alpha)
    q_pi = np.percentile(pi_mean_box, alpha)
    mask_top = (mi_mean_box > q_mi) | (pi_mean_box > q_pi)
if a is not None:
    mask_top = np.repeat(False, len(mi_mean_box))
    mask_top[:a] = True
    

x = np.arange(0, len(mi_mean_box[mask_top])*6)[::6]

plt.figure(figsize = (7, len(x)*0.45))
plt.barh(x, mi_mean_box[mask_top], alpha = 0.6, height = 1.2, color = 'blue', label = 'model')
eb = plt.errorbar(mi_mean_box[mask_top], x, xerr=mi_std_box[mask_top], fmt=".", color="blue", 
                  xlolims = True, alpha = 0.5)
eb[1][0].set_marker('|')
plt.barh(x+1, pi_mean_box[mask_top], alpha = 0.5, height = 1.2, color = 'orange', label = 'permutation')
eb = plt.errorbar(pi_mean_box[mask_top], x+1, xerr=pi_std_box[mask_top], fmt=".", color="orange", 
                  xlolims=True, alpha = 0.5)
eb[1][0].set_marker('|')

plt.barh(x+2, cont_mean_box[mask_top], alpha = 0.5, height = 1.2, color = 'green', label = 'contribution')
eb = plt.errorbar(cont_mean_box[mask_top], x+2, xerr=cont_std_box[mask_top], fmt=".", color="green", 
                  xlolims=True, alpha = 0.5)
eb[1][0].set_marker('|')

plt.yticks(x, tick_labels[mask_top], rotation = 0, color = 'k');
plt.grid(ls=':', alpha = 0.5)
plt.legend()
plt.title(f'top {mask_top.sum()} mean importance with std \n all months')
plt.tight_layout()
plt.savefig(f'{plots_dir}/importance_baplots_comb{mask_top.sum()}_all3.png',
             bbox_inches='tight')
plt.close()









### all model imp
plt.figure(figsize = (20, 5))
bx = plt.boxplot(mi_box, flierprops=dict(color='0.5', markersize = 2, markeredgecolor='0.5'));
for patch, color in zip(bx['boxes'], colors):
    patch.set_color(color)
plt.axhline(y=0, ls='--', color = 'r', lw = 0.5)
plt.xticks(np.arange(1, p+1), tick_labels, rotation = 90, color = 'k'); 
plt.ylim((-0.05,1.01))
#[plt.gca().get_xticklabels()[x].set_color("red") for x in np.where(box_mask)[0]]
plt.grid(ls=':', alpha = 0.5)
plt.title(f'model importance distributions per month+cv, all months')
plt.tight_layout()
plt.savefig(f'{plots_dir}/importance_boxplots_all.png',
             bbox_inches='tight')
plt.close()


### all permutation imp
plt.figure(figsize = (20, 5))
bx = plt.boxplot(pi_box, flierprops=dict(color='0.5', markersize = 2, markeredgecolor='0.5'));
for patch, color in zip(bx['boxes'], colors):
    patch.set_color(color)
plt.axhline(y=0, ls='--', color = 'r', lw = 0.5)
plt.xticks(np.arange(1, p+1), tick_labels, rotation = 90, color = 'k'); 
plt.ylim((-0.05,1.01))
#[plt.gca().get_xticklabels()[x].set_color("red") for x in np.where(box_mask)[0]]
plt.grid(ls=':', alpha = 0.5)
plt.title(f'permutation importance distributions per month+cv, all months')
plt.tight_layout()
plt.savefig(f'{plots_dir}/permutation_boxplots_all.png',
             bbox_inches='tight')
plt.close()




### top 
a = 20
feature_box_trunc = [x for x in mi_sorted[:a]]
colors = np.repeat('k', len(feature_box_trunc))
x = np.arange(0, a*3)[::3]
y = np.arange(1, a*3)[::3]


fig, ax = plt.subplots(figsize = (0.6*a, 5))
bx1 = ax.boxplot(feature_box_trunc, positions = x,
                 flierprops=dict(color='0.5', markersize = 2, 
                                 markeredgecolor='0.5'));
for patch, color in zip(bx['boxes'], colors):
    patch.set_color(color)
plt.axhline(y=0, ls='--', color = 'r', lw = 0.5)
plt.xticks(x, m_labels[:a], rotation = 90, color = 'k'); 
#[plt.gca().get_xticklabels()[x].set_color("red") for x in np.where(box_mask)[0]]


bx2 = ax.boxplot([x for x in pi_sorted[:a]], positions = y,
                 flierprops=dict(color='0.5', markersize = 2, markeredgecolor='0.5'));
for patch, color in zip(bx2['boxes'], colors):
    patch.set_color('blue')
plt.xticks(x, m_labels[:a], rotation = 90, color = 'k'); 
plt.grid(ls=':', alpha = 0.5)
ax.legend([bx1["boxes"][0], bx2["boxes"][0]], ['model', 'permutation'], loc='upper right')
plt.title(f'top {a} importance distributions per month+cv, all months')
plt.tight_layout()
plt.savefig(f'{plots_dir}/importance_boxplots_comb{a}_all.png',
             bbox_inches='tight')
plt.close()









#---------heatmap
months = ['dec', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov']
mi_monthly = []
for i in months:
    mask = np.in1d(months_list, i)
    mi_monthly.append(np.mean(mi_sorted[:,mask], axis = 1)) 
mi_monthly = np.row_stack(mi_monthly)

mask_nan = np.isnan(np.sum(mi_monthly, axis = 1))
mask_row = np.nansum(mi_monthly, axis = 0) > 1e-2
mi_monthly = mi_monthly[:, mask_row]

# D = pairwise_distances(mi_monthly.T)
# H = sch.linkage(D, method='average')
# d1 = sch.dendrogram(H, no_plot=True)
# idx = d1['leaves']
# X = mi_monthly[:, idx]

#biggest sum
new_tick_labels = tick_labels[mask_row] 
plt.figure(figsize = (20, 5))
plt.pcolor(mi_monthly)
plt.clim((0, 0.3))
plt.yticks(np.arange(len(months))+0.5, months)
plt.ylabel('month')
plt.xticks(np.arange(0, len(new_tick_labels))+0.5, new_tick_labels, rotation = 90)
plt.xlabel('feature')
plt.colorbar()
plt.tight_layout()
plt.title(f'subset of largest driver model importances, mean')
plt.savefig(f'{plots_dir}/importance_heatmap_all.png',
               bbox_inches='tight')
plt.close() 


a = 20
mi_monthly_trunc = mi_monthly[:, :a]
n = mi_monthly_trunc.shape[1]
plt.figure(figsize = (10, 5))
for z in range(len(mi_monthly_trunc)):
    plt.scatter(x = np.arange(1, n+1), y = np.repeat(z, n), 
                s = mi_monthly_trunc[z,:]*200, c = mi_monthly_trunc[z,:], 
                cmap = 'Reds')
plt.yticks(np.arange(len(months)), months)
plt.ylabel('month')
plt.xticks(np.arange(1, len(new_tick_labels[:a])+1), new_tick_labels[:a], rotation = 90)
plt.xlabel('feature')
plt.grid(ls=':', alpha = 0.5)
plt.colorbar()
plt.tight_layout()
plt.savefig(f'{plots_dir}/mi_bubble_subset.png',
               bbox_inches='tight')
plt.close() 





pi_monthly = []
for i in months:
    mask = np.in1d(months_list, i)
    pi_monthly.append(np.mean(pi_sorted[:,mask], axis = 1)) 
pi_monthly = np.row_stack(pi_monthly)

mask_nan = np.isnan(np.sum(pi_monthly, axis = 1))
mask_row = np.nansum(pi_monthly, axis = 0) > 1e-2
pi_monthly = pi_monthly[:, mask_row]

# D = pairwise_distances(mi_monthly.T)
# H = sch.linkage(D, method='average')
# d1 = sch.dendrogram(H, no_plot=True)
# idx = d1['leaves']
# X = mi_monthly[:, idx]

#biggest sum
new_tick_labels = tick_labels[mask_row] 
plt.figure(figsize = (20, 5))
plt.pcolor(pi_monthly)
plt.clim((0, 0.3))
plt.yticks(np.arange(len(months))+0.5, months)
plt.ylabel('month')
plt.xticks(np.arange(0, len(new_tick_labels))+0.5, new_tick_labels, rotation = 90)
plt.xlabel('feature')
plt.colorbar()
plt.tight_layout()
plt.title(f'subset of largest driver permutation importances, mean')
plt.savefig(f'{plots_dir}/permutation_heatmap_all.png',
               bbox_inches='tight')
plt.close() 




#bubble plots for contributions
a = 17
mi_monthly_trunc = cont_sorted.T[:, :a]
n = mi_monthly_trunc.shape[1]
plt.figure(figsize = (8, 5))
for z in range(len(mi_monthly_trunc)):
    plt.scatter(x = np.arange(1, n+1), y = np.repeat(z, n), 
                s = mi_monthly_trunc[z,:]*200, c = mi_monthly_trunc[z,:], 
                cmap = 'Reds')
plt.yticks(np.arange(len(months)), months)
plt.ylabel('month')
plt.xticks(np.arange(1, len(new_tick_labels[:a])+1), new_tick_labels[:a], rotation = 90)
plt.xlabel('feature')
plt.grid(ls=':', alpha = 0.5)
plt.colorbar()
plt.tight_layout()
plt.savefig(f'{plots_dir}/cont_bubble_subset.png',
               bbox_inches='tight')
plt.close() 
 
      
 
    

#spatial plots contributions
 
m = 7
var = 'momo.t'
data = xr.open_dataset(models[m])
data_var = np.array(data[var])
lon = data.lon
lat = data.lat
x, y = np.meshgrid(lon, lat, indexing='xy')

fig, ax = plt.subplots(figsize=(10, 8),
                           subplot_kw={'projection': ccrs.PlateCarree()})
plt.pcolor(x,y, data_var, cmap='coolwarm')
#plt.clim((-1, 1))
plt.colorbar()
ax.coastlines()
ax.stock_img()
ax.set_extent(bbox_dict['globe'], crs=ccrs.PlateCarree())  # NA region
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
plt.tight_layout()
plt.title(f'mean contribution, {var}, month {months[m]}')
plt.savefig(f'{model_dir}/plots/contributions/{r}/cont_map_{var}_{months[m]}.png',
            bbox_inches='tight')
plt.close()



#-------------------------
#---------CONTRIBUTION plots
pred_dir = f'{models_dir}/preds/'
cont_dir = f'{root_dir}/model/new/model_data/' 
models = glob.glob(f'{cont_dir}/*/*/test.contributions.mean.nc')

cont_data = []
for m in range(len(models)):
    data = xr.open_dataset(models[m])
    data_mean = data.mean()
    var_names = list(data_mean.keys())
    cont_data.append(data_mean.to_array().values)

cont = np.abs(np.column_stack(cont_data))
cont_norm = cont / np.max(cont, axis = 0, keepdims=True)
cont_mean = np.mean(cont_norm, axis = 1)

sort_idx = np.argsort(cont_mean)[::-1]
feature_box = [x for x in np.abs(cont_norm)[sort_idx, :]]
labels = np.hstack(var_names)[sort_idx]
labels = np.hstack(['.'.join(x.split('.')[1:]) for x in labels])

idx = np.hstack([np.where(tick_labels == x)[0] for x in labels])
feature_box = [feature_box[x] for x in idx]


#----boxplots per driver
#feature_box = [x for x in boxes.T]
plt.figure(figsize = (20, 5))
plt.boxplot(feature_box, flierprops={'marker': 'x', 'markersize': 0.5, 
                              'markerfacecolor': '0.5', 'markeredgecolor': '0.5',
                              'alpha': 0.7});
plt.xticks(np.arange(1, len(labels)+1), tick_labels, rotation = 90);
#plt.ylim((-5, 5))
plt.grid(ls=':', alpha = 0.5)
plt.axhline(y=0, color = 'r', ls=':')
plt.title(f'absolute contributions distributions, full year, all locations')
plt.tight_layout()
plt.savefig(f'{plots_dir}/contributions_boxplots_all.png',
             bbox_inches='tight')
plt.close()




# the simplest overlay: adding CONTRIBUTIONS
cmean_box = list(np.hstack([np.mean(x) for x in feature_box])[:a])
cstd_box = list(np.hstack([np.std(x) for x in feature_box])[:a])

plt.figure()
plt.bar(x, mmean_box, alpha = 0.6, width = 1.2, color = 'blue', label = 'model')
eb = plt.errorbar(x, mmean_box, yerr=mstd_box, fmt=".", color="blue", lolims = True, 
             alpha = 0.5)
eb[1][0].set_marker('_')
plt.bar(x, pmean_box, alpha = 0.5, width = 1.2, color = 'orange', label = 'permutation')
eb = plt.errorbar(x, pmean_box, yerr=pstd_box, fmt=".", color="orange", lolims=True,
             alpha = 0.5)
eb[1][0].set_marker('_')

plt.bar(x, cmean_box, alpha = 0.5, width = 1.2, color = 'green', label = 'contribution')
eb = plt.errorbar(x, cmean_box, yerr=cstd_box, fmt=".", color="green", lolims=True,
             alpha = 0.5)
eb[1][0].set_marker('_')

plt.xticks(x, tick_labels[:a], rotation = 90, color = 'k');
plt.grid(ls=':', alpha = 0.5)
plt.legend()
plt.title(f'top {a} mean importances with std, all months')
plt.tight_layout()
plt.savefig(f'{plots_dir}/importance_baplots_comb{a}_all2.png',
             bbox_inches='tight')
plt.close()




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
months = ['dec', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov']
cont_dir = f'{root_dir}/model/new/model_data/' 
pred_files = glob.glob(f'{cont_dir}/*/*/test.predict.nc')
true_files = glob.glob(f'{cont_dir}/*/*/test.target.nc')

months_pred = [x.split('/')[8] for x in pred_files]
idx = [np.where(np.hstack(months_pred) == x)[0][0] for x in months]
pred_files = [pred_files[x] for x in idx]
true_files = [true_files[x] for x in idx]

pred_list = []
truth_list = []
for m in range(len(pred_files)):
    data_pred = xr.open_dataset(pred_files[m])
    data_truth = xr.open_dataset(true_files[m])
    #data_mean = data.mean(dim = 'time')
    
    varname = list(data_pred.variables)[-1]
    pred = data_pred[varname].values
    truth = data_truth['target'].values
    mask_truth = np.isnan(truth)
    mask_pred = np.isnan(pred)
    mask = mask_truth | mask_pred
    
    pred_list.append(pred[~mask])
    truth_list.append(truth[~mask])



# plt.figure()
# plt.plot(pred_list[0], truth_list[0], '.', ms = 1)
# plt.xlabel('predicted value')
# plt.ylabel('true value')
# plt.grid(ls=':', alpha = 0.5) 
# plt.contour()

boxes = []
rmse = []
nrmse = []
mape = []
mae = []
for m in range(len(pred_list)):
    diff = truth_list[m] - pred_list[m]
    boxes.append(diff)
    
    error = np.sqrt(mean_squared_error(truth_list[m], pred_list[m]))
    #q = np.percentile(truth_list[m], [25, 75])
    nrmse.append(error / np.std(truth_list[m]))
    mae.append(mean_absolute_error(truth_list[m], pred_list[m]))
    rmse.append(error)
    #mean_absolute_percentage_error(truth_list[m], pred_list[m])
    
    
#rmse vs mae
plt.figure()
plt.plot(np.arange(1, len(rmse)+1), rmse, 'x', color = 'blue', label = 'rmse');
plt.plot(np.arange(1, len(rmse)+1), rmse, ':', alpha = 0.5, color = 'blue');
plt.plot(np.arange(1, len(mae)+1), mae, 'x', color = 'green', label = 'mae');
plt.plot(np.arange(1, len(mae)+1), mae, ':', alpha = 0.5, color = 'green');
plt.xticks(np.arange(1, len(months)+1), months)
plt.grid(ls=':', alpha = 0.5)
plt.ylabel('ppb')
plt.title(f'model rmse/mae')
plt.legend(frameon = True)
plt.savefig(f'{plots_dir}/rmse_mae_all.png', bbox_inches='tight')
plt.close()


plt.figure()
plt.plot(pred_list[m], boxes[m], '.', ms=0.8)
plt.grid(ls=':', alpha = 0.5)
plt.ylabel('residual')
plt.xlabel(f'predicted value')
plt.title(f'')
plt.savefig(f'{plots_dir}/rmse_all.png',
             bbox_inches='tight')
plt.close()




plt.figure()
bx = plt.boxplot(truth_list, flierprops=dict(color='0.5', 
                markersize = 2, markeredgecolor='0.5'));  
plt.grid(ls=':', alpha = 0.5)
plt.xticks(np.arange(1, len(months)+1), months)
plt.title(f'momo - toar true values')
plt.tight_layout()
plt.savefig(f'{plots_dir}/target_boxlots_all.png',
             bbox_inches='tight')
plt.close()

plt.figure()
bx = plt.boxplot(truth_list, showfliers = False, showmeans = True);  
plt.grid(ls=':', alpha = 0.5)
plt.xticks(np.arange(1, len(months)+1), months)
plt.title(f'momo - toar true values')
plt.tight_layout()
plt.savefig(f'{plots_dir}/target_boxlots_nofliers_all.png',
             bbox_inches='tight')
plt.close()



#residuals with errors
q = [np.percentile(x, [25, 75]) for x in boxes]
whisker_low = [q1 - (q3 - q1) * 1.5 for q1, q3 in q]
whisker_high = [q3 + (q3 - q1) * 1.5 for q1, q3 in q]
q1 = np.array(q)[:,0]
q3 = np.array(q)[:,1]

inds = range(1, len(boxes)+1)
fig, ax1 = plt.subplots()
bx = plt.violinplot(boxes, showmedians =True, showmeans = True,
                    showextrema=False);  
# ax.vlines(inds, np.array(q)[:,0], np.array(q)[:,1], 
#            color='b', linestyle='-', lw=5)
ax1.vlines(inds, whisker_low, whisker_high, color='b', linestyle=':', lw=1)
plt.ylim((-40, 40))
plt.grid(ls=':', alpha = 0.5)
plt.xticks(np.arange(1, len(months)+1), months)
plt.ylabel('bias residuals (ppb)')

ax2 = ax1.twinx()
#ax2.bar(np.arange(1, len(rmse)+1), rmse, width = 0.3, alpha = 0.5)
ax2.plot(np.arange(1, len(rmse)+1), rmse, 'x', color = 'blue', alpha = 0.5, label = 'rmse')
ax2.plot(np.arange(1, len(mae)+1), mae, 'x', color = 'green', alpha = 0.5, label = 'mae')
plt.ylim((-40, 40))
plt.legend()
ax2.tick_params(colors = 'blue')
plt.ylabel('rmse (ppb)')
plt.title(f'model residuals and root mean squared error (rmse)')
plt.tight_layout()
plt.savefig(f'{plots_dir}/residuals_rmse_violin_all.png',
             bbox_inches='tight')
plt.close()





#true bias with errors
alpha = 1.5
q = [np.percentile(x, [25, 75]) for x in truth_list]
whisker_low = [q1 - (q3 - q1) * alpha for q1, q3 in q]
whisker_high = [q3 + (q3 - q1) * alpha for q1, q3 in q]
q1 = np.array(q)[:,0]
q3 = np.array(q)[:,1]

inds = range(1, len(boxes)+1)
fig, ax1 = plt.subplots()
bx = plt.violinplot(truth_list, showmedians =True, showmeans = True,
                    showextrema=False);  
# ax.vlines(inds, np.array(q)[:,0], np.array(q)[:,1], 
#            color='b', linestyle='-', lw=5)
ax1.vlines(inds, whisker_low, whisker_high, color='b', linestyle=':', lw=1)
plt.grid(ls=':', alpha = 0.5)
plt.xticks(np.arange(1, len(months)+1), months)
plt.ylabel('bias (ppb)')
plt.ylim((-30, 40))
plt.ylabel('rmse (ppb)')
plt.title(f'true bias (momo - toar)')
plt.tight_layout()
plt.savefig(f'{plots_dir}/true_bias_violin_all.png', bbox_inches='tight')
plt.close()




m = 10
plt.figure()
plt.hist(truth_list[m], bins = 100, histtype = 'step', density = True);
plt.hist(pred_list[m], bins = 100, histtype = 'step', density = True);
plt.title(f'{months[m]}')


cmap = plt.get_cmap('coolwarm', 11)
colors = cmap(np.linspace(0,1, 12))
sidx = np.argsort(rmse)
clist = [[]] * len(colors)
for i in range(len(sidx)):
    clist[sidx[i]] = colors[i] 
plt.figure()
seaborn.violinplot(data=boxes, bw=.2, palette = clist)
plt.grid(ls=':', alpha = 0.5)
plt.axhline(y=0, color = 'r', ls=':')    
plt.xticks(np.arange(0, len(months)), months);
plt.title(f'residuals by month (true - predicted)')
plt.savefig(f'{plots_dir}/residuals_violin_all.png',
            bbox_inches='tight')
plt.close()           




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








