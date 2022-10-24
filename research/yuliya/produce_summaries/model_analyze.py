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
from scipy import stats

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
months = ['dec', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov']  

#set the directory for th60at month
models_dir = f'{root_dir}/models/2011-2015/bias-8hour/'
#set plot directory
summaries_dir = f'{root_dir}/summaries/2011-2015/bias-8hour/combined_data/'
plots_dir = f'{root_dir}/summaries/2011-2015/bias-8hour/summary_plots/'
# create one if it's not there
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    
  


#--------- LOAD importances
def load_importances(models_dir):
    output_files = glob.glob(f'{models_dir}/*/*/*/*importance.h5')
    
    mi = []
    pi = []
    mi_names = []
    pi_names = []
    pi_months = []
    for file in output_files:
        try:
            with closing(h5py.File(file, 'r')) as f:
                mi.append(f['model']['block0_values'][0,:])
                mi_names.append(f['model']['axis0'][:].astype(str))
                
                pi.append(f['permutation']['block0_values'][0, :])
                pi_names.append(f['permutation']['axis0'][:].astype(str))
                pi_months.append(file.split('/')[8])
        except:
            pi_months[-1] = np.nan
            continue
    
    months_list = np.hstack([x.split('/')[8] for x in output_files])
    mi_monthly_mean = []
    pi_monthly_mean = []
    for m in months:
        mask_month = months_list == m
        mi_monthly = [mi[a] for a in np.where(mask_month)[0]]
        mi_monthly_mean.append(np.column_stack(mi_monthly).mean(axis = 1))
        
        pi_monthly = [pi[a] for a in np.where(mask_month)[0]]
        pi_monthly_mean.append(np.column_stack(pi_monthly).mean(axis = 1))
     
    
    #sort everything to the first model importance order
    reference_labels = mi_names[0]
    #get labels and clean up names for plots
    tick_labels = np.hstack(['.'.join(x.split('.')[1:]) for x in reference_labels])
    tick_labels = np.hstack([x.split('2dsfc')[-1].split('.')[-1] for x in tick_labels])
    
    #sort by name to match
    mi_idx = [np.hstack([np.where(names == label)[0] for label in reference_labels]) 
              for names in mi_names]
    mi_sorted = np.column_stack([a[b] / a[b].max() for a, b in zip(mi_monthly_mean, mi_idx)])    
       
    pi_idx = [np.hstack([np.where(names == label)[0] for label in reference_labels]) 
              for names in pi_names]
    pi_sorted = np.column_stack([a[b] / a[b].max() for a, b in zip(pi_monthly_mean, pi_idx)])

    
    importances = {'model': 
                   {'norm': mi_sorted, 
                    'mean': mi_sorted.mean(axis = 1), 
                    'std': mi_sorted.std(axis = 1)}, 
                   'permutation': 
                       {'norm': pi_sorted, 
                        'mean': pi_sorted.mean(axis = 1), 
                        'std': pi_sorted.std(axis = 1)},
                   'labels': tick_labels,
                   'reference': reference_labels}
    
    return importances




#--------- load CONTRIBUTIONS
def load_contributions(summaries_dir, reference_labels):
    
    models = np.hstack(glob.glob(f'{summaries_dir}/*/test.contributions.mean.nc'))
    months_list_sort = np.hstack([x.split('/')[9] for x in models]) 
    sort_idx = np.hstack([np.where(months_list_sort == x) for x in months])[0]
    models = models[sort_idx]
    
    #read in contibutions and take absolute value
    cont_abs_mean = []
    cont_names = []
    for m in range(len(models)):
        data = xr.open_dataset(models[m])
        #mean of the absolute value
        cont_abs_mean.append(np.abs((data.to_array().values)).mean(axis = (1,2)))
        cont_names.append(np.hstack(list(data.keys())))
    
    #cont_names = np.hstack(cont_names).astype(str)
    
    #process contributions
    cont_norm = [a / a.max() for a in cont_abs_mean]
    cont_idx = [np.hstack([np.where(names == label)[0] for label in reference_labels]) 
              for names in cont_names]
    cont_sorted = [a[b] / a[b].max() for a, b in zip(cont_norm, cont_idx)]
    cont_sorted = np.column_stack(cont_sorted)
    
    contributions = {'norm': cont_sorted, 
                     'mean': cont_sorted.mean(axis = 1),
                     'std': cont_sorted.std(axis = 1)}
    
    return contributions


#get the mask for the top number a variables
def get_top_mask(a, var1, var2 = None, var3 = None):
    
    metrics_mean = np.column_stack([var1, var2, var3])
    mean_metrics = metrics_mean.mean(axis = 1)
    idx_sort = np.argsort(mean_metrics)[::-1]
    mask_top = idx_sort[:a]
    
    return mask_top
    


### ---------------- the simplest barplot and bubble plots for top X
importances = load_importances(models_dir)
contributions = load_contributions(summaries_dir, importances['reference'])


mask_top = get_top_mask(20, var1 = importances['model']['mean'],
                            var2 = importances['permutation']['mean'],
                            var3 = contributions['mean'])


plot_imp_barplot(labels = importances['labels'],
                 var1 = [importances['model']['mean'], importances['model']['std']],
                 var2 = [importances['permutation']['mean'], importances['permutation']['std']],
                 var3 = [contributions['mean'], contributions['std']], 
                 mask_top = mask_top)


plot_imp_bubble(labels = importances['labels'],
                var1 = importances['model']['norm'],
                var2 = importances['permutation']['norm'],
                var3 = contributions['norm'],
                mask_top = mask_top,
                plots_dir = None)


for metric in [importances['model']['norm'], importances['permutation']['norm'], contributions['norm']]:    
    plot_imp_fullbar(labels = importances['labels'],
                     var1 = metric,
                     key = 'model_importance',
                     mask_top = mask_top,
                     plots_dir = None) 


for metric in [importances['model']['norm'], importances['permutation']['norm'], contributions['norm']]: 
    plot_imp_bubble(labels = importances['labels'],
                    var1 = metric,
                    key = 'model_importance',
                    mask_top = None,
                    plots_dir = None)

   


#spatial plots contributions
 
m = 0

plot_cont_map(models[m], select_vars = ['momo.t'], plots_dir = None)

def plot_cont_map(model_dir, select_vars, plots_dir = None):
    #var = 'momo.t'
    #data = xr.open_dataset(models[m])
    data = xr.open_dataset(model_dir)
    #data_var = np.array(data[var])
    lon = data.lon
    lat = data.lat
    
    for var in select_vars:
        data_var = np.array(data[var])
        x, y = np.meshgrid(lon, lat, indexing='xy')
        fig, ax = plt.subplots(figsize=(10, 8),
                                   subplot_kw={'projection': ccrs.PlateCarree()})
        plt.pcolor(x,y, data_var, cmap='coolwarm')
        #plt.clim((-1, 1))
        ax.coastlines()
        ax.stock_img()
        ax.set_extent(bbox_dict['globe'], crs=ccrs.PlateCarree())  # NA region
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        im_ratio =320 / 160
        cb = plt.colorbar(fraction = 0.04 * im_ratio, pad = 0.1)
        plt.tight_layout()
        plt.title(f'mean contribution, {var}, month {months[m]}')
        if plots_dir is not None:
            plt.savefig(f'{plots_dir}/contributions/cont_map_{var}_{months[m]}.png',
                        bbox_inches='tight')
            plt.close()






# metrics_mean = np.column_stack([importances['model']['mean'], 
#                            importances['permutation']['mean'], 
#                            contributions['mean']])
# x = np.arange(0, len(importances['model']['mean'][mask_top])*6)[::6][::-1]


# ### --- 
# plt.figure(figsize = (7, len(x)*0.45))
# plt.barh(x, importances['model']['mean'][mask_top], alpha = 0.6, height = 1.2, color = 'blue', label = 'model')
# eb = plt.errorbar(importances['model']['mean'][mask_top], x, xerr=importances['model']['std'][mask_top], 
#                   fmt=".", color="blue", xlolims = True, alpha = 0.5)
# eb[1][0].set_marker('|')
# plt.barh(x+1, importances['permutation']['mean'][mask_top], alpha = 0.5, height = 1.2, color = 'orange', label = 'permutation')
# eb = plt.errorbar(importances['permutation']['mean'][mask_top], x+1, xerr=importances['permutation']['std'][mask_top], 
#                   fmt=".", color="orange", xlolims=True, alpha = 0.5)
# eb[1][0].set_marker('|')

# plt.barh(x+2, contributions['mean'][mask_top], alpha = 0.5, height = 1.2, color = 'green', label = 'contribution')
# eb = plt.errorbar(contributions['mean'][mask_top], x+2, xerr=contributions['std'][mask_top], 
#                   fmt=".", color="green", xlolims=True, alpha = 0.5)
# eb[1][0].set_marker('|')

# plt.yticks(x, importances['labels'][mask_top], rotation = 0, color = 'k');
# plt.grid(ls=':', alpha = 0.5)
# plt.legend()
# plt.title(f'top {a} mean importance with std \n all months')
# plt.tight_layout()
# plt.savefig(f'{plots_dir}/all_baplots_comb{a}_all.png',
#              bbox_inches='tight')
# plt.close()

    

### --- 
# metrics = np.dstack([importances['model']['norm'], 
#                            importances['permutation']['norm'], 
#                            contributions['norm']])

# mi_monthly_trunc = metrics.mean(axis = 2).T[:, mask_top]
# n = mi_monthly_trunc.shape[1]
# plt.figure(figsize = (a*0.35, 5))
# for z in range(len(mi_monthly_trunc)):
#     plt.scatter(x = np.arange(1, n+1), y = np.repeat(z, n), 
#                 s = mi_monthly_trunc[z,:]*200, c = mi_monthly_trunc[z,:], 
#                 cmap = 'Reds')
# plt.yticks(np.arange(len(months)), months)
# plt.ylabel('month')
# plt.xticks(np.arange(1, len(importances['labels'][mask_top])+1), 
#            importances['labels'][mask_top], rotation = 90)
# plt.xlabel('feature')
# plt.grid(ls=':', alpha = 0.5)
# plt.title(f'motnhly average importances')
# plt.colorbar()
# plt.tight_layout()
# plt.savefig(f'{plots_dir}/all_metrics_bubble_comb{a}.png',
#                bbox_inches='tight')
# plt.close() 







    



### ----------- full boxplots
### all model imp
# plt.figure(figsize = (20, 5))
# bx = plt.boxplot(list(importances['model']['norm']), flierprops=dict(color='0.5', 
#                 markersize = 2, markeredgecolor='0.5'));
# # for patch, color in zip(bx['boxes'], colors):
# #     patch.set_color(color)
# plt.axhline(y=0, ls='--', color = 'r', lw = 0.5)
# plt.xticks(np.arange(1, len(importances['labels'])+1), importances['labels'], rotation = 90, color = 'k'); 
# plt.ylim((-0.05,1.01))
# #[plt.gca().get_xticklabels()[x].set_color("red") for x in np.where(box_mask)[0]]
# plt.grid(ls=':', alpha = 0.5)
# plt.title(f'model importance distributions per month+cv, all months')
# plt.tight_layout()
# plt.savefig(f'{plots_dir}/importance_boxplots_all.png',
#              bbox_inches='tight')
# plt.close()


# ### all permutation imp
# plt.figure(figsize = (20, 5))
# bx = plt.boxplot(list(importances['permutation']['norm']), flierprops=dict(color='0.5', 
#             markersize = 2, markeredgecolor='0.5'));
# # for patch, color in zip(bx['boxes'], colors):
# #     patch.set_color(color)
# plt.axhline(y=0, ls='--', color = 'r', lw = 0.5)
# plt.xticks(np.arange(1, len(importances['labels'])+1), importances['labels'], rotation = 90, color = 'k'); 
# plt.ylim((-0.05,1.01))
# #[plt.gca().get_xticklabels()[x].set_color("red") for x in np.where(box_mask)[0]]
# plt.grid(ls=':', alpha = 0.5)
# plt.title(f'permutation importance distributions per month+cv, all months')
# plt.tight_layout()
# plt.savefig(f'{plots_dir}/permutation_boxplots_all.png',
#              bbox_inches='tight')
# plt.close()


# ### all permutation imp
# plt.figure(figsize = (20, 5))
# bx = plt.boxplot(list(contributions['norm']), flierprops=dict(color='0.5', 
#             markersize = 2, markeredgecolor='0.5'));
# # for patch, color in zip(bx['boxes'], colors):
# #     patch.set_color(color)
# plt.axhline(y=0, ls='--', color = 'r', lw = 0.5)
# plt.xticks(np.arange(1, len(importances['labels'])+1), importances['labels'], rotation = 90, color = 'k'); 

# plt.ylim((-0.05,1.01))
# #[plt.gca().get_xticklabels()[x].set_color("red") for x in np.where(box_mask)[0]]
# plt.grid(ls=':', alpha = 0.5)
# plt.title(f'contribution distributions per month+cv, all months')
# plt.tight_layout()
# plt.savefig(f'{plots_dir}/contribution_boxplots_all.png',
#              bbox_inches='tight')
# plt.close()




### ----------- full bubble plots
# mi_monthly_trunc = importances['model']['norm'].T
# n = mi_monthly_trunc.shape[1]
# plt.figure(figsize = (len(importances['labels'])*0.9, 5))
# for z in range(len(mi_monthly_trunc)):
#     plt.scatter(x = np.arange(1, n+1), y = np.repeat(z, n), 
#                 s = mi_monthly_trunc[z,:]*200, c = mi_monthly_trunc[z,:], 
#                 cmap = 'Reds')
# plt.yticks(np.arange(len(months)), months)
# plt.ylabel('month')
# plt.xticks(np.arange(1, len(importances['labels'])+1), importances['labels'], rotation = 90)
# plt.xlabel('feature')
# plt.grid(ls=':', alpha = 0.5)
# plt.colorbar()
# plt.tight_layout()
# plt.savefig(f'{plots_dir}/importance_bubble_allt.png',
#                bbox_inches='tight')
# plt.close() 


# mi_monthly_trunc = importances['permutation']['norm'].T
# n = mi_monthly_trunc.shape[1]
# plt.figure(figsize = (len(importances['labels'])*0.9, 5))
# for z in range(len(mi_monthly_trunc)):
#     plt.scatter(x = np.arange(1, n+1), y = np.repeat(z, n), 
#                 s = mi_monthly_trunc[z,:]*200, c = mi_monthly_trunc[z,:], 
#                 cmap = 'Reds')
# plt.yticks(np.arange(len(months)), months)
# plt.ylabel('month')
# plt.xticks(np.arange(1, len(importances['labels'])+1), importances['labels'], rotation = 90)
# plt.xlabel('feature')
# plt.grid(ls=':', alpha = 0.5)
# plt.colorbar()
# plt.tight_layout()
# plt.savefig(f'{plots_dir}/permutation_bubble_allt.png',
#                bbox_inches='tight')
# plt.close() 


# #bubble plots for contributions
# mi_monthly_trunc = contributions['norm'].T
# n = mi_monthly_trunc.shape[1]
# plt.figure(figsize = (len(importances['labels'])*0.5, 5))
# for z in range(len(mi_monthly_trunc)):
#     plt.scatter(x = np.arange(1, n+1), y = np.repeat(z, n), 
#                 s = mi_monthly_trunc[z,:]*200, c = mi_monthly_trunc[z,:], 
#                 cmap = 'Reds')
# plt.yticks(np.arange(len(months)), months)
# plt.ylabel('month')
# plt.xticks(np.arange(1, len(importances['labels'])+1), importances['labels'], rotation = 90)
# plt.xlabel('feature')
# plt.grid(ls=':', alpha = 0.5)
# plt.colorbar()
# plt.tight_layout()
# plt.savefig(f'{plots_dir}/contributions_bubble_allt.png',
#                bbox_inches='tight')
# plt.close() 
 
      









#---------heatmap
# mi_monthly = []
# for i in months:
#     mask = np.in1d(months_list, i)
#     mi_monthly.append(np.mean(mi_sorted[:,mask], axis = 1)) 
# mi_monthly = np.row_stack(mi_monthly)

# mask_nan = np.isnan(np.sum(mi_monthly, axis = 1))
# mask_row = np.nansum(mi_monthly, axis = 0) > 1e-2
# mi_monthly = mi_monthly[:, mask_row]

# # D = pairwise_distances(mi_monthly.T)
# # H = sch.linkage(D, method='average')
# # d1 = sch.dendrogram(H, no_plot=True)
# # idx = d1['leaves']
# # X = mi_monthly[:, idx]

# #biggest sum
# new_tick_labels = tick_labels[mask_row] 
# plt.figure(figsize = (20, 5))
# plt.pcolor(mi_monthly)
# plt.clim((0, 0.3))
# plt.yticks(np.arange(len(months))+0.5, months)
# plt.ylabel('month')
# plt.xticks(np.arange(0, len(new_tick_labels))+0.5, new_tick_labels, rotation = 90)
# plt.xlabel('feature')
# plt.colorbar()
# plt.tight_layout()
# plt.title(f'subset of largest driver model importances, mean')
# plt.savefig(f'{plots_dir}/importance_heatmap_all.png',
#                bbox_inches='tight')
# plt.close() 




#----boxplots per driver
#feature_box = [x for x in boxes.T]
# plt.figure(figsize = (20, 5))
# plt.boxplot(contributions, flierprops={'marker': 'x', 'markersize': 0.5, 
#                               'markerfacecolor': '0.5', 'markeredgecolor': '0.5',
#                               'alpha': 0.7});
# plt.xticks(np.arange(1, p+1), full_labels, rotation = 90);
# plt.ylim((-5, 5))
# plt.grid(ls=':', alpha = 0.5)
# plt.axhline(y=0, color = 'r', ls=':')
# plt.title(f'contributions distributions, full year, all locations')
# plt.tight_layout()
# plt.savefig(f'{model_dir}/plots/contributions_boxplots.png',
#              bbox_inches='tight')
# plt.close()


# std_idx = np.argsort([np.std(x) for x in contributions])[::-1] 
# a = 20
# #----boxplots per driver
# #feature_box = [x for x in boxes.T]
# plt.figure(figsize = ((20/80)*20, 5))
# plt.boxplot([contributions[x] for x in std_idx[:a]], flierprops={'marker': 'x', 'markersize': 0.8, 
#                               'markerfacecolor': '0.5', 'markeredgecolor': '0.5',
#                               'alpha': 0.7});
# plt.xticks(np.arange(1, p+1)[:a], full_labels[std_idx[:a]], rotation = 90);
# plt.ylim((-15, 15))
# plt.grid(ls=':', alpha = 0.5)
# plt.axhline(y=0, color = 'r', ls=':')
# plt.title(f'contributions distributions, top 20 std , \n full year, all locations')
# plt.tight_layout()
# plt.savefig(f'{model_dir}/plots/contributions_boxplots_sorted.png',
#              bbox_inches='tight')
# plt.close()


# stat_name = ['median', 'std']
# stat_clim = [(-0.2, 0.2), (0, 3)]
# for i, array in enumerate([mean_array, std_array]):
#     D = pairwise_distances(array.T)
#     H = sch.linkage(D, method='average')
#     d1 = sch.dendrogram(H, no_plot=True)
#     idx = d1['leaves']
#     X = array.T[idx,:]
    
#     mask = np.argsort(np.abs(X).sum(axis = 1))[::-1][:20]
#     x_labels = np.hstack([x.split('/')[-1] for x in REQUIRED_VARS])[idx]
#     #x_labels[mask] = ''
    
#     plt.figure(figsize = (20, 5))
#     plt.pcolor(X.T)
#     plt.yticks(np.arange(len(months))+0.5, months)
#     plt.ylabel('month')
#     plt.xticks(np.arange(0, p)+0.5, full_labels[idx], rotation = 90)
#     plt.xlabel('feature')
#     plt.clim(stat_clim[i])
#     plt.colorbar()
#     plt.tight_layout()
#     plt.title('full set of driver importances, {stat_name[i]}')
#     plt.savefig(f'{model_dir}/plots/contributions_heat_full_{stat_name[i]}.png',
#                    bbox_inches='tight')
#     plt.close()
    
#     plt.figure(figsize = (0.5*20, 5))
#     plt.pcolor(X[mask,:].T)
#     plt.clim(stat_clim[i])
#     plt.yticks(np.arange(len(months))+0.5, months)
#     plt.ylabel('month')
#     plt.xticks(np.arange(0, len(mask))+0.5, x_labels[mask], rotation = 90)
#     plt.xlabel('feature')
#     plt.colorbar()
#     plt.tight_layout()
#     plt.title('subset of drivers, with greatest cont sum')
#     plt.savefig(f'{model_dir}/plots/contributions_heat_subset_{stat_name[i]}.png',
#                    bbox_inches='tight')
#     plt.close() 






#--------- RESIDUAL plots
#months = ['dec', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov']
#cont_dir = f'{root_dir}/model/new/model_data/' 
pred_files = glob.glob(f'{summaries_dir}/*/test.predict.nc')
true_files = glob.glob(f'{summaries_dir}/*/test.target.nc')

months_pred = [x.split('/')[9] for x in pred_files]
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

# month = 'jul'
# pred_files = glob.glob(f'{models_dir}/{month}/rf/*/test.predict.nc')
# data_pred = xr.open_mfdataset(pred_files[7])
# data_pred = data_pred.to_array().stack({'loc': ["lon", "lat", 'time']})
# data_truth = xr.open_dataset(true_files[7])
# data_truth = data_truth.to_array().stack({'loc': ["lon", "lat", 'time']})
# pred = data_pred.values[0]
# truth = data_truth.values[0]
# mask = ~np.isnan(truth)

# np.sqrt(mean_squared_error(truth[mask], pred[mask]))

# plt.figure()
# plt.hist(truth, bins = 100, histtype='step', density = True, color = '0.5');
# plt.hist(pred[mask], bins = 100, histtype='step', density = True);
# plt.hist(pred[~mask], bins = 100, histtype='step', density = True);



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

rmse_total = np.sqrt(mean_squared_error(np.hstack(truth_list), 
                                        np.hstack(pred_list)))    
    
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



### residuals plot
fig, ax = plt.subplots(2, 6, figsize = (6*3, 2*3))
for m in range(len(months)):
    # H, xedges, yedges = np.histogram2d(pred_list[m], truth_list[m], bins = 200)
    # H[H == 0] = np.nan
    kernel  = stats.gaussian_kde([np.hstack(truth_list[m]), np.hstack(pred_list[m])])
    density = kernel([np.hstack(truth_list[m]), np.hstack(pred_list[m])])
    
    #plt.figure()
    plt.subplot(2, 6,m+1)
    plt.scatter(pred_list[m], truth_list[m], s = 3, c=density, alpha = 0.5, cmap = 'coolwarm')
    #plt.pcolor(xedges, yedges, H, cmap = 'coolwarm')
    plt.axvline(x=0, color = '0.5', alpha = 0.5, ls = '--')
    plt.axhline(y=0, color = '0.5', alpha = 0.5, ls = '--')
    plt.plot()
    plt.ylim((-80, 80))
    plt.xlim((-80, 80))
    plt.grid(ls = ':')
    plt.xlabel(f'predicted ppb')
    plt.ylabel(f'true ppb')
    #plt.colorbar()
    #plt.contour(H, levels = 5, cmap = 'coolwarm')
    plt.plot([-80,80], [-80,80], color = 'r', alpha = 0.2, ls = '--')
    plt.text(0.1, 0.9, f'{months[m]}({np.round(rmse[m], 1)})', 
             bbox=dict(facecolor='none', edgecolor='k'),
             transform=plt.gca().transAxes)
plt.suptitle(f'true vs predicted bias, per month')
plt.savefig(f'{plots_dir}/residuals_monthly.png',
             bbox_inches='tight')
plt.close()


### ------ all
# H, xedges, yedges = np.histogram2d(np.hstack(pred_list), 
#                                    np.hstack(truth_list), bins = 500)
# H[H == 0] = np.nan

kernel  = stats.gaussian_kde([np.hstack(truth_list), np.hstack(pred_list)])
density = kernel([np.hstack(truth_list), np.hstack(pred_list)])
plt.figure(figsize = (12,10))
plt.scatter(np.hstack(pred_list), np.hstack(truth_list), s = 3, c=density, alpha = 0.5, cmap = 'coolwarm')
#plt.pcolor(xedges, yedges, H, cmap = 'coolwarm')
plt.axvline(x=0, color = '0.5', alpha = 0.5, ls = '--')
plt.axhline(y=0, color = '0.5', alpha = 0.5, ls = '--')
plt.plot()
plt.ylim((-100, 100))
plt.xlim((-100, 100))
plt.grid(ls = ':')
plt.xlabel(f'predicted ppb')
plt.ylabel(f'true ppb')
plt.colorbar()
#plt.contour(H, levels = 5, cmap = 'coolwarm')
plt.plot([-100,100], [-100,100], color = 'r', alpha = 0.2, ls = '--')
plt.title(f'predicted vs true bias, all months, rmse total = {np.round(rmse_total, 2)}')
plt.savefig(f'{plots_dir}/residuals_all.png',
             bbox_inches='tight')
plt.close()




### ------ all
fig, ax = plt.subplots(2, 6, figsize = (6*3, 2*3))
for m in range(len(months)):
    
    #plt.figure()
    plt.subplot(2, 6,m+1)
    plt.hist(np.hstack(pred_list[m]), bins = 300, density = True, 
             histtype = 'step', label = f'predicted');
    plt.axvline(x = np.hstack(pred_list[m]).mean(), alpha = 0.5)
    plt.hist(np.hstack(truth_list[m]), bins = 300, density = True, 
             histtype = 'step', label = f'true');
    plt.axvline(x = np.hstack(truth_list[m]).mean(), color = 'orange',
                alpha = 0.5)
    plt.xlim((-50, 50))
    plt.legend(fontsize = 6)
    plt.grid(ls = ':')
    plt.xlabel(f'ppb')
    plt.text(0.1, 0.9, f'{months[m]}({np.round(rmse[m], 1)})', bbox=dict(facecolor='none', edgecolor='k'),
             transform=plt.gca().transAxes)
plt.suptitle(f'true vs predicted bias histograms, per month')
plt.savefig(f'{plots_dir}/residuals_hist_monthly.png',
             bbox_inches='tight')
plt.close()


plt.figure()
plt.hist(np.hstack(pred_list), bins = 300, density = True, 
         histtype = 'step', label = f'predicted');
plt.axvline(x = np.hstack(pred_list).mean(), alpha = 0.5)
plt.hist(np.hstack(truth_list), bins = 300, density = True, 
         histtype = 'step', label = f'true');
plt.axvline(x = np.hstack(truth_list).mean(), color = 'orange',
            alpha = 0.5)
plt.xlim((-50, 50))
plt.legend()
plt.grid(ls = ':')
plt.xlabel(f'ppb')
plt.title(f'predicted vs true bias, all months, rmse total = {np.round(rmse_total, 2)}')
plt.savefig(f'{plots_dir}/residuals_all.png',
             bbox_inches='tight')
plt.close()


plt.figure()
plt.hist(pred_list[7]);
plt.hist(truth_list[7]);


# plt.figure()
# bx = plt.boxplot(truth_list, flierprops=dict(color='0.5', 
#                 markersize = 2, markeredgecolor='0.5'));  
# plt.grid(ls=':', alpha = 0.5)
# plt.xticks(np.arange(1, len(months)+1), months)
# plt.title(f'momo - toar true values')
# plt.tight_layout()
# plt.savefig(f'{plots_dir}/target_boxlots_all.png',
#              bbox_inches='tight')
# plt.close()

# plt.figure()
# bx = plt.boxplot(truth_list, showfliers = False, showmeans = True);  
# plt.grid(ls=':', alpha = 0.5)
# plt.xticks(np.arange(1, len(months)+1), months)
# plt.title(f'momo - toar true values')
# plt.tight_layout()
# plt.savefig(f'{plots_dir}/target_boxlots_nofliers_all.png',
#              bbox_inches='tight')
# plt.close()



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




# m = 10
# plt.figure()
# plt.hist(truth_list[m], bins = 100, histtype = 'step', density = True);
# plt.hist(pred_list[m], bins = 100, histtype = 'step', density = True);
# plt.title(f'{months[m]}')


# cmap = plt.get_cmap('coolwarm', 11)
# colors = cmap(np.linspace(0,1, 12))
# sidx = np.argsort(rmse)
# clist = [[]] * len(colors)
# for i in range(len(sidx)):
#     clist[sidx[i]] = colors[i] 
# plt.figure()
# seaborn.violinplot(data=boxes, bw=.2, palette = clist)
# plt.grid(ls=':', alpha = 0.5)
# plt.axhline(y=0, color = 'r', ls=':')    
# plt.xticks(np.arange(0, len(months)), months);
# plt.title(f'residuals by month (true - predicted)')
# plt.savefig(f'{plots_dir}/residuals_violin_all.png',
#             bbox_inches='tight')
# plt.close()           




# cmap = plt.get_cmap('coolwarm', 11)
# colors = cmap(np.linspace(0,1, 11))
# sidx = np.argsort(rmse)
# clist = [[]] * len(colors)
# for i in range(len(sidx)):
#     clist[sidx[i]] = colors[i] 

# plt.figure()
# seaborn.violinplot(data=no_outliers, bw=.2, palette = clist)
# plt.grid(ls=':', alpha = 0.5)
# plt.axhline(y=0, color = 'r', ls=':')
# plt.xticks(np.arange(0, len(months)), months);
# plt.title(f'residuals by month (true - predicted)')
# plt.savefig(f'{model_dir}/plots/residuals_violin_all.png',
#                bbox_inches='tight')
# plt.close() 


   

      
#-------spatial maps
# bias_m = []
# for m in months.astype(str):
#     models_m = glob.glob(f'{pred_dir}/{m}/*.h5')
#     bias = []
#     for z in range(len(models_m)): 
#         with closing(h5py.File(models_m[z], 'r')) as f:
#             y_pred = f['prediction'][:]
#             y_true = f['truth'][:]
#             bias.append(y_true - y_pred)
        
#     bias_m.append(np.row_stack(bias))    
    
    


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








