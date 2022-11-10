#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 17:19:54 2022

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
from scipy import stats
from sklearn.metrics import mean_squared_error
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


MONTHS = ['dec', 'jan', 'feb', 'mar', 
          'apr', 'may', 'jun', 'jul', 
          'aug', 'sep', 'oct', 'nov'] 

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



#------- create a simple bar plot combing all importance metrics
def imp_barplotv(labels, var1, var2 = None, var3 = None, mask_top = None, 
                     plots_dir = None):
    
    if mask_top is None:
        mask_top = np.arange(len(var1[0]))
    
    a = len(mask_top)
    x = np.arange(0, len(var1[0][mask_top])*6)[::6]
    plt.figure(figsize = (len(x)*0.45, 7))
    plt.bar(x, var1[0][mask_top], alpha = 0.8, width = 1.2, color = 'blue', label = 'model')
    eb = plt.errorbar(x, var1[0][mask_top], yerr=var1[1][mask_top], 
                      fmt="none", color="blue", lolims = True, alpha = 0.2)
    eb[1][0].set_marker('_')
    eb[1][0].set_markersize(2)
    
    if var2 is not None:
        plt.bar(x+1, var2[0][mask_top], alpha = 0.8, width = 1.2, color = 'orange', label = 'permutation')
        eb = plt.errorbar(x+1, var2[0][mask_top], yerr=var2[1][mask_top], 
                          fmt="none", color="orange", lolims = True, alpha = 0.2)
        eb[1][0].set_marker('_')
        eb[1][0].set_markersize(2)

    if var3 is not None:
        plt.bar(x+2, var3[0][mask_top], alpha = 0.8, width = 1.2, color = 'green', label = 'contribution')
        eb = plt.errorbar(x+2, var3[0][mask_top], yerr=var3[1][mask_top], 
                          fmt="none", color="green", lolims = True, alpha = 0.2)
        eb[1][0].set_marker('_')
        eb[1][0].set_markersize(2)

    plt.xticks(x, labels[mask_top], rotation = 90, color = 'k');
    plt.grid(ls=':', alpha = 0.5)
    plt.legend()
    plt.title(f'top {a} mean importance with std \n all months')
    plt.tight_layout()
    if plots_dir is not None:
        plt.savefig(f'{plots_dir}/baplot_comb{a}_all.png',
                     bbox_inches='tight')
        plt.close()





#create a seasonal bubble plot for all importance metrics
def imp_bubble(labels, var1, key, var2 = None, var3 = None, mask_top = None, plots_dir = None):
    
    if mask_top is None:
        mask_top = np.arange(len(var1)) 
    a = len(mask_top)
    
    metrics = np.stack([v for v in list([var1, var2, var3]) if v is not None])
    #metrics = np.dstack([var1, var2, var3])
    if a < 50:
        scale = 200
    else:
        scale = 50

    mi_monthly_trunc = metrics.mean(axis = 0).T[:, mask_top]
    n = mi_monthly_trunc.shape[1]
    plt.figure(figsize = (a*0.35, 5))
    for z in range(len(mi_monthly_trunc)):
        plt.scatter(x = np.arange(1, n+1), y = np.repeat(z, n), 
                    s = mi_monthly_trunc[z,:]*scale, c = mi_monthly_trunc[z,:], 
                    cmap = 'Reds')
    plt.yticks(np.arange(len(MONTHS)), MONTHS)
    plt.ylabel('month')
    plt.xticks(np.arange(1, len(labels[mask_top])+1), 
               labels[mask_top], rotation = 90)
    plt.xlabel('feature')
    plt.grid(ls=':', alpha = 0.5)
    plt.title(f'motnhly average {key}')
    plt.colorbar()
    plt.tight_layout()
    if plots_dir is not None:
        plt.savefig(f'{plots_dir}/bubble_{key}_comb{a}.png',
                       bbox_inches='tight')
        plt.close() 



#create a boxplot
def imp_fullbar(labels, var1, key, mask_top= None, plots_dir = None):
    
    if mask_top is None:
        mask_top = np.arange(len(var1)) 
    a = len(mask_top)
    
    boxes = list(var1[mask_top])
    boxes = [x[~np.isnan(x)] for x in boxes]
    
    plt.figure(figsize = (0.5*a, 5))
    bx = plt.boxplot(boxes, flierprops=dict(color='0.5', 
                    markersize = 2, markeredgecolor='0.5'));
    # for patch, color in zip(bx['boxes'], colors):
    #     patch.set_color(color)
    plt.axhline(y=0, ls='--', color = 'r', lw = 0.5)
    plt.xticks(np.arange(1, len(labels[mask_top])+1), labels[mask_top], rotation = 90, color = 'k'); 
    plt.ylim((-0.05, 1.02))
    #[plt.gca().get_xticklabels()[x].set_color("red") for x in np.where(box_mask)[0]]
    plt.grid(ls=':', alpha = 0.5)
    plt.title(f'{key} distributions per month+cv, all months')
    plt.tight_layout()
    if plots_dir is not None:
        plt.savefig(f'{plots_dir}/boxplots_{key}_comb{a}.png',
                     bbox_inches='tight')
        plt.close()
        
        
#create a map
def cont_map(month, select_vars, summaries_dir, plots_dir = None):
    #var = 'momo.t'
    #data = xr.open_dataset(models[m])
    models = np.hstack(glob.glob(f'{summaries_dir}/*/test.contributions.mean.nc'))
    months_list_sort = np.hstack([x.split('/')[13] for x in models]) 
    idx = np.where(months_list_sort == month)[0][0]
    
    data = xr.open_dataset(models[idx])
    #data_var = np.array(data[var])
    lon = data.lon
    lat = data.lat
    
    for var in select_vars:
        data_var = np.array(data[var])
        x, y = np.meshgrid(lon, lat, indexing='xy')
        fig, ax = plt.subplots(figsize=(10, 8),
                                   subplot_kw={'projection': ccrs.PlateCarree()})
        plt.pcolor(x,y, data_var, cmap='coolwarm')
        plt.clim((-2, 2))
        ax.coastlines()
        ax.stock_img()
        ax.set_extent(bbox_dict['globe'], crs=ccrs.PlateCarree())  # NA region
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        im_ratio = len(lat) / len(lon)
        cb = plt.colorbar(fraction = 0.05 * im_ratio, pad = 0.1)
        plt.tight_layout()
        plt.title(f'mean contribution, {var}, month {month}')
        if plots_dir is not None:
            plt.savefig(f'{plots_dir}/contributions/map_{cont_var}_{month}.png',
                        bbox_inches='tight')
            plt.close()
    



#------- residual bubble plot
def residual_scatter(un_lons, un_lats, res, key, zlim = None, plots_dir = None):
    #var = 'momo.t'
    #data = xr.open_dataset(models[m])
    
    
    
    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    plt.scatter(x = un_lons, y = un_lats, s = np.abs(res)*0.5, c = res, cmap = 'bwr')
    if zlim is not None: 
        plt.clim(zlim)
    ax.coastlines()
    #ax.stock_img()
    ax.set_extent(bbox_dict['globe'], crs=ccrs.PlateCarree())  # NA region
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    #im_ratio = len(lat) / len(lon)
    cb = plt.colorbar(fraction = 0.025, pad = 0.05)
    plt.tight_layout()
    plt.title(f'mean {key} per location')
    if plots_dir is not None:
        plt.savefig(f'{plots_dir}/map_mean_{key}.png',
                    bbox_inches='tight', dpi = 200)
        plt.close()





#------- residual bubble plot
def predicted_kde(output, plots_dir = None):

    fig, ax = plt.subplots(2, 6, figsize = (6*3, 2*3))
    for m in range(len(MONTHS)):
        
        if len(output['pred'][m]) > 0:
            kernel  = stats.gaussian_kde([np.hstack(output['truth'][m]), 
                                          np.hstack(output['pred'][m])])
            density = kernel([np.hstack(output['truth'][m]), 
                              np.hstack(output['pred'][m])])
            
            rmse_m = np.sqrt(mean_squared_error(output['truth'][m], output['pred'][m]))
            #plt.figure()
            plt.subplot(2, 6,m+1)
            plt.scatter(output['pred'][m], output['truth'][m], s = 3, c=density, 
                        alpha = 0.5, cmap = 'coolwarm')
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
            plt.text(0.1, 0.9, f'{MONTHS[m]}({np.round(rmse_m, 1)})', 
                     bbox=dict(facecolor='none', edgecolor='k'),
                     transform=plt.gca().transAxes)
    plt.suptitle(f'true vs predicted bias, per month')
    if plots_dir is not None:
        plt.savefig(f'{plots_dir}/residuals_monthly.png',
                     bbox_inches='tight')
        plt.close()




#------- histrograms of predicted vs true
def predicted_hist(output, plots_dir = None):
    fig, ax = plt.subplots(2, 6, figsize = (6*3, 2*3))
    for m in range(len(MONTHS)):
        #plt.figure()
        
        if len(output['pred'][m]) > 0:
            rmse_m = np.sqrt(mean_squared_error(output['truth'][m], output['pred'][m]))
            
            plt.subplot(2, 6,m+1)
            plt.hist(np.hstack(output['pred'][m]), bins = 300, density = True, 
                     histtype = 'step', label = f'predicted');
            plt.axvline(x = np.hstack(output['pred'][m]).mean(), alpha = 0.5)
            plt.hist(np.hstack(output['truth'][m]), bins = 300, density = True, 
                     histtype = 'step', label = f'true');
            plt.axvline(x = np.hstack(output['truth'][m]).mean(), color = 'orange',
                        alpha = 0.5)
            plt.xlim((-50, 50))
            plt.legend(fontsize = 6)
            plt.grid(ls = ':')
            plt.xlabel(f'ppb')
            true_mean = np.hstack(output['truth'][m]).mean()
            plt.text(0.1, 0.8, f'mean true {np.round(true_mean, 2)}', 
                     bbox=dict(facecolor='none', edgecolor='none'), fontsize = 6,
                     transform=plt.gca().transAxes)
            plt.text(0.1, 0.9, f'{MONTHS[m]}({np.round(rmse_m, 1)})', 
                     bbox=dict(facecolor='none', edgecolor='k'),
                     transform=plt.gca().transAxes)
    plt.suptitle(f'true vs predicted bias histograms, per month')
    if plots_dir is not None:
        plt.savefig(f'{plots_dir}/hist_predicted_monthly.png',
                     bbox_inches='tight')
        plt.close()
    

    ### ------ histograms ALL
    rmse_total = np.sqrt(mean_squared_error(np.hstack(output['truth']), 
                                            np.hstack(output['pred'])))   
    plt.figure()
    plt.hist(np.hstack(output['pred']), bins = 300, density = True, 
             histtype = 'step', label = f'predicted');
    plt.axvline(x = np.hstack(output['pred']).mean(), alpha = 0.5)
    plt.hist(np.hstack(output['truth']), bins = 300, density = True, 
             histtype = 'step', label = f'true');
    plt.axvline(x = np.hstack(output['truth']).mean(), color = 'orange',
                alpha = 0.5)
    plt.xlim((-50, 50))
    plt.legend()
    plt.grid(ls = ':')
    plt.xlabel(f'ppb')
    true_mean = np.hstack(output['truth']).mean()
    plt.text(0.1, 0.9, f'mean true {np.round(true_mean, 2)}', 
             bbox=dict(facecolor='none', edgecolor='none'), fontsize = 8,
             transform=plt.gca().transAxes)
    plt.title(f'predicted vs true bias, all months, rmse total = {np.round(rmse_total, 2)}')
    if plots_dir is not None:
        plt.savefig(f'{plots_dir}/hist_predicted_all.png',
                     bbox_inches='tight')
        plt.close()


#------- histrograms of predicted vs true
def predicted_hist_single(output, plots_dir = None):

    ### ------ histograms ALL
    rmse_total = np.sqrt(mean_squared_error(np.hstack(output['truth']), 
                                            np.hstack(output['pred'])))   
    plt.figure()
    plt.hist(np.hstack(output['pred']), bins = 300, density = True, 
             histtype = 'step', label = f'predicted');
    plt.axvline(x = np.hstack(output['pred']).mean(), alpha = 0.5)
    plt.hist(np.hstack(output['truth']), bins = 300, density = True, 
             histtype = 'step', label = f'true');
    plt.axvline(x = np.hstack(output['truth']).mean(), color = 'orange',
                alpha = 0.5)
    plt.xlim((-50, 50))
    plt.legend()
    plt.grid(ls = ':')
    plt.xlabel(f'ppb')
    true_mean = np.hstack(output['truth']).mean()
    plt.text(0.1, 0.9, f'mean true {np.round(true_mean, 2)}', 
             bbox=dict(facecolor='none', edgecolor='none'), fontsize = 8,
             transform=plt.gca().transAxes)
    plt.title(f'predicted vs true bias, all months, rmse total = {np.round(rmse_total, 2)}')
    if plots_dir is not None:
        plt.savefig(f'{plots_dir}/hist_predicted_all.png',
                     bbox_inches='tight')
        plt.close()



#------- horizontal barplot
def imp_barplot(labels, var1, var2 = None, var3 = None, mask_top = None, 
                     plots_dir = None):
    
    if mask_top is None:
        mask_top = np.arange(len(var1[0]))
    
    a = len(mask_top)
    x = np.arange(0, len(var1[0][mask_top])*6)[::6][::-1]
    plt.figure(figsize = (7, len(x)*0.45))
    plt.barh(x, var1[0][mask_top], alpha = 0.6, height = 1.2, color = 'blue', label = 'model')
    eb = plt.errorbar(var1[0][mask_top], x, xerr=var1[1][mask_top], 
                      fmt=".", color="blue", xlolims = True, alpha = 0.5)
    eb[1][0].set_marker('|')
    
    if var2 is not None:
        plt.barh(x+1, var2[0][mask_top], alpha = 0.5, height = 1.2, color = 'orange', label = 'permutation')
        eb = plt.errorbar(var2[0][mask_top], x+1, xerr=var2[1][mask_top], 
                          fmt=".", color="orange", xlolims=True, alpha = 0.5)
        eb[1][0].set_marker('|')

    if var3 is not None:
        plt.barh(x+2, var3[0][mask_top], alpha = 0.5, height = 1.2, color = 'green', label = 'contribution')
        eb = plt.errorbar(var3[0][mask_top], x+2, xerr=var3[1][mask_top], 
                          fmt=".", color="green", xlolims=True, alpha = 0.5)
        eb[1][0].set_marker('|')

    plt.yticks(x, labels[mask_top], rotation = 0, color = 'k');
    plt.grid(ls=':', alpha = 0.5)
    plt.legend()
    plt.title(f'top {a} mean importance with std \n all months')
    plt.tight_layout()
    if plots_dir is not None:
        plt.savefig(f'{plots_dir}/all_baplots_comb{a}_all.png',
                     bbox_inches='tight')
        plt.close()



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




