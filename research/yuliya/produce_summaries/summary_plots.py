#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 17:19:54 2022

@author: marchett
"""
import os, glob
import sys
import numpy as np
import h5py
import xarray as xr
from tqdm import tqdm
from contextlib import closing
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from sklearn.metrics import pairwise_distances
from matplotlib import colors
import pandas as pd


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
    ver = plots_dir.split('/')[-3]
    plt.title(f'top {a} mean importance with std \n all months, {ver}')
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
        w = 0.35
    else:
        scale = 50
        w = 0.2

    mi_monthly_trunc = np.nanmean(metrics, axis = 0).T[:, mask_top]
    n = mi_monthly_trunc.shape[1]
    plt.figure(figsize = (a*w, 5))
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
    ver = plots_dir.split('/')[-3]
    plt.title(f'monthly average {key}, {ver}')
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
    ver = plots_dir.split('/')[-3]
    plt.title(f'{key} distributions per month+cv, all months, {ver}')
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
    

def make_unique_locs(dat, lons, lats, years, days):
    un_lons, un_lats = np.unique([lons, lats], axis = 1)
    res = np.zeros_like(un_lons)
    for s in range(len(un_lons)):
        mask1 = np.in1d(lons, un_lons[s])
        mask2 = np.in1d(lats, un_lats[s])
        mask3 = mask1 & mask2
        
        time = years[mask3] * 100 + days[mask3]
        sidx = np.argsort(time)
        dat_select = dat[mask3][sidx]
        res[s] = np.mean(dat_select)

    return res


#------- residual bubble plot
def residual_scatter(un_lons, un_lats, res, key, zlim = None, 
                     cmap = 'bwr', plots_dir = None):
    #var = 'momo.t'
    #data = xr.open_dataset(models[m])
    if zlim is not None:
        divnorm=colors.TwoSlopeNorm(vmin=zlim[0], vcenter=zlim[1], vmax=zlim[2])
    else:
        divnorm = None
    
    if zlim[2] > 40:
        scale = 0.05
    else:
        scale = 0.4
        
    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    plt.scatter(x = un_lons, y = un_lats, s = np.abs(res)*scale, c = res, cmap = cmap,
                norm=divnorm)
    ax.coastlines()
    #ax.stock_img()
    ax.set_extent(bbox_dict['globe'], crs=ccrs.PlateCarree())  # NA region
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    #im_ratio = len(lat) / len(lon)
    #plt.clim(zlim)
    cb = plt.colorbar(fraction = 0.025, pad = 0.05)
    plt.tight_layout()
    ver = plots_dir.split('/')[-3]
    plt.title(f'mean {key} per location, {ver}')
    if plots_dir is not None:
        plt.savefig(f'{plots_dir}/map_mean_{key}.png',
                    bbox_inches='tight', dpi = 200)
        plt.close()


import statsmodels.api as sm
def time_series_loc(lon_0, lat_0, output, plots_dir = None):
    
    lons = np.hstack(output['lon'])
    lats = np.hstack(output['lat'])
    lons = (lons + 180) % 360 - 180
    
    y = np.hstack(output['truth'])
    yhat = np.hstack(output['pred'])
    years_y = np.hstack(output['years'])
    months_y = np.hstack(output['months'])
    days_y = np.hstack(output['days'])
    
    time_full = years_y * 10000 + months_y * 100 + days_y
    time_line = np.sort(np.unique(time_full))
    time_formatted = pd.to_datetime(time_line, format='%Y%m%d')
    
    lowess = sm.nonparametric.lowess

    mask1 = np.in1d(lons, lon_0)
    mask2 = np.in1d(lats, lat_0)
    mask3 = mask1 & mask2
    
    time = years_y[mask3] * 10000 + months_y[mask3] * 100 + days_y[mask3]
    #time = years_y[mask3] * 100 + days_y[mask3]
    sidx = np.argsort(time)
    ys = y[mask3][sidx]
    ys_hat = yhat[mask3][sidx]
    mr = np.mean(ys - ys_hat)
    
    t = np.arange(0, len(ys))
    z = lowess(ys, t, frac = 0.1)[:,1]
    zhat = lowess(ys_hat, t, frac = 0.1)[:, 1]
    
    var = {'y': ys, 'yhat': ys_hat, 'z': z, 'zhat': zhat}
    var_matched = {'y': [], 'yhat': [], 'z': [], 'zhat': []}
    mask_gaps = np.in1d(time_line, time)
    
    for k in var.keys():
        var_matched[k] = np.zeros_like(time_line, dtype = float)
        var_matched[k][:] = np.nan
        var_matched[k][mask_gaps] = var[k]
    
    plt.figure(figsize = (10, 5))
    plt.plot(time_formatted, var_matched['y'], 
             '-', alpha = 0.8, lw = 1, label = 'true', color = '0.5') 
    plt.plot(time_formatted, var_matched['yhat'], 
             '-', alpha = 0.5, label = 'pred', color = 'blue')
    plt.plot(time_formatted, var_matched['z'], '-', alpha = 1., color = '0.4') 
    plt.plot(time_formatted, var_matched['zhat'], '-', alpha = 0.5, color = 'blue')
    #plt.ylim((res['y'].min(), res['y'].max()))
    plt.xlim((time_formatted.min(), time_formatted.max()))
    plt.legend()
    plt.xlabel(f'time (w/ gaps)')
    plt.grid(ls=':', alpha = 0.5)  
    #ver = plots_dir.split('/')[-3]
    plt.title(f'location: {lon_0}, {lat_0}, mean res: {np.round(mr,2)}')
    plt.tight_layout()
    if plots_dir is not None:
        plt.savefig(f'{plots_dir}/signal_{lon_0}_{lat_0}.png',
                    bbox_inches='tight')
        plt.close()




#------- residual bubble plot
def predicted_kde(output, lims = (-80, 80), key = 'bias', plots_dir = None):

    fig, ax = plt.subplots(2, 6, figsize = (6*3, 2*3))
    for m in range(len(MONTHS)):
        
        if len(output['pred'][m]) > 0:
            mask_nan = ~np.isnan(np.hstack(output['pred'][m]))
            
            kernel  = stats.gaussian_kde([np.hstack(output['truth'][m])[mask_nan], 
                                          np.hstack(output['pred'][m])[mask_nan]])
            density = kernel([np.hstack(output['truth'][m])[mask_nan], 
                              np.hstack(output['pred'][m])[mask_nan]])
            
            rmse_m = np.sqrt(mean_squared_error(output['truth'][m][mask_nan], 
                                                output['pred'][m][mask_nan]))
            #plt.figure()
            plt.subplot(2, 6,m+1)
            plt.scatter(output['pred'][m][mask_nan], output['truth'][m][mask_nan], 
                        s = 3, c=density, 
                        alpha = 0.5, cmap = 'coolwarm')
            #plt.pcolor(xedges, yedges, H, cmap = 'coolwarm')
            plt.axvline(x=0, color = '0.5', alpha = 0.5, ls = '--')
            plt.axhline(y=0, color = '0.5', alpha = 0.5, ls = '--')
            plt.plot()
            plt.ylim(lims)
            plt.xlim(lims)
            plt.grid(ls = ':')
            plt.xlabel(f'predicted ppb')
            plt.ylabel(f'true ppb')
            #plt.colorbar()
            #plt.contour(H, levels = 5, cmap = 'coolwarm')
            plt.plot(lims, lims, color = 'r', alpha = 0.2, ls = '--')
            plt.text(0.1, 0.9, f'{MONTHS[m]}', 
                     bbox=dict(facecolor='none', edgecolor='k'),
                     transform=plt.gca().transAxes)
            plt.text(0.1, 0.8, f'rmse {np.round(rmse_m, 2)}', 
                     bbox=dict(facecolor='none', edgecolor='none'), fontsize = 6,
                     transform=plt.gca().transAxes)
    
    ver = plots_dir.split('/')[-3]
    plt.suptitle(f'true vs predicted {key}, per month, {ver}')
    if plots_dir is not None:
        plt.savefig(f'{plots_dir}/residuals_monthly.png',
                     bbox_inches='tight')
        plt.close()




#------- histrograms of predicted vs true
def predicted_hist(output, lims = (-50, 50), key = 'bias', plots_dir = None):
    
    
    fig, ax = plt.subplots(2, 6, figsize = (6*3, 2*3))
    for m in range(len(MONTHS)):
        #plt.figure()
        
        if len(output['pred'][m]) > 0:
            
            mask_nan = ~np.isnan(np.hstack(output['pred'][m]))
            rmse_m = np.sqrt(mean_squared_error(output['truth'][m][mask_nan], 
                                                output['pred'][m][mask_nan]))
            r2_m = r2_score(output['truth'][m][mask_nan], output['pred'][m][mask_nan])
            
            plt.subplot(2, 6,m+1)
            plt.hist(np.hstack(output['pred'][m])[mask_nan], bins = 300, density = True, 
                     histtype = 'step', label = f'predicted');
            plt.axvline(x = np.nanmean(np.hstack(output['pred'][m])), alpha = 0.5)
            plt.hist(np.hstack(output['truth'][m]), bins = 300, density = True, 
                     histtype = 'step', label = f'true');
            plt.axvline(x = np.hstack(output['truth'][m]).mean(), color = 'orange',
                        alpha = 0.5)
            plt.xlim(lims)
            if key == 'bias':
                plt.ylim(top = 0.09)
            plt.legend(fontsize = 6)
            plt.grid(ls = ':')
            plt.xlabel(f'ppb')
            true_mean = np.hstack(output['truth'][m]).mean()
            plt.text(0.1, 0.8, f'mean true {np.round(true_mean, 2)}', 
                     bbox=dict(facecolor='none', edgecolor='none'), fontsize = 6,
                     transform=plt.gca().transAxes)
            plt.text(0.1, 0.9, f'{MONTHS[m]}', 
                     bbox=dict(facecolor='none', edgecolor='k'),
                     transform=plt.gca().transAxes)
            plt.text(0.1, 0.75, f'rmse ({np.round(rmse_m, 1)})', 
                     bbox=dict(facecolor='none', edgecolor='none'), fontsize = 6,
                     transform=plt.gca().transAxes)
            plt.text(0.1, 0.7, f'pve ({np.round(r2_m, 1)})', 
                     bbox=dict(facecolor='none', edgecolor='none'), fontsize = 6,
                     transform=plt.gca().transAxes)
    
    plt.suptitle(f'true vs predicted {key} histograms, per month')
    if plots_dir is not None:
        plt.savefig(f'{plots_dir}/hist_predicted_monthly.png',
                     bbox_inches='tight')
        plt.close()
    

    ### ------ histograms ALL
    mask_nan = ~np.isnan(np.hstack(output['pred']))
    rmse_total = np.sqrt(mean_squared_error(np.hstack(output['truth'])[mask_nan], 
                                            np.hstack(output['pred'])[mask_nan]))   
    pve_total = r2_score(np.hstack(output['truth'])[mask_nan], 
                                            np.hstack(output['pred'])[mask_nan])
    
    plt.figure()
    plt.hist(np.hstack(output['pred'])[mask_nan], bins = 300, density = True, 
             histtype = 'step', label = f'predicted');
    plt.axvline(x = np.hstack(output['pred']).mean(), alpha = 0.5)
    plt.hist(np.hstack(output['truth']), bins = 300, density = True, 
             histtype = 'step', label = f'true');
    plt.axvline(x = np.hstack(output['truth']).mean(), color = 'orange',
                alpha = 0.5)
    plt.xlim(lims)
    if key == 'bias':
        plt.ylim(top = 0.09)
    plt.legend()
    plt.grid(ls = ':')
    plt.xlabel(f'ppb')
    true_mean = np.hstack(output['truth']).mean()
    plt.text(0.1, 0.9, f'mean true {np.round(true_mean, 2)}', 
             bbox=dict(facecolor='none', edgecolor='none'), fontsize = 8,
             transform=plt.gca().transAxes)
    plt.text(0.1, 0.85, f'rmse ({np.round(rmse_total, 1)})', 
             bbox=dict(facecolor='none', edgecolor='none'), fontsize = 8,
             transform=plt.gca().transAxes)
    plt.text(0.1, 0.8, f'pve ({np.round(pve_total, 1)})', 
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
    
    mask_nan = ~np.isnan(np.hstack(output['pred']))
    
    plt.figure()
    plt.hist(np.hstack(output['pred'])[mask_nan], bins = 300, density = True, 
             histtype = 'step', label = f'predicted');
    plt.axvline(x = np.nanmean(np.hstack(output['pred'])), alpha = 0.5)
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




#------- clustered correlation matrix
def plot_correlations(corr_mat, var_names, key, max_corr = 0.9, plot_name = None):
    
    X0 = corr_mat.copy()
    X0[np.isnan(X0)] = 0.
    D = pairwise_distances(X0)
    H = sch.linkage(D, method='average')
    d1 = sch.dendrogram(H, no_plot=True)
    idx = d1['leaves']
    X = X0[idx,:][:, idx]
    var_names_X = np.hstack(var_names)[idx]
    
    X2 = X - np.eye(X.shape[0])
    X2_max = np.abs(X2).max(axis = 0)
    mask_corr = X2_max > max_corr
    labels_mask = np.hstack(var_names_X)[mask_corr]
    
    scale = 150
    
    tick_labels = np.zeros_like(labels_mask)
    for l, label in enumerate(labels_mask):
        label_split = np.hstack(label.split('.'))
        label_mask = ~np.in1d(label_split, ['momo', '2dsfc'])
        tick_labels[l] = '.'.join(label_split[label_mask])
    
    X2 = X[mask_corr, :][:, mask_corr]
    x, y = np.meshgrid(np.arange(len(X2)), np.arange(len(X2)))
    plt.figure(figsize = (12, 10))
    plt.scatter(x, y, s = np.abs(X2)*scale, c = X2, marker = 's', edgecolors = '0.8', 
                vmin = -0.85, vmax = 0.85, cmap = 'bwr')
    plt.xticks(np.arange(len(tick_labels)), tick_labels, fontsize = 9, rotation = 90);
    plt.yticks(np.arange(len(tick_labels)), tick_labels, fontsize = 9, rotation = 0);
    plt.colorbar()
    plt.title(f'momo {key} metric, truncated for {max_corr} max')
    plt.tight_layout()
    if plot_name is not None:
        plt.savefig(f'{plot_name}', dpi = 150, bbox_inches = 'tight')
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




