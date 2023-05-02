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
import matplotlib as mp
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pickle
import scipy.cluster.hierarchy as sch
from sklearn.metrics import pairwise_distances
import seaborn
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from scipy import stats
#sys.path.append('/Users/marchett/Documents/SUDSAQ/analysis_mount/code/suds-air-quality/research/yuliya/produce_summaries/')
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


    #--------- prediction and RESIDUAL plots
    output = read.load_predictions(summaries_dir)
    
    boxes = []
    mkeys = ['rmse', 'pve']
    metrics = {k: [None] * len(output['pred']) for k in mkeys}
    for m in range(len(output['pred'])):
        
        u_years = np.unique(output['years'][m])
        metrics['pve'][m] = []     
        metrics['rmse'][m] = [] 
        if len(output['pred'][m]) > 1:
            for j in range(len(u_years)):
                mask_years = output['years'][m] == u_years[j]
                diff = output['truth'][m] - output['pred'][m]
                boxes.append(diff)
                ys = output['truth'][m][mask_years]
                ys_hat = output['pred'][m][mask_years]
                mask_nan = ~np.isnan(ys_hat)
                
                error = np.sqrt(mean_squared_error(ys[mask_nan], ys_hat[mask_nan]))
                #q = np.percentile(truth_list[m], [25, 75])
                #nrmse.append(error / np.std(output['truth'][m][mask_years]))
                #mae.append(mean_absolute_error(output['truth'][m][mask_years], 
                                               #output['pred'][m][mask_years]))
                #pve.append(explained_variance_score(output['truth'][m], output['pred'][m]))
                metrics['pve'][m].append(r2_score(ys[mask_nan], ys_hat[mask_nan]))
                metrics['rmse'][m].append(error)
        else:
            metrics['pve'][m].append(np.nan)
            metrics['rmse'][m].append(np.nan)
    
    mask_nan = ~np.isnan(np.hstack(output['pred']))
    rmse_total = np.round(np.sqrt(mean_squared_error(np.hstack(output['truth'])[mask_nan], 
                                            np.hstack(output['pred'])[mask_nan])), 2)    
    r2_total =   np.round(r2_score(np.hstack(output['truth'])[mask_nan], 
                                            np.hstack(output['pred'])[mask_nan]) , 2) 
    
    #------rmse vs mae
    for mk in mkeys:
        metric = metrics[mk]
        metric_mean = np.hstack([np.mean(x) for x in metric])
        metric_err = np.hstack([np.std(x) for x in metric])
        
        pos = np.arange(1, len(metric)+1)
        plabels = [f'{y}\n{np.round(np.mean(x), 2)}' for y, x in zip(plots.MONTHS, metric)]
        
        plt.figure()
        plt.errorbar(pos, metric_mean, yerr= metric_err, color = '0.5', ls = '--')
        plt.plot(pos, metric_mean, 'o', label = '')
        plt.xticks(np.arange(1, len(plots.MONTHS)+1), plabels, color = 'k');
        if mk == 'pve':
            plt.ylim((0, 1))    
        plt.grid(ls = ':', alpha = 0.5)
        plt.title(f'{mk}');
        plt.savefig(f'{plots_dir}/{mk}_box_all.png', bbox_inches='tight')
        plt.close()
    
    
    
    gkeys = ['north_america', 'europe', 'asia', 'australia', 'east_europe']
    gmetrics = {m: {k: [None] * len(output['pred']) for k in gkeys} for m in mkeys}
    #rmse = {k: [None] * len(output['pred']) for k in gkeys}
    #n = {k: [None] * len(output['pred']) for k in gkeys}
    for k in gkeys:
        for m in range(len(output['pred'])):
            gmetrics['rmse'][k][m] = [] 
            gmetrics['pve'][k][m] = [] 
            #n[k][m] = []
            u_years = np.unique(output['years'][m])
            for j in range(len(u_years)):
                mask_years = output['years'][m] == u_years[j]
                lats = output['lat'][m][mask_years]
                lons = output['lon'][m][mask_years]
                lons = (lons + 180) % 360 - 180
                bbox = plots.bbox_dict[k]
                mask_lons = (lons > bbox[0]) & (lons < bbox[1])
                mask_lats = (lats > bbox[2]) & (lats < bbox[3])
                mask = mask_lons & mask_lats
                
                mask_nan = ~np.isnan(output['pred'][m][mask_years][mask])
                
                if (mask.sum() > 0) & (mask_nan.sum() > 0):
                    ys = output['truth'][m][mask_years][mask]
                    ys_hat = output['pred'][m][mask_years][mask]
                    mask_nan = ~np.isnan(output['pred'][m][mask_years][mask])
                    error = np.sqrt(mean_squared_error(ys[mask_nan], ys_hat[mask_nan]))
                    gmetrics['rmse'][k][m].append(error)
                    gmetrics['pve'][k][m].append(r2_score(ys[mask_nan], 
                                                          ys_hat[mask_nan]))
                else:
                    gmetrics['rmse'][k][m].append(np.nan)
                    gmetrics['pve'][k][m].append(np.nan)
                #n[k][m].append(mask.sum())
    
    c = plt.cm.rainbow(np.linspace(0,1, len(gkeys)))
    for mk in mkeys:
        plt.figure()
        for i, k in enumerate(gkeys):
            pos = np.arange(len(gmetrics[mk][k]) * len(gkeys))[::len(gkeys)]
            y_mean = [np.mean(x) for x in gmetrics[mk][k]]
            y_err = [np.std(x) for x in gmetrics[mk][k]]
            plt.errorbar(pos, y_mean, yerr= y_err, ls = '--', color = c[i])
            plt.plot(pos, y_mean, 'o', color = c[i], label = k)
        plt.grid(ls = ':', alpha = 0.5) 
        plt.legend()
        plt.xticks(pos, plots.MONTHS, color = 'k'); 
        if mk == 'pve':
            plt.ylim((0, 1))   
        plt.title(f'{mk} per region'); 
        plt.savefig(f'{plots_dir}/{mk}_regional_all.png', bbox_inches='tight')
        plt.close()
    
    
    model_type = np.hstack(summaries_dir.split('/'))
    mtype_bias = np.in1d(model_type, 'bias').sum()
    mtype_toar = np.in1d(model_type, 'toar').sum()
    mtype_emu = np.in1d(model_type, 'emulator').sum()
    if mtype_bias > 0:
        bins = [-20, -10, -5, -1, 0, 1, 5, 10, 20]
        pos = np.arange(len(bins))
        key = 'bias'
    else:
        if mtype_toar + mtype_emu > 0:
            bins = np.histogram(np.hstack(output['truth']), 12)[1][1:-2]
            pos = np.arange(len(bins))
            if mtype_toar > 0:
                key = 'toar'
            if mtype_emu > 0:
                key = 'emulator'
        else:
            print('not a valid model type')
    
    #rmse = {k: [] for k in range(len(plots.MONTHS))}
    rmse = []
    for m in range(len(output['pred'])):
        rmse.append([])
        idx = np.digitize(output['truth'][m], bins = bins, right= True)
        if len(idx) > 0:
            for i in range(len(bins)):
                bin_mask = idx == i
                if bin_mask.sum() > 1:
                    mask_nan = ~np.isnan(output['pred'][m][bin_mask])
                    if mask_nan.sum() > 0:
                        error = np.sqrt(mean_squared_error(output['truth'][m][bin_mask][mask_nan], 
                                                        output['pred'][m][bin_mask][mask_nan]))
                    else:
                        error = np.nan
                    rmse[-1].append(error)
                else:
                    rmse[-1].append(np.nan)   
        else:
            rmse[-1].append(np.repeat(np.nan, len(pos)))
    
    
    c = plt.cm.rainbow(np.linspace(0, 1, len(plots.MONTHS)))
    plt.figure()
    for m in range(len(plots.MONTHS)):
        plt.plot(pos, np.hstack(rmse[m]), ls = '--', color = c[m])
        plt.plot(pos, np.hstack(rmse[m]), 'o', color = c[m], label = f'{plots.MONTHS[m]}')
    plt.grid(ls = ':', alpha = 0.5) 
    plt.legend()
    xt = np.hstack(np.round(bins, 0)).astype(int)
    plt.xticks(pos, np.hstack([xt.astype(str)]), color = 'k', fontsize = 10); 
    plt.xlabel(f'ppb true value')
    plt.ylabel(f'rmse')
    plt.title(f'rmse per region'); 
    plt.savefig(f'{plots_dir}/rmse_binned_all.png', bbox_inches='tight')
    plt.close()
    
    

    #-------- large residuals
    print(f'plotting residual maps')
    lon = np.hstack(output['lon'])
    lat = np.hstack(output['lat'])
    lon = (lon + 180) % 360 - 180
    un_lons, un_lats = np.unique([lon, lat], axis = 1)
    
    import statsmodels.api as sm
    lowess = sm.nonparametric.lowess
    
    y = np.hstack(output['truth'])
    yhat = np.hstack(output['pred'])
    years_y = np.hstack(output['years'])
    months_y = np.hstack(output['months'])
    days_y = np.hstack(output['days'])
    
    keys = ['y', 'yhat', 'res', 'res_std', 'rmse', 'rmse_trend', 'std_y']
    res = {}
    for k in keys:
        res[k] = np.zeros_like(un_lons)
    for s in tqdm(range(len(un_lons)), desc = 'Computing trends'):
        mask1 = np.in1d(lon, un_lons[s])
        mask2 = np.in1d(lat, un_lats[s])
        mask3 = mask1 & mask2
        
        time = years_y[mask3] * 10000 + months_y[mask3] * 100 + days_y[mask3]
        #time = years_y[mask3] * 100 + days_y[mask3]
        sidx = np.argsort(time)
        ys = y[mask3][sidx]
        ys_hat = yhat[mask3][sidx]
        
        mask_nan = ~np.isnan(ys_hat)
        
        if mask_nan.sum() > 0:
            #t = np.arange(0, len(y[mask3]))
            res['y'][s] = np.mean(ys)
            res['yhat'][s] = np.mean(ys_hat)
            res['res'][s] = np.mean(ys - ys_hat)
            res['rmse'][s] = np.sqrt(mean_squared_error(ys, ys_hat))
            res['res_std'][s] = np.std(ys - ys_hat)
            
            #true variance per location
            t = np.arange(0, len(ys))
            z = lowess(ys, t, frac = 0.1)
            zhat = lowess(ys_hat, t, frac = 0.1)
            
            res['rmse_trend'][s] = np.sqrt(mean_squared_error(z[:,1], zhat[:,1]))
            res['std_y'][s] = np.sqrt(np.mean((ys - z[:,1])**2))
        else:
             for k in res.keys():
                 res[k][s] = np.nan 

    if key == 'bias':
        zlim_a = [-5., 0., 15.]
        zlim_b = [-10., 0., 10.]
    else:
        zlim_a = np.nanpercentile(res['y'], [10, 50, 90])
        min_b = np.nanpercentile(res['res'], 0)
        zlim_b = (min_b, 0, -min_b)

    #---------- residuals on maps
    plots.residual_scatter(un_lons, un_lats, res['y'], zlim = zlim_a, 
                           key = keys[0], plots_dir = plots_dir)
    plots.residual_scatter(un_lons, un_lats, res['yhat'], zlim = zlim_a, 
                           key = keys[1], plots_dir = plots_dir)
    plots.residual_scatter(un_lons, un_lats, res['res'], zlim = zlim_b, 
                           key = keys[2], plots_dir = plots_dir)   
    # plots.residual_scatter(un_lons, un_lats, res['res_std'], zlim = (-30, 30), 
    #                        key = keys[3], plots_dir = plots_dir)
    plots.residual_scatter(un_lons, un_lats, res['rmse'], zlim = (5, 7.5, 20), 
                           key = keys[4], cmap = 'YlOrRd', plots_dir = plots_dir)
    plots.residual_scatter(un_lons, un_lats, res['rmse_trend'], zlim = (5, 7.5, 20), 
                           key = keys[5], cmap = 'YlOrRd', plots_dir = plots_dir)
    plots.residual_scatter(un_lons, un_lats, res['std_y'], zlim = (5, 7.5, 20), 
                           key = keys[6], cmap = 'YlOrRd', plots_dir = plots_dir)
    
    a = np.nanpercentile(res['res'], [98])
    idx_res = np.where(np.abs(res['res']) > a)[0] 
    #s = idx_res[1]
    
    # toar = glob.glob(f'{root_dir}/data/toar/matched/201[1-5]/01.nc')
    # dat = xr.open_mfdataset(toar)
    # dat.coords['lon'] = (dat.coords['lon'] + 180) % 360 - 180
    
    # toar_lons = dat.coords['lon'].values
    # toar_lats = dat.coords['lat'].values
    # ilon = int(np.where(np.in1d(toar_lons, un_lons[s]))[0])
    # ilat = int(np.where(np.in1d(toar_lats, un_lats[s]))[0])
    
    # toar_dat = dat['toar.o3.dma8epa.median'].values
    # toar_dat[:, ilat, ilon]
    
    if not os.path.exists(f'{plots_dir}/residuals/large'):
        os.makedirs(f'{plots_dir}/residuals/large')
    for s in idx_res:
        plots.time_series_loc(un_lons[s], un_lats[s], output, 
                              plots_dir = f'{plots_dir}/residuals/large/')
        # mask1 = np.in1d(lon, un_lons[s])
        # mask2 = np.in1d(lat, un_lats[s])
        # mask3 = mask1 & mask2
        
        # time = years_y[mask3] * 10000 + months_y[mask3] * 100 + days_y[mask3]
        # #time = years_y[mask3] * 100 + days_y[mask3]
        # sidx = np.argsort(time)
        # ys = y[mask3][sidx]
        # ys_hat = yhat[mask3][sidx]
        # mr = np.mean(ys - ys_hat)
        
        # t = np.arange(0, len(ys))
        # z = lowess(ys, t, frac = 0.1)[:,1]
        # zhat = lowess(ys_hat, t, frac = 0.1)[:, 1]
        
        # plt.figure(figsize = (10, 5))
        # plt.plot(ys_hat, '-', alpha = 0.8, label = 'pred') 
        # plt.plot(zhat, ls = '-', alpha = 0.5, color = 'darkblue')
        # plt.plot(ys, '-', color = '0.5', alpha = 0.5, label = 'true') 
        # plt.plot(z, ls = '-', color = '0.5')
        # plt.legend()
        # plt.xlabel(f'time (w/ gaps)')
        # plt.grid(ls=':', alpha = 0.5)    
        # plt.title(f'location: {un_lons[s]}, {un_lats[s]}, mean res: {np.round(mr,2)}')
        # plt.tight_layout()
        # plt.savefig(f'{plots_dir}/large_residuals/signal_{un_lons[s]}_.{un_lats[s]}.png',
        #             bbox_inches='tight')
        # plt.close()

    #small residuals
    if not os.path.exists(f'{plots_dir}/residuals/small'):
        os.makedirs(f'{plots_dir}/residuals/small')
        
    idx_res_small = np.where((np.abs(res['res']) > 0) & (np.abs(res['res']) < 1))[0]
    np.random.seed(0)
    idx_res_small = np.random.choice(idx_res_small, 20, replace=False)
    for s in idx_res_small:
        plots.time_series_loc(un_lons[s], un_lats[s], output, 
                              plots_dir = f'{plots_dir}/residuals/small/')
    
    #------histrograms and kde
    print(f'plotting KDE and hist')
    if key == 'bias':
        hlims = [-50, 50]
    else:
        hlims = np.percentile(y, [0.1, 99.9])
    plots.predicted_hist(output, lims = hlims, key = key, plots_dir = plots_dir)
    #plots.predicted_kde(output, lims = (-hlims[1], hlims[1]), plots_dir = plots_dir)



          

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_dir', type=str, default = '/bias/local/8hr_median/v3/')
    parser.add_argument('--months', default = 'all', nargs = '*', type=str)
    #parser.add_argument('--parameter', type=str, default=None)

    args = parser.parse_args()
    main(**vars(args)) 





