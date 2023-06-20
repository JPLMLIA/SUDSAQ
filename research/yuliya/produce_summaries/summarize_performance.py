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
from tqdm import tqdm
from contextlib import closing
from datetime import datetime, timedelta, date
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mp
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import scipy.cluster.hierarchy as sch
from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from scipy import stats
sys.path.insert(0, '/Users/marchett/Documents/SUDS_AQ/analysis_mount/code/suds-air-quality/research/yuliya/produce_summaries/')
import summary_plots as plots
import read_output as read

# import importlib.machinery
# fullpath = '/Users/marchett/Documents/SUDS_AQ/analysis_mount/code/suds-air-quality/research/yuliya/produce_summaries/summary_plots.py'
# plots = importlib.machinery.SourceFileLoader('summary_plots', fullpath).load_module()


def main(sub_dir):
    
    #sub_dir = '/bias/local/8hr_median/v4.1/'
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

    ver = sub_dir.split('/')[-2]

    #--------- prediction and RESIDUAL plots
    output = read.load_predictions(summaries_dir)
    
    
    print(f'compute trends --->')
    lon = np.hstack(output['lon'])
    lat = np.hstack(output['lat'])
    #lon = (lon + 180) % 360 - 180
    un_lons, un_lats = np.unique([lon, lat], axis = 1)
    
    import statsmodels.api as sm
    lowess = sm.nonparametric.lowess
    
    y = np.hstack(output['truth'])
    yhat = np.hstack(output['pred'])
    years_y = np.hstack(output['years'])
    months_y = np.hstack(output['months'])
    days_y = np.hstack(output['days'])
    
    zhat = np.zeros(len(yhat),)
    z = np.zeros(len(yhat),)
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
            #true variance per location
            t = np.arange(0, len(ys))
            z[mask3] = lowess(ys, t, frac = 0.1)[:,1][sidx]
            zhat[mask3] = lowess(ys_hat, t, frac = 0.1)[:, 1][sidx]
  
        else:
            z[mask3] = np.repeat(np.nan, len(ys_hat))
            zhat[mask3] = np.repeat(np.nan, len(ys_hat))
    
    
    #---------- plot metrics of performance
    boxes = []
    mkeys = ['rmse', 'rmse_trend', 'pve', 'pve_trend']
    metrics = {k: [None] * len(output['pred']) for k in mkeys}
    
    u_months = np.arange(1, len(plots.MONTHS)+ 1)
    u_months = np.hstack([u_months[-1], u_months[:-1]])
    u_years = np.unique(years_y)
    for m in range(len(plots.MONTHS)):

        #u_years = np.unique(output['years'][m])
        mask_months = months_y == u_months[m]
        if mask_months.sum() > 0:
        
            metrics['pve'][m] = []     
            metrics['rmse'][m] = [] 
            metrics['rmse_trend'][m] = []
            metrics['pve_trend'][m] = []
            for j in range(len(u_years)):
                mask_years = years_y == u_years[j]
                diff = (y - yhat)[mask_years & mask_months]
                #boxes.append(diff)
                ys = y[mask_years & mask_months]
                ys_hat = yhat[mask_years & mask_months]
                mask_nan = ~np.isnan(ys_hat)
                
                zs = z[mask_years & mask_months]
                zs_hat = zhat[mask_years & mask_months]
                
                if mask_nan.sum() > 0:
                    error = np.sqrt(mean_squared_error(ys[mask_nan], ys_hat[mask_nan]))
                    error_trend = np.sqrt(mean_squared_error(zs[mask_nan], zs_hat[mask_nan]))
                    metrics['pve'][m].append(r2_score(ys[mask_nan], ys_hat[mask_nan]))
                    metrics['rmse'][m].append(error)
                    metrics['rmse_trend'][m].append(error_trend)
                    metrics['pve_trend'][m].append(r2_score(zs[mask_nan], zs_hat[mask_nan]))
                else:
                    metrics['pve'][m].append(np.nan)
                    metrics['rmse'][m].append(np.nan)
                    metrics['rmse_trend'][m].append(np.nan)
                    metrics['pve_trend'][m].append(np.nan)
            else:
                metrics['pve'][m].append(np.nan)
                metrics['rmse'][m].append(np.nan)
                metrics['rmse_trend'][m].append(np.nan)
                metrics['pve_trend'][m].append(np.nan)
    
    mask_nan = ~np.isnan(np.hstack(output['pred']))
    rmse_total = np.round(np.sqrt(mean_squared_error(y[mask_nan], yhat[mask_nan])), 2)    
    r2_total =   np.round(r2_score(y[mask_nan], yhat[mask_nan]) , 2) 
    
    
    # --- by value
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
    
    
    #------ plot metrics
    for mk in mkeys:
        metric = metrics[mk]
        metric_mean = np.hstack([np.nanmean(x) for x in metric])
        metric_err = np.hstack([np.nanstd(x) for x in metric])
        
        pos0 = np.arange(1, len(metric)+1)
        plabels = [f'{y}\n{np.round(np.nanmean(x), 2)}' for y, x in zip(plots.MONTHS, metric)]
        
        plt.figure()
        plt.errorbar(pos0, metric_mean, yerr= metric_err, color = '0.5', ls = '--')
        plt.plot(pos0, metric_mean, 'o', label = '')
        plt.xticks(np.arange(1, len(plots.MONTHS)+1), plabels, color = 'k');
        if (mk == 'pve') | (mk == 'pve_trend'):
            plt.ylim((0, 1))    
        plt.grid(ls = ':', alpha = 0.5)
        plt.title(f'{key} ({ver}) {mk}');
        plt.savefig(f'{plots_dir}/{mk}_box_all.png', bbox_inches='tight')
        plt.close()

    
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
    plt.title(f'{key} ({ver}) rmse per true value'); 
    plt.savefig(f'{plots_dir}/rmse_binned_all.png', bbox_inches='tight')
    plt.close()
    
    

    #-------- large residuals
    print(f'plotting residual maps --->')
    lon = np.hstack(output['lon'])
    lat = np.hstack(output['lat'])
    #lon = (lon + 180) % 360 - 180
    un_lons, un_lats = np.unique([lon, lat], axis = 1)
    
    import statsmodels.api as sm
    lowess = sm.nonparametric.lowess
    
    y = np.hstack(output['truth'])
    yhat = np.hstack(output['pred'])
    years_y = np.hstack(output['years'])
    months_y = np.hstack(output['months'])
    days_y = np.hstack(output['days'])
    
    keys = ['y', 'yhat', 'res', 'res_std', 'rmse', 'rmse_trend', 'std_y', 'count']
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
            res['count'][s] = len(ys)
            
            #true variance per location
            t = np.arange(0, len(ys))
            zs = lowess(ys, t, frac = 0.1)
            zs_hat = lowess(ys_hat, t, frac = 0.1)
            
            res['rmse_trend'][s] = np.sqrt(mean_squared_error(zs[:,1], zs_hat[:,1]))
            res['std_y'][s] = np.sqrt(np.mean((ys - zs[:,1])**2))
        else:
             for k in res.keys():
                 res[k][s] = np.nan 

    if key == 'bias':
        zlim_a = [-5., 0., 15.]
        zlim_b = [-10., 0., 10.]
        scale = 0.4
    else:
        zlim_a = np.nanpercentile(res['y'], [10, 50, 90])
        min_b = np.nanpercentile(res['res'], 0)
        zlim_b = (min_b, 0, -min_b)
        scale = 0.05

    

    #---------- residuals on maps
    plots.residual_scatter(un_lons, un_lats, res['y'], zlim = zlim_a, 
                           key = keys[0], scale = scale, plots_dir = plots_dir)
    plots.residual_scatter(un_lons, un_lats, res['yhat'], zlim = zlim_a, 
                           key = keys[1], scale = scale, plots_dir = plots_dir)
    plots.residual_scatter(un_lons, un_lats, res['res'], zlim = zlim_b, 
                           key = keys[2], scale = scale, plots_dir = plots_dir)   
    # plots.residual_scatter(un_lons, un_lats, res['res_std'], zlim = (-30, 30), 
    #                        key = keys[3], plots_dir = plots_dir)
    plots.residual_scatter(un_lons, un_lats, res['rmse'], zlim = (5, 7.5, 20), 
                           key = keys[4], cmap = 'YlOrRd', plots_dir = plots_dir)
    plots.residual_scatter(un_lons, un_lats, res['rmse_trend'], zlim = (5, 7.5, 20), 
                           key = keys[5], cmap = 'YlOrRd', plots_dir = plots_dir)
    plots.residual_scatter(un_lons, un_lats, res['std_y'], zlim = (5, 7.5, 20), 
                           key = keys[6], cmap = 'YlOrRd', plots_dir = plots_dir)
    
    plots.residual_scatter(un_lons, un_lats, res['count'], zlim = (100, 600, 1400), 
                           key = keys[7], cmap = 'cool', 
                           scale = 0.001, plots_dir = plots_dir)
    
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

    #small residuals
    if not os.path.exists(f'{plots_dir}/residuals/small'):
        os.makedirs(f'{plots_dir}/residuals/small')
        
    idx_res_small = np.where((np.abs(res['res']) > 0) & (np.abs(res['res']) < 1))[0]
    np.random.seed(0)
    idx_res_small = np.random.choice(idx_res_small, 20, replace=False)
    for s in idx_res_small:
        plots.time_series_loc(un_lons[s], un_lats[s], output, 
                              plots_dir = f'{plots_dir}/residuals/small/')
    
    
    bbox = plots.bbox_dict['east_europe1']
    #bbox = [20, 35, 40, 50]
    if not os.path.exists(f'{plots_dir}/residuals/east_europe'):
        os.makedirs(f'{plots_dir}/residuals/east_europe')
    reg_lons = (lon > bbox[0]) & (lon < bbox[1])
    reg_lats = (lat > bbox[2]) & (lat < bbox[3])
    mask_reg = reg_lons & reg_lats
    un_lons_reg, un_lats_reg = np.unique([lon[mask_reg], lat[mask_reg]], axis = 1)
    for s in range(len(un_lons_reg)):
        plots.time_series_loc(un_lons_reg[s], un_lats_reg[s], output,
                              plots_dir = f'{plots_dir}/residuals/east_europe/')
    
    
    
    #------histrograms and kde
    print(f'plotting KDE and hist --->')
    if key == 'bias':
        hlims = [-50, 50]
    else:
        hlims = np.percentile(y, [0.1, 99.9])
    plots.predicted_hist_total(output, lims = hlims, key = f'global_{key}', plots_dir = plots_dir)
    plots.predicted_hist_monthly(output, lims = hlims, key = f'global_{key}', plots_dir = plots_dir)
    #plots.predicted_kde(output, lims = (-hlims[1], hlims[1]), plots_dir = plots_dir)


    

    # -------- REGIONAL plots
    print('regional plots ---> ')
    gkeys = ['north_america', 'europe', 'asia', 'australia']
    gmetrics = {m: {k: [None] * len(output['pred']) for k in gkeys} for m in mkeys}
    u_months = np.arange(1, len(plots.MONTHS)+ 1)
    u_months = np.hstack([u_months[-1], u_months[:-1]])
    u_years = np.unique(years_y)
    for k in gkeys:
        for m in range(len(output['pred'])):
            gmetrics['rmse'][k][m] = [] 
            gmetrics['pve'][k][m] = [] 
            gmetrics['rmse_trend'][k][m] = [] 
            gmetrics['pve_trend'][k][m] = []
           
            mask_months = months_y == u_months[m]
            #n[k][m] = []
            #u_years = np.unique(output['years'][m])
            for j in range(len(u_years)):
                mask_years = years_y == u_years[j]
                
                if (mask_years & mask_months).sum() > 0:
                    lats = lat[mask_years & mask_months]
                    lons = lon[mask_years & mask_months]
                    if np.max(lons) > 180:
                        lons = (lons + 180) % 360 - 180
                    bbox = plots.bbox_dict[k]
                    mask_lons = (lons > bbox[0]) & (lons < bbox[1])
                    mask_lats = (lats > bbox[2]) & (lats < bbox[3])
                    mask = mask_lons & mask_lats
                    
                    mask_nan = ~np.isnan(yhat[mask_years & mask_months][mask])
                   
                    if (mask.sum() > 0) & (mask_nan.sum() > 0):
                        ys = y[mask_years & mask_months][mask]
                        ys_hat = yhat[mask_years & mask_months][mask]
                        mask_nan = ~np.isnan(yhat[mask_years & mask_months][mask])
                        error = np.sqrt(mean_squared_error(ys[mask_nan], ys_hat[mask_nan]))
                        gmetrics['rmse'][k][m].append(error)
                        gmetrics['pve'][k][m].append(r2_score(ys[mask_nan], 
                                                              ys_hat[mask_nan]))
                        
                        zs = z[mask_years & mask_months][mask]
                        zs_hat = zhat[mask_years & mask_months][mask]
                        error_trend = np.sqrt(mean_squared_error(zs[mask_nan], zs_hat[mask_nan]))
                        gmetrics['rmse_trend'][k][m].append(error_trend)
                        gmetrics['pve_trend'][k][m].append(r2_score(zs[mask_nan], 
                                                              zs_hat[mask_nan]))
                    
                    else:
                        gmetrics['rmse'][k][m].append(np.nan)
                        gmetrics['pve'][k][m].append(np.nan)
                        gmetrics['rmse_trend'][k][m].append(np.nan)
                        gmetrics['pve_trend'][k][m].append(np.nan)
    
                else:
                    gmetrics['rmse'][k][m].append(np.nan)
                    gmetrics['pve'][k][m].append(np.nan)
                    gmetrics['rmse_trend'][k][m].append(np.nan)
                    gmetrics['pve_trend'][k][m].append(np.nan)
        
    
    c = plt.cm.rainbow(np.linspace(0,1, len(gkeys)))
    for mk in mkeys:
        plt.figure()
        for i, k in enumerate(gkeys):
            pos = np.arange(len(gmetrics[mk][k]) * len(gkeys))[::len(gkeys)]
            y_mean = [np.nanmean(x) for x in gmetrics[mk][k]]
            y_err = [np.nanstd(x) for x in gmetrics[mk][k]]
            plt.errorbar(pos, y_mean, yerr= y_err, ls = '--', color = c[i])
            plt.plot(pos, y_mean, 'o', color = c[i], label = k)
        plt.grid(ls = ':', alpha = 0.5) 
        plt.legend()
        plt.xticks(pos, plots.MONTHS, color = 'k'); 
        if mk == 'pve':
            plt.ylim((0, 1))   
        plt.title(f'{key} ({ver}) {mk} per region'); 
        plt.savefig(f'{plots_dir}/{mk}_regional_all.png', bbox_inches='tight')
        plt.close()    


    for k in gkeys:
        bbox = plots.bbox_dict[k]
        reg_key = f'{k}_{key}_{ver}'
        try:
            plots.predicted_hist_total(output, bbox, hlims, reg_key, plots_dir = plots_dir)
            plots.predicted_hist_monthly(output, bbox, hlims, reg_key, plots_dir = plots_dir)
        except:
            print(f'no {k} data available')
    
        
    
    # -------- TOAR locs vs full
    #output_full = read.load_predictions_full(summaries_dir)   

          

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_dir', type=str, default = '/bias/local/8hr_median/v3/')
    parser.add_argument('--months', default = 'all', nargs = '*', type=str)
    #parser.add_argument('--parameter', type=str, default=None)

    args = parser.parse_args()
    main(**vars(args)) 





