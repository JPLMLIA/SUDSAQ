#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 11:43:31 2022

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

from scipy import stats
import read_data_momo


# data_root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
# momo_root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/MOMO/'
# toar_output = f'{data_root_dir}/processed/summary_dp/TOAR2/'
# momo_output = f'{data_root_dir}/processed/summary_dp/MOMO/'


def main(years, months, inputs, plotting):
    
    root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
    if not os.path.exists(root_dir):
        root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
    #momo_root_dir = f'{root_dir}/MOMO/'
    toar_output = f'{root_dir}/processed/summary_dp/TOAR2/'
    momo_output = f'{root_dir}/processed/summary_dp/MOMO/'
    
    # years = ['2012']
    # months = ['06', '07', '08']
    # names = ['o3', 't', 'q']
    summaries = ['mean', 'std', 'count']
    
    if inputs == 'all':
        inputs = ['t', 'q', 'ps', 'u', 'v']
    
    if len(months) == 0:
        months = [f'{x}'.zfill(2) for x in np.arange(1, 13)]
        
    for year in years:
        
        for month in tqdm(months):
            #save
            
            toar_file = f'{toar_output}/toar2_{year}_{month}.h5'
            with closing(h5py.File(toar_file, 'r')) as f:
                    lon = f['lon'][:]
                    lat = f['lat'][:]
                    toar_data = f['data'][:]
                    date = f['date'][:].astype(str)
                    station =  f['station'][:].astype(str)
            
            momo_file = f'{momo_output}/momo_{year}_{month}.h5' 
            momo = {}
            with closing(h5py.File(momo_file, 'r')) as f:
                keys = list(f)
                for k in keys:
                    momo[k] = f[k][:]
            
            # momo = read_data_momo.main(output_var = names[0], input_var = names[1,2], 
            #                 year = year, month = month)
            momo_lon = np.hstack([momo['lon'], 360.]) - 180
            momo_lat = np.hstack([momo['lat'], 90.])    
            x, y = np.meshgrid(momo['lon'], momo['lat'])
            
            #mdaily daytime average
            days = np.unique(momo['date'][:,2].astype(str))
            mean_hours = [f'{x}'.zfill(2) for x in np.arange(8, 17)]
            
            momo_in = {k: [] for k in inputs}
            momo_dat = []
            momo_ud = []
            toar_to_momo = {k: [] for k in summaries}
            for d in days:
                mask_d = np.in1d(momo['date'][:, 2], d)
                mask_h = np.in1d(np.hstack(momo['hour'])[mask_d], mean_hours)
                momo_dat.append(momo['o3'][mask_d, :, :][mask_h, :, :].mean(axis = 0))
                for k in inputs:
                    momo_in[k].append(momo[k][mask_d, :, :][mask_h, :, :].mean(axis = 0))
                momo_ud.append(np.unique(momo['date'][mask_d, :][mask_h, :], axis = 0))
            
                mask_toar = np.in1d(date[:, 2], d)
                # dlon = np.digitize(lon[mask_toar], momo_lon)
                # dlat = np.digitize(lat[mask_toar], momo_lat)
                #momo_matched = momo_da[dlat, dlon]
                
                for s in summaries:
                    ret = stats.binned_statistic_2d(lat[mask_toar], lon[mask_toar], 
                                                toar_data[mask_toar], s, 
                                                bins=[momo_lat, momo_lon],
                                                expand_binnumbers = True)
                  
                    toar_to_momo[s].append(ret.statistic)
                #momo_to_toar = momo_dat[-1][ret.binnumber[0], ret.binnumber[1]]
                
    
        momo_dat = np.dstack(momo_dat)
        momo_in = {k: np.dstack(momo_in[k]) for k in momo_in.keys()}
        #toar_to_momo = np.dstack(toar_to_momo)
        momo_ud = np.row_stack(momo_ud)
        toar_to_momo = {s: np.dstack(toar_to_momo[s]) for s in summaries}
        bias = momo_dat - toar_to_momo['mean']
        
       
        #save matched  
        new_file = f'{root_dir}/processed/summary_dp/coregistered/momo_matched_{year}_{month}.h5' 
        with closing(h5py.File(new_file, 'w')) as f:
            for s in summaries:
                f['toar/' + str(s)] = np.dstack(toar_to_momo[s])
            f['o3'] = momo_dat
            f['date'] = momo_ud
            for k in momo_in.keys():
                f[k] = momo_in[k]
            
        if plotting:
            cmin = np.nanmin(bias)
            cmax = np.nanmax(bias)
            cmax = np.min(np.abs([cmin, cmax])) *0.5
            cmin = -cmax
            x, y = np.meshgrid(momo['lon']-180, momo['lat'])
            for d in range(bias.shape[2]):
                fig = plt.figure(figsize=(18, 9))
                ax = plt.subplot(projection = ccrs.PlateCarree())
                #plt.contourf(lon-180, lat, (momo_dat - means), levels = 50, cmap = 'coolwarm')
                #plt.pcolor(x, y, toar_to_momo.mean(axis = 2), cmap = 'coolwarm')
                plt.pcolor(x, y, bias[:, :, d], cmap = 'coolwarm')
                plt.clim((cmin, cmax))
                plt.colorbar()
                ax.set_global()
                ax.coastlines()
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
                ax.stock_img()
                plt.title(f'daily bias (mean(momo 8-4) - toar), year = {year}, month = {month}, day = {days[d]}')
                ax.set_extent([-140, -50, 10, 80], crs=ccrs.PlateCarree())
                plt.savefig(f'{root_dir}/processed/plots/bias_{year}_{month}_{d}.png', 
                            bbox_inches = 'tight')
                plt.close()
        
            #montly mean
            fig = plt.figure(figsize=(18, 9))
            ax = plt.subplot(projection = ccrs.PlateCarree())
            #plt.contourf(lon-180, lat, (momo_dat - means), levels = 50, cmap = 'coolwarm')
            #plt.pcolor(x, y, toar_to_momo.mean(axis = 2), cmap = 'coolwarm')
            plt.pcolor(x, y, np.nanmean(bias, axis = 2), cmap = 'coolwarm')
            plt.clim(cmin, cmax)
            plt.colorbar()
            ax.set_global()
            ax.coastlines()
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=1, color='gray', alpha=0.5, linestyle='--')
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            ax.stock_img()
            plt.title(f'monthly mean bias, year = {year}, month = {month}')
            ax.set_extent([-140, -50, 10, 80], crs=ccrs.PlateCarree())
            plt.savefig(f'{root_dir}/processed/plots/bias_{year}_{month}_mean.png', 
                        bbox_inches = 'tight')
            plt.close()
            
            #montly std
            fig = plt.figure(figsize=(18, 9))
            ax = plt.subplot(projection = ccrs.PlateCarree())
            #plt.contourf(lon-180, lat, (momo_dat - means), levels = 50, cmap = 'coolwarm')
            #plt.pcolor(x, y, toar_to_momo.mean(axis = 2), cmap = 'coolwarm')
            plt.pcolor(x, y, np.nanstd(bias, axis = 2), cmap = 'BuPu')
            plt.clim(0, cmax*0.5)
            plt.colorbar()
            ax.set_global()
            ax.coastlines()
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=1, color='gray', alpha=0.5, linestyle='--')
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            ax.stock_img()
            plt.title(f'monthly mean bias, year = {year}, month = {month}, day = {d+1}')
            ax.set_extent([-140, -50, 10, 80], crs=ccrs.PlateCarree())
            plt.savefig(f'{data_root_dir}/processed/plots/bias_{year}_{month}_std.png', 
                        bbox_inches = 'tight')
            plt.close()
        
        
            #plot monthly average bias maps
            # x, y = np.meshgrid(momo['lon']-180, momo['lat'])
            # for year in years:
            #     for month in months:
            #         cmaps = ['jet', 'jet', 'YlOrRd']
                    
            #         fig = plt.subplots(1, len(summaries), figsize=(18, 9))
                    
            #         for s in range(len(summaries)):
            #             mask = (momo_ud[:, 0] == year) & (momo_ud[:, 1] == month)
            #             plot_array = toar_to_momo[summaries[s]].mean(axis =2)
            #             plot_array[plot_array == 0] = np.nan
                        
            #             #plt.subplot(1, len(summaries), d+1)
            #             ax = plt.subplot(1, len(summaries), s+1, projection = ccrs.PlateCarree())
            #             #plt.contourf(lon-180, lat, (momo_dat - means), levels = 50, cmap = 'coolwarm')
            #             #plt.pcolor(x, y, toar_to_momo.mean(axis = 2), cmap = 'coolwarm')
            #             plt.pcolor(x, y, plot_array, cmap = cmaps[s])
            #             plt.colorbar(shrink = 0.4)
            #             ax.set_global()
            #             ax.coastlines()
            #             gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
            #                               linewidth=1, color='gray', alpha=0.5, linestyle='--')
            #             gl.xformatter = LONGITUDE_FORMATTER
            #             gl.yformatter = LATITUDE_FORMATTER
            #             ax.stock_img()
            #             plt.title(f'{summaries[s]}')
            #             #north america
            #             ax.set_extent([-140, -50, 10, 80], crs=ccrs.PlateCarree())
            #             if summaries[s] == 'count':
            #                 plt.clim((1, 10))
            #             if summaries[s] == 'std':
            #                 plt.clim((0, 10))    
                    
            #         plt.tight_layout()    
            #         #ax.set_extent([-125, -112, 30, 40], crs=ccrs.PlateCarree())
            #         #ax.set_extent([-100, -90, 30, 40], crs=ccrs.PlateCarree())
            #         plt.suptitle('Ozone bias (momo - toar) and stats for {month}-{yeaf}')
            #         plt.savefig(f'{data_root_dir}/processsed/plots/momo_vs_toar_mean_{}' )
            #         plt.close()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    #parser.add_argument('data_root_dir', type=str)
    parser.add_argument('--years', default = ['2012'], type=str)
    parser.add_argument('--months', default = [], nargs = '*', type=str)
    parser.add_argument('--inputs', type=str, default='all')
    parser.add_argument('--plotting', type=bool, default=True)
    #parser.add_argument('out_file', type=str)

    args = parser.parse_args()
    main(**vars(args))




   
   



        
        
        
        
        
            
    
    
    
        
        
        



