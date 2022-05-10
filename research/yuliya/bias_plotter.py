#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 13:51:53 2022

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
import subprocess
from scipy.ndimage.filters import gaussian_filter


def main(years, which = 'bias'):
    
    root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
    if not os.path.exists(root_dir):
        root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
    #momo_root_dir = f'{root_dir}/MOMO/'
    toar_output = f'{root_dir}/processed/summary_dp/TOAR2/'
    momo_output = f'{root_dir}/processed/summary_dp/MOMO/'
    subdirs = glob.glob(root_dir + 'MOMO/inputs/*')
    inputs = [x.split('/')[-1] for x in subdirs]

    #years = ['2012', '2013', '2014', '2015']
    months = [f'{x}'.zfill(2) for x in np.arange(1, 13)]
    years = np.atleast_1d(years)
    
    if which == 'bias':
        plot_bias(years, months, root_dir)
     
    if which == 'inputs':
        plot_inputs(years)



def plot_inputs(years, months, inputs, root_dir):
 
    momo_lon = []
    momo_lat = []
    momo = {}
    toar_dat = []
    momo_dat = []
    momo_ud = [] 
    
    for year in years:
        new_file = f'{root_dir}/processed/coregistered/momo_matched_{year}_{month}.h5' 
        with closing(h5py.File(new_file, 'r')) as f:
            momo_dat.append(f['o3'][:])
            toar_dat.append(f['toar']['mean'][:])
            momo_ud.append(f['date'][:])
            momo_lon.append(f['lon'][:])
            momo_lat.append(f['lat'][:])
            for k in inputs:
                momo[k] = f[k][:]
        
    momo_dat = np.dstack(momo_dat)
    momo_ud = np.row_stack(momo_ud) 
    toar_dat = np.dstack(toar_dat)
    bias = momo_dat - toar_dat      
    
    momo_lon = momo_lon[0]
    momo_lat = momo_lat[0]
    
    days = np.unique(momo_ud[:,2].astype(str))
    
    for k in inputs:
        pdict = {'year': year, 'month': month, 'name': f'momo_{k}'}
        x, y = np.meshgrid(momo['lon']-180, momo['lat'])
        Z = momo[k].mean(axis = 0)
        mask = np.nanmean(bias, axis = 0)
        Z[np.isnan(mask)] = np.nan 
        plots.spatial_map(x, y, Z, name_params = pdict, 
                          subdir = 'inputs/momo/')
 
    
    


def plot_bias(years, months, root_dir):
    cmin = -20
    cmax = 20
    
    bbox_dict = {'globe':[0, 360, -90, 90],
                 'europe': [-20+180, 40+180, 25, 80],
                 'asia': [110+180, 160+180, 10, 70],
                 'australia': [130+180, 170+180, -50, -10],
                 'north_america': [-140+180, -50+180, 10, 80]}
    
    for year in years:
        for month in months:
             
            momo_lon = []
            momo_lat = []
            momo_dat = []
            toar_dat = []
            momo_ud = [] 
            new_file = f'{root_dir}/processed/coregistered/momo_matched_{year}_{month}.h5' 
            with closing(h5py.File(new_file, 'r')) as f:
                momo_dat.append(f['o3'][:])
                toar_dat.append(f['toar']['mean'][:])
                momo_ud.append(f['date'][:])
                momo_lon.append(f['lon'][:])
                momo_lat.append(f['lat'][:])
                
            momo_dat = np.dstack(momo_dat)
            momo_ud = np.row_stack(momo_ud) 
            toar_dat = np.dstack(toar_dat)
            bias = momo_dat - toar_dat      
            
            momo_lon = momo_lon[0]
            momo_lat = momo_lat[0]
            
            days = np.unique(momo_ud[:,2].astype(str))
    
            x, y = np.meshgrid(momo_lon-180, momo_lat)
            for b in bbox_dict.keys():
                bbox_ = bbox_dict[b]
                if not os.path.exists(f'{root_dir}/processed/plots/bias/{b}/'):
                    os.makedirs(f'{root_dir}/processed/plots/bias/{b}/')
                    
                for d in range(bias.shape[0]):
                    fig = plt.figure(figsize=(18, 9))
                    ax = plt.subplot(projection = ccrs.PlateCarree())
                    #plt.pcolor(x, y, toar_to_momo.msssean(axis = 2), cmap = 'coolwarm')
                    bias_plot = bias[d, :, :].copy()
                    bias_plot[np.isnan(bias_plot)] = 0.
                    bias_plot = gaussian_filter(bias_plot, sigma=1)
                    bias_plot[np.abs(bias_plot) < 0.5] = np.nan
                    #plt.pcolor(x, y, bias[:, :, d], cmap = 'coolwarm')
                    plt.pcolor(x, y, bias_plot, cmap = 'coolwarm')
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
                    ax.set_extent([bbox_[0]+180, bbox_[1]+180, bbox_[2], bbox_[3]], crs=ccrs.PlateCarree())
                    plt.savefig(f'{root_dir}/processed/plots/bias/{b}/bias_{year}_{month}_{days[d]}_{b}.png', 
                                bbox_inches = 'tight')
                    plt.close()

        
            #montly mean
            # for b in bbox_dict.keys():
            #     bbox_ = bbox_dict[b]
            #     fig = plt.figure(figsize=(18, 9))
            #     ax = plt.subplot(projection = ccrs.PlateCarree())
            #     #plt.contourf(lon-180, lat, (momo_dat - means), levels = 50, cmap = 'coolwarm')
            #     #plt.pcolor(x, y, toar_to_momo.mean(axis = 2), cmap = 'coolwarm')
            #     plt.pcolor(x, y, np.nanmean(bias, axis = 0), cmap = 'coolwarm')
            #     plt.clim(cmin, cmax)
            #     plt.colorbar()
            #     ax.set_global()
            #     ax.coastlines()
            #     gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
            #                       linewidth=1, color='gray', alpha=0.5, linestyle='--')
            #     gl.xformatter = LONGITUDE_FORMATTER
            #     gl.yformatter = LATITUDE_FORMATTER
            #     ax.stock_img()
            #     plt.title(f'monthly mean bias, year = {year}, month = {month}')
            #     ax.set_extent([bbox_[0]+180, bbox_[1]+180, bbox_[2], bbox_[3]], crs=ccrs.PlateCarree())
            #     plt.savefig(f'{root_dir}/processed/plots/bias_monthly/bias_{year}_{month}_mean_{b}.png', 
            #                 bbox_inches = 'tight')
            #     plt.close()
        
            #     #montly std
            #     fig = plt.figure(figsize=(18, 9))
            #     ax = plt.subplot(projection = ccrs.PlateCarree())
            #     #plt.contourf(lon-180, lat, (momo_dat - means), levels = 50, cmap = 'coolwarm')
            #     #plt.pcolor(x, y, toar_to_momo.mean(axis = 2), cmap = 'coolwarm')
            #     plt.pcolor(x, y, np.nanstd(bias, axis = 0), cmap = 'BuPu')
            #     plt.clim(0, cmax*0.5)
            #     plt.colorbar()
            #     ax.set_global()
            #     ax.coastlines()
            #     gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
            #                       linewidth=1, color='gray', alpha=0.5, linestyle='--')
            #     gl.xformatter = LONGITUDE_FORMATTER
            #     gl.yformatter = LATITUDE_FORMATTER
            #     ax.stock_img()
            #     plt.title(f'monthly mean bias, year = {year}, month = {month}, day = {d+1}')
            #     ax.set_extent([bbox_[0]+180, bbox_[1]+180, bbox_[2], bbox_[3]], crs=ccrs.PlateCarree())
            #     plt.savefig(f'{root_dir}/processed/plots/bias_monthly/bias_{year}_{month}_std_{b}.png', 
            #                 bbox_inches = 'tight')
            #     plt.close()

        #movies
        framepath = f'{root_dir}/processed/plots/bias/'
        mp4name = f'bias_movie_{year}'
        framepath_split = framepath.split('&')
        if len(framepath_split) > 1:
            framepath = r'\&'.join(framepath.split('&'))
            mp4name = r'\&'.join(mp4name.split('&'))
        #mp4file_path = f"{'/'.join(framepath.split('/')[:-2])}/{mp4name}.mp4"

        movie_path = glob.glob(f"{root_dir}/processed/plots/bias_movie_{year}*")
        if len(movie_path) > 0:
            os.remove(movie_path[0])

        # command = (f"cd {framepath} && "
        #            f"ffmpeg -r 4 -i *.png -c:v "
        #            f"libx264 -r 25 -pix_fmt "
        #            f"yuv420p ../{mp4name}.mp4")

        
        command = (f"cd {framepath} && "
                   f"ffmpeg -pattern_type glob -i 'bias_{year}*.png' "
                   f"-framerate 25 "
                   f"-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' "
                   f"-c:v libx264 -pix_fmt "
                   f"yuv420p ../{mp4name}.mp4")
        
        subprocess.check_call(command, shell=True)
        filelist = glob.glob(os.path.join(
            '&'.join(framepath.split('\\&')), 'bias_{year}*.png'))
        for f in filelist:
            os.remove(f)





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    #parser.add_argument('data_root_dir', type=str)
    parser.add_argument('--years', default = ['2012'], nargs = '*', type=str)
    #parser.add_argument('--months', default = [], nargs = '*', type=str)
    #parser.add_argument('--inputs', type=str, default='all')
    #parser.add_argument('--plotting', type=bool, default=True)
    #parser.add_argument('out_file', type=str)

    args = parser.parse_args()
    main(**vars(args))






           