#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 17:03:01 2022

@author: marchett
@author: kelsey - refactored for feature clustering
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

from reval.best_nclust_cv import FindBestClustCV
from sklearn.neighbors import KNeighborsClassifier

bbox = [0, 360, -90, 90]

bbox_dict = {'globe':[0, 360, -90, 90],
             'europe': [-20+180, 40+180, 25, 80],
             'asia': [110+180, 160+180, 10, 70],
             'australia': [130+180, 170+180, -50, -10],
             'north_america': [-140+180, -50+180, 10, 80]}

#-----------------read in data
root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
if not os.path.exists(root_dir):
    root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
#momo_root_dir = f'{root_dir}/MOMO/'
toar_output = f'{root_dir}/processed/summary_dp/TOAR2/'
momo_output = f'{root_dir}/processed/summary_dp/MOMO/'

years = ['2012', '2013', '2014', '2015']
months = [f'{x}'.zfill(2) for x in np.arange(1, 13)]

#processed clean dataset
momo_lon = []
momo_lat = []
momo_dat = []
toar_dat = []
momo_ud = [] 
for year in years:
    for month in months:
        new_file = f'{root_dir}/processed/coregistered/momo_matched_{year}_{month}.h5' 
        with closing(h5py.File(new_file, 'r')) as f:
            momo_dat.append(f['o3'][:])
            toar_dat.append(f['toar']['mean'][:])
            momo_ud.append(f['date'][:])
            momo_lon.append(f['lon'][:])
            momo_lat.append(f['lat'][:])
    
    
#x, y = np.meshgrid(momo_lon[0], momo_lat[0])
momo_dat = np.row_stack(momo_dat)
momo_ud = np.row_stack(momo_ud).astype(str)
toar_dat = np.row_stack(toar_dat)

mask_lon = (bbox[0] < momo_lon[0]) & (bbox[1] > momo_lon[0])
mask_lat = (bbox[3] > momo_lat[0]) & (bbox[2] < momo_lat[0]) 
#region_mask = mask_lon & mask_lat

toar_dat_region = toar_dat[:, mask_lat, :][:, :, mask_lon]
momo_dat_region = momo_dat[:, mask_lat, :][:, :, mask_lon]
momo_lon_region = momo_lon[0][mask_lon]
momo_lat_region = momo_lat[0][mask_lat]
nx = len(momo_lon_region)
ny = len(momo_lat_region)

mask_all = np.isnan(toar_dat_region)
mask = mask_all.sum(axis = 0) < toar_dat_region.shape[0]
idx = np.where(mask)



#plot one time series
# s = 100
# for s in range(len(idx[0])):
#     z1 = momo_dat_region[:, idx[0][s], idx[1][s]]
#     z2 = toar_dat_region[:, idx[0][s], idx[1][s]]
#     z1_wave = smooth_wavelet(z1)
#     z2_wave = smooth_wavelet(z2)
#     fig, ax = plt.subplots(figsize = (14, 3.5))
#     plt.plot(z1, '.-', alpha = 0.5, label = 'momo')
#     plt.plot(z1_wave, '-', color=plt.gca().lines[-1].get_color())
#     plt.plot(z2, '.-', alpha = 0.5, label = 'toar')
#     plt.plot(z2_wave, '-', color=plt.gca().lines[-1].get_color())
#     plt.legend()
#     plt.grid(ls=':', alpha = 0.5)
#     ticks = plt.gca().get_xticks().astype(int)
#     tick_d = momo_ud[ticks[1:-1], 1].astype(str)
#     tick_y = momo_ud[ticks[1:-1], 0].astype (str)
#     tick_labels = ['-'.join([x,y]) for x,y in zip(tick_d, tick_y)]
#     plt.xticks(ticks[1:-1], labels = tick_labels)
#     # ax2 = ax.twinx()
#     # plt.plot(smooth_wavelet(z1 - z2), '--', color = '0.5')
#     cx = momo_lon_region[idx[1][s]]
#     cy = momo_lat_region[idx[0][s]]
#     plt.title(f'lon {cx}, lat {cy}')    
#     plt.savefig(f'{root_dir}/processed/plots/time_series/temporal_{cx}_{cy}.png', 
#                 bbox_inches = 'tight')
#     plt.close()


#plot overlaid time series with smoothing
g = Gaussian1DKernel(stddev=10)
#overlay each year
s = 100
for s in range(len(idx[0])):    
    z1 = momo_dat_region[:, idx[0][s], idx[1][s]]
    z2 = toar_dat_region[:, idx[0][s], idx[1][s]]
    x = np.arange(366)
    bias_yearly = np.zeros((len(years), 366))
    mask_ly = momo_ud[:, 0] == '2012'
    mask_leap = (momo_ud[mask_ly, 1] == '02') & (momo_ud[mask_ly, 2] == '29')
    plt.figure(figsize = (10, 5))
    for j, year in enumerate(years):
        mask_y = momo_ud[:,0] == year
        check_leap = (momo_ud[mask_y, 1] == '02') & (momo_ud[mask_y, 2] == '29')
        bias_y = z1[mask_y] - z2[mask_y]
        if check_leap.sum() > 0:
            bias_yearly[j, :] = bias_y
        else:
            bias_yearly[j, ~mask_leap] = bias_y
        plt.plot(x, bias_yearly[j, :], alpha = 0.5, lw = 0.5, label = year)
        plt.plot(x, bias_yearly[j, :], '.', alpha = 0.5, color = plt.gca().lines[-1].get_color())
        #plt.plot(z1_wave[mask_y] - z2_wave[mask_y], lw = 0.9, label = year)
        # bmean_med = gaussian_filter1d( bias_yearly[j, :], 5)
        # plt.plot( bmean_med, color='k')
    
    bias_mean = np.nanmean(bias_yearly, axis = 0)
    bias_std = np.nanstd(bias_yearly, axis = 0)
    bmean_med = bias_mean.copy()
    mask_nan = np.isnan(bias_mean)
    #bmean_med[~mask_nan] = gaussian_filter1d(bias_mean[~mask_nan], 10)
    bmean_med = convolve(bias_mean, g, boundary='extend')
    bmean_med[bmean_med == 0] = np.nan
    
    bstd_med = bias_mean.copy()
    mask_nan = np.isnan(bias_std)
    #bstd_med[~mask_nan] = gaussian_filter1d(bias_std[~mask_nan], 10)
    bstd_med = convolve(bias_std, g, boundary='extend')
    bstd_med[bstd_med == 0] = np.nan
    
    plt.plot(bmean_med, '--', color='k')
    plt.fill_between(x, bmean_med-bstd_med, bmean_med+bstd_med,
                     color = '0.5', alpha = 0.5)
    plt.legend() 
    ticks = plt.gca().get_xticks().astype(int)
    tick_d = momo_ud[ticks, 1].astype(str)
    plt.xticks(ticks, labels = tick_d)
    plt.grid(ls=':', alpha = 0.5)
    plt.xlabel('month index')
    plt.ylim((-40, 80))
    
    cx = momo_lon_region[idx[1][s]]
    cy = momo_lat_region[idx[0][s]]
    plt.title(f'lon {cx}, lat {cy}')    
    plt.savefig(f'{root_dir}/processed/plots/time_series/annual_mean/temporal_mean_{cx}_{cy}.png', 
                bbox_inches = 'tight')
    plt.close()

    

#-----------------get smoothed characteristic trends
g = Gaussian1DKernel(stddev=10)
#get variability of seasonal trend map
std_map = np.zeros((ny, nx))
std_map[:] = np.nan
label = np.zeros((ny, nx))
label[:] = np.nan
bias_ = np.zeros((len(idx[0]), 366)) 
bias_[:] = np.nan
for s in range(len(idx[0])):    
    z1 = momo_dat_region[:, idx[0][s], idx[1][s]]
    z2 = toar_dat_region[:, idx[0][s], idx[1][s]]
   
    bias_yearly = np.zeros((len(years), 366))
    mask_ly = momo_ud[:, 0] == '2012'
    mask_leap = (momo_ud[mask_ly, 1] == '02') & (momo_ud[mask_ly, 2] == '29')
    for j, year in enumerate(years):
        mask_y = momo_ud[:,0] == year
        check_leap = (momo_ud[mask_y, 1] == '02') & (momo_ud[mask_y, 2] == '29')
        bias_y = z1[mask_y] - z2[mask_y]
        if check_leap.sum() > 0:
            bias_yearly[j, :] = bias_y
        else:
            bias_yearly[j, ~mask_leap] = bias_y
    
    bias_mean = np.nanmean(bias_yearly, axis = 0)
    bias_std = np.nanstd(bias_yearly, axis = 0)
    bmean_med = bias_mean.copy()
    mask_nan = np.isnan(bias_mean)
    #bmean_med[~mask_nan] = gaussian_filter1d(bias_mean[~mask_nan], 10)
    bmean_med = convolve(bias_mean, g, boundary='extend')
    bmean_med[bmean_med == 0] = np.nan
    
    bstd_med = bias_mean.copy()
    mask_nan = np.isnan(bias_std)
    #bstd_med[~mask_nan] = gaussian_filter1d(bias_std[~mask_nan], 10)
    bstd_med = convolve(bias_std, g, boundary='extend')
    bstd_med[bstd_med == 0] = np.nan
    
    bias_[s, :] = bmean_med
    
    std_map[idx[0][s], idx[1][s]] = np.nanmean(bstd_med)
    
    years_avail = np.isnan(bias_yearly).sum(axis = 1)
    if (years_avail < 320).sum() < 2:
    #years_avail = (~np.isnan(bias_yearly)).sum(axis = 1)
    #if (years_avail > 200).sum() > 2:    
        label[idx[0][s], idx[1][s]] = 1.

#plot variance of trends
for b in bbox_dict.keys():
    bbox_ = bbox_dict[b]
    x, y = np.meshgrid(momo_lon_region-180, momo_lat_region)
    fig = plt.figure(figsize=(18, 9))
    ax = plt.subplot(projection = ccrs.PlateCarree())
    #plt.contourf(lon-180, lat, (momo_dat - means), levels = 50, cmap = 'coolwarm')
    #plt.pcolor(x, y, toar_to_momo.mean(axis = 2), cmap = 'coolwarm')
    plt.pcolor(x, y, std_map, cmap = 'coolwarm')
    idx_single = np.where(label == 1)
    #plt.plot(momo_lon_region[idx_single[1]]-180,  momo_lat_region[idx_single[0]], 'kx')
    #plt.clim(cmin, cmax)
    plt.colorbar()
    ax.set_global()
    ax.coastlines()
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.stock_img()
    ax.set_extent([bbox_[0]+180, bbox_[1]+180, bbox_[2], bbox_[3]], crs=ccrs.PlateCarree()) 
    plt.title(f'seasonal variability of bias, {years}, \n n = {len(idx[0])}')
    plt.savefig(f'{root_dir}/processed/plots/time_series/seasonal_std_map_{b}.png', 
                 bbox_inches = 'tight')
    plt.close()



#-----------------process data for clustering
#take time series that have enough valid data
#remove several missing months
#and compute distance for clustering
mask_annual = momo_ud[:,0] == '2012' 
mask_month = ~np.in1d(momo_ud[mask_annual,1], ['01', '02', '11', '12'])
bias_ = bias_[:, mask_month]
n_nan = np.isnan(bias_).sum(axis =1)  
mask_nan = n_nan < 100
bias_valid = bias_[mask_nan, :]

lon_idx = idx[0][mask_nan]
lat_idx = idx[1][mask_nan]

M = bias_valid.shape[0]
dist = np.zeros((M, M))
for i in tqdm(range(M)):
    s1 = np.array(bias_valid[i, :], dtype=np.double)
    s1 = s1[~np.isnan(s1)]
    start_j = i+1
    for j in range(start_j, M):
        #d = pydtw.dtw(X[i,:], X[j, :]).get_dist()
        s2 = np.array(bias_valid[j, :], dtype=np.double)
        s2 = s2[~np.isnan(s2)]
        d = dt.dtw.distance_fast(s1, s2)
        dist[i, j] = d
        dist[j, i] = d



#-----------------cluster data with VRC
from sklearn import metrics
#try clusters by stability
#K_init = [0.2, 0.15, 0.1, 0.08, 0.05]
K_init = [50, 40, 30, 25, 20, 15]
K_final = [5, 6, 7, 8, 9, 10, 11]

X = bias_valid.copy()
X[np.isnan(X)] = 0.

wk = np.zeros((len(K_init), len(K_final)))
vrc = np.zeros((len(K_init), len(K_final), 3))
stability = np.zeros((len(K_init), len(K_final)))
for r0 in range(len(K_init)):
    #K0 = int(len(dist) * K_init[r0])
    K0 = K_init[r0]
    for r in range(len(K_final)):
    
        K = K_final[r]  
        
        #initial fine res clustering
        assignments, mask = cluster_data(dist, K0, True)
        # clustering = AgglomerativeClustering(n_clusters = K0, 
        #                                       affinity = 'precomputed', 
        #                                       linkage = 'average').fit(dist)
        
        ch0 = metrics.calinski_harabasz_score(X, assignments)
        vrc[r0, r, 0] = ch0
            
        dist_big = dist[mask, :][:, mask]
        new_assignments, _ = cluster_data(dist_big, K, False)
        unique_labels = np.unique(new_assignments)
        
        bias_clust = np.zeros((ny, nx))
        bias_clust[:] = np.nan
        bias_clust[lon_idx[mask], lat_idx[mask]] = new_assignments
        assignments[mask] = new_assignments
        assignments[~mask] = -1
        
        
        # clustering = AgglomerativeClustering(n_clusters = K, 
        #                                   affinity = 'precomputed', 
        #                                   linkage = 'average').fit(dist_big)
        # labels_uni = np.unique(clustering.labels_)
        # nc = np.hstack([(clustering.labels_ == x).sum() for x in labels_uni])
        # bias_clust = np.zeros((ny, nx))
        # bias_clust[:] = np.nan
        # bias_clust[idx[0][mask_nan][mask_big], idx[1][mask_nan][mask_big]] = clustering.labels_
        # assignments[mask_big] = clustering.labels_
        # assignments[~mask_big] = -1
        
        
        ch = metrics.calinski_harabasz_score(X, assignments)
        vrc[r0, r, 1] = ch
        n0 = (assignments == -1).sum()
        #print(f'{K0}, {ch0}, {K}, {ch}, {(assignments == -1).sum()}')
        
        #cluster means
        bias_big = bias_valid[mask]
        means = np.zeros((K, bias_big.shape[1]))
        for l in range(len(unique_labels)):
            mask_uni = new_assignments == unique_labels[l]
            means[l, :] = np.nanmean(bias_big[mask_uni], axis = 0)

        #assign small clusters
        assignments, labels_small = assign_data(bias_valid, mask, means)
        
        # idx_small = np.where(~mask)[0]
        # labels_small = np.zeros_like(bias_clust)
        # for i in idx_small:
        #     bias_i = bias_valid[i,:]
        #     s1 = np.array(bias_i, dtype=np.double)
        #     s1 = s1[~np.isnan(s1)]
        #     dist_i = np.zeros((len(means), ))
        #     for j in range(len(means)):
        #         s2 = np.array(means[j, :], dtype=np.double)
        #         s2 = s2[~np.isnan(s2)]
        #         d = dt.dtw.distance_fast(s1, s2)
        #         dist_i[j] = d
        #     assign = np.argmin(dist_i)
        #     bias_clust[lon_idx[i], lat_idx[i]] = assign
        #     labels_small[lon_idx[i], lat_idx[i]] = 1
        #     assignments[i] = assign

        chA = metrics.calinski_harabasz_score(X, assignments)
        vrc[r0, r, 2] = chA
        print(f'{K0}, {ch0}, {K}, {ch}, {n0}, {chA}')


        #bias_clust, assignments = clustering_momo(dist, K0, K)
        #plot assignments on map
        cmap = plt.cm.get_cmap('jet', K)
        x, y = np.meshgrid(momo_lon_region-180, momo_lat_region)
        fig = plt.figure(figsize=(18, 9))
        ax = plt.subplot(projection = ccrs.PlateCarree())
        #plt.contourf(lon-180, lat, (momo_dat - means), levels = 50, cmap = 'coolwarm')
        #plt.pcolor(x, y, toar_to_momo.mean(axis = 2), cmap = 'coolwarm')
        plt.pcolor(x, y, bias_clust, cmap = cmap)
        idx_single = np.where(labels_small == 1)
        plt.plot(momo_lon_region[idx_single[1]]-180,  
                 momo_lat_region[idx_single[0]], 'x', color = '0.5')
        plt.colorbar()
        ax.set_global()
        ax.coastlines()
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        ax.stock_img()
        plt.suptitle(f'monthly mean bias, year = {year}, month = {month}')
        plt.title(f'n locs = {M}, K = {K}, VRC = {chA}')
        ax.set_extent([bbox[0]+180, bbox[1]+180, bbox[2], bbox[3]], crs=ccrs.PlateCarree())
        plt.savefig(f'{root_dir}/processed/plots/cluster_analysis/clusters_{K0}_{K}.png', 
                    bbox_inches = 'tight')
        plt.close()
        

#plot vrc curves  
plt.figure()
for j in range(vrc.shape[0]):
    plt.plot(K_final, vrc[j,:,1], '.') 
    plt.plot(K_final, vrc[j,:,1], color = plt.gca().lines[-1].get_color(), label = K_init[j])  
plt.xlabel('number of clusters')
plt.ylabel('vrc')
plt.title(f'vrc, secondary clustering')
plt.legend()
plt.grid(ls=':', alpha=0.5)
plt.savefig(f'{root_dir}/processed/plots/cluster_analysis/vrc_step2.png', bbox_inches = 'tight')
plt.close()

plt.figure()
for j in range(vrc.shape[0]):
    plt.plot(K_final, vrc[j,:,2], '.')
    plt.plot(K_final, vrc[j,:,2], color = plt.gca().lines[-1].get_color(), label = K_init[j])  
plt.legend()
plt.xlabel('number of clusters')
plt.ylabel('vrc')
plt.title(f'vrc, after small cluster assignment')
plt.grid(ls=':', alpha=0.5)
plt.savefig(f'{root_dir}/processed/plots/cluster_analysis/vrc_step3.png', bbox_inches = 'tight')
plt.close()



#-----------------final clustering
K0 = 25
K = 6

assignments, mask = cluster_data(dist, K0, True)

dist_big = dist[mask, :][:, mask]
new_assignments, _ = cluster_data(dist_big, K, False)
unique_labels = np.unique(new_assignments)

bias_clust = np.zeros((ny, nx))
bias_clust[:] = np.nan
bias_clust[lon_idx[mask], lat_idx[mask]] = new_assignments
assignments[mask] = new_assignments
assignments[~mask] = -1

#cluster means
bias_big = bias_valid[mask]
means = np.zeros((K, bias_big.shape[1]))
for l in range(len(unique_labels)):
    mask_uni = new_assignments == unique_labels[l]
    means[l, :] = np.nanmean(bias_big[mask_uni], axis = 0)

#assign small clusters
assignments, labels_small = assign_data(bias_valid, mask, means)




#plot final clustering map
for b in bbox_dict.keys():
    bbox_ = bbox_dict[b]
    cmap = plt.cm.get_cmap('jet', K)
    x, y = np.meshgrid(momo_lon_region-180, momo_lat_region)
    fig = plt.figure(figsize=(18, 9))
    ax = plt.subplot(projection = ccrs.PlateCarree())
    #plt.contourf(lon-180, lat, (momo_dat - means), levels = 50, cmap = 'coolwarm')
    #plt.pcolor(x, y, toar_to_momo.mean(axis = 2), cmap = 'coolwarm')
    plt.pcolor(x, y, bias_clust, cmap = cmap)
    idx_single = np.where(labels_small == 1)
    plt.plot(momo_lon_region[idx_single[1]]-180,  
             momo_lat_region[idx_single[0]], 'x', color = '0.5')
    plt.colorbar()
    ax.set_global()
    ax.coastlines()
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.stock_img()
    plt.suptitle(f'monthly mean bias, year = {year}, month = {month}')
    plt.title(f'n locs = {M}, K = {K}, VRC = {chA}')
    #ax.set_extent([bbox[0]+180, bbox[1]+180, bbox[2], bbox[3]], crs=ccrs.PlateCarree())
    ax.set_extent([bbox_[0]+180, bbox_[1]+180, bbox_[2], bbox_[3]], crs=ccrs.PlateCarree())
    plt.savefig(f'{root_dir}/processed/plots/cluster_analysis/cluster_map_{b}_{K}.png', 
                bbox_inches = 'tight')
    plt.close()

#plot means
for l in range(len(unique_labels)):
    idx_labels = np.where(assignments == unique_labels[l])[0]
    plt.figure(figsize = (8,4))
    for i in idx_labels:
        plt.plot(bias_valid[i, :], color = '0.7')
        #plt.plot(bias_big[i, :], color = '0.7', alpha = 0.5)
    plt.plot(means[l, :], ls = '--', color = cmap(l))
    plt.grid(ls=':', alpha = 0.5)
    plt.ylim((np.nanmin(bias_valid)*1.1, np.nanmax(bias_valid)*1.1))
    ticks = plt.gca().get_xticks().astype(int)
    tick_d = momo_ud[ticks[1:-1], 1].astype(str)
    plt.xticks(ticks[1:-1], labels = tick_d)
    # tick_d = momo_ud[mask_year][ticks[1:-1], 1].astype(str)
    # tick_y = momo_ud[mask_year][ticks[1:-1], 0].astype(str)
    # tick_labels = ['-'.join([x,y]) for x,y in zip(tick_d, tick_y)]
    # plt.xticks(ticks[1:-1], labels = tick_labels)
    plt.title(f'region {l} biases and mean bias, n = {len(idx_labels)}')
    plt.ylim((np.nanmin(bias_), np.nanmax(bias_)))
    plt.savefig(f'{root_dir}/processed/plots/cluster_analysis/cluster_mean_{K}_{l}.png', 
                bbox_inches = 'tight')
    plt.close()
    




# --------------
def cluster_data(dist, K, initial):
    
    clustering = AgglomerativeClustering(n_clusters = K, 
                                          affinity = 'precomputed', 
                                          linkage = 'average').fit(dist)
    assignments = clustering.labels_
 
    if initial:
        labels_uni = np.unique(clustering.labels_)
        nc = np.hstack([(clustering.labels_ == x).sum() for x in labels_uni])
        mask_big = np.in1d(clustering.labels_, labels_uni[nc > 1])
    else:
        mask_big = np.repeat(True, len(assignments))
    
    return assignments, mask_big



# --------------
def assign_data(bias_valid, mask, means):
    
    idx_small = np.where(~mask)[0]
    labels_small = np.zeros_like(bias_clust)
    for i in idx_small:
        bias_i = bias_valid[i,:]
        s1 = np.array(bias_i, dtype=np.double)
        s1 = s1[~np.isnan(s1)]
        dist_i = np.zeros((len(means), ))
        for j in range(len(means)):
            s2 = np.array(means[j, :], dtype=np.double)
            s2 = s2[~np.isnan(s2)]
            d = dt.dtw.distance_fast(s1, s2)
            dist_i[j] = d
        assign = np.argmin(dist_i)
        bias_clust[idx[0][mask_nan][i], lat_idx[i]] = assign
        labels_small[idx[0][mask_nan][i], idx[1][mask_nan][i]] = 1
        assignments[i] = assign

    
    return assignments, labels_small



# --------------
def constant_interpolation(wave):
    mask_wave = np.isnan(wave)
    xi = np.arange(len(wave))[~mask_wave]
    x0 = np.arange(len(wave))[mask_wave]

    fit = sp.interpolate.interp1d(xi, wave[~mask_wave], bounds_error = False,
                                  fill_value = 'extrapolate')
    vals_interp = fit(x0)
    wave[mask_wave] = vals_interp
    
    return wave
    
    
# --------------
def smooth_wavelet(x, coef = 10., nl = 0.5):
    x_nan = x.copy()
    x_nan[:] = np.nan
    
    mask_nan = np.isnan(x)
    d4 = pywt.Wavelet('haar')
    coeffs = pywt.wavedec(x[~mask_nan], d4, mode='per')
    filteredCoeffs = copy.deepcopy(coeffs)
    keep_coefs = np.floor(len(filteredCoeffs) * nl)
    levels = -np.arange(1, keep_coefs + 1, dtype = int)
   
    for level in levels:
        filteredCoeffs[level] = np.zeros(len(filteredCoeffs[level]))
    recon = pywt.waverec(filteredCoeffs, d4, mode='constant')
    
    if len(recon) > len(x[~mask_nan]):
        recon = recon[:len(x[~mask_nan])]
   
    x_nan[~mask_nan] = recon
    
    return x_nan


# --------------
def compute_inertia(a, X):
    W = [np.mean(pairwise_distances(X[a == c, :])) for c in np.unique(a)]
    return np.mean(W)


# --------------
def compute_gap(data, dist, k_max=5, n_references=5):
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    reference = np.random.rand(*data.shape)
    reference_inertia = []
    for k in range(1, k_max+1):
        local_inertia = []
        for _ in range(n_references):
            #clustering.n_clusters = k
            assignments = AgglomerativeClustering(n_clusters = k, 
                                                 linkage = 'average').fit(reference).labels_
            local_inertia.append(compute_inertia(assignments, reference))
        reference_inertia.append(np.mean(local_inertia))
    
    ondata_inertia = []
    for k in range(1, k_max+1):
        #clustering.n_clusters = k
        assignments = AgglomerativeClustering(n_clusters = k, 
                                             affinity = 'precomputed', 
                                             linkage = 'average').fit(dist).labels_
        ondata_inertia.append(compute_inertia(assignments, data))
        
    gap = np.log(reference_inertia)-np.log(ondata_inertia)
    return gap, np.log(reference_inertia), np.log(ondata_inertia)


from sklearn.metrics import pairwise_distances
def compute_gap(clustering, data, k_max=5, n_references=5):
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    reference = np.random.rand(*data.shape)
    reference_inertia = []
    for k in range(1, k_max+1):
        local_inertia = []
        for _ in range(n_references):
            clustering.n_clusters = k
            assignments = clustering.fit_predict(reference)
            local_inertia.append(compute_inertia(assignments, reference))
        reference_inertia.append(np.mean(local_inertia))
    
    ondata_inertia = []
    for k in range(1, k_max+1):
        clustering.n_clusters = k
        assignments = clustering.fit_predict(data)
        ondata_inertia.append(compute_inertia(assignments, data))
        
    gap = np.log(reference_inertia)-np.log(ondata_inertia)
    return gap, np.log(reference_inertia), np.log(ondata_inertia)

k_max = 5
gap, reference_inertia, ondata_inertia = compute_gap(KMeans(), X, k_max)








