#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 13:10:36 2022

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
import summary_plots as plots
import read_output as read
from sklearn.preprocessing import StandardScaler

#root_dir = '/Users/marchett/Documents/SUDS_AQ/analysis_mount/'
#sub_dir = '/bias/local/8hr_median/v1/'

def main(sub_dir, months = 'all'):
    
    root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
    if not os.path.exists(root_dir):
        root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
    
    #set plot directory
    summaries_dir = f'{root_dir}/summaries/{sub_dir}/combined_data/' 
    # plots_dir = f'{root_dir}/summaries/{sub_dir}/summary_plots/'
    # # create one if it's not there
    # if not os.path.exists(plots_dir):
    #     os.makedirs(plots_dir)
    bbox = plots.bbox_dict['globe']
    if months == 'all':
        months = plots.MONTHS
    
    # make the full correlation matrix
    exclude = ['momo.hno3', 'momo.oh', 'momo.pan', 'momo.q2',
               'momo.sens', 'momo.so2', 'momo.T2', 'momo.taugxs',
               'momo.taugys', 'momo.taux', 'momo.tauy', 'momo.twpc',
               'momo.2dsfc.CFC11', 'momo.2dsfc.CFC113', 'momo.2dsfc.CFC12',
               'momo.ch2o', 'momo.cumf0', 'momo.2dsfc.dms']
    
    #run correlations for all months
    for month in months:
        
        
        #data_file = f'{summaries_dir}/{month}/test.data.mean.nc'
        data_file = f'{summaries_dir}/{month}/data.h5'
        print(data_file)
        if not os.path.isfile(data_file):
            continue
         
        #data = xr.open_dataset(data_file)
        # data.coords['lon'] = (data.coords['lon'] + 180) % 360 - 180
        # data = data.sortby(data.lon)
        # var_names = list(data.keys())
        
        #crop the data for the region
        # data_cropped = data.sel(lat=slice(bbox[2], bbox[3]), 
        #                         lon=slice(bbox[0], bbox[1]))
        # data_stacked = data_cropped.stack(z=('lon', 'lat'))
        
        # #extract the values and remove non-TOAR locs
        # data_array = data_stacked.to_array().values
        
        #option2: optionally can run on contributions
        if raw:
            with closing(h5py.File(data_file, 'r')) as f:
                var_names = f['var_names'][:].astype(str)
                data_array = f['X'][:]
        
        if contributions:    
            data = xr.open_dataset(f'{models_dir}/test.contributions.mean.nc')
            data.coords['lon'] = (data.coords['lon'] + 180) % 360 - 180
            data = data.sortby(data.lon)
            var_names = list(data.keys())
            data_cropped = data.sel(lat=slice(bbox[2], bbox[3]), 
                                    lon=slice(bbox[0], bbox[1]))
            data_stacked = data_cropped.stack(z=('lon', 'lat'))
            data_array = data_stacked.to_array().values
            
        
        # with closing(h5py.File(data_file, 'r')) as f:
        #     var_names = f['var_names'][:].astype(str)
        #     data_array = f['X'][:]
        

        #extract the values and remove non-TOAR locs
        counts_nan = np.isnan(data_array).sum(axis = 0)
        mask_locs = counts_nan < len(var_names)
        
        # some variables have all zeros, mask them
        counts_zero = (data_array == 0).sum(axis = 0)
        mask_zero =  counts_zero < len(data_array)
    
        mask_vars = ~np.in1d(np.hstack(var_names), exclude) & mask_zero
        #mask_ = mask_zero & mask_vars
        
        #scale the data
        scaler = StandardScaler()
        data_stand = scaler.fit(data_array).transform(data_array)

        #optional, log transform
        # data_log = data_array.copy()
        # for p in range(data_array.shape[0]):
        #     count_zero = (data_array[p, :] < 0).sum()
        #     if count_zero > 0:
        #         continue
        #     else:
        #         data_log[p, :] = np.log(data_array[p, :]+1)
        

        corr_mat = np.corrcoef(data_stand[:, mask_vars].T)
        var_names = np.hstack(var_names)[mask_vars]
        output_file = f'{summaries_dir}/{month}/var_corr_matrix.h5'
        with closing(h5py.File(output_file, 'w')) as f:
            f['corr_mat'] = corr_mat
            f['var_names'] = var_names.astype(np.string_)   
        
        #clean up variable names to make them shorter
        #labels = np.hstack([x.split('.')[-1] for x in var_names])
        
        #plot full correlation matrix
        #cluster the correlation matrix to see groups better
        X0 = corr_mat.copy()
        X0[np.isnan(X0)] = 0.
        D = pairwise_distances(X0)
        H = sch.linkage(D, method='average')
        d1 = sch.dendrogram(H, no_plot=True)
        idx = d1['leaves']
        X = X0[idx,:][:, idx]
        var_names_X = np.hstack(var_names)[idx]
        
        mc = 0.9
        X2 = X - np.eye(X.shape[0])
        X2_max = np.abs(X2).max(axis = 0)
        mask_corr = X2_max > mc
        labels_mask = np.hstack(var_names_X)[mask_corr]
        
        X2 = X[mask_corr, :][:, mask_corr]
        x, y = np.meshgrid(np.arange(len(X2)), np.arange(len(X2)))
        plt.figure(figsize = (12, 10))
        plt.scatter(x, y, s = np.abs(X2)*10, c = X2, cmap = 'bwr')
        plt.xticks(np.arange(len(labels_mask)), labels_mask, fontsize = 7, rotation = 90);
        plt.yticks(np.arange(len(labels_mask)), labels_mask, fontsize = 7, rotation = 0);
        plt.colorbar()
        plt.tight_layout()
        plt.title(f'momo variable correlations, truncated for {mc} max corr, {month}')
        plt.savefig(f'{summaries_dir}/{month}/variable_corr_matrix.png', 
                    dpi = 150, bbox = 'tight')
        plt.close()
        
        name1 = 'momo.2dsfc.Cl2'
        name2 = 'momo.2dsfc.dflx.hno3'
        name1_idx = np.where(np.in1d(var_names_X, name1))[0]
        name2_idx = np.where(np.in1d(var_names_X, name2))[0]
        plt.figure()
        plt.plot(X[:, name1_idx], X[:, name2_idx], '.')
        
        cluster_ids = sch.fcluster(H, 0.9, criterion="distance")
        idx, cluster_counts = np.unique(cluster_ids, return_counts=True)
        np.where(cluster_counts > 1)
        var_names[cluster_ids == idx[8]]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_dir', type=str, default = '/bias/local/8hr-median/v1/')
    parser.add_argument('--months', default = 'all', nargs = '*', type=str)
    #parser.add_argument('--parameter', type=str, default=None)

    args = parser.parse_args()
    main(**vars(args)) 
    
    
    

#just one variable correlation
# name = 'momo.2dsfc.doxphy'
# def corr_one_var(name, var_names, corr_mat):
#     name_idx = np.where(np.in1d(var_names, name))[0]
#     corr_name = corr_mat[name_idx]
#     corr_name[np.isnan(corr_name)] = 0
#     sort_idx = np.argsort(np.abs(corr_name))[0][::-1]
#     x = np.arange(len(var_names))
    
#     plt.figure(figsize = (18, 7))
#     plt.bar(x, height = corr_name[:,sort_idx].ravel())
#     plt.xticks(x, np.hstack(var_names)[sort_idx], rotation = 90, color = 'k');
#     plt.grid(ls= ':', alpha = 0.5)
#     plt.title(f'{name}')
#     plt.tight_layout()
#     plt.savefig(f'{plots_dir}/correlations_bar_total_{name}.png',
#                  bbox_inches='tight')
#     plt.close()
    
    

# counts = [] 
# for z in range(mask_.sum()):
#     counts.append((data_array[mask_, :][z, :] < 0).sum())
# mask2_ = np.hstack(counts) > 0


# for z in range(mask_.sum()):
    
#     plt.figure()
#     plt.hist(data_array[mask_, :][z, :], bins = 100, 
#              density = True, histtype = 'step');
#     plt.grid(ls=':', alpha = 0.5)
#     plt.title(f'{var_names[z]}')
    
#     a1 = data_array[mask_, :][z, :].mean()
#     a2 = data_array[mask_, :][z, :].std()
#     plt.figure()
#     plt.hist((data_array[mask_, :][z, :] -a1)/a2 , bins = 100, 
#              density = True, histtype = 'step');
#     plt.grid(ls=':', alpha = 0.5)
#     plt.title(f'{var_names[z]}')
   
#     data_log = np.log(data_array[mask_, :][z, :])
    
#     plt.figure()
#     plt.hist(data_log[~np.isinf(data_log)], bins = 100, 
#              density = True, histtype = 'step');
#     plt.grid(ls=':', alpha = 0.5)
#     plt.title(f'{var_names[z]}')
#     plt.savefig(f'{plots_dir}/variables/{var_names[z]}.png', dpi = 150, bbox = 'tight')
#     plt.close()




# from scipy.cluster.hierarchy import ward, fcluster

# X0 = corr_mat.copy()
# X0[np.isnan(X0)] = 0.
# D = 1 - np.abs(X0)
# H = sch.linkage(D, method='complete')

# assign = fcluster(H, t=0.8, criterion='distance')
# order = np.argsort(assign)

# plt.figure(figsize = (12, 10))
# plt.scatter(x, y, s = np.abs(X0)*2, c = X0[:,order], cmap = 'bwr')


# from sklearn.manifold import MDS

# embedding = MDS(n_components=2)
# X_transformed = embedding.fit_transform(X)

# plt.figure()
# plt.plot(X_transformed[:, 0], X_transformed[:, 1], '.')















    