#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 13:10:36 2022

@author: marchett
"""
import os, glob
import numpy as np
import h5py
import xarray as xr
from tqdm import tqdm
from contextlib import closing
from datetime import datetime, timedelta, date
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import scipy.cluster.hierarchy as sch
from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy import stats
import summary_plots as plots
import read_output as read
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
simplefilter("ignore", ClusterWarning)

#root_dir = '/Users/marchett/Documents/SUDS_AQ/analysis_mount/'
#sub_dir = '/bias/local/8hr_median/v1/'

def main(sub_dir, months = 'all', max_corr = 0.9, 
         regions = ['globe', 'north_america', 'europe', 'asia'],
         raw = False):
    
    root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
    if not os.path.exists(root_dir):
        root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
    
    #set plot directory
    summaries_dir = f'{root_dir}/summaries/{sub_dir}/combined_data/' 
    plots_dir = f'{root_dir}/summaries/{sub_dir}/summary_plots/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    if months == 'all':
        months = plots.MONTHS
    key = plots.get_model_type(sub_dir)    
        
    for region in tqdm(regions, desc = 'Computing correlations'):
        bbox = plots.bbox_dict[region]
        # make the full correlation matrix
        exclude = []
        #run correlations for all months
        collect_data = []
        for month in months:
            #option2: optionally can run on contributions
            if raw:
                key = 'variable'
                data_file = f'{summaries_dir}/{month}/data.h5'
                with closing(h5py.File(data_file, 'r')) as f:
                    var_names = f['var_names'][:].astype(str)
                    data_array = f['X'][:]
            
            else:
                key = 'contributions'
                try:
                    files_cont = glob.glob(f'{summaries_dir}/{month}/test.contributions.mean.nc')[0]
                    #files_cont = glob.glob(f'{models_dir}/{month}/*/test.contributions.nc')
                    data = xr.open_dataset(files_cont)
                    #data.coords['lon'] = (data.lon + 180) % 360 - 180
                    #data = data.sortby('lon')
                    var_names = list(data.keys())
                    data_cropped = data.sel(lat = slice(bbox[2], bbox[3]), 
                                            lon = slice(bbox[0], bbox[1]))
                #try:
                    data_stacked = data_cropped.stack(z=('lon', 'lat'))
                    data_array = data_stacked.to_array().values.T
                except:
                    data_array = []
                    continue
                #data_array = data_stacked.to_array().values.T
                
                if not os.path.exists(f'{plots_dir}/contributions'):
                    os.makedirs(f'{plots_dir}/contributions')

            if len(data_array) == 0:
                continue
            # with closing(h5py.File(data_file, 'r')) as f:
            #     var_names = f['var_names'][:].astype(str)
            #     data_array = f['X'][:]
            
            #extract the values and remove non-TOAR locs
            counts_nan = np.isnan(data_array).sum(axis = 1)
            mask_locs = counts_nan > 0
            
            data_array = data_array[~mask_locs,:]
            
            # some variables have all zeros, mask them
            counts_zero = (data_array == 0).sum(axis = 0)
            mask_zero =  counts_zero < len(data_array)
            
            #mask_vars = ~np.in1d(np.hstack(var_names), exclude) & mask_zero
            if mask_zero.sum() < data_array.shape[1]:
                print(f'Warning: some variables have contributions of exactly ZERO.')
                print(f'Total {mask_zero.shape[0]} but un-masked {mask_zero.sum()}.')
                print(f'{region, month}')
            mask_vars = ~np.in1d(np.hstack(var_names), exclude)
            #mask_ = mask_zero & mask_vars
            
            #scale the data
            scaler = StandardScaler()
            data_stand = scaler.fit(data_array).transform(data_array)
            collect_data.append(data_stand[:, mask_vars])
    
            corr_mat = np.corrcoef(data_stand[:, mask_vars].T)
            var_names = np.hstack(var_names)[mask_vars]
            # output_file = f'{summaries_dir}/{month}/{key}_corr_matrix.h5'
            # with closing(h5py.File(output_file, 'w')) as f:
            #     f['corr_mat'] = corr_mat
            #     f['var_names'] = var_names.astype(np.string_)   
            
            
            #clean up variable names to make them shorter
            #labels = np.hstack([x.split('.')[-1] for x in var_names])
            #plot full correlation matrix
            #cluster the correlation matrix to see groups better
            plots.plot_correlations(corr_mat, var_names, f' {key} correlation {region}',  
                                    max_corr = 0.5 if key == 'bias' else 0.7, 
                                    new = False, plot_name = None)
            plt.text(0.87, 1.015, f', {month}', fontsize = 12,
                     bbox=dict(facecolor='none', edgecolor='none'),
                     transform=plt.gca().transAxes)
            plt.savefig(f'{plots_dir}/contributions/{key}_corr_{region}_{month}.png', 
                        dpi = 150, bbox = 'tight')
            plt.close()
            
            # name1 = 'momo.2dsfc.Cl2'
            # name2 = 'momo.2dsfc.dflx.hno3'
            # name1_idx = np.where(np.in1d(var_names_X, name1))[0]
            # name2_idx = np.where(np.in1d(var_names_X, name2))[0]
            # plt.figure()
            # plt.plot(X[:, name1_idx], X[:, name2_idx], '.')
            
            # cluster_ids = sch.fcluster(H, 0.9, criterion="distance")
            # idx, cluster_counts = np.unique(cluster_ids, return_counts=True)
            # np.where(cluster_counts > 1)
            # var_names[cluster_ids == idx[8]]
           
        if len(collect_data) == 0:
            continue 
         
        collect_data = np.row_stack(collect_data)
        corr_mat = np.corrcoef(collect_data.T)
        var_names = np.hstack(var_names)[mask_vars]
        # output_file = f'{summaries_dir}/{month}/{key}_corr_matrix.h5'
        # with closing(h5py.File(output_file, 'w')) as f:
        #     f['corr_mat'] = corr_mat
        #     f['var_names'] = var_names.astype(np.string_)   
      
        plots.plot_correlations(corr_mat, var_names, f' {key} correlation',  
                                max_corr = 0.3 if key == 'bias' else 0.5, 
                                new = False, plot_name = None)
        plt.text(0.87, 1.015, f', {month}', fontsize = 12,
                 bbox=dict(facecolor='none', edgecolor='none'),
                 transform=plt.gca().transAxes)
        plt.savefig(f'{plots_dir}/contributions/{key}_corr_{region}.png', 
                    dpi = 150, bbox = 'tight')
        plt.close()    
        




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















    