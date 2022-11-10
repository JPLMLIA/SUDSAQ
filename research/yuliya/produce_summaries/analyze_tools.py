#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:17:26 2022

@author: marchett
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr



#------- get correlations for one variable and barplot
def corr_one_var(month, sub_dir, name, plots_dir = None):
    
    root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
    if not os.path.exists(root_dir):
        root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
    
    #set plot directory
    models_dir = f'{root_dir}/models/{sub_dir}'
    summaries_dir = f'{root_dir}/summaries/{sub_dir}/combined_data/'
    
    data = xr.open_dataset(f'{summaries_dir}/{month}/test.data.mean.nc')
    data.coords['lon'] = (data.coords['lon'] + 180) % 360 - 180
    data = data.sortby(data.lon)
    var_names = list(data.keys())
    
    data_cropped = data.sel(lat=slice(bbox[2], bbox[3]), 
                            lon=slice(bbox[0], bbox[1]))
    data_stacked = data_cropped.stack(z=('lon', 'lat'))

    
    name_idx = np.where(np.in1d(var_names, name))[0]
    corr_name = corr_mat[name_idx]
    corr_name[np.isnan(corr_name)] = 0
    sort_idx = np.argsort(np.abs(corr_name))[0][::-1]
    x = np.arange(len(var_names))
    
    plt.figure(figsize = (18, 7))
    plt.bar(x, height = corr_name[:,sort_idx].ravel())
    plt.xticks(x, np.hstack(var_names)[sort_idx], rotation = 90, color = 'k');
    plt.grid(ls= ':', alpha = 0.5)
    plt.title(f'{name}')
    plt.tight_layout()
    if plots_dir is not None:
        plt.savefig(f'{plots_dir}/correlations_bar_total_{name}.png', bbox_inches='tight')
        plt.close()
    
    return corr_name[:,sort_idx]
    