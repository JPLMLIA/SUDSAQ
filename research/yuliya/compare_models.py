#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:00:21 2023

@author: marchett
"""
import os, glob
import sys
import numpy as np
import h5py
from tqdm import tqdm
from contextlib import closing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import scipy as sp
from scipy import stats
import xarray as xr
from tqdm import tqdm
from scipy import stats
from treeinterpreter import treeinterpreter as ti
sys.path.insert(0, '/Users/marchett/Documents/SUDS_AQ/analysis_mount/code/suds-air-quality/research/yuliya/produce_summaries')
import summary_plots as plots
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.metrics import pairwise_distances
from sklearn.metrics import r2_score
import pandas as pd

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF, WhiteKernel



sub_dir = '/bias/gattaca.v4.bias-median'
sub_dir_comp = '/bias/gattaca.v5.bias-median'
root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
if not os.path.exists(root_dir):
    root_dir = '/data/MLIA_active_data/data_SUDSAQ/'

#set plot directory
models_dir = f'{root_dir}/models/{sub_dir}'
summaries_dir = f'{root_dir}/summaries/{sub_dir}/combined_data/'
plots_dir = f'{root_dir}/summaries/{sub_dir}/summary_plots/'


month = 'jan'
#testing_dir = f'{research_dir}/{month}/'

save_file = f'{summaries_dir}/{month}/data.h5'

output = {'X': [], 'pred': [], 'truth': [], 'lon': [], 'lat': [], 'years': [], 
          'months': [], 'days': []}
with closing(h5py.File(save_file, 'r')) as f:
    output['X'].append(f['X'][:])
    output['truth'].append(f['y'][:])
    output['pred'].append(f['y0'][:])
    output['lon'].append(f['lons'][:])
    output['lat'].append(f['lats'][:])
    output['years'].append(f['years'][:])
    output['months'].append(f['months'][:])
    output['days'].append(f['days'][:])  
    var_names = f['var_names'][:].astype(str)


comp_dir = f'{root_dir}/summaries/{sub_dir_comp}/combined_data/'
save_file_comp = f'{comp_dir}/{month}/data.h5'

output_comp = {'X': [], 'pred': [], 'truth': [], 'lon': [], 'lat': [], 'years': [], 
          'months': [], 'days': []}
with closing(h5py.File(save_file_comp, 'r')) as f:
    output_comp['X'].append(f['X'][:])
    output_comp['truth'].append(f['y'][:])
    output_comp['pred'].append(f['y0'][:])
    output_comp['lon'].append(f['lons'][:])
    output_comp['lat'].append(f['lats'][:])
    output_comp['years'].append(f['years'][:])
    output_comp['months'].append(f['months'][:])
    output_comp['days'].append(f['days'][:])  
    var_names_comp = f['var_names'][:].astype(str)



plt.figure()
plt.hist(output['truth'][0], bins = 100, density = True, 
         alpha = 0.2, color = '0.5', label = 'truth');
plt.hist(output['pred'][0], histtype = 'step', bins = 100, 
         density = True, label = 'pred-v4');
plt.hist(output_comp['pred'][0], histtype = 'step', bins = 100, 
         density = True, label = 'pred-v5');
plt.grid(ls=':', alpha = 0.5)
plt.legend()
plt.xlim((-50, 55))
plt.title(f'')
plt.text(0.05, 0.9, s=f'rmse-r = {rmse_r}, pve-r = {pve_r}', 
         fontsize = 8, transform = plt.gca().transAxes)
plt.text(0.05, 0.86, s=f'rmse-s = {rmse_s}, pve-s = {pve_s}', 
         fontsize = 8, transform = plt.gca().transAxes)
plt.savefig(f'{testing_dir}/hist_comp_preds.png')
plt.close()





