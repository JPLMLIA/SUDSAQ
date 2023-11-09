#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 11:31:45 2023

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
import seaborn as sns


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
        
    for region in regions:
        bbox = plots.bbox_dict[region]
        # make the full correlation matrix
        exclude = []
        #run correlations for all months
        collect_data = []
        for month in months:
            #option2: optionally can run on contributions
            key = 'contributions'
            files_cont = glob.glob(f'{summaries_dir}/{month}/test.contributions.mean.nc')[0]
            #files_cont = glob.glob(f'{models_dir}/{month}/*/test.contributions.nc')
            data = xr.open_dataset(files_cont)
            data.coords['lon'] = (data.lon + 180) % 360 - 180
            data = data.sortby('lon')
            var_names = list(data.keys())
            data_cropped = data.sel(lat = slice(bbox[2], bbox[3]), 
                                    lon = slice(bbox[0], bbox[1]))
            data_stacked = data_cropped.stack(z=('lon', 'lat'))
            data_array = data_stacked.to_array().values.T
            
            if not os.path.exists(f'{plots_dir}/contributions'):
                os.makedirs(f'{plots_dir}/contributions')
            
            plt.figure()    
            ax = sns.swarmplot(data=data_array[0:500,0:10], orient = 'h')
            ax = sns.boxplot(x="day", y="total_bill", data=tips,
                    showcaps=False,boxprops={'facecolor':'None'},
                    showfliers=False,whiskerprops={'linewidth':0})
    
            
            
            
            
            