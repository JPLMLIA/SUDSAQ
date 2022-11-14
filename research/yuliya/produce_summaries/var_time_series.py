#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 16:42:53 2022

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
    
    month = 'jul'
    name = 'momo.snow'
    name2 = 'momo.aerosol.nh4'
    data_file = f'{summaries_dir}/{month}/data.h5'     
    with closing(h5py.File(data_file, 'r')) as f:
        var_names = f['var_names'][:].astype(str)
        mask_name = np.where(var_names == name)[0]
        mask_name2 = np.where(var_names == name2)[0]
        data_array = f['X'][:, mask_name]
        data_array2 = f['X'][:, mask_name2]
        days = f['days'][:]
        lats = f['lats'][:]
        lons = f['lons'][:]
        years = f['years'][:]
        
        data = f['X'][:]
    
    boxes = []
    for d in np.unique(days):
        mask = (years == 2011) & (days == d)
        boxes.append(data_array[mask, 0])    
    
    means = np.hstack([x.mean() for x in boxes])
    
    plt.figure()
    plt.plot(data_array, data_array2, '.')
    np.corrcoef(data_array.T, data_array2.T) 

    corr = np.corrcoef(data.T)       
            
    corr[mask_name, mask_name2]        
    
    
    
    
    
    
    
    