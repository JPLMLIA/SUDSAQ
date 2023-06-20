#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 12:01:39 2023

@author: marchett
"""
import os, glob
import sys
import numpy as np
import h5py
import xarray as xr
from tqdm import tqdm
from contextlib import closing
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from sklearn.metrics import pairwise_distances
from matplotlib import colors
import pandas as pd
import statsmodels.api as sm


MONTHS = ['dec', 'jan', 'feb', 'mar', 
          'apr', 'may', 'jun', 'jul', 
          'aug', 'sep', 'oct', 'nov'] 

#everything is in -180 to 180 lon
bbox_dict = {'globe':[-180, 180, -90, 90],
            'europe': [-20, 40, 25, 80],
            'asia': [110, 160, 10, 70],
            'australia': [130, 170, -50, -10],
            'north_america': [-140, -50, 10, 80],
            'west_europe': [-20, 10, 25, 80],
            'east_europe': [10, 40, 25, 80],
            'west_na': [-140, -95, 10, 80],
            'east_na': [-95, -50, 10, 80], }



def make_toar_mask(summaries_dir):
    
    files = glob.glob(f'{summaries_dir}/*/test.target.nc')
    
    toar_mask = []
    for m in range(len(files)):
        dat = xr.open_dataset(f'{files[m]}/test.target.nc')
        #dat_ = dat.to_array().stack({'loc': ["lon", "lat"]})
        vals = dat.target.values
        toar_mask.append(np.isnan(vals).sum(axis = 2) < vals.shape[2])
    toar_mask_ = np.dstack(toar_mask).sum(axis = 2)
    
    with closing(h5py.File(f'{summaries_dir}/toar_mask.h5', 'w')) as f:
        f['toar_mask'] = toar_mask_
    
    



    
    
