#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 11:47:09 2022

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
 
 

year = '2012'
parameter = 'o3'
#data_root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/TOAR2/'


def main(parameter, years, months):
    
    root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
    if not os.path.exists(root_dir):
        root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
    if not os.path.exists(root_dir):
        print('[ERROR] Data root directory does not exist.')
        sys.exit(1)
    
    #data_root_dir = f'{root_dir}/TOAR2/'  
    # if months == 'all':
    #     months = [f'{x}'.zfill(2) for x in np.arange(1, 13)]
    
    # years = np.atleast_1d(years)
    # months = np.atleast_1d(months)
    # for year in years:
    #     for month in months:
    #         dat = get_toar(root_dir, parameter, year, month)
            
            
            
def get_toar_meta(root_dir, parameter, year, month):
    
    parameter = 'o3'
    root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
    if not os.path.exists(root_dir):
        root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
    if not os.path.exists(root_dir):
        print('[ERROR] Data root directory does not exist.')
        sys.exit(1)
    
    data_root_dir = f'{root_dir}/TOAR2/'    
    network_names = [f for f in os.listdir(data_root_dir)
                         if os.path.isdir(os.path.join(data_root_dir, f))]
    
    
    meta_dict = {}
    for i in range(len(network_names)):
        #data_dict.setdefault(network_name, dict())
        network_name = network_names[i]
        station_ids = os.listdir(os.path.join(data_root_dir, network_name))
        for j in tqdm(range(len(station_ids)), desc=network_name):
                
            station_id = station_ids[j]
            data_dir = os.path.join(os.path.abspath(data_root_dir),
                                    '%s/%s' % (network_name, station_id))
    
            # Get the station lat/lon info from downloaded JSON file
            if parameter is None:
                station_files = [f for f in os.listdir(data_dir)
                                 if f.endswith('.json')]
            else:
                station_files = [f for f in os.listdir(data_dir)
                                 if f.endswith('.json') and parameter in f]
    
            if len(station_files) == 0:
                continue
    
            station_file = os.path.join(data_dir, station_files[0])
            with open(station_file, 'r') as f:
                station_data = json.load(f)
    
            if not isinstance(station_data, dict):
                print('[WARN] Unexpected format for %s. Skipped.' %
                      station_file)
                continue
    
            if 'metadata' not in station_data.keys():
                print('[WARN] Station data %s does not contain metadata field. '
                      'Skipped.' % station_file)
                continue
            
            station_metadata = station_data['metadata']
            if 'station_lat' not in station_metadata.keys() or \
                    'station_lon' not in station_metadata.keys():
                print('[WARN] Station metadata %s does not contain station_lat'
                      'or station_lon field. Skipped.' % station_file)
                continue
            
            for key in list(station_metadata):
                if (j in [0,1]) & (i in [0]):
                    meta_dict[key] = [station_metadata[key]]
                else:
                    meta_dict[key].append(station_metadata[key])
                    
                    
    #save
    non_valid_names = ['station_name', 'station_state', 'station_comments', 'station_local_id',
                       'station_country', 'parameter_original_units', 'parameter_calibration',
                       'parameter_contributor', 'comments', 'parameter_pi']
    data_output_dir = f'{root_dir}/processed/summary_dp/TOAR2/'
    ofile = f'toar2_metadata.h5'
    with closing(h5py.File(data_output_dir + ofile, 'w')) as f:
            for key in meta_dict.keys():
                if np.in1d(key, non_valid_names):
                    continue
                if type(meta_dict[key][0]) == str:
                    f[key] = np.hstack(meta_dict[key]).astype(np.string_)
                else:
                    f[key] = np.hstack(meta_dict[key])
    
    # meta_dict = {}        
    # with closing(h5py.File(data_output_dir + ofile, 'r')) as f:
    #         for key in list(f):
    #             meta_dict[key] = f[key][:]
                    
                    
            
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    #parser.add_argument('root_dir', type=str)
    parser.add_argument('--years', default = ['2012'], nargs = '*', type=str)
    parser.add_argument('--months', default = 'all', nargs = '*', type=str)
    parser.add_argument('--parameter', type=str, default=None)
    #parser.add_argument('out_file', type=str)

    args = parser.parse_args()
    main(**vars(args))            
            
            
            
            
            
            
            
            