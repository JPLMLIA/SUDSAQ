#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 13:12:41 2022

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
import matplotlib.pyplot as plt
import pandas as pd
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


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
    if months == 'all':
        months = [f'{x}'.zfill(2) for x in np.arange(1, 13)]
    
    years = np.atleast_1d(years)
    months = np.atleast_1d(months)
    for year in years:
        for month in months:
            dat = get_toar(root_dir, parameter, year, month)
    
    
    
    
def get_toar(root_dir, parameter, year, month):
    
    # root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
    # if not os.path.exists(root_dir):
    #     root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
    # if not os.path.exists(root_dir):
    #     print('[ERROR] Data root directory does not exist.')
    #     sys.exit(1)
    
    data_root_dir = f'{root_dir}/TOAR2/'    
    network_names = [f for f in os.listdir(data_root_dir)
                         if os.path.isdir(os.path.join(data_root_dir, f))]
    
    
    lon_collect = []
    lat_collect = []
    station_collect = []
    network_collect = []
    data_collect = []
    dates_collect = []
    
    meta_names = ['station_alt', 'station_population_density']
    meta_data = {k: [] for k in meta_names}

    for network_name in tqdm(network_names, desc='Getting networks'):
        #data_dict.setdefault(network_name, dict())
        
        station_ids = os.listdir(os.path.join(data_root_dir, network_name))
        for station_id in tqdm(station_ids, desc=network_name):
            
        
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
    
            station_metadata['station_alt']
            if 'dma8epa' not in station_data.keys():
                print('[WARN] Station data %s does not contain ozone EPA 8h mean field. '
                      'Skipped.' % station_file)
                continue
                
            toar_dat = np.hstack(station_data['dma8epa'])
            station_date = station_data['datetime']    
            dates = np.row_stack([a.split(' ')[0].split('-') for a in station_date])
            
                
            mask_year = dates[:, 0] == year
            mask_month = dates[:, 1] == month
            mask = mask_year & mask_month
            dat_monthly = toar_dat[mask]
            
            network_rep = np.repeat(network_name, mask.sum())
            station_lon = np.repeat(station_metadata['station_lon'], mask.sum())
            station_lat = np.repeat(station_metadata['station_lat'], mask.sum())
            station_rep = np.repeat(station_id, mask.sum())
            
            for k in meta_names:
                meta_key = np.repeat(station_metadata[k], mask.sum())
                meta_data[k].append(meta_key)
                
            data_collect.append(dat_monthly)
            network_collect.append(network_rep)
            station_collect.append(station_rep)
            lon_collect.append(station_lon)
            lat_collect.append(station_lat)
            dates_collect.append(dates[mask])
            
    #save
    data_output_dir = f'{root_dir}/processed/summary_dp/TOAR2/'
    ofile = f'toar2_{parameter}_{year}_{month}.h5'
    with closing(h5py.File(data_output_dir + ofile, 'w')) as f:
            f['network'] =  np.hstack(network_collect).astype(np.string_)
            f['station'] =  np.hstack(station_collect).astype(np.string_)
            f['lon'] = np.hstack(lon_collect)
            f['lat'] = np.hstack(lat_collect)
            f['data'] = np.hstack(data_collect)
            f['date'] = np.row_stack(dates_collect).astype(np.string_)
            

            
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


            