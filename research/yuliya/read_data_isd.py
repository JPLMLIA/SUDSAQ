#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:54:39 2022

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
#from datetime import datetime, timedelta, date
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER



def main(datatype, year):
    root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
    if not os.path.exists(root_dir):
        root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
    
    #get stations
    stations_file = f'{root_dir}/ISD/stations_file.txt'
    stations_data = pd.read_csv(stations_file, sep="\s+|;|:", header=None, error_bad_lines=False);
    all_stations = np.array(stations_data[0]).astype(str)
    station_lons =  np.array(stations_data[2])
    station_lats =  np.array(stations_data[1])
    
    #get data
    file = f'{root_dir}/ISD/GHCND/{year}.csv.gz'
    data = pd.read_csv(file, compression='gzip', header = None, error_bad_lines=False)
    station = np.array(data[0]).astype(str)
    dtype = np.array(data[2])
    date = np.array(data[1])
    
    #get data for a type
    mask_dtype = dtype == datatype
    station_dtype = station[mask_dtype]
    unique_station = np.unique(station_dtype)
    
    #add coords
    lons = np.zeros(len(station_dtype))
    lons[:] = np.nan
    lats = np.zeros(len(station_dtype))
    lats[:] = np.nan
    for u in tqdm(range(len(unique_station)), desc = 'Assigning lon/lat'):
        mask_u = all_stations == unique_station[u]
        if mask_u.sum() > 0:
            mask_i = station_dtype == unique_station[u]
            lons[mask_i] = station_lons[mask_u]
            lats[mask_i] = station_lats[mask_u]
     
    
    date_str = date[mask_dtype].astype(str)    
    station_year = np.hstack([''.join(list(a)[0:4]) for a in date_str])
    station_month = np.hstack([''.join(list(a)[4:6]) for a in date_str])
    station_day = np.hstack([''.join(list(a)[6:8]) for a in date_str])
    
    data_dtype = np.array(data[3])[mask_dtype]
    
    months = np.unique(station_month)
    data_output_dir = f'{root_dir}/processed/summary_dp/ISD/{datatype}/'
    if not os.path.exists(data_output_dir):
        os.makedirs(data_output_dir)
    for month in months:
        mask_m = station_month == month  
        ymd = np.column_stack([station_year[mask_m], station_month[mask_m], station_day[mask_m]])
        ofile = f'ghcnd_{year}_{month}.h5'
        with closing(h5py.File(data_output_dir + ofile, 'w')) as f:
                f['station'] =  np.hstack(station_dtype[mask_m]).astype(np.string_)
                f['lon'] = lons[mask_m]
                f['lat'] = lats[mask_m]
                f['data'] = data_dtype[mask_m]
                f['date'] = ymd.astype(np.string_)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--datatype', default = 'PRCP', type = str)
    parser.add_argument('--year', default = '2012', type = str)

    args = parser.parse_args()
    main(**vars(args))






# def get_isd(root_dir, parameter, year, month):
    
#     # root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
#     # if not os.path.exists(root_dir):
#     #     root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
#     # if not os.path.exists(root_dir):
#     #     print('[ERROR] Data root directory does not exist.')
#     #     sys.exit(1)
    
#     dataset_ids = ['TAVG', 'PRCP', 'SNOW', 'SNWD', 'TMAX',
#                   'TMIN', 'ACMC', 'ACMH', 'ACSH', 'AWND',
#                   'FRGB', 'FRGT', 'FRTH', 'MNPN', 'MXPN',
#                   'PSUN', 'THIC', 'TOBS', 'TSUN', 'WDF1',
#                   'WDFI', 'WDFG', 'WDFM', 'WDMV', 'WESD',
#                   'WESF', 'WSF1', 'WSFG', 'WSFI', 'WSFM',
#                   'WT01', 'WT02', 'WT03', 'WT04', 'WT05',
#                   'WT06', 'WT07', 'WT08', 'WT09', 'WT10',
#                   'WT11', 'WT12', 'WT13', 'WT14', 'WT15',
#                   'WT16', 'WT17', 'WT18', 'WT19', 'WT20']
    
#     parameter = 'TAVG'
    
#     data_root_dir = f'{root_dir}/ISD/GHCND/'    
    
#     lon_collect = []
#     lat_collect = []
#     station_collect = []
#     network_collect = []
#     data_collect = []
#     dates_collect = []
    
#     #isd_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
#     station_ids = glob.glob(f'{data_root_dir}*.h5')
#     for station_id in tqdm(station_ids):
        
#         test_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
#         with closing(h5py.File(f'{station_id}', 'r')) as f:
#             dataset_keys = np.hstack(list(f))
#             # dict_lon = f['lon'][:]
#             # dict_lat = f['lat'][:]
#             mask = np.in1d(dataset_keys, ['lon', 'lat', 'elev'])
#             for k in dataset_keys:
#                 if k in dataset_keys[~mask]:
#                     test_dict[k]['values'] = f[k]['values'][:]
#                     test_dict[k]['date'] = f[k]['date'][:]
#                 else:
#                       test_dict[k] = f[k][:]
        
#         keys = np.hstack(test_dict.keys())
#         if parameter in keys:

#             station_dates = test_dict[parameter]['date'].astype(str)
#             dates_split = np.array([x.split('T')[0].split('-') for x in station_dates])
#             mask_year = dates_split[:, 0] == year
#             mask_month = dates_split[:, 1] == month
#             mask = mask_year & mask_month
            
#             dates_collect.append(dates_split[mask])
#             data_collect.append(test_dict[parameter]['values'][mask])
            
#             station_rep = np.repeat(station_id, len(dates_split[mask]))
#             station_collect.append(station_rep)
            
#             station_lon = np.repeat(test_dict['lon'], len(dates_split[mask]))
#             station_lat = np.repeat(test_dict['lat'], len(dates_split[mask]))
#             lon_collect.append(station_lon)
#             lat_collect.append(station_lat)
        
#      #save
#      # data_output_dir = f'{root_dir}/processed/summary_dp/ISD/'
#      # ofile = f'isd_{year}_{month}_{parameter}.h5'
#      # with closing(h5py.File(data_output_dir + ofile, 'w')) as f:
#      #         #f['network'] =  np.hstack(network_collect).astype(np.string_)
#      #         f['station'] =  np.hstack(station_collect).astype(np.string_)
#      #         f['lon'] = np.hstack(lon_collect)
#      #         f['lat'] = np.hstack(lat_collect)
#      #         f['data'] = np.hstack(data_collect)
#      #         f['date'] = np.row_stack(dates_collect).astype(np.string_)    
              
     
    
#     #all avaialable stations
#     station_ids = glob.glob(f'{data_root_dir}*.h5')
#     station_lons = []
#     station_lats = []
#     for station_id in tqdm(station_ids):
#         with closing(h5py.File(f'{station_id}', 'r')) as f:
#             station_lons.append(f['lon'][:])
#             station_lats.append(f['lat'][:])
        
        
#     #montly mean
#     fig = plt.figure(figsize=(18, 9))
#     ax = plt.subplot(projection = ccrs.PlateCarree())
#     #plt.pcolor(x, y, np.nanmean(bias, axis = 0), cmap = 'coolwarm')
#     plt.plot(station_lons, station_lats, 'x')
#     #plt.colorbar()
#     ax.set_global()
#     ax.coastlines()
#     gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                       linewidth=1, color='gray', alpha=0.5, linestyle='--')
#     gl.xformatter = LONGITUDE_FORMATTER
#     gl.yformatter = LATITUDE_FORMATTER
#     ax.stock_img()
#     plt.title(f'ISD stations {parameter}, year = {year}, month = {month}')
#     ax.set_extent([-140, -50, 10, 80], crs=ccrs.PlateCarree())         
              
                
              
                
              
                
              
                      
                      
        