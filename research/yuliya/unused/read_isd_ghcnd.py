#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:50:12 2022

@author: marchett
"""
import requests
import json, os
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from tqdm import tqdm
import numpy as np
from contextlib import closing
import h5py
from collections import defaultdict
import time
import pandas as pd




# file = '/Users/marchett/Desktop/isd_2012/725280-14733-2012'
# import isd.io
# data_frame = isd.io.read_to_data_frame(file)
# data_frame.columns
# data_frame['wind_speed']


import shutil
import urllib.request as request
from contextlib import closing

ftp_link = 'https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/'
file = '2012.csv.gz'
file = 'ghcnd-stations.txt'

stations_file = '/Users/marchett/Documents/SUDS_AQ/stations_file.txt'
with closing(request.urlopen(f'{ftp_link}{file}')) as r:
    with open(stations_file, 'wb') as f:
        shutil.copyfileobj(r, f)
        
        



def main(lon_min, lon_max, d = 0.5):

    root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
    if not os.path.exists(root_dir):
        root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
    
    url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/'
    token = 'dtiXGoeTJxKBwqyeJtwZkxuUUKrAGceT'
    
    #lon_edge = np.arange(-140, -50, 0.5)
    lon_edge = np.arange(lon_min, lon_max, d)
    datasetid = 'GHCND'
    
    # req_type = 'datasets'
    # r = requests.get(url + req_type, headers=dict(token=token))
    # datasetlist = r.json()['results']
    # req_type = 'datatypes'
    # payload = {'datasetid': datasetid, 'limit': 1000}
    # r = requests.get(url + req_type, params = payload, headers=dict(token=token))
    # datasets = r.json()['results']
    # datasets = [x['id'] for x in datasets]
   
    datatypeid = ['TAVG', 'PRCP', 'SNOW', 'SNWD', 'TMAX',
                  'TMIN', 'ACMC', 'ACMH', 'ACSH', 'AWND',
                  'FRGB', 'FRGT', 'FRTH', 'MNPN', 'MXPN',
                  'PSUN', 'THIC', 'TOBS', 'TSUN', 'WDF1',
                  'WDFI', 'WDFG', 'WDFM', 'WDMV', 'WESD',
                  'WESF', 'WSF1', 'WSFG', 'WSFI', 'WSFM',
                  'WT01', 'WT02', 'WT03', 'WT04', 'WT05',
                  'WT06', 'WT07', 'WT08', 'WT09', 'WT10',
                  'WT11', 'WT12', 'WT13', 'WT14', 'WT15',
                  'WT16', 'WT17', 'WT18', 'WT19', 'WT20']

   
    #years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
    years = ['2012']
        
    #dat_dict = defaultdict(lambda: defaultdict(list))
    #dat_dict = defaultdict(defaultdict(dict))
    for z in tqdm(range(12, len(lon_edge)-1)):  
        extent = [30, lon_edge[z], 60, lon_edge[z+1]]  
        dat_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for t in tqdm(range(len(datatypeid))):
            print(f'dataset ----> {datatypeid[t]}')
            dtype = datatypeid[t]
            for year in years:
                print(f'year --> {year}')
                req_type = 'stations'
                payload = {'datasetid': datasetid, 
                           'datatypeid': dtype,
                           'startdate': f'{year}-01-01', 'enddate': f'{year}-12-31',  
                           'extent': extent, 'limit': 1000}
                
                time.sleep(5)
                r = requests.get(url + req_type, params = payload, headers=dict(token=token))
                
                
                if r.status_code != 200: 
                    continue
                
                if len(r.json()) < 1:
                    continue
                
                results = r.json()['results']
                
                keys_results = [list(x) for x in results]
                mask_elev = np.hstack(['elevation' not in x for x in keys_results])
                
                
                station_ids = np.hstack([x['id'] for x in results])
                station_lon = np.hstack([x['longitude'] for x in results])
                station_lat = np.hstack([x['latitude'] for x in results])
                
                #some stations miss elevation
                if mask_elev.sum() > 0:
                    station_elev = np.zeros(len(results), )
                    
                    for xi in range(len(results)):
                        if mask_elev[xi]:
                            station_elev[xi] = np.nan
                        else:
                            station_elev[xi] = results[xi]['elevation']
                else:    
                    station_elev = np.hstack([x['elevation'] for x in results]) 

                
                #print(f'region {z} --> stations {len(station_ids)}')
                for s in range(len(station_ids)):
                    station = station_ids[s]
                    req_type = 'data'
                    payload = {'datasetid': datasetid, 'stationid': station_ids[s], 
                               'datatypeid': datatypeid[t],
                               'startdate': f'{year}-01-01', 'enddate': f'{year}-12-31',  
                               'extent': extent, 'limit': 1000}
                    
                    time.sleep(5)
                    r = requests.get(url + req_type, params = payload, headers=dict(token=token))
                    
                    if r.status_code != 200: 
                        continue
                    
                    if len(r.json()) < 1:
                        continue
                    
                    results = r.json()['results']
                    
                    vals = np.hstack([x['value'] for x in results])
                    dates = np.hstack([x['date'] for x in results])
                    dat_dict[station][dtype]['values'].append(vals)
                    dat_dict[station][dtype]['date'].append(dates)
                    dat_dict[station]['lon'] = np.atleast_1d(station_lon[s])
                    dat_dict[station]['lat'] = np.atleast_1d(station_lat[s])
                    dat_dict[station]['elev'] = np.atleast_1d(station_elev[s])
        
        station_keys = list(dat_dict)
        for k in station_keys:
            name = k.split(':')[-1]
            ofile = f'{name}_{year}.h5'
            with closing(h5py.File(f'{root_dir}/ISD/{datasetid}/{ofile}', 'w')) as f:
                f['lon'] = dat_dict[k]['lon']
                f['lat'] = dat_dict[k]['lat']
                f['elev'] = dat_dict[k]['elev']
                dataset_keys = np.hstack(list(dat_dict[k]))
                mask = np.in1d(dataset_keys, ['lon', 'lat', 'elev'])
                for subk in dataset_keys[~mask]:
                    f[f'{subk}/values'] = np.hstack(dat_dict[k][subk]['values'])  
                    f[f'{subk}/date'] = np.hstack(dat_dict[k][subk]['date']).astype(np.string_)
   
       # test_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
       # with closing(h5py.File(f'{root_dir}/ISD/{datasetid}/{ofile}', 'r')) as f:
       #     dataset_keys = np.hstack(list(f))
       #     mask = np.in1d(dataset_keys, ['lon', 'lat', 'elev'])
       #     for k in dataset_keys:
       #         if k in dataset_keys[~mask]:
       #             test_dict[k]['values'] = f[k]['values'][:]
       #             test_dict[k]['date'] = f[k]['date'][:]
       #         else:
       #              test_dict[k] = f[k][:]
                
                
                
                

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lon_min', default = -140)
    parser.add_argument('--lon_max', default = -50)
    parser.add_argument('--d', default = 0.5)
    #parser.add_argument('out_file', type=str)

    args = parser.parse_args()
    main(**vars(args))
    
    


def get_stations_isd():
   #get all the stations per region
   lon_edge = np.arange(-140, -50, 0.5)
   station_ids = []
   station_lon = []
   station_lat = []
   stations_elev = []
   for z in tqdm(range(len(lon_edge)-1)):
       extent = [30, lon_edge[z], 80, lon_edge[z+1]]
       payload = {'datasetid': datasetid, 'datatypeid': datatypeid[0], 
              'extent': extent, 'limit': 1000}
       req_type = 'stations'
       r = requests.get(url + req_type, params = payload, headers=dict(token=token))
       if len(r.json()) == 0:
           continue
       
       results = r.json()['results']
       if len(results) >= 1000:
           #print(f'number of stations > 1000 at {z}')
           break
       for i in range(len(results)):
           station_ids.append(results[i]['id'])
           station_lon.append(results[i]['longitude'])
           station_lat.append(results[i]['latitude'])
           station_elev.append(results[i]['elevation'])
       
   station_ids = np.hstack(station_ids)
   station_lon = np.hstack(station_lon)
   station_lat = np.hstack(station_lat)
   station_elev = np.hstack(station_elev)
   
   #plot station locations
   fig = plt.figure(figsize=(18, 9))
   ax = plt.subplot(projection = ccrs.PlateCarree())
   plt.plot(station_lon, station_lat, 'x', ms = 0.5)
   ax.set_global()
   ax.coastlines()
   gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                     linewidth=1, color='gray', alpha=0.5, linestyle='--')
   gl.xformatter = LONGITUDE_FORMATTER
   gl.yformatter = LATITUDE_FORMATTER
   ax.stock_img()
   plt.title(f'isd stations reporting {datatypeid}')
   ax.set_extent([-140, -50, 10, 80], crs=ccrs.PlateCarree())
   plt.savefig(f'{root_dir}/ISD/isd_tavg_stations.png', 
               bbox_inches = 'tight')
   plt.close()
  
   
   ofile = f'isd_stations_bbox.h5'
   with closing(h5py.File(f'{root_dir}/ISD/{ofile}', 'w')) as f:
       f['lon'] = station_lon
       f['lat'] = station_lat
       f['id'] = station_ids
       f['elev'] = station_elev
   
    
   

class NOAAData(object):
    def __init__(self, token):
        # NOAA API Endpoint
        self.url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/'
        self.h = dict(token=token)

    def poll_api(self, req_type, payload):
        # Initiate http request - kwargs are constructed into a dict and passed as optional parameters
        # Ex (limit=100, sortorder='desc', startdate='1970-10-03', etc)
        r = requests.get(self.url + req_type, headers=self.h, params=payload)

        if r.status_code != 200:  # Handle erroneous requests
            print("Error: " + str(r.status_code))
        else:
            r = r.json()
            try:
                return r['results']  # Most JSON results are nested under 'results' key
            except KeyError:
                return r  # for non-nested results, return the entire JSON string

    # Fetch available datasets
    # http://www.ncdc.noaa.gov/cdo-web/webservices/v2#datasets
    def datasets(self, **kwargs):
        req_type = 'datasets'
        return self.poll_api(req_type, kwargs)

    # Fetch data categories
    # http://www.ncdc.noaa.gov/cdo-web/webservices/v2#dataCategories
    def data_categories(self, **kwargs):
        req_type = 'datacategories'
        return self.poll_api(req_type, kwargs)

    # Fetch data types
    # http://www.ncdc.noaa.gov/cdo-web/webservices/v2#dataTypes
    def data_types(self, **kwargs):
        req_type = 'datatypes'
        return self.poll_api(req_type, kwargs)

    # Fetch available location categories
    # http://www.ncdc.noaa.gov/cdo-web/webservices/v2#locationCategories
    def location_categories(self, **kwargs):
        req_type = 'locationcategories'
        return self.poll_api(req_type, kwargs)

    # Fetch all available locations
    # http://www.ncdc.noaa.gov/cdo-web/webservices/v2#locations
    def locations(self, **kwargs):
        req_type = 'locations'
        return self.poll_api(req_type, kwargs)

    # Fetch All available stations
    # http://www.ncdc.noaa.gov/cdo-web/webservices/v2#stations
    def stations(self, h, p, **kwargs):
        req_type = 'stations'
        return self.poll_api(req_type, kwargs)

    # Fetch information about specific dataset
    def dataset_spec(self, set_code, **kwargs):
        req_type = 'datacategories/' + set_code
        return self.poll_api(req_type, kwargs)

    # Fetch data
    # http://www.ncdc.noaa.gov/cdo-web/webservices/v2#data
    def fetch_data(self, **kwargs):
        req_type = 'data'
        return self.poll_api(req_type, kwargs)

    