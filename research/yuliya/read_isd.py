#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:50:12 2022

@author: marchett
"""
import requests
import json
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from tqdm import tqdm

def get_noaa_data(url, dtype, header):

    root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
    if not os.path.exists(root_dir):
        root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
        
    
    req_type = 'datasets'
    r = requests.get(url + req_type, headers=dict(token=token))
    datasetlist = r.json()['results']
    
    req_type = 'datatypes'
    payload = {'datasetid': datasetid, 'limit': 1000}
    r = requests.get(url + req_type, params = payload, headers=dict(token=token))
    typeslist = r.json()['results']
    types_ids = [x['id'] for x in typeslist]
    
    datasetid = 'GHCND'
    datatypeid = 'TAVG'
    
    lon_edge = np.arange(-140, -50, 0.5)
    station_ids = []
    station_lon = []
    station_lat = []
    for z in tqdm(range(len(lon_edge)-1)):
        extent = [30, lon_edge[z], 80, lon_edge[z+1]]
        payload = {'datasetid': datasetid, 'datatypeid': datatypeid, 
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
        
    station_ids = np.hstack(station_ids)
    station_lon = np.hstack(station_lon)
    station_lat = np.hstack(station_lat)
    
    
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
   
    
    req_type = 'data'
    for j in range(len(station_ids)):
        stationid = station_ids[j]
        payload = {'datasetid': datasetid, 'stationid': stationid, 'datatypeid': datatypeid,
                   'startdate': '2019-01-01', 'enddate': '2019-12-31',  'limit': 1000}
        r = requests.get(url + req_type, params = payload, headers=dict(token=token))
        
        # if r.status_code != 200:  # Handle erroneous requests
        #     print("Error: " + str(r.status_code))
        # else:
        #     r = r.json()
        if len(r.json()) < 1:
            continue
            
        dat = r.json()['results']
        np.hstack([x['value'] for x in dat])
        np.hstack([x['date'] for x in dat])

        
    
   


if __name__ == '__main__':

    token = 'dtiXGoeTJxKBwqyeJtwZkxuUUKrAGceT'
    header = dict(token=token)
    dtype = 'dataset'
    url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/'

    get_noaa_data(url, dtype, header)
    
    
    


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

    