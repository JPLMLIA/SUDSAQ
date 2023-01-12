#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:28:35 2022

@author: marchett
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 10:05:28 2022

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

import wget
import read_data_isd as isd
import match_grid_isd as match


def main(years, months, plotting):
    
    root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
    if not os.path.exists(root_dir):
        root_dir = '/data/MLIA_active_data/data_SUDSAQ/'
    
    years = np.atleast_1d(years)
    data_output_dir = f'{root_dir}/ISD/GHCND/' 
    for year in years:
        filename = f'{year}.csv.gz'
        #check if data exists
        if not os.path.isfile(data_output_dir + filename):
            #download from ftp 
            ftp_address = 'https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/'
            output_file = f'{data_output_dir}/{filename}' 
            wget.download(ftp_address + filename, out = output_file)
        
        if len(months) == 0:
            months = [f'{x}'.zfill(2) for x in np.arange(1, 13)]
            
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
        
        for dt in datatypeid:
            outdir = f'{root_dir}/processed/summary_dp/ISD/{dt}'
            dt_files = glob.glob(f'{outdir}/*{year}*')
            if len(dt_files) == len(months):
                continue
            print(f'datatype {dt} ----->')
            isd.main(dt, year)
            match.main(year, months, dt, plotting)
     
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    #parser.add_argument('data_root_dir', type=str)
    parser.add_argument('--years', default = ['2012'], nargs = '*', type=str)
    parser.add_argument('--months', default = [], nargs = '*', type=str)
    #parser.add_argument('--dtype', type=str, default='PRCP')
    parser.add_argument('--plotting', type=bool, default=True)
    #parser.add_argument('--read_first', type=bool, default=True)
    #parser.add_argument('out_file', type=str)

    args = parser.parse_args()
    main(**vars(args))
    
    
    