
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
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd

import read_data_toar as read
import match_grid_toar as match



def main(years, months, plotting):
    
    root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
    if not os.path.exists(root_dir):
        root_dir = '/data/MLIA_active_data/data_SUDSAQ/'

    parameter = ['o3']
    
    #all momo input variables
    subdirs = glob.glob(f'{root_dir}/MOMO/inputs/*')
    inputs = [x.split('/')[-1] for x in subdirs]

    years = np.atleast_1d(years)
    #data_output_dir = f'{root_dir}/TOAR2/' 
    for year in years:
        for p in parameter:
            #create summery dp first
            read.main(p, year, months)
            #then, match to momo
            match.main(year, months, p, inputs, plotting)
            
            


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    #parser.add_argument('data_root_dir', type=str)
    parser.add_argument('--years', default = ['2012'], nargs = '*', type=str)
    parser.add_argument('--months', default = [], nargs = '*', type=str)
    #parser.add_argument('--dtype', type=str, default='PRCP')
    parser.add_argument('--plotting', type=bool, default=True)
    #parser.add_argument('out_file', type=str)

    args = parser.parse_args()
    main(**vars(args))
                
            
            