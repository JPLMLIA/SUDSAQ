#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:54:56 2023

@author: marchett
"""

import os, glob
import sys
import numpy as np
import h5py
import xarray as xr
from tqdm import tqdm
from contextlib import closing
from datetime import datetime, timedelta, date
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from adjustText import adjust_text
sys.path.insert(0, '/home/marchett/code/suds-air-quality/research/yuliya/produce_summaries/')
import summary_plots as plots
import read_output as read

root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
if not os.path.exists(root_dir):
    root_dir = '/data/MLIA_active_data/data_SUDSAQ/'

# #set plot directory
# models_dir = f'{root_dir}/models/{sub_dir}'
# summaries_dir = f'{root_dir}/summaries/{sub_dir}/combined_data/'
# plots_dir = f'{root_dir}/summaries/{sub_dir}/summary_plots/'

#get the mask for the top number a variables
def get_top_mask(a, var1, var2 = None, var3 = None):
    
    metrics_mean = np.stack([v for v in list([var1, var2, var3]) if v is not None])
    #metrics_mean = np.column_stack([var1, var2, var3])
    mean_metrics = np.nanmean(metrics_mean, axis = 0)
    mean_metrics[np.isnan(mean_metrics)] = 0. 
    idx_sort = np.argsort(mean_metrics)[::-1]
    mask_top = idx_sort[:a]
    
    return mask_top


def compare_monthly(m, imp_var1, cont_var1, imp_var2, cont_var2):
    #### ------ compare bias and toar, average model and permute
    a = imp_var1['model']['monthly_mean'][:, m]
    b = imp_var1['permutation']['monthly_mean'][:, m]
    c = cont_var1['monthly_mean'][:, m]
    dat1 = np.mean([a, b, c], axis = 0)
    
    a = imp_var2['model']['monthly_mean'][:, m]
    b = imp_var2['permutation']['monthly_mean'][:, m]
    c = cont_var2['monthly_mean'][:, m]
    dat2 = np.mean([a, b, c], axis = 0)
    
    var2_idx = np.hstack([np.where(imp_var2['labels'] == x)[0] for x in imp_var1['labels']])
    dat2 = dat2[var2_idx]
    
    return dat1, dat2




def main(version, var1, var2):

    
    regions = ['globe', 'Europe', 'Asia', 'NorthAmerica']
    box_version = 'v2'
    
    for region in regions:
        if region == 'globe':    
            if var1 == 'bias':
                var1_dir = f'/bias/gattaca.{version}.bias-median'
            if var1 == 'toar':
                var1_dir = f'/toar/gattaca.{version}.toar'
            if var1 == 'emu':
                var1_dir = f'/emulator/utc/mda8/toar-limited/{version}'
            if var2 == 'bias':
                var2_dir = f'/bias/gattaca.{version}.bias-median'
            if var2 == 'toar':
                var2_dir = f'/toar/gattaca.{version}.toar'
            if var2 == 'emu':
                var2_dir = f'/emulator/utc/mda8/toar-limited/{version}'    
        else:
            if var1 == 'bias':
                var1_dir = f'/bias/gattaca.{version}.bias-median.{region}-{box_version}'
            if var1 == 'toar':
                var1_dir = f'/toar/gattaca.{version}.toar.{region}-{box_version}'
            if var1 == 'emu':
                var1_dir = f'/emulator/gattaca.{version}.mda8.{region}-{box_version}'
            if var2 == 'bias':
                var2_dir = f'/bias/gattaca.{version}.bias-median.{region}-{box_version}'
            if var2 == 'toar':
                var2_dir =  f'/toar/gattaca.{version}.toar.{region}-{box_version}'
            if var2 == 'emu':
                var2_dir = f'/emulator/gattaca.{version}.mda8.{region}-{box_version}' 
                
        ### ---------------- the simplest barplot and bubble plots for top X
        #sub_dir = '/bias/local/8hr_median/v4.1'
        imp_var1 = read.load_importances(f'{root_dir}/summaries/{var1_dir}/combined_data/')
        cont_var1 = read.load_contributions(f'{root_dir}/summaries/{var1_dir}/combined_data/', 
                                            imp_var1['reference'])
        
        # mask_top = get_top_mask(20, var1 = imp_var1['model']['mean'],
        #                             var2 = imp_var1['permutation']['mean'],
        #                             var3 = cont_var1['mean'])
    
    
        imp_var2 = read.load_importances(f'{root_dir}/summaries/{var2_dir}/combined_data/')
        cont_var2 = read.load_contributions(f'{root_dir}/summaries/{var2_dir}/combined_data/', 
                                            imp_var2['reference'])
        
        #### ------ compare bias and toar, average model and permute
        a = imp_var1['model']['mean']
        b = imp_var1['permutation']['mean']
        c = cont_var1['mean']
        dat1 = np.mean([a, b, c], axis = 0)
        
        a = imp_var2['model']['mean']
        b = imp_var2['permutation']['mean']
        c = cont_var2['mean']
        dat2 = np.mean([a, b, c], axis = 0)
        
        # a = imp_bias['model']['mean'] / imp_bias['model']['mean'].max()
        # b = imp_bias['permutation']['mean'] / imp_bias['permutation']['mean'].max()
        # dat2 = np.mean([a, b], axis = 0)
        
        var2_idx = np.hstack([np.where(imp_var2['labels'] == x)[0] for x in imp_var1['labels']])
        dat2 = dat2[var2_idx]
        
        plt.figure(figsize = (10,8))
        plt.plot(dat1, dat2, 's', color = 'r')
        plt.ylim((0, 1.1)); plt.xlim((0, 1.1))
        mask1 = dat2 > 0.18
        mask2 = dat1 > 0.18
        mask = mask1 | mask2
        texts = []
        for i, txt in enumerate(imp_var1['labels'][mask]):
            texts.append(plt.annotate(txt, (dat1[mask][i], dat2[mask][i]), color = '0.0'))
        adjust_text(texts)
        plt.axline((1,1,), slope = 1, color = '0.5', ls = '--')
        plt.grid(ls = ':', alpha = 0.5)
        plt.xlabel(f'{var1}'); plt.ylabel(f'{var2}')
        plt.title(f'driver importance comparisons ({var1} vs {var2}), {region}')
        plt.savefig(f'{root_dir}/summaries/other/{var1}_vs_{var2}_imp_{region}.png', 
                    bbox_inches='tight')
        plt.close()
        
        
        ####------ compare the same monthly
        for m in range(len(plots.MONTHS)):
            dat1, dat2 = compare_monthly(m, imp_var1, cont_var1, imp_var2, cont_var2)
            plt.figure(figsize = (10,8))
            plt.plot(dat1, dat2, 's', color = 'r')
            plt.ylim((0, 1.1)); plt.xlim((0, 1.1))
            mask1 = dat2 > 0.18
            mask2 = dat1 > 0.18
            mask = mask1 | mask2
            texts = []
            for i, txt in enumerate(imp_var1['labels'][mask]):
                texts.append(plt.annotate(txt, (dat1[mask][i], dat2[mask][i]), color = '0.0'))
            adjust_text(texts)
            plt.axline((1,1,), slope = 1, color = '0.5', ls = '--')
            plt.grid(ls = ':', alpha = 0.5)
            plt.xlabel(f'{var1}'); plt.ylabel(f'{var2}')
            plt.title(f'driver importance comparisons ({var1} vs {var2}), {region}, {plots.MONTHS[m]}')
            plt.savefig(f'{root_dir}/summaries/other/{var1}_vs_{var2}_imp_{region}_{plots.MONTHS[m]}.png', 
                        bbox_inches='tight')
            plt.close()
        
        
    
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default = 'v4')
    parser.add_argument('--var1', type=str, default = 'bias')
    parser.add_argument('--var2', type=str, default = 'toar')
    #parser.add_argument('--months', default = 'all', nargs = '*', type=str)
    #parser.add_argument('--parameter', type=str, default=None)

    args = parser.parse_args()
    main(**vars(args)) 

    
    
 