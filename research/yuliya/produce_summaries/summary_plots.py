#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 17:19:54 2022

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


MONTHS = ['dec', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov'] 


#create a simple bar plot combing all importance metrics
def plot_imp_barplot(labels, var1, var2 = None, var3 = None, mask_top = None, plots_dir = None):
    
    if mask_top is None:
        mask_top = np.arange(len(var1[0]))
    
    a = len(mask_top)
    x = np.arange(0, len(var1[0][mask_top])*6)[::6][::-1]
    plt.figure(figsize = (7, len(x)*0.45))
    plt.barh(x, var1[0][mask_top], alpha = 0.6, height = 1.2, color = 'blue', label = 'model')
    eb = plt.errorbar(var1[0][mask_top], x, xerr=var1[1][mask_top], 
                      fmt=".", color="blue", xlolims = True, alpha = 0.5)
    eb[1][0].set_marker('|')
    
    if var2 is not None:
        plt.barh(x+1, var2[0][mask_top], alpha = 0.5, height = 1.2, color = 'orange', label = 'permutation')
        eb = plt.errorbar(var2[0][mask_top], x+1, xerr=var2[1][mask_top], 
                          fmt=".", color="orange", xlolims=True, alpha = 0.5)
        eb[1][0].set_marker('|')

    if var3 is not None:
        plt.barh(x+2, var3[0][mask_top], alpha = 0.5, height = 1.2, color = 'green', label = 'contribution')
        eb = plt.errorbar(var3[0][mask_top], x+2, xerr=var3[1][mask_top], 
                          fmt=".", color="green", xlolims=True, alpha = 0.5)
        eb[1][0].set_marker('|')

    plt.yticks(x, labels[mask_top], rotation = 0, color = 'k');
    plt.grid(ls=':', alpha = 0.5)
    plt.legend()
    plt.title(f'top {a} mean importance with std \n all months')
    plt.tight_layout()
    if plots_dir is not None:
        plt.savefig(f'{plots_dir}/all_baplots_comb{a}_all.png',
                     bbox_inches='tight')
        plt.close()


#create a seasonal bubble plot for all importance metrics
def plot_imp_bubble(labels, var1, key, var2 = None, var3 = None, mask_top = None, plots_dir = None):
    
    if mask_top is None:
        mask_top = np.arange(len(var1)) 
    a = len(mask_top)
    
    metrics = np.stack([v for v in list([var1, var2, var3]) if v is not None])
    #metrics = np.dstack([var1, var2, var3])
    if a < 50:
        scale = 200
    else:
        scale = 50

    mi_monthly_trunc = metrics.mean(axis = 0).T[:, mask_top]
    n = mi_monthly_trunc.shape[1]
    plt.figure(figsize = (a*0.35, 5))
    for z in range(len(mi_monthly_trunc)):
        plt.scatter(x = np.arange(1, n+1), y = np.repeat(z, n), 
                    s = mi_monthly_trunc[z,:]*scale, c = mi_monthly_trunc[z,:], 
                    cmap = 'Reds')
    plt.yticks(np.arange(len(MONTHS)), MONTHS)
    plt.ylabel('month')
    plt.xticks(np.arange(1, len(labels[mask_top])+1), 
               labels[mask_top], rotation = 90)
    plt.xlabel('feature')
    plt.grid(ls=':', alpha = 0.5)
    plt.title(f'motnhly average {key}')
    plt.colorbar()
    plt.tight_layout()
    if plots_dir is not None:
        plt.savefig(f'{plots_dir}/{key}_metrics_bubble_comb{a}.png',
                       bbox_inches='tight')
        plt.close() 



#create a boxplot
def plot_imp_fullbar(labels, var1, key, mask_top= None, plots_dir = None):
    
    if mask_top is None:
        mask_top = np.arange(len(var1)) 
    a = len(mask_top)
    
    plt.figure(figsize = (0.5*a, 5))
    bx = plt.boxplot(list(var1[mask_top]), flierprops=dict(color='0.5', 
                    markersize = 2, markeredgecolor='0.5'));
    # for patch, color in zip(bx['boxes'], colors):
    #     patch.set_color(color)
    plt.axhline(y=0, ls='--', color = 'r', lw = 0.5)
    plt.xticks(np.arange(1, len(labels[mask_top])+1), labels[mask_top], rotation = 90, color = 'k'); 
    plt.ylim((-0.05, 1.02))
    #[plt.gca().get_xticklabels()[x].set_color("red") for x in np.where(box_mask)[0]]
    plt.grid(ls=':', alpha = 0.5)
    plt.title(f'{key} distributions per month+cv, all months')
    plt.tight_layout()
    if plots_dir is not None:
        plt.savefig(f'{plots_dir}/{key}_boxplots_all.png',
                     bbox_inches='tight')
        plt.close()
        
        
    



