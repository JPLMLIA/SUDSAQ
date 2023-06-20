#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 09:42:39 2023

@author: marchett
"""
import os, glob, sys
import numpy as np
import h5py
from tqdm import tqdm
from contextlib import closing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib as mp
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from scipy import stats
import xarray as xr
from scipy import stats
from treeinterpreter import treeinterpreter as ti
import statsmodels.api as sm
sys.path.insert(0, '/Users/marchett/Documents/SUDS_AQ/analysis_mount/code/suds-air-quality/research/yuliya/produce_summaries')
import summary_plots as plots
import read_output as read
import summarize_explanations as se
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import scipy.cluster.hierarchy as sch
from sklearn.metrics import pairwise_distances


sub_dir = '/bias/local/8hr_median/v4.1/'
root_dir = '/Volumes/MLIA_active_data/data_SUDSAQ/'
if not os.path.exists(root_dir):
    root_dir = '/data/MLIA_active_data/data_SUDSAQ/'

#set plot directory
models_dir = f'{root_dir}/models/{sub_dir}'
summaries_dir = f'{root_dir}/summaries/{sub_dir}/combined_data/'
plots_dir = f'{root_dir}/summaries/{sub_dir}/summary_plots/'


month = 'jul'

save_file = f'{summaries_dir}/{month}/data.h5'

output = {'X': [], 'pred': [], 'truth': [], 'lon': [], 'lat': [], 'years': [], 
          'months': [], 'days': []}
with closing(h5py.File(save_file, 'r')) as f:
    output['X'].append(f['X'][:])
    output['truth'].append(f['y'][:])
    output['pred'].append(f['y0'][:])
    output['lon'].append(f['lons'][:])
    output['lat'].append(f['lats'][:])
    output['years'].append(f['years'][:])
    output['months'].append(f['months'][:])
    output['days'].append(f['days'][:])  
    var_names = f['var_names'][:].astype(str)


files_x = glob.glob(f'{models_dir}/{month}/*/test.data.nc')
data = xr.open_mfdataset(files_x, parallel=True)


scaler = StandardScaler()
XX = scaler.fit(output['X'][0]).transform(output['X'][0])

bbox = plots.bbox_dict['north_america']
lons = (output['lon'][0] + 180) % 360 - 180
#lons = (output['lon'][0])
lats = output['lat'][0]
mask_lons = (lons > bbox[0]) & (lons < bbox[1])
mask_lats = (lats > bbox[2]) & (lats < bbox[3])
mask_reg = mask_lons & mask_lats 


XX = XX[mask_reg]
y = output['truth'][0][mask_reg] 
years = output['years'][0][mask_reg] 

lon = np.hstack(output['lon'][0][mask_reg])
lat = np.hstack(output['lat'][0][mask_reg])
lon = (lon + 180) % 360 - 180
un_lons, un_lats = np.unique([lon, lat], axis = 1)

for k in output.keys():
    output[k][0] = output[k][0][mask_reg]

i = 10
mask, idx = plots.unique_loc(un_lons[i], un_lats[i], output)

plt.figure()
plt.plot(output['truth'][0][mask])


from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans

coords = np.column_stack([un_lons, un_lats])
kmeans = KMeans(n_clusters=10)
kmeans.fit(coords)


plt.figure()
for l in np.unique(kmeans.labels_):
    idx1 = kmeans.labels_ == l
    plt.plot(coords[idx1, 0], coords[idx1, 1], '.')



from scipy.cluster.hierarchy import fcluster 

rf = RandomForestRegressor(n_estimators = 20,
                            min_samples_leaf=20,
                            max_features = 0.33,
                            oob_score = True,
                            random_state=6789)
rf.fit(output['X'][0], output['truth'][0])
np.sqrt(mean_squared_error(output['truth'][0], rf.oob_prediction_))

new_labels = np.arange(len(coords))
yhat_all = np.zeros_like(output['truth'][0])
rmse_total_total = []
rmse_new = []
K = [340]
dist = 50

sizes = [plots.unique_loc(x[0], x[1], output)[0].sum() for x in coords]
plt.figure()
plt.hist(np.hstack(sizes), bins = 50, histtype = 'step');

for z in range(10):

    print(f'{z} ------->')    
    rmse_total_total = []
    rmse0 = np.zeros((len(coords), len(np.unique(new_labels))))
    for l, label in tqdm(enumerate(np.unique(new_labels))):
        mask_cluster = new_labels == label
        coords_cluster = coords[mask_cluster]
        mask, idx = plots.unique_loc(coords_cluster[:, 0], coords_cluster[:, 1], output)
        if mask.sum() < 10:
            continue
        
        XX = output['X'][0][mask]
        y = output['truth'][0][mask]
        years = output['years'][0][mask] 
        
        X0 = output['X'][0][~mask]
        y0 = output['truth'][0][~mask]
        
        un_years = np.unique(years)
        yhat = np.zeros_like(output['truth'][0])
        #rmse = []
        
        #for i in tqdm(range(len(un_years))):
            # mask_test = years == un_years[i]
            # XX_train = XX[~mask_test,:]
            # y_train = y[~mask_test]
        
            # XX_test = XX[mask_test, :]
            # y_test = y[mask_test]
        
        rf = RandomForestRegressor(n_estimators = 20,
                                    min_samples_leaf=20,
                                    max_features = 0.33,
                                    oob_score = True,
                                    random_state=6789)
        rf.fit(XX, y)
        #idx_test = np.where(mask)[0][mask_test]
        yhat[mask] = rf.oob_prediction_
        
        yhat[~mask] = rf.predict(X0)
        rmse_ext = np.sqrt(mean_squared_error(output['truth'][0][~mask], yhat[~mask]))
        rmse_int = np.sqrt(mean_squared_error(output['truth'][0][mask], yhat[mask]))
        
        yhat_all[mask] = yhat[mask]
        
        rmse_total = np.sqrt(mean_squared_error(output['truth'][0], yhat))
        rmse_total_total.append(rmse_total)
      
        
        #rmse0 = np.zeros(len(coords))
        for i in range(len(coords)):
            mask1, idx1 = plots.unique_loc(coords[i, 0], coords[i, 1], output)
            rmse0[i, l] = np.sqrt(mean_squared_error(output['truth'][0][mask1], yhat[mask1]))
        
        # plt.figure()
        # plt.hist(rmse0[:, l], bins = 40, histtype = 'step', label = f'{l}')
        # plt.legend()
    
    
    rmse_new.append(np.sqrt(mean_squared_error(output['truth'][0], yhat_all)))
    #plt.figure()
    #plt.plot(K, rmse_new)
 
    X0 = rmse0.copy()
    X0[np.isnan(X0)] = 0.
    D = pairwise_distances(X0)
    H = sch.linkage(D, method='average')
    # d1 = sch.dendrogram(H, no_plot=True)
    # idx = d1['leaves']
    # X = X0[idx,:][:, idx]
    
    new_K = K[-1]
    while new_K >= K[-1]:
        out = np.percentile(H[:,2], 99)
        med_trimmed = np.percentile(H[H[:,2] < out,2], 95)
        dist = np.min([med_trimmed, dist + 40])
        clusters = fcluster(H, dist, criterion='distance')
        new_K = len(np.unique(clusters))
    
    # clusters = fcluster(H, dist, criterion='distance')
    # new_K = len(np.unique(clusters))
    # if new_K >= K[-1]:
    #     out = np.percentile(H[:,2], 95)
    #     med_trimmed = np.percentile(H[H[:,2] < out,2], 50)
    #     dist = np.min([med_trimmed, dist + 20])
    #     clusters = fcluster(H, dist, criterion='distance')
    #     new_K = len(np.unique(clusters))
    K.append(new_K)
    print(f'new K: {new_K}')
    
    plt.figure(1)
    plt.hist(H[:,2], bins = 100, histtype = 'step', density = True)
    plt.axvline(dist)
    plt.title(f'{dist}, {new_K}')
    
    cmap = cm.get_cmap('Spectral', len(np.unique(clusters)))
    cmap.set_under("0.8")

    _, idx, n = np.unique(clusters, return_index = True, return_counts = True)
    mask_c = n < 2
    colors = clusters.copy()
    colors[idx[mask_c]] = -1
    plt.figure(figsize = (8,6))
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, 
                edgecolors = 'none', vmin =0, cmap = cmap, marker='s')
    plt.colorbar()
    plt.grid(ls=':', alpha = 0.5)
    plt.xlim((bbox[0], bbox[1]))
    plt.ylim((bbox[2], bbox[3]))
    plt.title(f'{new_K}, {dist}, {np.round(rmse_new[-1],2)}, K > 1: {(np.hstack(n)>1).sum()}')
    
    new_labels = clusters.copy()
    
    # colors = cm.get_cmap('Spectral', len(np.unique(clusters)))
    # for i, l in enumerate(np.unique(clusters)):
    #     #plt.figure()
    #     idx1 = clusters == l
    #     n.append(idx1.sum())
    #     if idx1.sum() > 1:
    #         ec = 'none'
    #         sc = plt.scatter(coords[idx1, 0], coords[idx1, 1], color=colors(i), 
    #                     edgecolors = ec, cmap = colors, marker='s')
    #     else:
    #         ec = 'none'
    #     #cb = plt.scatter(coords[idx1, 0], coords[idx1, 1], color=colors(i), 
    #                 #edgecolors = ec, cmap = 'Spectral', marker='s')
    # plt.xlim((bbox[0], bbox[1]))
    # plt.ylim((bbox[2], bbox[3]))
    # plt.title(f'{new_K}, {dist}, K > 1: {(np.hstack(n)>1).sum()}')

    
    
    # values, base = np.histogram(H[:,2], bins=40);
    # #evaluate the cumulative
    # cumulative = np.cumsum(values)
    # #plt.figure()
    # plt.plot(base[:-1], cumulative, c='blue')
    # plt.axvline(dist)




