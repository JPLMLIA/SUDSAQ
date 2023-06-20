#!/usr/bin/env python3
# Clustering with the following selected MOMOChem variables
# 1.  momo.no
# 2.  momo.no2
# 3.  momo.co
# 4.  momo.ch2o
# 5.  momo.u
# 6.  momo.v
# 7.  momo.t
# 8.  momo.ps
# 9.  momo.2dsfc.NH3
# 10. momo.2dsfc.BrOX
# 11. momo.hno3
# 12. momo.slrc
# 13. momo.2dsfc.HO2
# 14. momo.2dsfc.C2H6
# 15. momo.2dsfc.C5H8
#
# Steven Lu
# May 30, 2023

import numpy as np
import xarray as xr
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# List of MOMOChem variables provided by Kazu
SELECTED_MOMO_VARS = [
    'momo.no',
    'momo.no2',
    'momo.co',
    'momo.ch2o',
    'momo.u',
    'momo.v',
    'momo.t',
    'momo.ps',
    'momo.2dsfc.NH3',
    'momo.2dsfc.BrOX',
    'momo.hno3',
    'momo.slrc',
    'momo.2dsfc.HO2',
    'momo.2dsfc.C2H6',
    'momo.2dsfc.C5H8'
]


def kmeans_plot(data_arr, latlon_arr, n_clusters, out_file):
    mbk = MiniBatchKMeans(
        init='k-means++', n_clusters=n_clusters, batch_size=100, n_init=10,
        max_no_improvement=10, verbose=0)
    cluster_labels = mbk.fit_predict(data_arr)

    fig, ax = plt.subplots(figsize=(24, 18),
                           subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cf.COASTLINE)
    ax.add_feature(cf.BORDERS)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.xlabel_style = {'size': 15, 'color': 'gray'}

    colors = iter(cm.rainbow(np.linspace(0, 1, n_clusters)))

    for ind, k in enumerate(range(n_clusters)):
        in_group = cluster_labels == k
        color = next(colors)
        ax.scatter(latlon_arr[in_group, 1], latlon_arr[in_group, 0],
                   c=color)

    plt.savefig(out_file)


def dbscan_plot(data_arr, latlon_arr, eps, out_file):
    clustering = DBSCAN(eps=eps)
    cluster_labels = clustering.fit_predict(data_arr)
    unique_labels = np.unique(clustering.labels_)
    n_clusters = len(unique_labels)

    fig, ax = plt.subplots(figsize=(24, 18),
                           subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cf.COASTLINE)
    ax.add_feature(cf.BORDERS)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.xlabel_style = {'size': 15, 'color': 'gray'}

    colors = iter(cm.rainbow(np.linspace(0, 1, n_clusters)))

    for ind, k in enumerate(unique_labels):
        if k == -1:
            color = '#000000'  # Black for noise points
            label = 'Noise'
        else:
            color = next(colors)
            label = f'Cluster {ind}'

        in_group = cluster_labels == k
        ax.scatter(latlon_arr[in_group, 1], latlon_arr[in_group, 0],
                   c=color, label=label)

    ax.legend(loc='upper right')
    plt.savefig(out_file)


def main(momo_nc_file, out_file, method, n_clusters, dbscan_eps):
    momo_ds = xr.open_mfdataset(momo_nc_file, parallel=True, lock=False)

    lat_arr = np.array(momo_ds['lat'])
    lon_arr = np.array(momo_ds['lon'])
    var_list = list()

    for momo_var in SELECTED_MOMO_VARS:
        var_data = np.array(momo_ds[momo_var])

        if np.isnan(var_data).any():
            print(f'Variable {momo_var} contains nan values. Skip.')
            continue

        var_mean = np.mean(var_data, axis=0)
        var_std = np.std(var_data, axis=0)
        var_list.append(var_mean)
        var_list.append(var_std)

    var_arr = np.array(var_list)
    data_arr = np.zeros((len(lat_arr) * len(lon_arr), len(var_arr)))
    latlon_arr = np.zeros((len(lat_arr) * len(lon_arr), 2))

    data_ind = 0
    for lat_ind, lat_val in enumerate(lat_arr):
        for lon_ind, lon_val in enumerate(lon_arr):
            data_arr[data_ind, :] = var_arr[:, lat_ind, lon_ind]
            latlon_arr[data_ind, :] = np.array([lat_val, lon_val])
            data_ind += 1

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    data_arr = scaler.fit_transform(data_arr)

    if method.lower() == 'kmeans':
        kmeans_plot(data_arr, latlon_arr, n_clusters, out_file)
    elif method.lower() == 'dbscan':
        dbscan_plot(data_arr, latlon_arr, dbscan_eps, out_file)
    else:
        pass


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('momo_nc_file', type=str)
    parser.add_argument('out_file', type=str)
    parser.add_argument('--method', type=str, choices=['kmeans', 'dbscan'],
                        default='kmeans')
    parser.add_argument('--n_clusters', type=int, default=10,
                        help='Number of clusters for k-means clustering')
    parser.add_argument('--dbscan_eps', type=float, default=0.5)

    args = parser.parse_args()
    main(**vars(args))
