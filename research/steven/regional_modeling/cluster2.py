#!/usr/bin/env python3
#
# Steven Lu
# October 10, 2023

import os
import sys
import numpy as np
import xarray as xr
from region import Dataset
from region import MOMO_V4_VARS_SEL
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


# Variables for making clusters
CLUSTER_VARS = [
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


def main(momo_files, toar_mask_file, n_clusters, out_dir, feature_set):
    for f in momo_files:
        if not os.path.exists(f):
            print(f'Input file does not exist: {f}')
            sys.exit(1)

    # Load in MOMO files
    ds = xr.open_mfdataset(momo_files, engine='netcdf4', lock=False, parallel=True)
    ds = Dataset(ds)

    if feature_set == '15vars':
        ds = ds[CLUSTER_VARS]
    elif feature_set == 'momo_v4':
        ds = ds[MOMO_V4_VARS_SEL]
    else:
        raise Exception('Undefined feature set')

    ds = ds.sortby('lon')
    lat_arr = np.array(ds['lat'])
    lon_arr = np.array(ds['lon'])
    var_list = list()

    # Load in toar binary mask file
    mask = np.load(toar_mask_file)

    for momo_var in list(ds):
        data = np.array(ds[momo_var])

        if np.isnan(data).any():
            print(f'Variable {momo_var} contains nan values. Skip.')
            continue

        var_list.append(np.mean(data, axis=0))
        var_list.append(np.std(data, axis=0))

    var_arr = np.array(var_list)
    data_arr = np.zeros((len(lat_arr) * len(lon_arr), len(var_arr)))
    latlon_arr = np.zeros((len(lat_arr) * len(lon_arr), 2))
    mask_arr = np.zeros(len(lat_arr) * len(lon_arr), dtype=bool)

    data_ind = 0
    for lat_ind, lat_val in enumerate(lat_arr):
        for lon_ind, lon_val in enumerate(lon_arr):
            data_arr[data_ind, :] = var_arr[:, lat_ind, lon_ind]
            latlon_arr[data_ind, :] = np.array([lat_val, lon_val])
            mask_arr[data_ind] = mask[lat_ind, lon_ind]
            data_ind += 1

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    data_arr = scaler.fit_transform(data_arr)

    # Perform clustering
    mbk = MiniBatchKMeans(
        init='k-means++', n_clusters=n_clusters, batch_size=100, n_init=10,
        max_no_improvement=10, verbose=0, random_state=398)
    labels = mbk.fit_predict(data_arr)

    # Set up the layout for the overall cluster map
    cluster_map_file = os.path.join(out_dir, f'cluster_map_k{n_clusters}.png')
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
    colors = iter(cm.tab20(np.linspace(0, 1, n_clusters)))
    for ind, k in enumerate(range(n_clusters)):
        in_group = labels == k
        ax.scatter(latlon_arr[in_group, 1], latlon_arr[in_group, 0],
                   c=next(colors), label=f'Cluster {ind}')

    ax.scatter(latlon_arr[:, 1], latlon_arr[:, 0], mask_arr.astype(int), c='black',
               label='TOAR2 stations')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True,
              shadow=True, ncol=5, fontsize=20)
    plt.savefig(cluster_map_file)
    plt.close()

    cluster_centroids = dict()
    colors = iter(cm.tab20(np.linspace(0, 1, n_clusters)))
    for ind, k in enumerate(range(n_clusters)):
        # Set up the layout for individual cluster maps
        fig, ax = plt.subplots(figsize=(24, 18),
                               subplot_kw={'projection': ccrs.PlateCarree()})
        ax.add_feature(cf.COASTLINE)
        ax.add_feature(cf.BORDERS)
        ax.set_global()
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 15, 'color': 'gray'}
        gl.xlabel_style = {'size': 15, 'color': 'gray'}

        in_group = labels == k
        cluster_mask = mask_arr & in_group
        n_total = np.sum(in_group)
        n_toar = np.sum(cluster_mask)

        cluster_centroids[f'c{k}'] = {
            'n_total': n_total,
            'n_toar': n_toar,
            'centroid': mbk.cluster_centers_[k]
        }
        cluster_data = data_arr[cluster_mask, :]
        cluster_lat = latlon_arr[cluster_mask, 0]
        cluster_lon = latlon_arr[cluster_mask, 1]
        np.save(os.path.join(out_dir, f'cluster_{k}_data.npy'), cluster_data)
        np.save(os.path.join(out_dir, f'cluster_{k}_lat.npy'), cluster_lat)
        np.save(os.path.join(out_dir, f'cluster_{k}_lon.npy'), cluster_lon)
        np.save(os.path.join(out_dir, f'cluster_{k}_mask.npy'), cluster_mask)

        ax.scatter(latlon_arr[in_group, 1], latlon_arr[in_group, 0],
                   c=next(colors), label=f'Cluster {ind}')
        ax.scatter(latlon_arr[cluster_mask, 1], latlon_arr[cluster_mask, 0],
                   c='black', label=f'TOAR2 locations ({n_toar}/{n_total})')
        ax.legend(loc='upper right', fontsize=30)
        plt.savefig(os.path.join(out_dir, f'cluster_{k}.png'))
        plt.close()
    np.savez(os.path.join(out_dir, f'cluster.npz'), **cluster_centroids)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--momo_files', nargs='+', required=True)
    parser.add_argument('--toar_mask_file', type=str, required=True)
    parser.add_argument('--n_clusters', type=int, default=15)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--feature_set', type=str, choices=['15vars', 'momo_v4'])

    args = parser.parse_args()
    main(**vars(args))
