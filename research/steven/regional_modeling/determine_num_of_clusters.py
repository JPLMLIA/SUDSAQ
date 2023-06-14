#!/usr/bin/env python3
# Determine the optimal number of clusters with the Elbow method
# https://www.analyticsvidhya.com/blog/2021/05/k-mean-getting-the-optimal-number-of-clusters/
#
# Steven Lu
# May 2, 2023

import numpy as np
import xarray as xr
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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


def main(momo_nc_file, out_plot):
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

    sqrt_dist_sum = []
    K = range(1, 50)

    for n_clusters in K:
        mbk = MiniBatchKMeans(
            init='k-means++', n_clusters=n_clusters, batch_size=100, n_init=10,
            max_no_improvement=10, verbose=0)
        mbk.fit(data_arr)
        sqrt_dist_sum.append(mbk.inertia_)

    plt.plot(K, sqrt_dist_sum, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of squared distances')
    plt.savefig(out_plot)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('momo_nc_file', type=str)
    parser.add_argument('out_plot', type=str)

    args = parser.parse_args()
    main(**vars(args))
