#!/usr/bin/env python3
# Perform PCA dimensionality reduction
#
# Steven Lu
# April 11, 2023

import os
import numpy as np
import xarray as xr
from sklearn.decomposition import PCA


def main(momo_nc_file, out_dir):
    momo_ds = xr.open_mfdataset(momo_nc_file, parallel=True, lock=False)

    # Work with North American regional
    na_ds = momo_ds.sel(lat=slice(10, 80), lon=slice(220, 310))

    momo_vars = list(na_ds.keys())
    lat_arr = np.array(na_ds['lat'])
    lon_arr = np.array(na_ds['lon'])
    var_list = list()

    for momo_var in momo_vars:
        var_data = np.array(na_ds[momo_var])

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

    # Run PCA to reduce dimensionality
    pca = PCA(n_components=10)
    reduced_data = pca.fit_transform(data_arr)

    # Save reduced data, lat, and lon arrays
    data_file = os.path.join(out_dir, 'pca_reduced_data.npy')
    latlon_file = os.path.join(out_dir, 'latlon.npy')
    np.save(data_file, reduced_data)
    np.save(latlon_file, latlon_arr)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('momo_nc_file', type=str)
    parser.add_argument('out_dir', type=str)

    args = parser.parse_args()
    main(**vars(args))
