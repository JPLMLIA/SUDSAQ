#!/usr/bin/env python3
#
# Steven Lu
# October 19, 2023

import os
import sys
import pickle
import numpy as np
import xarray as xr
from region import Dataset
from region import gen_true_pred_plot
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def main(rf_file, data_file, lat_file, lon_file, toar_file, mask_file, out_dir):
    for f in [rf_file, data_file, toar_file, lat_file, lon_file, mask_file]:
        if not os.path.exists(f):
            print(f'Input file does not exist: {f}')
            sys.exit(1)

    # Load in files
    with open(rf_file, 'rb') as f:
        rf = pickle.load(f)
    data_arr = np.load(data_file)
    lat_arr = np.load(lat_file)
    lon_arr = np.load(lon_file)
    mask_arr = np.load(mask_file)
    toar_ds = xr.open_mfdataset(toar_file, engine='netcdf4', lock=False, parallel=True)
    toar_ds = Dataset(toar_ds)
    toar_ds = toar_ds.sortby('lon')
    toar_arr = np.array(toar_ds['toar.mda8.mean'])[0, :, :]
    toar_mda8_mean = np.zeros((toar_arr.shape[0] * toar_arr.shape[1]), dtype=float)

    data_ind = 0
    for lat_ind in range(toar_arr.shape[0]):
        for lon_ind in range(toar_arr.shape[1]):
            toar_mda8_mean[data_ind] = toar_arr[lat_ind, lon_ind]
            data_ind += 1
    toar_mda8_mean = toar_mda8_mean[mask_arr]
    m = ~np.isnan(toar_mda8_mean)
    toar_mda8_mean = toar_mda8_mean[m]
    data_arr = data_arr[m]
    lat_arr = lat_arr[m]
    lon_arr = lon_arr[m]

    if len(data_arr) > 0:
        preds = rf.predict(data_arr)
        out_perf_plot = os.path.join(out_dir, 'pref-test.png')
        gen_true_pred_plot(toar_mda8_mean, preds, out_perf_plot, sub_sample=False)
        preds_map = os.path.join(out_dir, 'preds-map.png')
        gen_map(lat_arr, lon_arr, preds, preds_map, min_v=0, max_v=50)
        diffs_map = os.path.join(out_dir, 'diffs-map.png')
        gen_map(lat_arr, lon_arr, toar_mda8_mean - preds, diffs_map,
                cmap='seismic', min_v=-50, max_v=50, bgcolor=True)

    ground_truth_map = os.path.join(out_dir, 'ground-truth-map.png')
    gen_map(lat_arr, lon_arr, toar_mda8_mean, ground_truth_map, min_v=0, max_v=50)


def gen_map(lat_arr, lon_arr, value_arr, out_file, cmap=None, min_v=None, max_v=None, bgcolor=False):
    fig, ax = plt.subplots(figsize=(24, 18),
                           subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cf.COASTLINE)
    ax.add_feature(cf.BORDERS)
    ax.set_global()
    if bgcolor:
        ax.set_facecolor('gray')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    if cmap is not None:
        plt.scatter(lon_arr, lat_arr, c=value_arr, cmap=cmap)
    else:
        plt.scatter(lon_arr, lat_arr, c=value_arr, cmap='jet')
    plt.colorbar()
    if min_v is not None and max_v is not None:
        plt.clim(min_v, max_v)
    plt.savefig(out_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('rf_file', type=str)
    parser.add_argument('data_file', type=str)
    parser.add_argument('lat_file', type=str)
    parser.add_argument('lon_file', type=str)
    parser.add_argument('toar_file', type=str)
    parser.add_argument('mask_file', type=str)
    parser.add_argument('out_dir', type=str)

    args = parser.parse_args()
    main(**vars(args))

