#!/usr/bin/env python3
#
# Steven Lu
# December 5, 2023

import os
import sys
import pickle
import numpy as np
import xarray as xr
from region import Dataset
from region import MOMO_V4_VARS_SEL
from region import gen_true_pred_plot
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cluster_train_momo_v4 import EXCLUDE_VARS


def main(rf_file, momo_file, toar_file, cluster_mask_file, out_dir):
    for f in [rf_file, momo_file, toar_file, cluster_mask_file]:
        if not os.path.exists(f):
            print(f'Input file does not exist: {f}')
            sys.exit(1)

    # Load in random forest model file
    with open(rf_file, 'rb') as f:
        rf = pickle.load(f)

    # Load in MOMO file
    momo_ds = xr.open_mfdataset(momo_file, engine='netcdf4', lock=False, parallel=True)
    momo_ds = Dataset(momo_ds)
    momo_ds = momo_ds[MOMO_V4_VARS_SEL]
    momo_ds = momo_ds.drop_vars(EXCLUDE_VARS)
    momo_ds = momo_ds.sortby('lon')
    momo_ds.coords['time'] = momo_ds.time.dt.floor('1D')
    momo_ds = momo_ds.groupby('time').mean()

    # Load in TOAR2 file
    toar_ds = xr.open_mfdataset(toar_file, engine='netcdf4', lock=False, parallel=True)
    toar_ds = Dataset(toar_ds)
    toar_da = toar_ds['toar.mda8.mean']
    toar_da = toar_da.sortby('lon')
    toar_da = toar_da.sortby('time')

    # Load in cluster mask file
    mask = np.load(cluster_mask_file)
    mask = np.reshape(mask, (160, 320))
    mask_da = xr.DataArray(mask, dims=['lat', 'lon'],
                           coords={'lat': momo_ds['lat'], 'lon': momo_ds['lon']})

    # Keep only the data points defined by the cluster mask
    momo_ds = momo_ds.where(mask_da, drop=True)
    toar_da = toar_da.where(mask_da, drop=True)

    # Prepare data for making predictions
    x = momo_ds.to_array()
    x = x.stack({'loc': ['time', 'lat', 'lon']})
    x = x.transpose('loc', 'variable')
    y = toar_da.stack({'loc': ['time', 'lat', 'lon']})

    # Remove NaNs
    x = x.where(np.isfinite(x), np.nan)
    x = x.dropna('loc')
    y = y.where(np.isfinite(y), np.nan)
    y = y.dropna('loc')
    x, y = xr.align(x, y, copy=False)

    # Make predictions
    if x.shape[0] == 0:
        print('There are no data available in this cluster. '
              'Try using another cluster.')
        sys.exit()
    pred_y = rf.predict(x)

    # Evaluation
    basename = os.path.splitext(os.path.basename(cluster_mask_file))[0]
    t = basename.split('_')
    out_perf_plot = os.path.join(out_dir, f'{t[0]}_{t[1]}_perf.png')
    gen_true_pred_plot(y, pred_y, out_perf_plot, sub_sample=False)

    # Plot label distribution
    plt.clf()
    plt.hist(y, bins=100, density=True, histtype='step')
    plt.savefig(os.path.join(out_dir, f'{t[0]}_{t[1]}_pdf.png'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('rf_file', type=str)
    parser.add_argument('momo_file', type=str)
    parser.add_argument('toar_file', type=str)
    parser.add_argument('cluster_mask_file', type=str)
    parser.add_argument('out_dir', type=str)

    args = parser.parse_args()
    main(**vars(args))
