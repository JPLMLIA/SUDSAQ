#!/usr/bin/env python3
#
# Steven Lu
# October 19, 2023

import os
import sys
import pickle
import numpy as np
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
from region import Dataset
from region import gen_true_pred_plot


def main(data_file, toar_file, mask_file, out_dir):
    for f in [data_file, toar_file, mask_file]:
        if not os.path.exists(f):
            print(f'Input file does not exist: {f}')
            sys.exit(1)

    # Load in files
    data_arr = np.load(data_file)
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

    clf = RandomForestRegressor()
    clf.fit(data_arr, toar_mda8_mean)
    preds = clf.predict(data_arr)
    out_perf_plot = os.path.join(out_dir, 'perf-train.png')
    gen_true_pred_plot(toar_mda8_mean, preds, out_perf_plot, sub_sample=False)

    out_model_file = os.path.join(out_dir, 'rf.pickle')
    with open(out_model_file, 'wb') as f:
        pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=str)
    parser.add_argument('toar_file', type=str)
    parser.add_argument('mask_file', type=str)
    parser.add_argument('out_dir', type=str)

    args = parser.parse_args()
    main(**vars(args))
