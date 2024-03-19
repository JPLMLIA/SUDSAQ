#!/usr/bin/env python3
#
# Steven Lu
# December 4, 2023


import os
import sys
import pickle
import numpy as np
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
from region import Dataset
from region import gen_true_pred_plot
from region import MOMO_V4_VARS_SEL


EXCLUDE_VARS = [
    'momo.2dsfc.BrONO2',
    'momo.2dsfc.BrOX',
    'momo.2dsfc.C5H8',
    'momo.2dsfc.C10H16',
    'momo.2dsfc.CCl4',
    'momo.2dsfc.CFC11',
    'momo.2dsfc.CFC12',
    'momo.2dsfc.CFC113',
    'momo.2dsfc.CH2O',
    'momo.2dsfc.CH3Br',
    'momo.2dsfc.CH3CCl3',
    'momo.2dsfc.CH3Cl',
    'momo.2dsfc.CH3COCH3',
    'momo.2dsfc.CH3COOOH',
    'momo.2dsfc.CH3O2',
    'momo.2dsfc.CH3OH',
    'momo.2dsfc.CH3OOH',
    'momo.2dsfc.ClONO2',
    'momo.2dsfc.H2O2',
    'momo.2dsfc.H1211',
    'momo.2dsfc.H1301',
    'momo.2dsfc.HACET',
    'momo.2dsfc.HCFC22',
    'momo.2dsfc.ISOOH',
    'momo.2dsfc.LR.OY',
    'momo.2dsfc.LR.SO2',
    'momo.2dsfc.MACR',
    'momo.2dsfc.MACROOH',
    'momo.2dsfc.mc.pm25.dust',
    'momo.2dsfc.MGLY',
    'momo.2dsfc.MPAN',
    'momo.2dsfc.OCS',
    'momo.2dsfc.OH',
    'momo.2dsfc.PROD.HOX',
    'momo.evap',
    'momo.mda8',
    'momo.precw',
    'momo.q',
    'momo.slrc',
    'momo.snow'
]


def main(momo_file, toar_file, cluster_mask_files, out_dir):
    for f in [momo_file, toar_file] + cluster_mask_files:
        if not os.path.exists(f):
            print(f'Input file does not exist: {f}')
            sys.exit(1)

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

    # Load in cluster mask files
    mask = np.zeros(160 * 320, dtype=bool)
    for cluster_mask_file in cluster_mask_files:
        mask = mask | np.load(cluster_mask_file)

    mask = np.reshape(mask, (160, 320))
    mask_da = xr.DataArray(mask, dims=['lat', 'lon'],
                           coords={'lat': momo_ds['lat'], 'lon': momo_ds['lon']})

    # Keep only the data points defined by the cluster mask
    momo_ds = momo_ds.where(mask_da, drop=True)
    toar_da = toar_da.where(mask_da, drop=True)

    # Prepare data for training
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

    # Train random forest
    # clf = RandomForestRegressor(n_estimators=150, max_features=50, max_depth=5, min_samples_leaf=1)
    clf = RandomForestRegressor(n_estimators=50, max_features=20, max_depth=15)
    clf.fit(x, y)

    # Make prediction
    pred_y = clf.predict(x)
    out_perf_plot = os.path.join(out_dir, 'perf-train.png')
    gen_true_pred_plot(y, pred_y, out_perf_plot, sub_sample=True)

    # Save trained random forest model
    out_model = os.path.join(out_dir, 'rf.pickle')
    with open(out_model, 'wb') as f:
        pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--momo_file', type=str, required=True)
    parser.add_argument('--toar_file', type=str, required=True)
    parser.add_argument('--cluster_mask_files', nargs='+', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)

    args = parser.parse_args()
    main(**vars(args))
