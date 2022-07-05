#!/usr/bin/env python
#
# Steven Lu
# May 9, 2022

import os
import sys
import glob
import h5py
import calendar
import sklearn_xarray
import datetime as dt
import numpy as np
import xarray as xr
from joblib import dump
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from treeinterpreter import treeinterpreter as ti


# Note: The feature `mda8` must be placed at the end of this list.
MOMO_FEATURES = [
    '2dsfc/Br2', '2dsfc/BrCl', '2dsfc/BrONO2', '2dsfc/BrOX', '2dsfc/C10H16',
    '2dsfc/C2H5OOH', '2dsfc/C2H6', '2dsfc/C3H6', '2dsfc/C3H7OOH', '2dsfc/C5H8',
    '2dsfc/CCl4', '2dsfc/CFC11', '2dsfc/CFC113', '2dsfc/CFC12', '2dsfc/CH2O',
    '2dsfc/CH3Br', '2dsfc/CH3CCl3', '2dsfc/CH3CHO', '2dsfc/CH3COCH3',
    '2dsfc/CH3COO2', '2dsfc/CH3COOOH', '2dsfc/CH3Cl', '2dsfc/CH3O2',
    '2dsfc/CH3OH', '2dsfc/CH3OOH', '2dsfc/CHBr3', '2dsfc/Cl2', '2dsfc/ClONO2',
    '2dsfc/ClOX', '2dsfc/DCDT/HOX', '2dsfc/DCDT/OY', '2dsfc/DCDT/SO2',
    '2dsfc/DMS', '2dsfc/H1211', '2dsfc/H1301', '2dsfc/H2O2', '2dsfc/HACET',
    '2dsfc/HBr', '2dsfc/HCFC22', '2dsfc/HCl', '2dsfc/HNO3', '2dsfc/HNO4',
    '2dsfc/HO2', '2dsfc/HOBr', '2dsfc/HOCl', '2dsfc/HOROOH', '2dsfc/ISON',
    '2dsfc/ISOOH', '2dsfc/LR/HOX', '2dsfc/LR/OY', '2dsfc/LR/SO2', '2dsfc/MACR',
    '2dsfc/MACROOH', '2dsfc/MGLY', '2dsfc/MPAN', '2dsfc/N2O5', '2dsfc/NALD',
    '2dsfc/NH3', '2dsfc/NH4', '2dsfc/OCS', '2dsfc/OH', '2dsfc/ONMV', '2dsfc/PAN',
    '2dsfc/PROD/HOX', '2dsfc/PROD/OY', '2dsfc/SO2', '2dsfc/SO4', '2dsfc/dflx/bc',
    '2dsfc/dflx/dust', '2dsfc/dflx/hno3', '2dsfc/dflx/nh3', '2dsfc/dflx/nh4',
    '2dsfc/dflx/oc', '2dsfc/dflx/salt', '2dsfc/dms', '2dsfc/doxdyn',
    '2dsfc/doxphy', '2dsfc/mc/bc', '2dsfc/mc/dust', '2dsfc/mc/nh4',
    '2dsfc/mc/nitr', '2dsfc/mc/oc', '2dsfc/mc/pm25/dust', '2dsfc/mc/pm25/salt',
    '2dsfc/mc/salt', '2dsfc/mc/sulf', '2dsfc/taut', 'T2', 'ccover', 'ccoverh',
    'ccoverl', 'ccoverm', 'cumf', 'cumf0', 'dqcum', 'dqdad', 'dqdyn', 'dqlsc',
    'dqvdf', 'dtcum', 'dtdad', 'dtdyn', 'dtlsc', 'dtradl', 'dtrads', 'dtvdf',
    'evap', 'olr', 'olrc', 'osr', 'osrc', 'prcp', 'prcpc', 'prcpl', 'precw',
    'q2', 'sens', 'slrc', 'slrdc', 'snow', 'ssrc', 'taugxs', 'taugys', 'taux',
    'tauy', 'twpc', 'u10', 'uvabs', 'v10', 'aerosol/nh4', 'aerosol/no3',
    'aerosol/sul', 'ch2o', 'co', 'hno3', 'oh', 'pan', 'ps', 'q', 'so2', 't',
    'u', 'v', 'mda8'
]

TOAR_FEATURES = ['toar/o3/dma8epa/mean']


def get_days(year_list, month):
    all_days = []
    for year in year_list:
        n_days = calendar.monthrange(year, month)[1]
        days = [dt.date(year, month, day).strftime('%Y-%m-%d')
                for day in range(1, n_days+1)]

        # If a leap year is included in the year_list, we remove 2/29.
        if len(days) == 29:
            days = days[:-1]

        all_days.extend(days)

    return all_days


def create_data_groups(momo_ns, year_list):
    groups = np.empty((0,), dtype=int)
    lat_dim = momo_ns['lat'].shape[0]
    lon_dim = momo_ns['lon'].shape[0]
    for year in momo_ns['time.year'].values:
        index = year_list.index(year)
        groups = np.hstack((groups, [index] * lat_dim * lon_dim))

    return groups


def create_mask(momo_arr, toar_arr):
    toar_mask = ~np.isnan(toar_arr)
    momo_mask = [True] * len(momo_arr)
    for ind in range(len(momo_arr[0])):
        mask = ~np.isnan(momo_arr[:, ind])
        momo_mask = np.logical_and(momo_mask, mask)

    full_mask = np.logical_and(toar_mask, momo_mask)

    return full_mask


def reconstruct_array(arr, mask):
    if len(arr.shape) == 1:
        reconstructed_arr = np.ones(mask.shape)
        reconstructed_arr[mask == False] = np.nan
        reconstructed_arr[mask == True] = arr
    elif len(arr.shape) == 2:
        n_row, n_col = arr.shape
        reconstructed_arr = np.ones((mask.shape[0], n_col))
        reconstructed_arr[mask == False] = [np.nan] * n_col
        reconstructed_arr[mask == True] = arr
    else:
        print(f'Does not support reconstruct arrays with more than 2 dimensions')
        sys.exit(1)

    return reconstructed_arr


def main(momo_dir, toar_dir, exp_out_dir, year_list, month_list, model_mode):
    if not os.path.exists(momo_dir):
        print(f'[ERROR] momo data dir does not exist: '
              f'{os.path.abspath(momo_dir)}')
        sys.exit(1)

    if not os.path.exists(toar_dir):
        print(f'[ERROR] toar data dir does not exist: '
              f'{os.path.abspath(toar_dir)}')
        sys.exit(1)

    if not os.path.exists(exp_out_dir):
        os.mkdir(exp_out_dir)
        print(f'[INFO] Created experiment output dir: '
              f'{os.path.abspath(exp_out_dir)}')

    models_sub_dir = os.path.join(exp_out_dir, 'models')
    if not os.path.exists(models_sub_dir):
        os.mkdir(models_sub_dir)

    preds_sub_dir = os.path.join(exp_out_dir, 'preds')
    if not os.path.exists(preds_sub_dir):
        os.mkdir(preds_sub_dir)

    plots_sub_dir = os.path.join(exp_out_dir, 'plots')
    if not os.path.exists(plots_sub_dir):
        os.mkdir(plots_sub_dir)

    for month in month_list:
        models_month_dir = os.path.join(models_sub_dir, str(month))
        if not os.path.exists(models_month_dir):
            os.mkdir(models_month_dir)

        preds_month_dir = os.path.join(preds_sub_dir, str(month))
        if not os.path.exists(preds_month_dir):
            os.mkdir(preds_month_dir)

        plots_month_dir = os.path.join(plots_sub_dir, str(month))
        if not os.path.exists(plots_month_dir):
            os.mkdir(plots_month_dir)

        # Load MOMO data
        momo_files = glob.glob('%s/**/%02d.nc' % (momo_dir, month))
        for momo_file in momo_files:
            momo_year = int(momo_file.split('/')[-2])
            if momo_year not in year_list:
                momo_files.remove(momo_file)
        momo_ds = xr.open_mfdataset(momo_files, engine='scipy', parallel=True)
        momo_ns = momo_ds.sel(time=get_days(year_list, month))
        momo_ns = momo_ns[MOMO_FEATURES]
        momo_ns.load()

        # Load TOAR2 data
        toar_files = glob.glob('%s/**/%02d.nc' % (toar_dir, month))
        toar_ds = xr.open_mfdataset(toar_files, engine='scipy', parallel=True)
        toar_ns = toar_ds.sel(time=get_days(year_list, month))
        toar_ns = toar_ns[TOAR_FEATURES]
        toar_ns.load()

        # Convert xarray datasets to numpy arrays
        momo_arr = sklearn_xarray.utils.convert_to_ndarray(momo_ns)
        toar_arr = sklearn_xarray.utils.convert_to_ndarray(toar_ns)
        time_dim, lat_dim, lon_dim, var_dim = momo_arr.shape
        momo_arr = momo_arr.reshape((time_dim * lat_dim * lon_dim, var_dim))
        toar_arr = toar_arr.reshape((time_dim * lat_dim * lon_dim))

        # Create data groups for k-fold cross validation
        data_groups = create_data_groups(momo_ns, year_list)

        # Prepare data for training
        x = momo_arr[:, :-1]
        if model_mode == 'bias':
            y = momo_arr[:, -1].flatten() - toar_arr
        elif model_mode == 'toar2':
            y = toar_arr
        else:
            y = momo_arr[:, -1].flatten()

        group_kfold = GroupKFold(n_splits=len(year_list))
        for cv_ind, (train_ind, test_ind) in enumerate(group_kfold.split(x, y, data_groups)):
            train_x, test_x = x[train_ind], x[test_ind]
            train_y, test_y = y[train_ind], y[test_ind]

            train_mask = create_mask(train_x, train_y)
            test_mask = create_mask(test_x, test_y)

            rf = RandomForestRegressor()
            rf.fit(train_x[train_mask], train_y[train_mask])

            out_model = os.path.join(models_month_dir,
                                     'rf_%s_cv%d.joblib' % (model_mode, cv_ind))
            dump(rf, out_model)
            print(f'[INFO] Saved trained model: {out_model}')

            pred_y, bias, contribution = ti.predict(rf, test_x[test_mask])

            # Insert nan values back so that these arrays can be reshaped into
            # the shape that matches (time, lat, lon)
            pred_y = pred_y.flatten()
            pred_y = reconstruct_array(pred_y, test_mask)
            bias = reconstruct_array(bias, test_mask)
            contribution = reconstruct_array(contribution, test_mask)

            test_y = test_y.reshape((time_dim // len(year_list), lat_dim, lon_dim))
            pred_y = pred_y.reshape((time_dim // len(year_list), lat_dim, lon_dim))

            out_pred = os.path.join(preds_month_dir, 'cv%d_preds.h5' % cv_ind)
            out_file = h5py.File(out_pred, 'w')
            out_file.create_dataset('truth', data=test_y)
            out_file.create_dataset('prediction', data=pred_y)
            out_file.create_dataset('lat', data=momo_ns['lat'].to_numpy())
            out_file.create_dataset('lon', data=momo_ns['lon'].to_numpy())
            out_file.create_dataset('bias', data=bias)
            out_file.create_dataset('contribution', data=contribution)

            year = year_list[cv_ind]
            n_days = calendar.monthrange(year, month)[1]
            date = np.array([[year, month, day] for day in range(1, n_days + 1)])
            out_file.create_dataset('date', data=np.array(date))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('momo_dir', type=str)
    parser.add_argument('toar_dir', type=str)
    parser.add_argument('exp_out_dir', type=str,
                        help='Experiment output directory')
    parser.add_argument('-yl', '--year_list', type=int, nargs='+',
                        help='The years (in YYYY format) to be included '
                             'in the experiment')
    parser.add_argument('-ml', '--month_list', type=int, nargs='+',
                        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                        help='The months to be included in the experiment')
    parser.add_argument('-mm', '--model_mode', type=str,
                        choices=['bias', 'toar2', 'momo'])

    args = parser.parse_args()
    main(**vars(args))
