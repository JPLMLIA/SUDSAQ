#!/usr/bin/env python
#
# Steven Lu
# May 9, 2022

import os
import sys
import h5py
import glob
import cv_analysis
import numpy as np
from joblib import dump
from utils import REQUIRED_VARS
from utils import ISD_FEATURES
from utils import TRAIN_FEATURES
from utils import format_data_v2
from utils import format_data_v3
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from treeinterpreter import treeinterpreter as ti


MONTHS_DICT = {
    'jan': '01',
    'feb': '02',
    'mar': '03',
    'apr': '04',
    'may': '05',
    'jun': '06',
    'jul': '07',
    'aug': '08',
    'sep': '09',
    'oct': '10',
    'nov': '11',
    'dec': '12'
}


def main(data_dir, exp_out_dir, year_list, month_list, model_mode, isd_data_dir):
    if not os.path.exists(data_dir):
        print(f'[ERROR] Data dir does not exist: {os.path.abspath(data_dir)}')
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

    data_dict = dict()
    for month in month_list:
        data_dict.setdefault(month, dict())
        data_dict[month].setdefault('momo', list())
        if isd_data_dir is not None:
            data_dict[month].setdefault('isd', list())

        for year in year_list:
            data_file = os.path.join(
                data_dir, 'momo_matched_%s_%s.h5' % (year, MONTHS_DICT[month]))
            data_dict[month]['momo'].append(data_file)

            if not os.path.exists(data_file):
                print(f'[ERROR] Input data file does not exist: {data_file}')
                sys.exit(1)

            if isd_data_dir is not None:
                isd_file = os.path.join(
                    isd_data_dir, 'isd_matched_%s_%s.h5' % (year, MONTHS_DICT[month]))
                data_dict[month]['isd'].append(isd_file)

    if model_mode == 'bias':
        do_bias_model = True
    elif model_mode == 'toar2':
        do_bias_model = False
    else:
        print(f'[ERROR] Unexpected model_mode')
        sys.exit(1)

    for month in data_dict.keys():
        models_month_dir = os.path.join(models_sub_dir, month)
        if not os.path.exists(models_month_dir):
            os.mkdir(models_month_dir)

        preds_month_dir = os.path.join(preds_sub_dir, month)
        if not os.path.exists(preds_month_dir):
            os.mkdir(preds_month_dir)

        plots_month_dir = os.path.join(plots_sub_dir, month)
        if not os.path.exists(plots_month_dir):
            os.mkdir(plots_month_dir)

        data_files = data_dict[month]['momo']
        data_groups = np.arange(len(data_files))
        isd_files = []
        if isd_data_dir is not None:
            isd_files = data_dict[month]['isd']
            x = np.empty((0, len(TRAIN_FEATURES + ISD_FEATURES)),
                         dtype=np.float32)
        else:
            x = np.empty((0, len(TRAIN_FEATURES)), dtype=np.float32)
        y = np.empty((0,), dtype=np.float32)
        mask = np.empty((0,), dtype=bool)
        groups = np.empty((0,), dtype=int)
        lat_list = list()
        lon_list = list()
        date_list = list()

        if len(isd_files) == 0:
            for data_file, data_group in zip(data_files, data_groups):
                data = h5py.File(data_file, 'r')

                for var in REQUIRED_VARS:
                    if var not in list(data.keys()):
                        print(f'[ERROR] Required variable {var} is not '
                              f'included in the input data set.')
                        sys.exit(1)

                data_x, data_y, toar_mask, lat, lon = format_data_v2(
                    data, latitude_min=10, latitude_max=80, longitude_min=-140,
                    longitude_max=-50, bias_format=do_bias_model
                )

                x = np.vstack((x, data_x))
                y = np.hstack((y, data_y))
                mask = np.hstack((mask, toar_mask))
                groups = np.hstack((groups, [data_group] * len(data_y)))
                lat_list.append(lat)
                lon_list.append(lon)
                date_list.append(np.array(data['date']))
        else:
            for data_file, isd_file, data_group in zip(data_files, isd_files, data_groups):
                data = h5py.File(data_file, 'r')
                isd_data = h5py.File(isd_file, 'r')

                data_x, data_y, min_mask, lat, lon = format_data_v3(
                    data, isd_data, latitude_min=10, latitude_max=80,
                    longitude_min=-140, longitude_max=-50, bias_format=do_bias_model
                )

                x = np.vstack((x, data_x))
                y = np.hstack((y, data_y))
                mask = np.hstack((mask, min_mask))
                groups = np.hstack((groups, [data_group] * len(data_y)))
                lat_list.append(lat)
                lon_list.append(lon)
                date_list.append(np.array(data['date']))

        group_kfold = GroupKFold(n_splits=len(data_files))
        for cv_index, (train_index, test_index) in enumerate(group_kfold.split(x, y, groups)):
            train_x, test_x = x[train_index], x[test_index]
            train_y, test_y = y[train_index], y[test_index]
            train_mask, test_mask = mask[train_index], mask[test_index]

            rf_predictor = RandomForestRegressor()
            rf_predictor.fit(train_x[train_mask],
                             train_y[train_mask])

            out_model = os.path.join(models_month_dir,
                                     'rf_%s_cv%d.joblib' % (model_mode, cv_index))
            dump(rf_predictor, out_model)
            print(f'[INFO] Saved trained model: {out_model}')

            pred_y, bias, contribution = ti.predict(rf_predictor, test_x)
            pred_y = pred_y.flatten()

            test_y = test_y.reshape((len(date_list[cv_index]),
                                     len(lat_list[cv_index]),
                                     len(lon_list[cv_index])))
            pred_y = pred_y.reshape((len(date_list[cv_index]),
                                     len(lat_list[cv_index]),
                                     len(lon_list[cv_index])))

            out_pred = os.path.join(preds_month_dir, 'cv%d_preds.h5' % cv_index)
            out_file = h5py.File(out_pred, 'w')
            out_file.create_dataset('true_bias', data=test_y)
            out_file.create_dataset('pred_bias', data=pred_y)
            out_file.create_dataset('lat', data=lat_list[cv_index])
            out_file.create_dataset('lon', data=lon_list[cv_index])
            out_file.create_dataset('date', data=date_list[cv_index])
            out_file.create_dataset('bias', data=bias)
            out_file.create_dataset('contribution', data=contribution)

        cv_analysis.main(glob.glob('%s/*_preds.h5' % preds_month_dir),
                         plots_month_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
                        help='Directory under which the coregistered data sets '
                             'are stored')
    parser.add_argument('exp_out_dir', type=str,
                        help='Experiment output directory')
    parser.add_argument('-yl', '--year_list', type=str, nargs='+',
                        help='The years (in YYYY format) to be included '
                             'in the experiment')
    parser.add_argument('-ml', '--month_list', type=str, nargs='+',
                        choices=list(MONTHS_DICT.keys()),
                        help='The months to be included in the experiment')
    parser.add_argument('-mm', '--model_mode', type=str,
                        choices=['bias', 'toar2'],
                        help='Specify the mode of the ML model. If bias is '
                             'provided, the ML model will be constructed as '
                             'a bias model (i.e., MOMOChem - toar2); if toar2 '
                             'is provided, the ML model will be constructed as '
                             'a model to directly estimate toar2 values')
    parser.add_argument('--isd_data_dir', type=str, required=False, default=None,
                        help='Optional ISD data set for the experiment. If a '
                             'valid ISD data directory is provided, the '
                             'variables in the ISD files will be included in '
                             'the experiment')

    args = parser.parse_args()
    main(**vars(args))
