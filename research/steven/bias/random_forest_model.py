#!/usr/bin/env python
# Train Random Forest model for the MOMO-TOAR bias predictor
# Example input data for this script can be found in the following directory on
# MLIA machines.
# /data/MLIA_active_data/data_SUDSAQ/processed/coregistered/
#
# Steven Lu
# April 12, 2022

import os
import sys
import h5py
import glob
import numpy as np
from joblib import dump
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from utils import REQUIRED_VARS
from utils import TRAIN_FEATURES
from utils import format_data
from utils import gen_true_pred_plot


def main(in_dir, out_dir, hyperparameter_tuning):
    if not os.path.exists(in_dir):
        print(f'[ERROR] Input directory does not exist: '
              f'{os.path.abspath(in_dir)}')
        sys.exit(1)

    train_files = glob.glob('%s/*.h5' % in_dir)
    train_x = np.empty((0, len(TRAIN_FEATURES)), dtype=np.float32)
    train_y = np.empty((0, ), dtype=np.float32)
    for train_file in tqdm(train_files, desc=f'Process training file'):
        train_data = h5py.File(train_file, 'r')

        # Checking to make sure the variables we use in this script are included
        # in the data set
        for var in REQUIRED_VARS:
            if var not in list(train_data.keys()):
                print(f'[ERROR] Required variable {var} is not included in the '
                      f'input data set.')
                sys.exit(1)

        # Get training inputs and targets
        data_x, data_y, _, _ = format_data(
            train_data, apply_toar_mask=True, latitude_min=10, latitude_max=80,
            longitude_min=-140, longitude_max=-50
        )

        train_x = np.vstack((train_x, data_x))
        train_y = np.hstack((train_y, data_y))

    # Parameter search
    if hyperparameter_tuning:
        print('Parameter tuning ...')
        params = [{
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [3, 4, 5]
        }]
        kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
        clf = GridSearchCV(
            RandomForestRegressor(), params, cv=kfold,
            scoring='neg_root_mean_squared_error', n_jobs=10, error_score='raise'
        )
        clf.fit(train_x, train_y)
        print('Parameter selection:')
        print(f'n_estimators: {clf.best_params_["n_estimators"]}')
        print(f'max_depth: {clf.best_params_["max_depth"]}')

        # Create the Gradient Boosting predictor
        print('Training ...')
        rf_predictor = RandomForestRegressor(
            n_estimators=clf.best_params_['n_estimators'],
            max_depth=clf.best_params_['max_depth'],
            random_state=1234
        )
    else:
        rf_predictor = RandomForestRegressor()

    rf_predictor.fit(train_x, train_y)

    # Save trained model
    print('Saving trained model ...')
    out_model = os.path.join(out_dir, 'random_forest_predictor.joblib')
    dump(rf_predictor, out_model)

    # Make predictions on training set and plot true_y v.s. pred_y
    print('Generating evaluation plot ...')
    preds_y = rf_predictor.predict(train_x)
    out_plot = os.path.join(out_dir, 'train_results.png')
    gen_true_pred_plot(train_y, preds_y, out_plot)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', type=str,
                        help='The input directory contains training data '
                             '(.h5 format)')
    parser.add_argument('out_dir', type=str,
                        help='The output directory to save trained model and '
                             'evaluation plot')
    parser.add_argument('--hyperparameter_tuning', action='store_true',
                        help='If this parameter is provided, hyperparameter '
                             'tuning will be enabled. Default is False.')

    args = parser.parse_args()
    main(**vars(args))
