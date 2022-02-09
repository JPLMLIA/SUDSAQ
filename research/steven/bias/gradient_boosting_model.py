#!/usr/bin/env python
# Train gradient boosting model for the MOMO-TOAR bias predictor
# Example input data for this script can be found in the following directory on
# MLIA machines.
# /data/MLIA_active_data/data_SUDSAQ/processed/coregistered/
#
# Steven Lu
# February 1, 2022

import os
import sys
import h5py
from joblib import dump
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from utils import REQUIRED_VARS
from utils import format_data


def main(in_data, out_model):
    if not os.path.exists(in_data):
        print(f'[ERROR] Input file does not exist: {os.path.abspath(in_data)}')
        sys.exit(1)

    data = h5py.File(in_data, 'r')

    # Checking to make sure the variables we use in this script are included in
    # the data set
    for var in REQUIRED_VARS:
        if var not in list(data.keys()):
            print(f'[ERROR] Required variable {var} is not included in the '
                  f'input data set.')
            sys.exit(1)

    # Get training inputs and targets
    train_x, train_y, _, _ = format_data(
        data, apply_toar_mask=True, latitude_min=10, latitude_max=80,
        longitude_min=-140, longitude_max=-50
    )

    # Parameter search
    params = [{
        'loss': ['squared_error', 'absolute_error'],
        'learning_rate': [0.001, 0.01, 0.1],
        'n_estimators': [50, 75, 100, 125, 150],
        'max_depth': [2, 3, 4, 5, 6, 7]
    }]
    kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
    clf = GridSearchCV(
        GradientBoostingRegressor(), params, cv=kfold,
        scoring='neg_mean_squared_error', n_jobs=10, error_score='raise'
    )
    clf.fit(train_x, train_y)
    print('Parameter selection:')
    print(f'loss: {clf.best_params_["loss"]}')
    print(f'learning_rate: {clf.best_params_["learning_rate"]}')
    print(f'n_estimators: {clf.best_params_["n_estimators"]}')
    print(f'max_depth: {clf.best_params_["max_depth"]}')

    # Create the Gradient Boosting predictor
    gb_predictor = GradientBoostingRegressor(
        loss=clf.best_params_['loss'],
        learning_rate=clf.best_params_['learning_rate'],
        n_estimators=clf.best_params_['n_estimators'],
        max_depth=clf.best_params_['max_depth'],
        random_state=1234
    )
    gb_predictor.fit(train_x, train_y)

    # Save trained model
    dump(gb_predictor, out_model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('in_data', type=str, help='The input data file.')
    parser.add_argument('out_model', type=str, help='The output gradient '
                                                    'boosting model file')

    args = parser.parse_args()
    main(**vars(args))
