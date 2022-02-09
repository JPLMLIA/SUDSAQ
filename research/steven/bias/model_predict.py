#!/usr/bin/env python
# Predict biases using trained and saved models. The predictions will be saved
# in a h5py file for further analysis.
#
# Steven Lu
# February 1, 2022

import os
import sys
import h5py
import numpy as np
from joblib import load
from utils import REQUIRED_VARS
from utils import format_data


def main(in_model, in_data, out_pred_file):
    if not os.path.exists(in_model):
        print(f'[ERROR] Input model file does not exist: '
              f'{os.path.abspath(in_model)}')
        sys.exit(1)

    # Load the saved model
    model = load(in_model)

    if not os.path.exists(in_data):
        print(f'[ERROR] Input data file does not exist: '
              f'{os.path.abspath(in_data)}')
        sys.exit(1)

    data = h5py.File(in_data, 'r')

    # Checking to make sure the variables we use in this script are included in
    # the data set
    for var in REQUIRED_VARS:
        if var not in list(data.keys()):
            print(f'[ERROR] Required variable {var} is not included in the '
                  f'input data set.')
            sys.exit(1)

    # Format test data
    test_x, test_y, lat, lon = format_data(
        data, apply_toar_mask=False, latitude_min=10, latitude_max=80,
        longitude_min=-140, longitude_max=-50
    )

    # Make predictions
    pred_y = model.predict(test_x)

    # Reshape arrays back into 2d
    test_y = test_y.reshape((len(lat), len(lon), len(data['date'])))
    pred_y = pred_y.reshape((len(lat), len(lon), len(data['date'])))

    # Save the predictions
    out_file = h5py.File(out_pred_file, 'w')
    out_file.create_dataset('true_bias', data=test_y)
    out_file.create_dataset('pred_bias', data=pred_y)
    out_file.create_dataset('lat', data=lat)
    out_file.create_dataset('lon', data=lon)
    out_file.create_dataset('date', data=np.array(data['date']))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('in_model', type=str)
    parser.add_argument('in_data', type=str)
    parser.add_argument('out_pred_file', type=str)

    args = parser.parse_args()
    main(**vars(args))
