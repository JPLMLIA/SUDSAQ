#!/usr/bin/env python
# Generate plots for bias v.s. ML model input feature.
#
# Example input data for this script can be found in the following directory on
# MLIA machines.
# /data/MLIA_active_data/data_SUDSAQ/processed/coregistered/
#
# Steven Lu
# February 15, 2022

import os
import sys
import h5py
import glob
import numpy as np
from scipy import stats
from tqdm import tqdm
from utils import REQUIRED_VARS
from utils import TRAIN_FEATURES
from utils import format_data
import matplotlib.pyplot as plt


def main(in_dir, out_dir):
    if not os.path.exists(in_dir):
        print(f'[ERROR] Input directory does not exist: '
              f'{os.path.abspath(in_dir)}')
        sys.exit(1)

    in_files = glob.glob('%s/*.h5' % in_dir)
    feature = np.empty((0, len(TRAIN_FEATURES)), dtype=np.float32)
    bias = np.empty((0,), dtype=np.float32)
    for in_file in tqdm(in_files, desc=f'Process training file'):
        data = h5py.File(in_file, 'r')

        # Checking to make sure the variables we use in this script are included
        # in the data set
        for var in REQUIRED_VARS:
            if var not in list(data.keys()):
                print(f'[ERROR] Required variable {var} is not included in the '
                      f'input data set.')
                sys.exit(1)

        # Get training inputs and targets
        data_feature, data_bias, _, _ = format_data(
            data, apply_toar_mask=True, latitude_min=10, latitude_max=80,
            longitude_min=-140, longitude_max=-50
        )

        feature = np.vstack((feature, data_feature))
        bias = np.hstack((bias, data_bias))

    for ind, feature_name in enumerate(TRAIN_FEATURES):
        fig, ax = plt.subplots()
        r, _ = stats.pearsonr(bias, feature[:, ind])
        ax.scatter(bias, feature[:, ind], s=(72.0 / fig.dpi) ** 2, lw=0,
                   label='Pearson correlation r = %.3f' % r)
        plt.xlabel('Ground Truth Bias')
        plt.ylabel(feature_name)
        plt.legend()
        plt.savefig('%s/bias-vs-%s.png' % (out_dir, feature_name))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', type=str,
                        help='The input directory contains training data '
                             '(.h5 format)')
    parser.add_argument('out_dir', type=str,
                        help='The output directory to save plots')

    args = parser.parse_args()
    main(**vars(args))
