#!/usr/bin/env python
# This script generates feature importance plot using the saved Random Forest
# models.
#
# Steven Lu
# March 14, 2022

import os
import sys
import numpy as np
from joblib import load
import matplotlib.pyplot as plt

from utils import TRAIN_FEATURES


def main(rf_model_file, out_plot):
    if not os.path.exists(rf_model_file):
        print(f'[ERROR] Input Random Forest model not found: '
              f'{os.path.abspath(rf_model_file)}')
        sys.exit(1)

    # Load Random Forest model
    rf_model = load(rf_model_file)
    feature_importance_arr = rf_model.feature_importances_
    if len(feature_importance_arr) != len(TRAIN_FEATURES):
        print('[ERROR] Incompatible Random Forest model w.r.t. the number of '
              'features used when the model was trained.')
        sys.exit(1)

    # Normalize the feature importance array
    feature_importance_arr = feature_importance_arr / np.max(feature_importance_arr)

    plt.barh(TRAIN_FEATURES, feature_importance_arr)
    plt.title('Random Forest Feature Importance')
    plt.ylabel('Features')
    plt.xlabel('Feature Importance')
    plt.grid(linestyle='dotted')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_plot)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('rf_model_file', type=str,
                        help='Saved Random Forest model')
    parser.add_argument('out_plot', type=str,
                        help='Output Random Forest feature importance plot')

    args = parser.parse_args()
    main(**vars(args))