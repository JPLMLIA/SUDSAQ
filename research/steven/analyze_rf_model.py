#!/usr/bin/env python3
# This script visualizes the first tree of a random forest model in a png image
#
# November 16th, 2022
# Steven Lu

from joblib import load
from sklearn import tree
import matplotlib.pyplot as plt
import xarray as xr


def main(in_model, test_data_nc, out_tree):
    rf_model = load(in_model)
    dt = rf_model.estimators_[0]
    ds = xr.open_mfdataset(test_data_nc)
    var_names = list(ds.keys())

    tree.plot_tree(dt, feature_names=var_names, filled=True)
    plt.savefig(out_tree)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('in_model')
    parser.add_argument('test_data_nc')
    parser.add_argument('out_tree')

    args = parser.parse_args()
    main(**vars(args))
