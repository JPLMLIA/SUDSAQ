#!/usr/bin/env python3
# This script generates a correlation plot of two MOMOChem variable
#
# November 16th, 2022
# Steven Lu

import xarray as xr
import matplotlib.pyplot as plt


def main(momo_files, var_name_1, var_name_2, out_plot):
    momo_ds = xr.open_mfdataset(momo_files, parallel=True, engine='netcdf4',
                                lock=False)
    var1_ds = momo_ds[var_name_1].load()
    var2_ds = momo_ds[var_name_2].load()
    var1_arr = var1_ds.to_numpy().flatten()
    var2_arr = var2_ds.to_numpy().flatten()

    plt.scatter(var2_arr, var1_arr)
    plt.xlabel(var_name_2)
    plt.ylabel(var_name_1)
    plt.savefig(out_plot)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--momo_files', nargs='+', required=True)
    parser.add_argument('--var_name_1', type=str, required=True)
    parser.add_argument('--var_name_2', type=str, required=True)
    parser.add_argument('--out_plot', type=str, required=True)

    args = parser.parse_args()
    main(**vars(args))
