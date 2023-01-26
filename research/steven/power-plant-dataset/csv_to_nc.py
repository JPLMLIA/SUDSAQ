#!/usr/bin/env python3
# Convert the power plant global data set from CSV to NetCDF format. During the
# conversion, certain columns (e.g., IDs, columns with missing values) will be
# dropped.
#
# December 12, 2022
# Steven Lu

import os
import sys
import numpy as np
import xarray as xr
import pandas as pd


def main(pp_csv_file, out_nc_file):
    if not os.path.exists(pp_csv_file):
        print('[ERROR] Input power plant database file does not exist')
        sys.exit(1)

    data_columns = ['capacity_mw', 'latitude', 'longitude', 'primary_fuel']
    pp_df = pd.read_csv(pp_csv_file, usecols=data_columns, index_col=False)

    out_nc = xr.Dataset()
    latitude = np.array(pp_df['latitude']).astype(float)
    longitude = np.array(pp_df['longitude']).astype(float)
    longitude[longitude < 0] = longitude[longitude < 0] + 360
    out_nc['pp_latitude'] = latitude
    out_nc['pp_longitude'] = longitude
    out_nc['capacity_mw'] = np.array(pp_df['capacity_mw'])
    out_nc['primary_fuel'] = np.array(pp_df['primary_fuel'])
    print(np.unique(np.array(pp_df['primary_fuel'])))

    out_nc.to_netcdf(out_nc_file, mode='w', engine='netcdf4')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('pp_csv_file', type=str,
                        help='Input power plant database in csv format')
    parser.add_argument('out_nc_file', type=str,
                        help='Output netcdf file')

    args = parser.parse_args()
    main(**vars(args))
