#!/usr/bin/env python3
# Match the preprocessed Power Plant data set onto MOMOChem grid
#
# Steven Lu
# February 10, 2023

import os
import re
import sys
import numpy as np
import xarray as xr
from scipy import stats

# The weighting factors were obtained from the following paper:
# https://www.osti.gov/biblio/1045758
PRIMARY_FUEL_DICT = {
    'Biomass': 0.9267,
    'Coal': 1.1410,
    'Cogeneration': 0.001,
    'Gas': 0.1175,
    'Geothermal': 0.001,
    'Hydro': 0.001,
    'Nuclear': 0.001,
    'Oil': 4.4825,
    'Other': 0.5,
    'Petcoke': 0.001,
    'Solar': 0.001,
    'Storage': 0.001,
    'Waste': 0.001,
    'Wave and Tidal': 0.001,
    'Wind': 0.001
}


def main(momo_file, pp_file, out_file):
    for f in [momo_file, pp_file]:
        if not os.path.exists(f):
            print(f'[ERROR] Input file does not exist: {os.path.abspath(f)}')
            sys.exit(1)

    momo_data = xr.open_mfdataset(momo_file, parallel=True)
    pp_data = xr.open_mfdataset(pp_file, parallel=True)
    out_data = xr.Dataset(
        coords={
            'lat': momo_data['lat'],
            'lon': momo_data['lon']
        }
    )

    momo_lat = np.hstack([momo_data['lat'], 90.])
    momo_lon = np.hstack([momo_data['lon'], 360.])
    pp_lat = np.array(pp_data['pp_latitude'])
    pp_lon = np.array(pp_data['pp_longitude'])
    pp_primary_fuel = np.array(pp_data['primary_fuel'])
    pp_capacity = np.array(pp_data['capacity_mw'])

    def _compute_impact_factor(binned_values):
        weighted_impact = 0.0

        # If there is no power plant in the MOMOChem cell, assign 0 factor
        if len(binned_values) == 0:
            return weighted_impact

        for string_value in binned_values:
            tokens = re.split(r'(\d+)', string_value)
            primary_fuel = tokens[0]
            capacity = float(''.join(tokens[1:]))

            weighted_impact += PRIMARY_FUEL_DICT[primary_fuel] * capacity

        return weighted_impact

    # Note: We have to concatenate the primary_fuel and capacity because scipy
    # binned_statistic_2d's statistic argument only allows passing in one 1d
    # array/list for the callable functions.
    pp_list = []
    for pf, c in zip(pp_primary_fuel, pp_capacity):
        pp_list.append(pf + str(c))

    ret = stats.binned_statistic_2d(
        pp_lat,
        pp_lon,
        pp_list,
        statistic=_compute_impact_factor,
        bins=[momo_lat, momo_lon],
        expand_binnumbers=True
    )

    out_data[f'pp.weighted_impact'] = (('lat', 'lon'), ret.statistic)
    out_data.to_netcdf(out_file, mode='w', engine='netcdf4')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('momo_file', type=str)
    parser.add_argument('pp_file', type=str)
    parser.add_argument('out_file', type=str)

    args = parser.parse_args()
    main(**vars(args))
