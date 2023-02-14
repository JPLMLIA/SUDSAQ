#!/usr/bin/env python3
# Interpolation of the matched power plant data sets using a Gaussian kernel
#
# Steven Lu
# February 10, 2023

import os
import sys
import numpy as np
import xarray as xr
from scipy.signal import convolve2d


def gaussian_kernel(size, theta, mu=0, sigma=1):
    # Round theta to the nearest 45 degrees (in radians)
    rad_45 = np.deg2rad(45)
    theta_round = round(theta / rad_45) * rad_45
    x, y = np.meshgrid(np.linspace(-1, 1, size),
                       np.linspace(-1, 1, size))
    dst = np.sqrt(x ** 2 + y ** 2)
    gaussian = np.exp(-((dst - mu) ** 2 / (2.0 * sigma ** 2)))
    half_size = size // 2

    u = np.linspace(-half_size, half_size, size)
    v = np.linspace(half_size, -half_size, size)
    for u_ind, u_val in enumerate(u):
        for v_ind, v_val in enumerate(v):
            if v_val == 0 and u_val == 0:
                continue

            if np.arctan2(v_val, u_val) != theta_round:
                gaussian[v_ind, u_ind] = 0

    return gaussian


def main(matched_pp_nc_file, momo_nc_file, out_nc_file):
    for f in [matched_pp_nc_file, momo_nc_file]:
        if not os.path.exists(f):
            print(f'[ERROR] Input file does not exist: {os.path.abspath(f)}')
            sys.exit()

    pp_data = xr.open_mfdataset(matched_pp_nc_file, parallel=True, lock=False)
    momo_data = xr.open_mfdataset(momo_nc_file, parallel=True, lock=False)
    out_data = xr.Dataset(
        coords={
            'lat': pp_data['lat'],
            'lon': pp_data['lon'],
            'time': momo_data['time']
        }
    )

    momo_var_u = np.array(momo_data['momo.u'])    # Horizontal wind
    momo_var_v = np.array(momo_data['momo.v'])    # Vertical wind
    pp_impact = np.array(pp_data['pp.weighted_impact'])
    pp_list = []

    for time_ind, time in enumerate(np.array(momo_data['time'])):
        u = momo_var_u[time_ind, :, :]
        v = momo_var_v[time_ind, :, :]
        d = np.sqrt(u**2 + v**2).astype(int)
        theta = np.arctan2(v, u)
        ys, xs = np.nonzero(pp_impact)
        pp_arr = np.zeros(u.shape, dtype=float)
        rows, cols = u.shape
        for y, x in zip(ys, xs):
            m = np.abs(d[y, x])
            ks = 2 * m + 1

            if y - m < 0 or y + m >= rows or x - m < 0 or x + m >= cols:
                pp_arr[y, x] = pp_impact[y, x]
                continue

            gaussian = gaussian_kernel(ks, theta[y, x])
            pp_arr[y-m:y+m, x-m:x+m] += convolve2d(pp_impact[y-m:y+m, x-m:x+m],
                                                   gaussian, mode='same')
        pp_list.append(pp_arr)

    out_data[f'pp.weighted_impact'] = (('time', 'lat', 'lon'), pp_list)
    out_data.to_netcdf(out_nc_file, mode='w', engine='netcdf4')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('matched_pp_nc_file', type=str)
    parser.add_argument('momo_nc_file', type=str)
    parser.add_argument('out_nc_file', type=str)

    args = parser.parse_args()
    main(**vars(args))
