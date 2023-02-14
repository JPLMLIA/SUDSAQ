#!/usr/bin/env python3
# Generate maps for the MOMO-Chem's u (horizontal wind) and v (vertical wind)
# variables
#
# Steven Lu
# February 13, 2023

import os
import numpy as np
import xarray as xr
from tqdm import tqdm
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def main(momo_file, out_dir):
    momo_data = xr.open_mfdataset(momo_file, parallel=True, lock=False)
    momo_u = np.array(momo_data['momo.u'])
    momo_v = np.array(momo_data['momo.v'])
    lat = np.array(momo_data['lat'])
    lat = block_reduce(lat, 2, np.mean)
    lon = np.array(momo_data['lon'])
    lon = block_reduce(lon, 2, np.mean)
    time = np.array(momo_data['time'])

    for ind, time in tqdm(enumerate(time), total=len(time)):
        u = momo_u[ind, :, :]
        u = block_reduce(u, (2, 2), np.mean)
        v = momo_v[ind, :, :]
        v = block_reduce(v, (2, 2), np.mean)

        fig, ax = plt.subplots(figsize=(18, 9),
                               subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_global()
        ax.coastlines()
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        ax.quiver(lon, lat, u, v)

        out_file = os.path.join(out_dir, 'momo_wind_map_%s.png' % time)
        plt.savefig(out_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('momo_file', type=str)
    parser.add_argument('out_dir', type=str)

    args = parser.parse_args()
    main(**vars(args))
