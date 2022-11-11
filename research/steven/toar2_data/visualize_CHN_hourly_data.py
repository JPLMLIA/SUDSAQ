#!/usr/bin/env python
# Read and visualize TOAR-2 China hourly data
#
# Steven Lu
# January 14, 2021

import os
import sys
import numpy as np
import netCDF4 as nc4
from datetime import datetime
from datetime import timedelta
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def main(in_data, out_dir):
    if not os.path.exists(in_data):
        print('[ERROR] Input file not found: %s' % os.path.abspath(in_data))
        sys.exit(1)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print('[INFO] Created output dir: %s' % os.path.abspath(out_dir))

    data = nc4.Dataset(in_data)
    print('Data set variables: %s' % ','.join(data.variables.keys()))

    start_time_all = data['time'].__dict__['units']
    start_time_tokens = start_time_all.split(' ')
    start_time_str = ' '.join(start_time_tokens[2:])
    start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
    time_arr = np.array([
        (start_time + timedelta(hours=int(h))).strftime('%Y-%m-%d %H:%M:%S')
        for h in data['time'][:].data
    ])

    site_lat = data['site_lat'][:].data.astype(float)
    site_lon = data['site_lon'][:].data.astype(float)
    ozone = data['O3'][:, :].data
    ozone_max = np.nanmax(ozone)
    ozone_normalized = ozone / ozone_max

    for ind, time_str in enumerate(time_arr):
        fig, ax = plt.subplots(figsize=(18, 9),
                               subplot_kw={'projection': ccrs.PlateCarree()})
        ax.coastlines()
        ax.add_feature(cf.BORDERS)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        plt.scatter(site_lon, site_lat, c=ozone_normalized[:, ind], cmap='jet',
                    marker='s', s=2)

        plt.colorbar(ticks=[])
        plt.title('TOAR-2 China ozone %s' % time_str)
        plt.savefig('%s/TOAR2_CHN_hourly_%s.png' % (out_dir, time_str))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('in_data', type=str, help='TOAR-2 China hourly data')
    parser.add_argument('out_dir', type=str)

    args = parser.parse_args()
    main(**vars(args))