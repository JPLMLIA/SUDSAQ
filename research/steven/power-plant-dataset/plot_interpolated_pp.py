#!/usr/bin/env python3
# Generate maps for the interpolated power plant data set.
#
# Steven Lu
# February 13, 2023

import os
import numpy as np
import xarray as xr
from tqdm import tqdm
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def main(pp_interpolated_file, out_dir):
    pp_data = xr.open_mfdataset(pp_interpolated_file, parallel=True, lock=False)
    pp_impact_all = np.array(pp_data['pp.weighted_impact'])
    min_impact = np.min(pp_impact_all)
    max_impact = np.max(pp_impact_all)
    pp_time = np.array(pp_data['time'])

    for ind, time in tqdm(enumerate(pp_time), total=len(pp_time)):
        pp_impact = pp_impact_all[ind, :, :]
        pp_impact[pp_impact == 0] = np.nan

        fig, ax = plt.subplots(figsize=(18, 9),
                               subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_global()
        ax.coastlines()
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        plt.pcolor(pp_data['lon'], pp_data['lat'], pp_impact, cmap='jet')
        plt.colorbar()
        plt.clim((min_impact, max_impact))

        out_file = os.path.join(out_dir, 'pp_interpolated_map_%s.png' % time)
        plt.savefig(out_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('pp_interpolated_file', type=str)
    parser.add_argument('out_dir', type=str)

    args = parser.parse_args()
    main(**vars(args))

