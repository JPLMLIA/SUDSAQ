#!/usr/bin/env python3
# Generate a map for the power plant data set matched to the MOMO-Chem grid.
#
# Steven Lu
# February 13, 2023


import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def main(pp_file, out_file):
    pp_data = xr.open_mfdataset(pp_file, parallel=True, lock=False)
    pp_impact = np.array(pp_data['pp.weighted_impact'])
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

    plt.savefig(out_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('pp_file', type=str)
    parser.add_argument('out_file', type=str)

    args = parser.parse_args()
    main(**vars(args))
