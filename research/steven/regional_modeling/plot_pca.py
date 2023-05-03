#!/usr/bin/env python3
#
# Steven Lu
# May 2, 2023

import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def main(pca_npy_file, latlon_npy_file, out_map):
    data_arr = np.load(pca_npy_file)
    latlon_arr = np.load(latlon_npy_file)

    first_component = data_arr[:, 0].flatten()
    lat_arr = latlon_arr[:, 0].flatten()
    lon_arr = latlon_arr[:, 1].flatten()

    fig, ax = plt.subplots(figsize=(24, 18),
                           subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([220, 310, 10, 80])
    # ax.coastlines()
    ax.add_feature(cf.COASTLINE)
    ax.add_feature(cf.BORDERS)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.xlabel_style = {'size': 15, 'color': 'gray'}

    cm = plt.cm.get_cmap('jet')
    sc = ax.scatter(lon_arr, lat_arr, c=first_component, cmap=cm, marker='s', s=60)
    plt.colorbar(sc)

    plt.savefig(out_map)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('pca_npy_file', type=str)
    parser.add_argument('latlon_npy_file', type=str)
    parser.add_argument('out_map', type=str)

    args = parser.parse_args()
    main(**vars(args))
