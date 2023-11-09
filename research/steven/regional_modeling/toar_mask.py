#!/usr/bin/env python3
# This script generates a binary mask for TOAR2 stations
#
# Steven Lu
# October 10, 2023


import os
import sys
import numpy as np
import xarray as xr
from tqdm import tqdm
from region import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def main(toar_files, out_dir):
    for f in toar_files:
        if not os.path.exists(f):
            print(f'Input file does not exist: {f}')
            sys.exit(1)

    # Set up map layout
    fig, ax = plt.subplots(figsize=(24, 18),
                           subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cf.COASTLINE)
    ax.add_feature(cf.BORDERS)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.xlabel_style = {'size': 15, 'color': 'gray'}

    # Generate a binary mask for each of the toar data set
    mask_list = []
    for toar_file in tqdm(toar_files):
        ds = xr.open_mfdataset(toar_file, engine='netcdf4', lock=False, parallel=True)
        ds = Dataset(ds)
        ds = ds['toar.mda8.mean']
        ds = ds.sortby('lon')
        mask = ~np.isnan(ds).all(dim='time').values
        mask_list.append(mask)

    # Combine the masks
    mask = np.ones(mask_list[0].shape, dtype=bool)
    for m in mask_list:
        mask = mask & m

    # Plot the mask map
    plt.pcolor(ds.lon, ds.lat, mask, cmap='binary')
    plt.title('Locations of TOAR2 stations')
    plt.savefig(os.path.join(out_dir, 'mask.png'))

    # Save the mask as a npy array
    np.save(os.path.join(out_dir, 'mask.npy'), mask)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--toar_files', nargs='+', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)

    args = parser.parse_args()
    main(**vars(args))
