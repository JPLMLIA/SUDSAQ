#!/usr/bin/env python3
#
# Steven Lu
# April 11, 2023

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


# Pre-defined colors for clusters
COLORS = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
          '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
          '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
          '#000075', '#808080', '#ffffff', '#000000']


def main(pca_npy_file, latlon_npy_file, out_map, n_clusters):
    data_arr = np.load(pca_npy_file)
    latlon_arr = np.load(latlon_npy_file)

    mbk = MiniBatchKMeans(
        init='k-means++', n_clusters=n_clusters, batch_size=100, n_init=10,
        max_no_improvement=10, verbose=0)
    cluster_labels = mbk.fit_predict(data_arr)

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

    for ind, k in enumerate(range(n_clusters)):
        in_group = cluster_labels == k
        ax.scatter(latlon_arr[in_group, 1], latlon_arr[in_group, 0],
                   c=COLORS[ind], marker='s')

    plt.savefig(out_map)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('pca_npy_file', type=str)
    parser.add_argument('latlon_npy_file', type=str)
    parser.add_argument('out_map', type=str)
    parser.add_argument('--n_clusters', type=int, default=3)

    args = parser.parse_args()
    main(**vars(args))
