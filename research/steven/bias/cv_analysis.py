#!/usr/bin/env python
# Analyze cross-validated prediction results .
#
# Steven Lu
# April 12, 2022

import os
import sys
import h5py
import numpy as np
from utils import TRAIN_FEATURES
from utils import gen_true_pred_plot
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def main(in_pred_files, out_dir):
    for f in in_pred_files:
        if not os.path.exists(f):
            print(f'[ERROR] Input file does not exist: {os.path.abspath(f)}')
            sys.exit()

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    first_data = h5py.File(in_pred_files[0], 'r')
    first_mask = np.isnan(np.array(first_data['true_bias']))
    cont_rows, cont_cols = first_data['contribution'].shape
    cont_arrs = np.empty((len(in_pred_files), cont_rows, cont_cols),
                         dtype=np.float32)

    for index, pred_file in enumerate(in_pred_files):
        data = h5py.File(pred_file, 'r')
        cont_arrs[index, :, :] = data['contribution']

        true_bias = np.array(data['true_bias'])
        pred_bias = np.array(data['pred_bias'])

        mask = ~np.isnan(true_bias)
        true_bias_masked = true_bias[mask]
        pred_bias_masked = pred_bias[mask]

        # Plot true bias vs predicted bias
        base_name = os.path.splitext(os.path.basename(pred_file))[0]
        out_plot = os.path.join(out_dir, '%s_plot.png' % base_name)
        gen_true_pred_plot(true_bias_masked, pred_bias_masked, out_plot,
                           sub_sample=False)
    cont_mean = np.mean(cont_arrs, axis=0)

    for ind, feature_name in enumerate(TRAIN_FEATURES):
        mean_arr = cont_mean[:, ind].reshape((len(first_data['date']),
                                              len(first_data['lat']),
                                              len(first_data['lon'])))
        mean_arr = np.ma.masked_array(mean_arr, first_mask)

        plt.clf()
        fig, ax = plt.subplots(figsize=(18, 9),
                               subplot_kw={'projection': ccrs.PlateCarree()})
        plt.pcolor(
            first_data['lon'], first_data['lat'], np.nanmean(mean_arr, axis=0),
            cmap='coolwarm'
        )
        plt.clim((-15, 15))
        plt.colorbar()
        ax.coastlines()
        ax.stock_img()
        ax.set_extent([-140, -50, 10, 80], crs=ccrs.PlateCarree())  # NA region
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        plt.title(f'Feature contribution (2012 - 2015) - {feature_name}')
        plt.savefig(f'{out_dir}/feature-contribution-{feature_name}.png',
                    bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_pred_files', nargs='+')
    parser.add_argument('--out_dir', type=str)

    args = parser.parse_args()
    main(**vars(args))
