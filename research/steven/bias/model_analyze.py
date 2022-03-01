#!/usr/bin/env python
# Analyze prediction results (generated by model_analyze.py).
#
# Steven Lu
# February 8, 2022

import os
import sys
import h5py
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from utils import gen_true_pred_plot


def main(in_pred_file, out_dir):
    if not os.path.exists(in_pred_file):
        print(f'[ERROR] Input prediction file does not exist: '
              f'{os.path.abspath(in_pred_file)}')
        sys.exit(1)

    data = h5py.File(in_pred_file, 'r')
    true_bias = np.array(data['true_bias'])
    pred_bias = np.array(data['pred_bias'])

    mask = ~np.isnan(true_bias)
    true_bias_masked = true_bias[mask]
    pred_bias_masked = pred_bias[mask]

    # Plot true bias vs predicted bias
    base_name = os.path.splitext(os.path.basename(in_pred_file))[0]
    out_plot = os.path.join(out_dir, '%s_plot.png' % base_name)
    gen_true_pred_plot(true_bias_masked, pred_bias_masked, out_plot,
                       sub_sample=False)

    # Generate daily bias map
    diff_bias = true_bias - pred_bias
    min_bias = np.nanmin(diff_bias)
    max_bias = np.nanmax(diff_bias)
    for ind, (year, month, day) in enumerate(data['date']):
        year = year.decode('UTF-8')
        month = month.decode('UTF-8')
        day = day.decode('UTF-8')
        fig, ax = plt.subplots(figsize=(18, 9),
                               subplot_kw={'projection': ccrs.PlateCarree()})
        plt.pcolor(data['lon'], data['lat'], diff_bias[:, :, ind],
                   cmap='coolwarm')
        plt.clim((min_bias, max_bias))
        plt.colorbar()
        ax.coastlines()
        ax.stock_img()
        ax.set_extent([-140, -50, 10, 80], crs=ccrs.PlateCarree())  # NA region
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        plt.title(f'True Bias v.s. ML Predicted Bias - {month}/{day}/{year}')
        plt.savefig(f'{out_dir}/bias_diff_{year}-{month}-{day}.png',
                    bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('in_pred_file', type=str)
    parser.add_argument('out_dir', type=str)

    args = parser.parse_args()
    main(**vars(args))
