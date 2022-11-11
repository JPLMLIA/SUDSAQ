#!/usr/bin/env python
# Analyze cross-validated prediction results .
#
# Steven Lu
# April 12, 2022

import os
import sys
import h5py
import numpy as np
from rf_temporal_cv_nc import MOMO_FEATURES
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
    first_mask = np.isnan(np.array(first_data['truth']))
    cont_rows, cont_cols = first_data['contribution'].shape
    cont_arrs = np.empty((len(in_pred_files), cont_rows, cont_cols),
                         dtype=np.float32)

    for index, pred_file in enumerate(in_pred_files):
        data = h5py.File(pred_file, 'r')
        cont_arrs[index, :, :] = data['contribution']

        truth = np.array(data['truth'])
        prediction = np.array(data['prediction'])

        mask1 = ~np.isnan(truth)
        truth_masked = truth[mask1]
        prediction_masked = prediction[mask1]

        mask2 = ~np.isnan(prediction_masked)
        truth_masked = truth_masked[mask2]
        prediction_masked = prediction_masked[mask2]

        # Plot true bias vs predicted bias
        base_name = os.path.splitext(os.path.basename(pred_file))[0]
        out_plot = os.path.join(out_dir, '%s_plot.png' % base_name)
        gen_true_pred_plot(truth_masked, prediction_masked, out_plot,
                           sub_sample=False)
    cont_mean = np.nanmean(cont_arrs, axis=0)

    for ind, feature_name in enumerate(MOMO_FEATURES[:-1]):
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
        plt.clim((-5, 5))
        # plt.clim((-15, 15))
        plt.colorbar()
        ax.coastlines()
        ax.stock_img()
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        feature_name = feature_name.replace('/', '-')
        plt.title(f'Feature contribution - {feature_name}')
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
