#!/usr/bin/env python3
# Run regional experiment
#
# Steven Lu
# June 23, 2023

import os
import re
import joblib
import numpy as np
import xarray as xr
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy import stats
from sklearn.metrics import mean_squared_error

# Variables for making clusters
CLUSTER_VARS = [
    'momo.no',
    'momo.no2',
    'momo.co',
    'momo.ch2o',
    'momo.u',
    'momo.v',
    'momo.t',
    'momo.ps',
    'momo.2dsfc.NH3',
    'momo.2dsfc.BrOX',
    'momo.hno3',
    'momo.slrc',
    'momo.2dsfc.HO2',
    'momo.2dsfc.C2H6',
    'momo.2dsfc.C5H8'
]

# Variables to drop for v4 data set
MOMO_V4_VARS_SEL = '.+\.(?!hno3|oh|pan|q2|sens|so2|T2|taugxs|taugys|taux|tauy|twpc|2dsfc.CFC11|2dsfc.CFC113|2dsfc.CFC12|ch2o|cumf0|2dsfc.dms|2dsfc.HCFC22|2dsfc.H1211|2dsfc.H1301|2dsfc.mc.oc|2dsfc.dflx.bc|2dsfc.dflx.oc|2dsfc.LR.SO2|2dsfc.CH3COOOH|prcpl|2dsfc.C3H7OOH|dqlsc|2dsfc.mc.pm25.salt|2dsfc.CH3COO2|u10|2dsfc.dflx.nh4|2dsfc.mc.nh4|2dsfc.dflx.dust|2dsfc.mc.pm25.dust|osr|osrc|ssrc|v10|2dsfc.OCS|2dsfc.taut|ccoverl|ccoverm|2dsfc.DCDT.HOX|2dsfc.DCDT.OY|2dsfc.DCDT.SO2|slrdc|uvabs|dqcum|dqdad|dqdyn|dqvdf|dtdad|cumf|ccoverh|prcpc|2dsfc.BrCl|2dsfc.Br2|dtcum|2dsfc.mc.sulf|2dsfc.HOBr|dtlsc|2dsfc.Cl2|2dsfc.CH3CCl3|2dsfc.CH3Br|2dsfc.ONMV|2dsfc.MACROOH|2dsfc.MACR|2dsfc.HBr).*'


# Copied from sudsaq/data.py
class Dataset(xr.Dataset):
    """
    Small override of xarray.Dataset that enables regex matching names in the variables
    list
    """
    # TODO: Bad keys failing to report which keys are bad: KeyError: 'momo'
    __slots__ = () # Required for subclassing

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError as e:
            if isinstance(key, str):
                keys = [var for var in self.variables.keys() if re.fullmatch(key, var)]
                if keys:
                    return super().__getitem__(keys)
            raise e


def gen_true_pred_plot(true_y, pred_y, out_plot, sub_sample=False):
    fig, _ = plt.subplots()
    maxx = np.max(true_y)
    maxy = np.max(pred_y)
    max_value = np.max([maxx, maxy])
    plt.plot((0, max_value), (0, max_value), '--', color='gray', linewidth=1)

    # Compute Pearson correlation coefficient R and RMSE
    r, _ = stats.pearsonr(true_y, pred_y)
    rmse = mean_squared_error(true_y, pred_y, squared=False)

    if sub_sample and len(true_y) > 10000:
        true_y = true_y[::100]
        pred_y = pred_y[::100]

    xy = np.vstack([true_y, pred_y])
    z = stats.gaussian_kde(xy)(xy)
    idx = z.argsort()
    d_true_y = true_y[idx]
    d_pred_y = pred_y[idx]
    z = z[idx]

    plt.scatter(d_true_y, d_pred_y, c=z, cmap=plt.cm.jet,
                label='RMSE = %.3f \n r = %.3f' % (rmse, r), s=0.5)
    cbar = plt.colorbar()
    cbar.set_ticks([])
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.xlim((0, max_value))
    plt.ylim((0, max_value))
    plt.legend(loc='upper right')
    plt.savefig(out_plot)


def main(momo_train_files, momo_test_files, n_clusters, out_dir):
    train_ds = xr.open_mfdataset(momo_train_files, engine='netcdf4', lock=False, parallel=True)
    test_ds = xr.open_mfdataset(momo_test_files, engine='netcdf4', lock=False, parallel=True)

    # Preparing data for K-means clustering
    lat_arr = np.array(train_ds['lat'])
    lon_arr = np.array(train_ds['lon'])
    var_list = list()

    for momo_var in CLUSTER_VARS:
        data = np.array(train_ds[momo_var])

        if np.isnan(data).any():
            print(f'Variable {momo_var} contains nan values. Skip.')
            continue

        var_list.append(np.mean(data, axis=0))
        var_list.append(np.std(data, axis=0))

    var_arr = np.array(var_list)
    data_arr = np.zeros((len(lat_arr) * len(lon_arr), len(var_arr)))
    latlon_arr = np.zeros((len(lat_arr) * len(lon_arr), 2))

    data_ind = 0
    for lat_ind, lat_val in enumerate(lat_arr):
        for lon_ind, lon_val in enumerate(lon_arr):
            data_arr[data_ind, :] = var_arr[:, lat_ind, lon_ind]
            latlon_arr[data_ind, :] = np.array([lat_val, lon_val])
            data_ind += 1

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    data_arr = scaler.fit_transform(data_arr)

    # Perform clustering
    mbk = MiniBatchKMeans(
        init='k-means++', n_clusters=n_clusters, batch_size=100, n_init=10,
        max_no_improvement=10, verbose=0, random_state=398)
    cluster_labels = mbk.fit_predict(data_arr)

    # Cluster map
    cluster_map_file = os.path.join(out_dir, f'cluster_map_k{n_clusters}.png')
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
    colors = iter(cm.tab20(np.linspace(0, 1, n_clusters)))

    # Prepare ML training data
    train_ds = Dataset(train_ds)    # Cast xarray dataset to custom Dataset
    train_ds = train_ds[MOMO_V4_VARS_SEL]        # Drop variables

    train_x = train_ds['momo.(?!o3|mda8).*']     # Select training variables
    train_x = train_x.to_array()
    train_x = train_x.stack({'latlon': ['lat', 'lon']}, create_index=False)
    train_x = train_x.transpose('latlon', 'time', 'variable')

    train_y = train_ds['momo.mda8']              # Select target variable
    train_y = train_y.stack({'latlon': ['lat', 'lon']}, create_index=False)
    train_y = train_y.transpose('latlon', 'time')

    # # Prepare ML test data
    test_ds = Dataset(test_ds)
    test_ds = test_ds[MOMO_V4_VARS_SEL]

    test_x = test_ds['momo.(?!o3|mda8).*']
    test_x = test_x.to_array()
    test_x = test_x.stack({'latlon': ['lat', 'lon']}, create_index=False)
    test_x = test_x.transpose('latlon', 'time', 'variable')

    test_y = test_ds['momo.mda8']
    test_y = test_y.stack({'latlon': ['lat', 'lon']}, create_index=False)
    test_y = test_y.transpose('latlon', 'time')

    # Regional modeling
    for ind, k in enumerate(range(n_clusters)):
        in_group = cluster_labels == k

        # Get data points in the current cluster
        train_x_group = train_x[in_group]
        train_x_group = train_x_group.stack({'loc': ['latlon', 'time']})
        train_x_group = train_x_group.transpose('loc', 'variable')
        train_y_group = train_y[in_group]
        train_y_group = train_y_group.stack({'loc': ['latlon', 'time']})

        test_x_group = test_x[in_group]
        test_x_group = test_x_group.stack({'loc': ['latlon', 'time']})
        test_x_group = test_x_group.transpose('loc', 'variable')
        test_y_group = test_y[in_group]
        test_y_group = test_y_group.stack({'loc': ['latlon', 'time']})

        # Remove NaNs
        train_x_group = train_x_group.where(np.isfinite(train_x_group), np.nan)
        train_y_group = train_y_group.where(np.isfinite(train_y_group), np.nan)
        train_x_group = train_x_group.dropna('loc')
        train_y_group = train_y_group.dropna('loc')
        train_x_group, train_y_group = xr.align(train_x_group, train_y_group, copy=False)

        test_x_group = test_x_group.where(np.isfinite(test_x_group), np.nan)
        test_y_group = test_y_group.where(np.isfinite(test_y_group), np.nan)
        test_x_group = test_x_group.dropna('loc')
        test_y_group = test_y_group.dropna('loc')
        test_x_group, test_y_group = xr.align(test_x_group, test_y_group, copy=False)

        # Train ML model
        clf = RandomForestRegressor()
        clf.fit(train_x_group, train_y_group)

        # Make predictions
        pred_y = clf.predict(test_x_group)

        # Generate plot
        # out_plot = os.path.join(out_dir, f'cluster{ind}_results.png')
        # gen_true_pred_plot(test_y_group, pred_y, out_plot, sub_sample=True)
        r, _ = stats.pearsonr(test_y_group, pred_y)
        rmse = mean_squared_error(test_y_group, pred_y, squared=False)

        # Plot cluster data
        color = next(colors)
        ax.scatter(latlon_arr[in_group, 1], latlon_arr[in_group, 0],
                   c=color, label=f'Cluster {ind} ({rmse:.3f})')

        # Save intermediate products
        out_npz = os.path.join(out_dir, f'cluster_{ind}_data.npz')
        np.savez(
            out_npz,
            train_x_group=train_x_group,
            train_y_group=train_y_group,
            test_x_group=test_x_group,
            test_y_group=test_y_group,
            pred_y=pred_y
        )

        print(f'Model {ind}: r={r}, RMSE={rmse}')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True,
              shadow=True, ncol=5, fontsize=20)
    plt.savefig(cluster_map_file)

    # ML model with all data
    train_x = train_x.stack({'loc': ['latlon', 'time']})
    train_x = train_x.transpose('loc', 'variable')
    train_y = train_y.stack({'loc': ['latlon', 'time']})

    test_x = test_x.stack({'loc': ['latlon', 'time']})
    test_x = test_x.transpose('loc', 'variable')
    test_y = test_y.stack({'loc': ['latlon', 'time']})

    train_x = train_x.where(np.isfinite(train_x), np.nan)
    train_y = train_y.where(np.isfinite(train_y), np.nan)
    train_x = train_x.dropna('loc')
    train_y = train_y.dropna('loc')
    train_x, train_y = xr.align(train_x, train_y, copy=False)

    test_x = test_x.where(np.isfinite(test_x), np.nan)
    test_y = test_y.where(np.isfinite(test_y), np.nan)
    test_x = test_x.dropna('loc')
    test_y = test_y.dropna('loc')
    test_x, test_y = xr.align(test_x, test_y, copy=False)

    clf = RandomForestRegressor()
    clf.fit(train_x, train_y)
    joblib.dump(clf, os.path.join(out_dir, 'global_rf.pb'))

    pred_y = clf.predict(test_x)
    r, _ = stats.pearsonr(test_y, pred_y)
    rmse = mean_squared_error(test_y, pred_y, squared=False)

    print(f'Whole model: r={r}, RMSE={rmse}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--momo_train_files', nargs='+', required=True)
    parser.add_argument('--momo_test_files', nargs='+', required=True)
    parser.add_argument('--n_clusters', type=int, default=15)
    parser.add_argument('--out_dir', type=str, required=True)

    args = parser.parse_args()
    main(**vars(args))
