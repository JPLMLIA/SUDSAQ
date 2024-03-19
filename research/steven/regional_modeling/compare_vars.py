import os
import sys
import numpy as np
import xarray as xr
from region import Dataset
from region import MOMO_V4_VARS_SEL
import matplotlib.pyplot as plt
from tqdm import tqdm


def main(momo_file, toar_file, cluster_mask_file1, cluster_mask_file2, out_dir):
    # Load in MOMO file
    momo_ds = xr.open_mfdataset(momo_file, engine='netcdf4', lock=False, parallel=True)
    momo_ds = Dataset(momo_ds)
    momo_ds = momo_ds[MOMO_V4_VARS_SEL]
    momo_ds = momo_ds.sortby('lon')
    momo_ds.coords['time'] = momo_ds.time.dt.floor('1D')
    momo_ds = momo_ds.groupby('time').mean()

    # Load in TOAR2 file
    toar_ds = xr.open_mfdataset(toar_file, engine='netcdf4', lock=False, parallel=True)
    toar_ds = Dataset(toar_ds)
    toar_da = toar_ds['toar.mda8.mean']
    toar_da = toar_da.sortby('lon')
    toar_da = toar_da.sortby('time')

    # Load in cluster mask files
    mask1 = np.load(cluster_mask_file1)
    mask1 = np.reshape(mask1, (160, 320))
    mask_da1 = xr.DataArray(mask1, dims=['lat', 'lon'],
                            coords={'lat': momo_ds['lat'], 'lon': momo_ds['lon']})

    mask2 = np.load(cluster_mask_file2)
    mask2 = np.reshape(mask2, (160, 320))
    mask_da2 = xr.DataArray(mask2, dims=['lat', 'lon'],
                            coords={'lat': momo_ds['lat'], 'lon': momo_ds['lon']})

    # Keep only the data points defined by the cluster masks
    momo_ds1 = momo_ds.where(mask_da1, drop=True)
    toar_da1 = toar_da.where(mask_da1, drop=True)

    momo_ds2 = momo_ds.where(mask_da2, drop=True)
    toar_da2 = toar_da.where(mask_da2, drop=True)

    # Prepare data for making predictions
    x1 = momo_ds1.to_array()
    x1 = x1.stack({'loc': ['time', 'lat', 'lon']})
    x1 = x1.transpose('loc', 'variable')
    y1 = toar_da1.stack({'loc': ['time', 'lat', 'lon']})

    x2 = momo_ds2.to_array()
    x2 = x2.stack({'loc': ['time', 'lat', 'lon']})
    x2 = x2.transpose('loc', 'variable')
    y2 = toar_da2.stack({'loc': ['time', 'lat', 'lon']})

    # Remove NaNs
    x1 = x1.where(np.isfinite(x1), np.nan)
    x1 = x1.dropna('loc')
    y1 = y1.where(np.isfinite(y1), np.nan)
    y1 = y1.dropna('loc')
    x1, y1 = xr.align(x1, y1, copy=False)

    x2 = x2.where(np.isfinite(x2), np.nan)
    x2 = x2.dropna('loc')
    y2 = y2.where(np.isfinite(y2), np.nan)
    y2 = y2.dropna('loc')
    x2, y2 = xr.align(x2, y2, copy=False)

    var_names = x1.coords['variable'].values.tolist()
    if var_names != x2.coords['variable'].values.tolist():
        sys.exit()

    for ind, var_name in tqdm(enumerate(var_names)):
        plt.clf()
        plt.hist(x1.variable[:, ind], bins=100, histtype='step', density=True,
                 color='blue',
                 label=f'{os.path.splitext(os.path.basename(cluster_mask_file1))[0]} '
                       f'({len(x1.variable[:, ind])})')
        plt.hist(x2.variable[:, ind], bins=100, histtype='step', density=True,
                 color='orange',
                 label=f'{os.path.splitext(os.path.basename(cluster_mask_file2))[0]} '
                       f'({len(x2.variable[:, ind])})')
        plt.title(var_name)
        plt.legend(loc='best')
        out_file = os.path.join(out_dir, f'{var_name}.png')
        plt.savefig(out_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('momo_file', type=str)
    parser.add_argument('toar_file', type=str)
    parser.add_argument('cluster_mask_file1', type=str)
    parser.add_argument('cluster_mask_file2', type=str)
    parser.add_argument('out_dir', type=str)

    args = parser.parse_args()
    main(**vars(args))