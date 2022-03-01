# Utility functions
#
# Steven Lu
# February 1, 2022


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


REQUIRED_VARS = ['aero_nh4', 'aero_no3', 'aero_sul', 'ch2o', 'co', 'date',
                 'hno3', 'lat', 'lon', 'o3', 'oh', 'pan', 'ps', 'q', 'so2', 't',
                 'toar', 'u', 'v']

TRAIN_FEATURES = ['aero_nh4', 'aero_no3', 'aero_sul', 'ch2o', 'co', 'hno3',
                  'oh', 'pan', 'ps', 'q', 'so2', 't', 'u', 'v']


def format_data(data, apply_toar_mask=True, latitude_min=-90, latitude_max=90,
                longitude_min=-180, longitude_max=180):
    # Define latitude and longitude masks
    lat = np.array(data['lat'])
    lon = np.array(data['lon']) - 180
    lat_mask = np.logical_and(lat >= latitude_min, lat <= latitude_max)
    lon_mask = np.logical_and(lon >= longitude_min, lon <= longitude_max)

    # The biases between ozone measurements of MOMO-Chem and TOAR are the target
    # values.
    o3 = filter_latlon(np.array(data['o3']), lat_mask, lon_mask)
    h, w, t = o3.shape
    o3 = o3.flatten()

    toar_mean = filter_latlon(np.array(data['toar/mean']), lat_mask, lon_mask)
    toar_mean = toar_mean.flatten()

    # Construct input features
    train_x = np.empty((h * w * t, len(TRAIN_FEATURES)),
                       dtype=np.float32)
    for ind, feat in enumerate(TRAIN_FEATURES):
        train_x[:, ind] = filter_latlon(np.array(data[feat]), lat_mask,
                                        lon_mask).flatten()

    if apply_toar_mask:
        toar_mask = ~np.isnan(toar_mean)
        train_x = train_x[toar_mask]
        train_y = o3[toar_mask] - toar_mean[toar_mask]
    else:
        train_y = o3 - toar_mean

    return train_x, train_y, lat[lat_mask], lon[lon_mask]


def filter_latlon(data, lat_mask, lon_mask):
    assert len(data.shape) == 3
    assert len(lat_mask.shape) == 1
    assert len(lon_mask.shape) == 1

    data = data[lat_mask]
    data = np.moveaxis(data, 0, 1)
    data = data[lon_mask]
    data = np.moveaxis(data, 0, 1)

    return data


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
    plt.xlabel('Ground Truth Bias (unit: ppb)')
    plt.ylabel('Predicted Bias (unit: ppb)')
    plt.xlim((0, max_value))
    plt.ylim((0, max_value))
    plt.legend(loc='upper right')
    plt.savefig(out_plot)
