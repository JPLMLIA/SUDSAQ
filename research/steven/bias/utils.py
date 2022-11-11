# Utility functions
#
# Steven Lu
# February 1, 2022


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import pairwise_distances
import matplotlib
import seaborn as sns
import scipy.cluster.hierarchy as sch


REQUIRED_VARS = ['aero_nh4', 'aero_no3', 'aero_sul', 'ch2o', 'co', 'date',
                 'hno3', 'lat', 'lon', 'o3', 'oh', 'pan', 'ps', 'q', 'so2', 't',
                 'toar', 'u', 'v']

TRAIN_FEATURES = ['aero_nh4', 'aero_no3', 'aero_sul', 'ch2o', 'co', 'hno3',
                  'oh', 'pan', 'ps', 'q', 'so2', 't', 'u', 'v']

ISD_FEATURES = ['WESD/mean', 'MNPN/mean', 'WT06/mean', 'SNOW/mean', 'PRCP/mean',
                'WT11/mean', 'TMAX/mean', 'AWND/mean', 'TOBS/mean', 'WESF/mean',
                'TAVG/mean', 'WT01/mean', 'WDFG/mean', 'WT02/mean', 'TSUN/mean',
                'WT03/mean', 'WSFG/mean', 'SNWD/mean', 'WDMV/mean', 'THIC/mean',
                'WT04/mean', 'PSUN/mean', 'TMIN/mean', 'WSFI/mean', 'WT08/mean',
                'MXPN/mean', 'WT05/mean']


def dendro_plot(tot_cont, feature_names, label):
    tot_cont = tot_cont[::100, :]
    nF = len(feature_names)
    D = pairwise_distances(tot_cont)
    H = sch.linkage(D, method='centroid')
    d1 = sch.dendrogram(H, orientation='right', no_plot=True)
    idx1 = d1['leaves']
    X = tot_cont[idx1, :]
    cmap1 = matplotlib.colors.ListedColormap(sns.color_palette('RdBu_r', 12))
    fsize = 16
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(X.T, aspect='auto', interpolation = 'none', cmap=cmap1)
    #plt.pcolor(X, cmap=cmap1)
    plt.yticks(np.arange(nF), feature_names)
    plt.clim([-0.3, 0.3])
    plt.xlabel('pixel index', fontsize=fsize)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fsize)
    plt.tick_params(labelsize=fsize)
    plt.title(label + ', n = ' + str(len(tot_cont)) + ', RF contributions')
    plt.tight_layout()
    return fig


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


# Additionally, returns toar mask
def format_data_v2(data, latitude_min=-90, latitude_max=90, longitude_min=-180,
                   longitude_max=180, bias_format=True):
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

    if bias_format:
        train_y = o3 - toar_mean
    else:
        train_y = toar_mean

    toar_mask = ~np.isnan(toar_mean)

    return train_x, train_y, toar_mask, lat[lat_mask], lon[lon_mask]


# With ISD data
def format_data_v3(data, isd_data, latitude_min=-90, latitude_max=90,
                   longitude_min=-180, longitude_max=180, bias_format=True):
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
    train_x = np.empty((h * w * t, len(TRAIN_FEATURES + ISD_FEATURES)),
                       dtype=np.float32)
    for ind, feat in enumerate(TRAIN_FEATURES):
        train_x[:, ind] = filter_latlon(np.array(data[feat]), lat_mask,
                                        lon_mask).flatten()

    mask = ~np.isnan(toar_mean)

    for ind, feat in enumerate(ISD_FEATURES):
        isd_ind = len(TRAIN_FEATURES) + ind
        train_x[:, isd_ind] = filter_latlon(np.array(isd_data[feat]), lat_mask,
                                            lon_mask).flatten()
        isd_mask = ~np.isnan(train_x[:, isd_ind])
        mask = np.logical_and(mask, isd_mask)

    print(train_x[0, :])

    if bias_format:
        train_y = o3 - toar_mean
    else:
        train_y = toar_mean

    return train_x, train_y, mask, lat[lat_mask], lon[lon_mask]


def filter_latlon(data, lat_mask, lon_mask):
    assert len(data.shape) == 3
    assert len(lat_mask.shape) == 1
    assert len(lon_mask.shape) == 1

    data = np.moveaxis(data, 0, 1)
    data = data[lat_mask]
    data = np.moveaxis(data, 0, 1)
    data = np.moveaxis(data, 2, 0)
    data = data[lon_mask]
    data = np.moveaxis(data, 0, 2)

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



