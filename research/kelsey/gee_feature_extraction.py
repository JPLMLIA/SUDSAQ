'''
Take GEE hdf files and extract features from them,
then match to TOAR station xarray
This really only works for TOAR locations,
any bigger of a dataset and python can't handle with df
@author: kdoerksen
'''

import h5py
import os
import xarray as xr
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Generate Features for RF from GEE data')
parser.add_argument("--dataset", help="Dataset type, must be one of modis or fire")
parser.add_argument("--year", help="Year to reference to extract data from")
parser.add_argument("--month", help="Month to reference to extract data from")
args = parser.parse_args()

yr = args.year
dataset = args.dataset
month = args.month

momo_lat = np.array([-89.142, -88.029, -86.911, -85.791, -84.67, -83.549, -82.428,
                -81.307, -80.185, -79.064, -77.943, -76.821, -75.7, -74.578,
                -73.457, -72.336, -71.214, -70.093, -68.971, -67.85, -66.728,
                -65.607, -64.485, -63.364, -62.242, -61.121, -60., -58.878,
                -57.757, -56.635, -55.514, -54.392, -53.271, -52.149, -51.028,
                -49.906, -48.785, -47.663, -46.542, -45.42, -44.299, -43.177,
                -42.056, -40.934, -39.813, -38.691, -37.57, -36.448, -35.327,
                -34.205, -33.084, -31.962, -30.841, -29.719, -28.598, -27.476,
                -26.355, -25.234, -24.112, -22.991, -21.869, -20.748, -19.626,
                -18.505, -17.383, -16.262, -15.14, -14.019, -12.897, -11.776,
                -11.654,  -9.5327, -8.4112,  -7.2897,  -6.1682,  -5.0467,
                -3.9252,  -2.8037, -1.6822, -0.56074, 0.56074, 1.6822, 2.8037,
                3.9252, 5.0467, 6.1682, 7.2897, 8.4112, 9.5327, 10.654, 11.776,
                12.897,  14.019,  15.14,  16.262,  17.383, 18.505, 19.626, 20.748,
                21.869,  22.991,  24.112, 25.234, 26.355, 27.476, 28.598, 29.719,
                30.841, 31.962, 33.084, 34.205, 35.327, 36.448, 37.57, 38.691,
                39.813, 40.934, 42.056, 43.177, 44.299, 45.42, 46.542, 47.663,
                48.785,  49.906,  51.028, 52.149, 53.271, 54.392, 55.514, 56.635,
                57.757, 58.878, 60., 61.121, 62.242, 63.364, 64.485, 65.607,
                66.728,  67.85,  68.971,  70.093,  71.214, 72.336, 73.457, 74.578,
                75.7,  76.821,  77.943, 79.064, 80.183, 81.307, 82.428, 83.549,
                84.67, 85.791, 86.911, 88.029, 89.142])

# MOMO lon values
momo_lon = np.array([-180., -178.875, -177.75, -176.625, -175.5, -174.375, -173.25,
                -172.125, -171., -169.875, -168.75, -167.625, -166.5, -165.375,
                -164.25, -163.125, -162., -160.875, -159.75, -158.625, -157.5,
                -156.375, -155.25, -154.125, -153., -151.875, -150.75, -149.625,
                -148.5, -147.375, -146.25, -145.125, -144., -142.875, -141.75,
                -140.625, -139.5, -138.375, -137.25, -136.125, -135., -133.875,
                -132.75, -131.625, -130.5, -129.375, -128.25, -127.125, -126.,
                -124.875, -123.75, -122.625, -121.5, -120.375, -119.25, -118.125,
                -117., -115.875, -114.75, -113.625, -112.5, -111.375, -110.25,
                -109.125, -108., -106.875, -105.75, -104.625, -103.5, -102.375,
                -101.25, -100.125, -99., -97.875, -96.75, -95.625,  -94.5, -93.375,
                -92.25, -91.125, -90., -88.875,  -87.75, -86.625, -85.5, -84.375,
                -83.25, -82.125,  -81., -79.875, -78.75, -77.625, -76.5, -75.375,
                -74.25, -73.125, -72., -70.875, -69.75, -68.625,  -67.5, -66.375,
                -65.25, -64.125, -63., -61.875,  -60.75, -59.625, -58.5, -57.375,
                -56.25, -55.125,  -54., -52.875, -51.75, -50.625, -49.5, -48.375,
                -47.25, -46.125, -45., -43.875, -42.75, -41.625,  -40.5, -39.375,
                -38.25, -37.125, -36., -34.875,  -33.75, -32.625, -31.5, -30.375,
                -29.25, -28.125,  -27., -25.875, -24.75, -23.625, -22.5, -21.375,
                -20.25, -19.125, -18., -16.875, -15.75, -14.625, -13.5, -12.375,
                -11.25, -10.125, -9., -7.875, -6.75, -5.625, -4.5, -3.375, -2.25,
                -1.125, 0., 1.125, 2.25, 3.375, 4.5, 5.625, 6.75 , 7.875, 9.,
                10.125, 11.25, 12.375, 13.5, 14.625, 15.75, 16.875, 18., 19.125,
                20.25, 21.375, 22.5, 23.625, 24.75, 25.875, 27., 28.125, 29.25,
                30.375, 31.5, 32.625, 33.75, 34.875, 36., 37.125,   38.25, 39.375,
                40.5, 41.625, 42.75, 43.875, 45., 46.125, 47.25, 48.375, 49.5,
                50.625, 51.75, 52.875, 54., 55.125, 56.25, 57.375, 58.5, 59.625,
                60.75, 61.875,63., 64.125, 65.25, 66.375, 67.5, 68.625, 69.75,
                70.875, 72., 73.125, 74.25, 75.375, 76.5, 77.625, 78.75, 79.875,
                81., 82.125, 83.25, 84.375, 85.5, 86.625, 87.75, 88.875, 90.,
                91.125, 92.25, 93.375, 94.5, 95.625, 96.75, 97.875, 99., 100.125,
                101.25, 102.375, 103.5, 104.625, 105.75, 106.875, 108., 109.125,
                110.25, 111.375, 112.5, 113.625, 114.75, 115.875, 117., 118.125,
                119.25, 120.375, 121.5,122.625, 123.75, 124.875, 126., 127.125,
                128.25, 129.375, 130.5, 131.625, 132.75, 133.875, 135., 136.125,
                137.25, 138.375, 139.5, 140.625, 141.75, 142.875, 144., 145.125,
                146.25, 147.375,  148.5, 149.625, 150.75, 151.875, 153., 154.125,
                155.25, 156.375, 157.5, 158.625, 159.75, 160.875, 162., 163.125,
                164.25, 165.375, 166.5, 167.625,  168.75 , 169.875, 171., 172.125,
                173.25, 174.375, 175.5, 176.625, 177.75, 178.875])


# Mapping class value to name for modis
modis_lc = {
    1: {'class': 'evg_conif',
        'pct_cov': 0},
    2: {'class': 'evg_broad',
        'pct_cov': 0},
    3: {'class': 'dcd_needle',
        'pct_cov': 0},
    4: {'class': 'dcd_broad',
        'pct_cov': 0},
    5: {'class': 'mix_forest',
        'pct_cov': 0},
    6: {'class': 'cls_shrub',
        'pct_cov': 0},
    7: {'class': 'open_shrub',
        'pct_cov': 0},
    8: {'class': 'woody_savanna',
        'pct_cov': 0},
    9: {'class': 'savanna',
        'pct_cov': 0},
    10: {'class': 'grassland',
         'pct_cov': 0},
    11: {'class': 'perm_wetland',
         'pct_cov': 0},
    12: {'class': 'cropland',
         'pct_cov': 0},
    13: {'class': 'urban',
         'pct_cov': 0},
    14: {'class': 'crop_nat_veg',
         'pct_cov': 0},
    15: {'class': 'perm_snow',
         'pct_cov': 0},
    16: {'class': 'barren',
         'pct_cov': 0},
    17: {'class': 'water_bds',
         'pct_cov': 0}
}

# Mapping class value to name for modis burned area
fire_lc = {
    0: {'class': 'crop_rain',
        'pct_cov': 0},
    20: {'class': 'crop_irr',
         'pct_cov': 0},
    30: {'class': 'crop_veg',
         'pct_cov': 0},
    40: {'class': 'veg_crop',
         'pct_cov': 0},
    50: {'class': 'broad_ever',
         'pct_cov': 0},
    60: {'class': 'broad_decid',
         'pct_cov': 0},
    70: {'class': 'needle_ever',
         'pct_cov': 0},
    80: {'class': 'needle_decid',
         'pct_cov': 0},
    90: {'class': 'tree_mixed',
         'pct_cov': 0},
    100: {'class': 'tree_shrub_herb',
          'pct_cov': 0},
    110: {'class': 'herb_tree_shrub',
          'pct_cov': 0},
    120: {'class': 'shrubland',
          'pct_cov': 0},
    130: {'class': 'grassland',
          'pct_cov': 0},
    140: {'class': 'lichen_moss',
          'pct_cov': 0},
    150: {'class': 'sparse_veg',
          'pct_cov': 0},
    170: {'class': 'tree_flooded',
          'pct_cov': 0},
    180: {'class': 'shrub_herb_flood',
          'pct_cov': 0}
}

def get_mode(x):
    '''
    Get mode of array --> useful for land cover datasets
    '''
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m]


def get_pct_coverage(img, lc_type):
    '''
    Get the percentage coverage per img
    for modis/fire landcover
    '''

    if lc_type == 'modis':
        feature_dict = modis_lc
    if lc_type == 'fire':
        feature_dict = fire_lc

    if img.ndim == 1:
        for k, v in feature_dict.items():
            feature_dict[k]['pct_cov'] = np.nan
    else:
        rows = len(img[:, 0])
        cols = len(img[0, :])
        total_pixels = rows * cols
        values, counts = np.unique(img, return_counts=True)

        count_dict = {}
        for j in range(len(values)):
            count_dict[values[j]] = counts[j]

        for k, v in count_dict.items():
            pct_cov = (count_dict[k]/total_pixels)
            feature_dict[k]['pct_cov'] = pct_cov

    new_dict = {}
    for k, v in feature_dict.items():
        new_key = feature_dict[k]['class'] + '_pct_cov'
        new_val = feature_dict[k]['pct_cov']
        new_dict[new_key] = new_val

    return new_dict


df_var = pd.DataFrame(index=momo_lat, columns=momo_lon)
df_mode = pd.DataFrame(index=momo_lat, columns=momo_lon)
modis_df_list = []
for i in range(len(modis_lc)):
    modis_df_list.append(pd.DataFrame(index=momo_lat, columns=momo_lon))

toar_dir = '/Users/kelseydoerksen/gee/{}/{}_{}_TOAR'.format(dataset, month, yr)

all_data = []
for fn in os.listdir(toar_dir):
    img_dict = {'lat': None, 'lon': None, 'var': None}
    file = os.path.join(toar_dir, fn)
    with h5py.File(os.path.join(toar_dir, file), 'r') as f:
        data = f['gee data']
        try:
            data_arr = data[:]
        except:
            data_arr = np.array([np.NaN])

        lat = data.attrs['lat']
        lon = data.attrs['lon']

        data_var = np.var(data_arr)
        data_mode = get_mode(data_arr)
        pct_cov_dict = get_pct_coverage(data_arr, dataset)

        img_dict['lat'] = lat
        img_dict['lon'] = lon
        img_dict['var'] = data_var
        img_dict['mode'] = data_mode
        img_dict.update(pct_cov_dict)

    all_data.append(img_dict)

ds_list = []
if dataset == "modis":
    for item in all_data:
        df_var.at[item['lat'], item['lon']] = item['var']
        df_mode.at[item['lat'], item['lon']] = item['mode']
        for df in range(len(modis_df_list)):
            var_name = modis_lc[df + 1]['class'] + '_pct_cov'
            modis_df_list[df].at[item['lat'], item['lon']] = item[var_name]
            ds = xr.Dataset(data_vars=dict(data_var=(['lat', 'lon'], modis_df_list[df].values)),
                            coords=dict(lon=momo_lon, lat=momo_lat))
            ds = ds.rename({'data_var': var_name})
            ds_list.append(ds)

elif dataset == "fire":
    for item in all_data:
        df_var.at[item['lat'], item['lon']] = item['var']
        df_mode.at[item['lat'], item['lon']] = item['mode']
        df_1.at[item['lat'], item['lon']] = item['crop_rain']
        df_2.at[item['lat'], item['lon']] = item['crop_irr']
        df_3.at[item['lat'], item['lon']] = item['crop_veg']
        df_4.at[item['lat'], item['lon']] = item['veg_crop']
        df_5.at[item['lat'], item['lon']] = item['broad_ever']
        df_6.at[item['lat'], item['lon']] = item['broad_decid']
        df_7.at[item['lat'], item['lon']] = item['needle_ever']
        df_8.at[item['lat'], item['lon']] = item['needle_decid']
        df_9.at[item['lat'], item['lon']] = item['tree_mixed']
        df_10.at[item['lat'], item['lon']] = item['tree_shrub_herb']
        df_11.at[item['lat'], item['lon']] = item['herb_tree_shrub']
        df_12.at[item['lat'], item['lon']] = item['shrubland']
        df_13.at[item['lat'], item['lon']] = item['grassland']
        df_14.at[item['lat'], item['lon']] = item['lichen_moss']
        df_15.at[item['lat'], item['lon']] = item['sparse_veg']
        df_16.at[item['lat'], item['lon']] = item['tree_flooded']
        df_17.at[item['lat'], item['lon']] = item['shrub_herb_flood']

    ds_1 = xr.Dataset(data_vars=dict(crop_rain=(['lat', 'lon'], df_1.values)), coords=dict(lon=momo_lon, lat=momo_lat))
    ds_2 = xr.Dataset(data_vars=dict(crop_irr=(['lat', 'lon'], df_2.values)), coords=dict(lon=momo_lon, lat=momo_lat))
    ds_3 = xr.Dataset(data_vars=dict(crop_veg=(['lat', 'lon'], df_3.values)), coords=dict(lon=momo_lon, lat=momo_lat))
    ds_4 = xr.Dataset(data_vars=dict(veg_crop=(['lat', 'lon'], df_4.values)), coords=dict(lon=momo_lon, lat=momo_lat))
    ds_5 = xr.Dataset(data_vars=dict(broad_ever=(['lat', 'lon'], df_5.values)), coords=dict(lon=momo_lon, lat=momo_lat))
    ds_6 = xr.Dataset(data_vars=dict(broad_decid=(['lat', 'lon'], df_6.values)), coords=dict(lon=momo_lon, lat=momo_lat))
    ds_7 = xr.Dataset(data_vars=dict(needle_ever=(['lat', 'lon'], df_7.values)), coords=dict(lon=momo_lon, lat=momo_lat))
    ds_8 = xr.Dataset(data_vars=dict(needle_decid=(['lat', 'lon'], df_8.values)), coords=dict(lon=momo_lon, lat=momo_lat))
    ds_9 = xr.Dataset(data_vars=dict(tree_mixed=(['lat', 'lon'], df_9.values)), coords=dict(lon=momo_lon, lat=momo_lat))
    ds_10 = xr.Dataset(data_vars=dict(tree_shrub_herb=(['lat', 'lon'], df_10.values)), coords=dict(lon=momo_lon, lat=momo_lat))
    ds_11 = xr.Dataset(data_vars=dict(herb_tree_shrub=(['lat', 'lon'], df_11.values)), coords=dict(lon=momo_lon, lat=momo_lat))
    ds_12 = xr.Dataset(data_vars=dict(shrubland=(['lat', 'lon'], df_12.values)), coords=dict(lon=momo_lon, lat=momo_lat))
    ds_13 = xr.Dataset(data_vars=dict(grassland=(['lat', 'lon'], df_13.values)), coords=dict(lon=momo_lon, lat=momo_lat))
    ds_14 = xr.Dataset(data_vars=dict(lichen_moss=(['lat', 'lon'], df_14.values)), coords=dict(lon=momo_lon, lat=momo_lat))
    ds_15 = xr.Dataset(data_vars=dict(sparse_veg=(['lat', 'lon'], df_15.values)), coords=dict(lon=momo_lon, lat=momo_lat))
    ds_16 = xr.Dataset(data_vars=dict(tree_flooded=(['lat', 'lon'], df_16.values)), coords=dict(lon=momo_lon, lat=momo_lat))
    ds_17 = xr.Dataset(data_vars=dict(shrub_herb_flood=(['lat', 'lon'], df_17.values)), coords=dict(lon=momo_lon, lat=momo_lat))


ds_var = xr.Dataset(data_vars=dict(lc_var=(['lat', 'lon'], df_var.values)), coords=dict(lon=momo_lon, lat=momo_lat))
ds_mode = xr.Dataset(data_vars=dict(lc_mode=(['lat', 'lon'], df_mode.values)), coords=dict(lon=momo_lon, lat=momo_lat))
ds = ds_var.merge(ds_mode)
for xr_data in range(len(ds_list)):
    ds = ds.merge(ds_list[xr_data])

ds.to_netcdf('/Users/kelseydoerksen/exp_runs/rf/{}/all_gee_added/{}/{}_{}_lc_no_time.nc'.format(month, yr, yr,
                                                                                                dataset))