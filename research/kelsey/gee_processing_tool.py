'''
Queries GEE dataset of interest and processes it to match momochem
grid, returns .nc file over AOI specified
@author: kdoerksen
'''

import os
import numpy as np
import ee
import argparse
import xarray as xr
import warnings
import h5py

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

parser = argparse.ArgumentParser(description='GEE Processing')
parser.add_argument("--gee_data", help="Google Earth Engine Dataset of interest. Must be one of: "
                                       "modis, pop, or fire", required=True)
parser.add_argument("--region", help="Boundary region on Earth to take data. Must be one of: "
                                     "globe, europe, asia, australia, north_america, west_europe, "
                                     "east_europe, west_na, east_na.")
parser.add_argument("--year", help="Year of query", required=True)
parser.add_argument("--month", help="Month of query", required=True)
parser.add_argument("--analysis_type", help="Type of analysis. Must be one of: collection, images", required=True)
parser.add_argument("--toar_only", help="Specify for only processing for TOAR locations.")

args = parser.parse_args()

# GEE datasets to query, can add more if you want to grab them!
gee_datasets = {
    'modis': {'name': 'modis',
              'prefix': 'modis',
              'data': 'MODIS/006/MCD12Q1',
              'band': 'LC_Type1',
              't_cadence': 'yearly'},
    'pop': {'name': 'population_density',
            'prefix': 'pop',
            'data': 'CIESIN/GPWv411/GPW_Population_Density',
            'band': 'population_density',
            't_cadence': 'yearly'},
    'fire': {'name': 'fire',
             'prefix': 'fire',
             'data': 'ESA/CCI/FireCCI/5_1',
             'band': 'LandCover',
             't_cadence': 'monthly'},
    'night_light': {'name': 'nightlight',
                    'data': 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG',
                    'prefix': 'nightlight',
                    'band': 'avg_rad',
                    't_cadence': 'monthly'}
}

name = gee_datasets[args.gee_data]['name']
gee_data = gee_datasets[args.gee_data]['data']
band = gee_datasets[args.gee_data]['band']
cadence = gee_datasets[args.gee_data]['t_cadence']
prefix = gee_datasets[args.gee_data]['prefix']
year = int(args.year)
analysis_type = args.analysis_type
month = args.month
#toar_analysis = args.toar_analysis

# Set boundaries to analyze by region
bbox_dict = {'globe': [-180, 180, -90, 90],
             'europe': [-20, 40, 25, 80],
             'asia': [110, 160, 10, 70],
             'australia': [130, 170, -50, -10],
             'north_america': [-140, -50, 10, 80],
             'west_europe': [-20, 10, 25, 80],
             'east_europe': [10, 40, 25, 80],
             'west_na': [-140, -95, 10, 80],
             'east_na': [-95, -50, 10, 80], }

region = args.region
#bbox = bbox_dict[region]

modis_lc = {
    0: {'class': 'filler',
        'pct_cov': 0},
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
          'pct_cov': 0},
    160: {'class': 'unburnt',
          'pct_cov': 0}
}

months_dict = {
    'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sept': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12,
}

# MOMO lat values
momo_lat = np.array([-89.142, -88.029, -86.911])

# MOMO lon values
momo_lon = np.array([-180., -178.875, -177.75])


def format_lon(x):
    '''
    Format longitude to use for subdividing regions
    Input: xarray dataset
    Output: xarray dataset
    '''
    x.coords['lon'] = (x.coords['lon'] + 180) % 360 - 180
    return x.sortby(x.lon)


def get_mode(x):
    '''
    Get mode of array --> useful for land cover datasets
    '''
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m]


def get_img_from_collect(data, collect_cadence, analysis_month, analysis_year):
    '''
    Get img of interest from collection.
    If yearly cadence, grab first img in collection
    If monthly, query over that month and grab first
    '''
    if collect_cadence == 'yearly':
        img = data.first()
    if collect_cadence == 'monthly':
        mon = months_dict[analysis_month]
        if mon != 'dec':
            im = data.filterDate('{}-{}-01'.format(analysis_year, mon),
                                 '{}-{}-01'.format(analysis_year, int(mon+1)))
        else:
            im = data.filterDate('{}-{}-01'.format(analysis_year, mon),
                                 '{}-01-01'.format(int(analysis_year+1)))
        img = im.first()

    return img


def get_perc_cov(img, lc_type):
    '''
    Calculate percent coverage of lc per class
    '''

    if lc_type == 'modis':
        feature_dict = modis_lc
    if lc_type == 'fire':
        feature_dict = fire_lc

    rows = len(img[:, 0])
    cols = len(img[0, :])
    total_pixels = rows * cols
    values, counts = np.unique(img, return_counts=True)

    values = values.tolist()
    counts = counts.tolist()

    # Removing data that is not categorized, this I think is due to bilinear interpolation
    new_vals = []
    new_counts = []
    for i in range(len(values)):
        if feature_dict.get(values[i]):
            new_vals.append(values[i])
            new_counts.append(counts[i])

    count_dict = {}
    for j in range(len(new_vals)):
        count_dict[new_vals[j]] = new_counts[j]

    for k, v in count_dict.items():
        pct_cov = (count_dict[k] / total_pixels)
        feature_dict[k]['pct_cov'] = pct_cov

    return feature_dict


def make_dataset(array, var, lats, lons):
    '''
    Makes an xarray dataset from an array with given
    lat, lon for specified variable
    '''
    xr_data = xr.DataArray(array, coords={'lat': lats, 'lon': lons})
    xr_dataset = xr_data.to_dataset(name='{}'.format(var))
    return xr_dataset


def process_collection(collection, c_band, c_cadence, lat, lon, c_month, analysis_year, dataset_name):
    '''
    Process GEE collection to return xarray of features
    '''
    # arrays for features from GEE
    mode_dat = []
    var_dat = []

    if dataset_name == 'nightlight' or 'pop':
        max_dat = []
        min_dat = []
        mean_dat = []

    if dataset_name == 'modis' or 'fire':
        pct_cov_1, pct_cov_2, pct_cov_3, pct_cov_4 = [], [], [], []
        pct_cov_5, pct_cov_6, pct_cov_7, pct_cov_8 = [], [], [], []
        pct_cov_9, pct_cov_10, pct_cov_11, pct_cov_12 = [], [], [], []
        pct_cov_13, pct_cov_14, pct_cov_15, pct_cov_16 = [], [], [], []
        pct_cov_17 = []
        punburnt_dat = []
        pct_cov_filler = []
    if dataset_name == 'modis':
        lc_pct_cov = [pct_cov_filler, pct_cov_1, pct_cov_2, pct_cov_3, pct_cov_4, pct_cov_5, pct_cov_6,
                   pct_cov_7, pct_cov_8, pct_cov_9, pct_cov_10, pct_cov_11, pct_cov_12,
                   pct_cov_13, pct_cov_14, pct_cov_15, pct_cov_16, pct_cov_17]

    mon = c_month
    # Select band type
    data = collection.select(c_band)
    # Get img from collection based on temporal cadence
    img = get_img_from_collect(data, c_cadence, mon, analysis_year)
    if dataset_name == 'fire':
        # resampling to 500m/pixel so array can be made otherwise too many pixels
        crs = 'EPSG:4326'
        img = img.resample('bilinear').reproject(crs=crs, scale=2000)

    for k in range(len(lat)):
        print('Processing for lat {} for all lon'.format(lat[k]))
        mode_arr = []
        var_arr = []
        mean_arr = []
        max_arr = []
        min_arr = []

        if dataset_name == 'modis' or 'fire':
            p1_arr, p2_arr, p3_arr, p4_arr, p5_arr = [], [], [], [], []
            p6_arr, p7_arr, p8_arr, p9_arr, p10_arr = [], [], [], [], []
            p11_arr, p12_arr, p13_arr, p14_arr, p15_arr = [], [], [], [], []
            p16_arr, p17_arr, punburnt_arr, pfiller = [], [], [], []
            lc_arrs = [pfiller, p1_arr, p2_arr, p3_arr, p4_arr, p5_arr, p6_arr, p7_arr,
                       p8_arr, p9_arr, p10_arr, p11_arr, p12_arr, p13_arr, p14_arr,
                       p15_arr, p16_arr, p17_arr]

        for m in range(len(lon)):
            print('Processing lon {}...'.format(lon[m]))
            point = ee.Geometry.Point(lon[m], lat[k])
            # Create 55km buffer will approx match the 1x1deg momo grid
            roi = point.buffer(55500)
            # Sample img over roi
            if dataset_name == 'modis':
                sq_extent = img.sampleRectangle(region=roi, defaultValue=0)
            if dataset_name == 'fire':
                sq_extent = img.sampleRectangle(region=roi, defaultValue=160)
            if dataset_name == 'nightlight' or 'pop':
                sq_extent = img.sampleRectangle(region=roi, defaultValue=0)

            # Get band of interest
            band_arr = sq_extent.get(c_band)
            np_arr = np.array(band_arr.getInfo())

            # Get basic stats
            mode_arr.append(get_mode(np_arr))
            var_arr.append(np.var(np_arr))
            mean_arr.append(np.mean(np_arr))
            max_arr.append(np.max(np_arr))
            min_arr.append(np.min(np_arr))

            if dataset_name == 'modis' or 'fire':
                pct_cov_all = get_perc_cov(np_arr, dataset_name)
            if dataset_name == 'modis':
                for i in range(len(lc_arrs)):
                    lc_arrs[i].append(pct_cov_all[i]['pct_cov'])
            if dataset_name == 'fire':
                p1_arr.append(pct_cov_all[0]['pct_cov'])
                p2_arr.append(pct_cov_all[20]['pct_cov'])
                p3_arr.append(pct_cov_all[30]['pct_cov'])
                p4_arr.append(pct_cov_all[40]['pct_cov'])
                p5_arr.append(pct_cov_all[50]['pct_cov'])
                p6_arr.append(pct_cov_all[60]['pct_cov'])
                p7_arr.append(pct_cov_all[70]['pct_cov'])
                p8_arr.append(pct_cov_all[80]['pct_cov'])
                p9_arr.append(pct_cov_all[90]['pct_cov'])
                p10_arr.append(pct_cov_all[100]['pct_cov'])
                p11_arr.append(pct_cov_all[110]['pct_cov'])
                p12_arr.append(pct_cov_all[120]['pct_cov'])
                p13_arr.append(pct_cov_all[130]['pct_cov'])
                p14_arr.append(pct_cov_all[140]['pct_cov'])
                p15_arr.append(pct_cov_all[150]['pct_cov'])
                p16_arr.append(pct_cov_all[170]['pct_cov'])
                p17_arr.append(pct_cov_all[180]['pct_cov'])
                punburnt_arr.append(pct_cov_all[160]['pct_cov'])

        mode_dat.append(mode_arr)
        var_dat.append(var_arr)

        if dataset_name == 'nightlight' or 'pop':
            mean_dat.append(mean_arr)
            min_dat.append(min_arr)
            max_dat.append(max_arr)

        if dataset_name == 'modis':
            for i in range(len(lc_pct_cov)):
                lc_pct_cov[i].append(lc_arrs[i])

        if dataset_name == 'fire':
            pct_cov_1.append(p1_arr)
            pct_cov_2.append(p2_arr)
            pct_cov_3.append(p3_arr)
            pct_cov_4.append(p4_arr)
            pct_cov_5.append(p5_arr)
            pct_cov_6.append(p6_arr)
            pct_cov_7.append(p7_arr)
            pct_cov_8.append(p8_arr)
            pct_cov_9.append(p9_arr)
            pct_cov_10.append(p10_arr)
            pct_cov_11.append(p11_arr)
            pct_cov_12.append(p12_arr)
            pct_cov_13.append(p13_arr)
            pct_cov_14.append(p14_arr)
            pct_cov_15.append(p15_arr)
            pct_cov_16.append(p16_arr)
            pct_cov_17.append(p17_arr)
            punburnt_dat.append(punburnt_arr)

    mode_all = np.array(mode_dat)
    var_all = np.array(var_dat)

    if dataset_name == 'fire':
        lc = 'fire'
        pct_1_all, pct_2_all, pct_3_all = np.array(pct_cov_1), np.array(pct_cov_2), np.array(pct_cov_3)
        pct_4_all, pct_5_all, pct_6_all = np.array(pct_cov_4), np.array(pct_cov_5), np.array(pct_cov_6)
        pct_7_all, pct_8_all, pct_9_all = np.array(pct_cov_7), np.array(pct_cov_8), np.array(pct_cov_9)
        pct_10_all, pct_11_all, pct_12_all = np.array(pct_cov_10), np.array(pct_cov_11), np.array(pct_cov_12)
        pct_13_all, pct_14_all, pct_15_all = np.array(pct_cov_13), np.array(pct_cov_14), np.array(pct_cov_15)
        pct_16_all, pct_17_all = np.array(pct_cov_16), np.array(pct_cov_17)
        punburnt_all = np.array(punburnt_dat)

        # Make xarray dataset
        mode_xr = make_dataset(mode_all, '{}.mode'.format(lc), lat, lon)
        var_xr = make_dataset(var_all, '{}.var'.format(lc), lat, lon)

        pct_1_xr = make_dataset(pct_1_all, lc + '.' + fire_lc[0]['class'], lat, lon)
        pct_2_xr = make_dataset(pct_2_all, lc + '.' + fire_lc[20]['class'], lat, lon)
        pct_3_xr = make_dataset(pct_3_all, lc + '.' + fire_lc[30]['class'], lat, lon)
        pct_4_xr = make_dataset(pct_4_all, lc + '.' + fire_lc[40]['class'], lat, lon)
        pct_5_xr = make_dataset(pct_5_all, lc + '.' + fire_lc[50]['class'], lat, lon)
        pct_6_xr = make_dataset(pct_6_all, lc + '.' + fire_lc[60]['class'], lat, lon)
        pct_7_xr = make_dataset(pct_7_all, lc + '.' + fire_lc[70]['class'], lat, lon)
        pct_8_xr = make_dataset(pct_8_all, lc + '.' + fire_lc[80]['class'], lat, lon)
        pct_9_xr = make_dataset(pct_9_all, lc + '.' + fire_lc[90]['class'], lat, lon)
        pct_10_xr = make_dataset(pct_10_all, lc + '.' + fire_lc[100]['class'], lat, lon)
        pct_11_xr = make_dataset(pct_11_all, lc + '.' + fire_lc[110]['class'], lat, lon)
        pct_12_xr = make_dataset(pct_12_all, lc + '.' + fire_lc[120]['class'], lat, lon)
        pct_13_xr = make_dataset(pct_13_all, lc + '.' + fire_lc[130]['class'], lat, lon)
        pct_14_xr = make_dataset(pct_14_all, lc + '.' + fire_lc[140]['class'], lat, lon)
        pct_15_xr = make_dataset(pct_15_all, lc + '.' + fire_lc[150]['class'], lat, lon)
        pct_16_xr = make_dataset(pct_16_all, lc + '.' + fire_lc[170]['class'], lat, lon)
        pct_17_xr = make_dataset(pct_17_all, lc + '.' + fire_lc[180]['class'], lat, lon)
        punburnt_xr = make_dataset(punburnt_all, lc + '.' + fire_lc[160]['class'], lat, lon)

        # Combine datasets
        combined_xr = var_xr.merge(pct_1_xr).merge(pct_2_xr).merge(pct_3_xr).merge(pct_4_xr).merge(
            pct_5_xr).merge(pct_6_xr).merge(pct_7_xr).merge(pct_8_xr).merge(pct_9_xr).merge(pct_10_xr).merge(
            pct_11_xr).merge(pct_12_xr).merge(pct_13_xr).merge(pct_14_xr).merge(pct_15_xr).merge(pct_16_xr).merge(
            pct_17_xr).merge(punburnt_xr).merge(mode_xr)

    if dataset_name == 'modis':
        lc = 'modis'
        # Make xarray dataset
        mode_xr = make_dataset(mode_all, '{}.mode'.format(lc), lat, lon)
        var_xr = make_dataset(var_all, '{}.var'.format(lc), lat, lon)
        combo_xr = mode_xr.merge(var_xr)
        pct_all = []
        pct_all_xr = []
        for i in range(len(lc_pct_cov)):
            pct_all.append(np.array(lc_pct_cov[i]))
            pct_all_xr.append(make_dataset(pct_all[i], lc + '.' + modis_lc[i]['class'], lat, lon))
            combined_xr = combo_xr.merge(pct_all_xr[0]).merge(pct_all_xr[1]).merge(pct_all_xr[2]).\
            merge(pct_all_xr[3]).merge(pct_all_xr[4]).merge(pct_all_xr[5]).merge(pct_all_xr[6]).\
            merge(pct_all_xr[7]).merge(pct_all_xr[8]).merge(pct_all_xr[9]).merge(pct_all_xr[10]).\
            merge(pct_all_xr[11]).merge(pct_all_xr[12]).merge(pct_all_xr[13]).merge(pct_all_xr[14]).\
            merge(pct_all_xr[15]).merge(pct_all_xr[16]).merge(pct_all_xr[17])

    if dataset_name == 'pop' or 'nightlight':
        mean_all = np.array(mean_dat)
        min_all = np.array(min_dat)
        max_all = np.array(max_dat)

        mode_xr = make_dataset(mode_all, '{}.mode'.format(dataset_name), lat, lon)
        var_xr = make_dataset(var_all, '{}.var'.format(dataset_name), lat, lon)
        mean_xr = make_dataset(mean_all, '{}.mean'.format(dataset_name), lat, lon)
        max_xr = make_dataset(max_all, '{}.max'.format(dataset_name), lat, lon)
        min_xr = make_dataset(min_all, '{}.min'.format(dataset_name), lat, lon)
        combined_xr = var_xr.merge(mode_xr).merge(mean_xr).merge(max_xr).merge(min_xr)

    return combined_xr

def process_collection_for_img(collection, img_band, img_cadence, lats, lons, save_dir, analysis_month, analysis_year):
    '''
    Processes image collection and saves images per lat, lon
    '''
    # Select band type
    data = collection.select(img_band)
    mon = analysis_month
    # Get img from collection based on temporal cadence
    img = get_img_from_collect(data, img_cadence, mon, analysis_year)
    for k in range(len(lats)):
        for m in range(len(lons)):
            point = ee.Geometry.Point(lons[m], lats[k])
            # Create 55km buffer will match the 1x1deg momo grid
            roi = point.buffer(55500)
            # Sample img over roi
            sq_extent = img.sampleRectangle(region=roi)
            # Get band of interest
            band_arr = sq_extent.get(band)
            try:
                np_arr = np.array(band_arr.getInfo())
            except:
                np_arr = np.nan
                continue

            # Read to hdf file
            with h5py.File('{}/{}_{}_data.hdf'.format(save_dir, lats[k], lons[m]), 'w') as outfile:
                h5_dataset = outfile.create_dataset('gee data', data=np_arr)
                h5_dataset.attrs['lat'] = lats[k]
                h5_dataset.attrs['lon'] = lons[m]
                h5_dataset.attrs['year'] = analysis_year


def get_toar_locations(x):
    '''
    Grab toar stations from xarray ds
    return list
    '''
    dataset = format_lon(x)
    points = dataset.to_dataframe().dropna().reset_index()[['lat', 'lon']].values
    station_list = []
    seen = set()
    for item in points:
        t = tuple(item)
        if not t in seen:
            station_list.append(item)
            seen.add(t)

    return station_list


# Authenticate ee
ee.Authenticate()

# Initialize ee
ee.Initialize()

# Grab data
dataset = ee.ImageCollection(gee_data).filterDate('{}-01-01'.
                                                  format(year), '{}-01-01'.format(year + 1))
print('processing {} for {} {}'.format(gee_data, month, year))

# Extract and save gee images for TOAR locations only -- old
'''
if toar_analysis:
    data_name = gee_datasets[args.gee_data]['name']
    toar_dir = '/Users/kelseydoerksen/gee/{}/jan_{}_TOAR'.format(data_name, year)
    toar_ds = xr.open_dataset('/Users/kelseydoerksen/exp_runs/rf/jan/all_gee_added/{}/test.target.nc'.format(year))
    toar_stations = get_toar_locations(toar_ds)
'''

# Process images and save individually
if analysis_type == 'images':
    # direct can be updated to whatever save directory you want
    direct = '/Users/kelseydoerksen/gee/{}/{}/{}'.format(name, year, month)
    process_collection_for_img(dataset, band, cadence, momo_lat, momo_lon, direct, month, year)

if analysis_type == 'collection':
    # Process data and save entire collection as nc
    processed_data = process_collection(dataset, band, cadence, momo_lat, momo_lon, month, year, name)

    # Save to Desktop for now
    processed_data.to_netcdf('/Users/kelseydoerksen/Desktop/{}_{}_{}.nc'.format(month, year, name))

    # Save file
    '''
    root_path = os.getcwd()
    processed_data.to_netcdf('{}/{}_{}_{}_mode.nc'.format(root_path, year, name, region))
    '''
