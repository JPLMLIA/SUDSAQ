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
                                       "modis, pop, or fire")
parser.add_argument("--region", help="Boundary region on Earth to take data. Must be one of: "
                                     "globe, europe, asia, australia, north_america, west_europe, "
                                     "east_europe, west_na, east_na.")
parser.add_argument("--year", help="Year of query")
parser.add_argument("--month", help="Month of query")
parser.add_argument("--analysis_type", help="Type of analysis. Stats for returning statistical features, "
                                            "full for full image return")

args = parser.parse_args()


# GEE datasets to query, can add more if you want to grab them!
gee_datasets = {
    'modis': {'name': 'modis',
              'data': 'MODIS/006/MCD12Q1',
              'band': 'LC_Type1',
              't_cadence': 'yearly'},
    'pop': {'name': 'population_density',
                   'data': 'CIESIN/GPWv411/GPW_Population_Density',
                   'band': 'population_density',
                   't_cadence': 'yearly'},
    'fire': {'name': 'fire',
             'data': 'ESA/CCI/FireCCI/5_1',
             'band': 'LandCover',
             't_cadence': 'monthly'}
}

name = gee_datasets[args.gee_data]['name']
gee_data = gee_datasets[args.gee_data]['data']
band = gee_datasets[args.gee_data]['band']
cadence = gee_datasets[args.gee_data]['t_cadence']
year = int(args.year)
analysis_type = args.analysis_type
month = args.month

# Set boundaries to analyze by region
bbox_dict = {'globe':[-180, 180, -90, 90],
            'europe': [-20, 40, 25, 80],
            'asia': [110, 160, 10, 70],
            'australia': [130, 170, -50, -10],
            'north_america': [-140, -50, 10, 80],
            'west_europe': [-20, 10, 25, 80],
            'east_europe': [10, 40, 25, 80],
            'west_na': [-140, -95, 10, 80],
            'east_na': [-95, -50, 10, 80], }

region = args.region
bbox = bbox_dict[region]

months_dict = {
    'jan': {
        'val': '01',
        'last_day': '31'
            },
    'feb': {
        'val': '02',
        'last_day': '28'},
    'mar': {
        'val': '03',
        'last_day': '31'
    },
    'apr': {
        'val': '04',
        'last_day': '30'
    },
    'may': {
        'val': '05',
        'last_day': '31'
    },
    'jun': {
        'val': '06',
        'last_day': '30'
    },
    'jul': {
        'val': '07',
        'last_day': '31'},
    'aug': {
        'val': '08',
        'last_day': '31'},
    'sept': {
        'val': '09',
        'last_day': '30'},
    'oct': {
        'val': '10',
        'last_day': '31'
    },
    'nov': {
        'val': '11',
        'last_day': '30'},
    'dec': {
        'val': '12',
        'last_day': '31'}
}

# To-do, can put these lat, lon into text files to grab for more generalized
# MOMO lat values
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
    else take mean
    '''
    if collect_cadence == 'yearly':
        img = data.first()
    if collect_cadence == 'monthly':
        mon = months_dict[analysis_month]['val']
        last_day = months_dict[analysis_month]['last_day']
        im = data.filterDate('{}-{}-01'.format(analysis_year, mon),
                              '{}-{}-{}'.format(analysis_year, mon, last_day))
        img = im.first()

    return img


def make_dataset(array, var, lats, lons):
    '''
    Makes an xarray dataset from an array with given
    lat, lon for specified variable
    '''

    xr_data = xr.DataArray(array, coords={'lat': lats, 'lon': lons})
    xr_dataset = xr_data.to_dataset(name='{}'.format(var))
    return xr_dataset


def process_collection(collection, band, cadence, lat, lon, month, analysis_year):
    '''
    Process GEE collection to return ndarray
    '''
    # arrays for statistical features from GEE
    all_dat = []
    mode_dat = []
    mon = month
    # Select band type
    data = collection.select(band)
    # Get img from collection based on temporal cadence
    img = get_img_from_collect(data, cadence, mon, analysis_year)
    for k in range(len(lat)):
        print('Processing for lat {} for all lon'.format(lat[k]))
        arr = []
        mode_arr = []
        for m in range(len(lon)):
            point = ee.Geometry.Point(lon[m], lat[k])
            # Create 55km buffer will match the 1x1deg momo grid
            roi = point.buffer(55500)
            # Sample img over roi
            sq_extent = img.sampleRectangle(region=roi)
            # Get band of interest
            band_arr = sq_extent.get(band)
            try:
                np_arr = np.array(band_arr.getInfo())
                import ipdb
                ipdb.set_trace()
                arr.append(np_arr)
                mode_arr.append(get_mode(np_arr))
            except:
                arr.append(np.nan)
                mode_arr.append(np.nan)

        all_dat.append(arr)
        mode_dat.append(mode_arr)

    arr_all = np.array(all_dat)
    mode_all = np.array(mode_dat)

    # Make xr datasets
    mode_xr = make_dataset(mode_all, 'mode', lat, lon)

    # Combine datasets
    combined_xr = mode_xr

    return combined_xr

def process_collection_for_img(collection, band, cadence, locations, toar_dir, month, analysis_year):
    '''
    Processes image collection and saves image for the toar locations
    '''
    # Select band type
    data = collection.select(band)
    mon = month
    # Get img from collection based on temporal cadence
    img = get_img_from_collect(data, cadence, mon, analysis_year)
    print('Number of stations is: {}'.format(len(locations)))
    for station in locations:
        lat = station[0]
        lon = station[1]
        point = ee.Geometry.Point(lon, lat)
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

        # Read to hdf file
        with h5py.File('{}/{}_{}_data.hdf'.format(toar_dir, lat, lon), 'w') as outfile:
            h5_dataset = outfile.create_dataset('gee data', data= np_arr)
            h5_dataset.attrs['lat'] = lat
            h5_dataset.attrs['lon'] = lon
            h5_dataset.attrs['year'] = analysis_year

def get_toar_locations(dataset):
    '''
    Grab toar stations from xarray
    return list
    '''
    dataset = format_lon(dataset)
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
                                                  format(year), '{}-01-01'.format(year+1))
print('processing {} for {} {}'.format(gee_data, month, year))


# Extract and save gee data for TOAR locations to begin with
data_name = gee_datasets[args.gee_data]['name']
data_name = 'pop'
toar_dir = '/Users/kelseydoerksen/gee/{}/jan_{}_TOAR'.format(data_name, year)
toar_ds = xr.open_dataset('/Users/kelseydoerksen/exp_runs/rf/jan/all_gee_added/{}/test.target.nc'.format(year))
toar_stations = get_toar_locations(toar_ds)
process_collection_for_img(dataset, band, cadence, toar_stations, toar_dir, month, year)
'''

# Process data with calculations
processed_data = process_collection(dataset, band, cadence, momo_lat, momo_lon, month, year)

# Save file
root_path = os.getcwd()
processed_data.to_netcdf('{}/{}_{}_{}_mode.nc'.format(root_path, year, name, region))
'''