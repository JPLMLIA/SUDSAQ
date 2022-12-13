#!/usr/bin/env python3
# Step 2 feature extraction script:
# Compute feature values (e.g., mean, std, etc for continuous variables;
# entropy and skewness for discrete variables) using the .h5 file generated
# from the step 1 feature extraction script (i.e., extract_metadata_from_toar.py)
#
# Steven Lu

import h5py
import numpy as np
import xarray as xr
from scipy import stats
from sklearn.preprocessing import LabelBinarizer


def feature_extraction_continous_var(toar_data, feature_name, out_data,
                                     momo_lat, momo_lon, station_lat, station_lon):
    # The strings in the statistics list must match the statistic argument of
    # scipy's binned_statistic_2d
    statistics = ['mean', 'std', 'median', 'count', 'min', 'max']

    for stat in statistics:
        ret = stats.binned_statistic_2d(
            station_lat,
            station_lon,
            np.array(toar_data[feature_name]),
            statistic=stat,
            bins=[momo_lat, momo_lon],
            expand_binnumbers=True
        )

        out_data[f'toar.{feature_name}.{stat}'] = (('lat', 'lon'), ret.statistic)


def feature_extraction_discrete_var(toar_data, feature_name, out_data, momo_lat,
                                    momo_lon, station_lat, station_lon):
    try:
        all_values = np.array(toar_data[feature_name]).astype(np.str_)
        unique_values = np.unique(all_values).astype(np.str_)
    except Exception:
        all_values = np.array(toar_data[feature_name]).astype(np.str_)
        unique_values = np.unique(all_values)
        unique_values = np.array([uv.decode('utf-8') for uv in unique_values])
        unique_values = unique_values.astype(np.str_)

    lb = LabelBinarizer()
    lb.fit(unique_values)

    def _feature_entropy(binned_values):
        onehot = lb.transform(binned_values)

        hist = np.sum(onehot, axis=0)
        entropy = stats.entropy(hist, base=2)

        return entropy

    def _feature_skewness(binned_values):
        onehot = lb.transform(binned_values)

        hist = np.sum(onehot, axis=0)
        skewness = stats.skew(hist)

        return skewness

    ret = stats.binned_statistic_2d(
        station_lat,
        station_lon,
        all_values,
        statistic=_feature_entropy,
        bins=[momo_lat, momo_lon],
        expand_binnumbers=True
    )
    out_data[f'toar.{feature_name}.entropy'] = (('lat', 'lon'), ret.statistic)

    ret = stats.binned_statistic_2d(
        station_lat,
        station_lon,
        all_values,
        statistic=_feature_skewness,
        bins=[momo_lat, momo_lon],
        expand_binnumbers=True
    )
    out_data[f'toar.{feature_name}.skewness'] = (('lat', 'lon'), ret.statistic)


def main(momo_file, toar_metadata_file, out_file):
    momo_data = xr.open_mfdataset(momo_file, parallel=True)
    toar_data = h5py.File(toar_metadata_file, 'r')
    out_data = xr.Dataset(
        coords={
            'lat': momo_data['lat'],
            'lon': momo_data['lon']
        }
    )

    lat = np.hstack([momo_data['lat'], 90.])
    lon = np.hstack([momo_data['lon'], 360.])

    # Convert TOAR2 longitude from (-180, 180) to (0, 360)
    station_lat = np.array(toar_data['station_lat'])
    station_lon = np.array(toar_data['station_lon'])
    station_lon[station_lon < 0] = station_lon[station_lon < 0] + 360

    # Discrete variables
    feature_extraction_discrete_var(
        toar_data, 'station_type', out_data, lat, lon, station_lat, station_lon)
    feature_extraction_discrete_var(
        toar_data, 'station_type_of_area', out_data, lat, lon, station_lat, station_lon)
    feature_extraction_discrete_var(
        toar_data, 'station_category', out_data, lat, lon, station_lat, station_lon)
    # feature_extraction_discrete_var(
        # toar_data, 'station_country', out_data, lat, lon, station_lat, station_lon)
    # feature_extraction_discrete_var(
    #     toar_data, 'station_state', out_data, lat, lon, station_lat, station_lon)
    feature_extraction_discrete_var(
        toar_data, 'station_climatic_zone', out_data, lat, lon, station_lat, station_lon)
    feature_extraction_discrete_var(
        toar_data, 'station_toar_category', out_data, lat, lon, station_lat, station_lon)
    feature_extraction_discrete_var(
        toar_data, 'station_htap_region', out_data, lat, lon, station_lat, station_lon)
    feature_extraction_discrete_var(
        toar_data, 'station_alt_flag', out_data, lat, lon, station_lat, station_lon)
    feature_extraction_discrete_var(
        toar_data, 'station_coordinate_status', out_data, lat, lon, station_lat, station_lon)
    feature_extraction_discrete_var(
        toar_data, 'station_dominant_landcover', out_data, lat, lon, station_lat, station_lon)

    # Continuous variables
    feature_extraction_continous_var(
        toar_data, 'station_alt', out_data, lat, lon, station_lat, station_lon)
    feature_extraction_continous_var(
        toar_data, 'station_nightlight_5km', out_data, lat, lon, station_lat, station_lon)
    feature_extraction_continous_var(
        toar_data, 'station_wheat_production', out_data, lat, lon, station_lat, station_lon)
    feature_extraction_continous_var(
        toar_data, 'station_rice_production', out_data, lat, lon, station_lat, station_lon)
    feature_extraction_continous_var(
        toar_data, 'station_nox_emissions', out_data, lat, lon, station_lat, station_lon)
    feature_extraction_continous_var(
        toar_data, 'station_omi_no2_column', out_data, lat, lon, station_lat, station_lon)
    feature_extraction_continous_var(
        toar_data, 'station_reported_alt', out_data, lat, lon, station_lat, station_lon)
    feature_extraction_continous_var(
        toar_data, 'station_google_alt', out_data, lat, lon, station_lat, station_lon)
    feature_extraction_continous_var(
        toar_data, 'station_etopo_alt', out_data, lat, lon, station_lat, station_lon)
    feature_extraction_continous_var(
        toar_data, 'station_etopo_min_alt_5km', out_data, lat, lon, station_lat, station_lon)
    feature_extraction_continous_var(
        toar_data, 'station_etopo_relative_alt', out_data, lat, lon, station_lat, station_lon)
    feature_extraction_continous_var(
        toar_data, 'station_max_population_density_25km', out_data, lat, lon, station_lat, station_lon)
    feature_extraction_continous_var(
        toar_data, 'station_nightlight_1km', out_data, lat, lon, station_lat, station_lon)
    feature_extraction_continous_var(
        toar_data, 'station_population_density', out_data, lat, lon, station_lat, station_lon)
    feature_extraction_continous_var(
        toar_data, 'google_resolution', out_data, lat, lon, station_lat, station_lon)
    feature_extraction_continous_var(
        toar_data, 'station_max_population_density_5km', out_data, lat, lon, station_lat, station_lon)

    out_data.to_netcdf(out_file, mode='w', engine='netcdf4')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('momo_file', type=str,
                        help='This momo file must have the lat/lon coordinates '
                             'for the TOAR2 file to match to')
    parser.add_argument('toar_metadata_file', type=str)
    parser.add_argument('out_file', type=str)

    args = parser.parse_args()
    main(**vars(args))
