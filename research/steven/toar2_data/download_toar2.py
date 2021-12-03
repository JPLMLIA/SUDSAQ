#!/usr/bin/env python
# This script provides functionalities to download TOAR-2 data set using the
# REST APIs at https://join.fz-juelich.de/services/rest/surfacedata/
#
# Steven Lu
# December 1, 2021

import os
import sys
import yaml
import json
from tqdm import tqdm
from urllib.parse import quote
from urllib.request import urlopen


BASE_URL = 'https://join.fz-juelich.de/services/rest/surfacedata/'


def check_required_arguments(config):
    if 'parameters' not in config.keys():
        print('[ERROR] "parameters" is a required argument in the config file.')
        sys.exit(1)

    if 'networks' not in config.keys():
        print('[ERROR] "networks" is a required argument in the config file. ')
        sys.exit(1)

    if 'stations' not in config.keys():
        print('[ERROR] "stations" is a required argument in the config file.')
        sys.exit(1)

    if 'metrics' not in config.keys():
        print('[ERROR] "metrics" is a required argument in the config file.')
        sys.exit(1)


def get_all_parameters(base_url):
    query_url = '%s/parameters/?format=json' % base_url
    response = urlopen(query_url)

    if response.code != 200:
        print('[ERROR] parameter query failed. Please verify the REST API %s' %
              query_url)
        sys.exit(1)

    parameters = json.load(response)

    return parameters


def get_all_networks(base_url):
    query_url = '%s/networks/?format=json' % base_url
    response = urlopen(query_url)

    if response.code != 200:
        print('[ERROR] network query failed. Please verify the REST API %s' %
              query_url)
        sys.exit(1)

    networks = json.load(response)

    return networks


def get_all_stations(base_url):
    query_url = '%s/stations/?format=json' % base_url
    response = urlopen(query_url)

    if response.code != 200:
        print('[ERROR] station query failed. Please verify the REST API %s' %
              query_url)
        sys.exit(1)

    stations = json.load(response)
    stations = [s[1] for s in stations]

    return stations


# Construct series using parameters. The resulting series will be further
# filtered by networks and stations to remove networks and stations from which
# we don't want to download.
def construct_series(parameters, networks, stations, base_url):
    series_dict = dict()
    for param in tqdm(parameters, total=len(parameters),
                      desc='Constructing series'):
        query_url = '%s/series/?parameter_name=%s&format=json&as_dict=True' % \
                    (base_url, param)
        response = urlopen(query_url)
        if response.code != 200:
            print('[ERROR] Cannot query parameter %s. Please verify the REST '
                  'API %s' % (param, query_url))
            sys.exit(1)

        resp_series = json.load(response)
        series_list = list()

        # Filter series list to remove unwanted networks and stations
        for record in tqdm(resp_series, total=len(resp_series), desc=param):
            if record['network_name'] in networks and \
                    record['station_id'] in stations:
                series_list.append(record)

        series_dict.setdefault(param, series_list)

    return series_dict


def create_hierarchical_output_directories(series_dict, out_dir):
    for records in tqdm(series_dict.values(),
                        desc='Creating hierarchical output directories'):
        for r in records:
            sub_dir = os.path.join(out_dir, '%s/%s/' % (r['network_name'],
                                                        r['station_id']))
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)


def main(config_file, out_dir, base_url, force_overwrite):
    if not os.path.exists(config_file):
        print('[ERROR] Config file not found: %s' % os.path.abspath(config_file))
        sys.exit(1)

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    check_required_arguments(config)

    # Get required arguments
    parameters = config['parameters']
    networks = config['networks']
    stations = config['stations']
    sampling = config['stats_sampling']
    metrics = ','.join(config['metrics'])

    # Get optional arguments
    date_range = ''
    if 'start_date' in config.keys() and 'end_date' in config.keys():
        date_range = '[%s,%s]' % (quote(config['start_date']),
                                  quote(config['end_date']))

    if parameters == 'all':
        parameters = get_all_parameters(base_url)

    if networks == 'all':
        networks = get_all_networks(base_url)

    if stations == 'all':
        stations = get_all_stations(base_url)

    # Construct series which will be needed to download the data sets
    series_dict = construct_series(parameters, networks, stations, base_url)

    # Download data sets
    out_failed = os.path.join(out_dir, 'failed.txt')
    for param, records in series_dict.items():
        for r in tqdm(records, total=len(records),
                      desc='Downloading %s data sets' % param):
            id = r['id']
            network_name = r['network_name']
            station_id = r['station_id']

            # Create sub directory
            sub_dir = os.path.join(out_dir, '%s/%s/' % (network_name,
                                                        station_id))
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

            out_file = os.path.join(sub_dir, '%s-%s-%s.json' % (network_name,
                                                                station_id,
                                                                param))

            # If the data set already exists and the "force_overwrite" option
            # isn't turned on, we will skip this data set.
            if os.path.exists(out_file) and not force_overwrite:
                continue

            # Create query URL
            if len(date_range) == 0:
                query_url = '%s/stats/?sampling=%s&statistics=%s&id=%s&' \
                            'format=json' % (base_url, sampling, metrics, id)
            else:
                query_url = '%s/stats/?sampling=%s&statistics=%s&daterange=%s&'\
                            'id=%s&format=json' % (base_url, sampling, metrics,
                                                   date_range, id)
            try:
                response = urlopen(query_url)
                data = json.load(response)

                with open(out_file, 'w') as f:
                    json.dump(data, f)
            except Exception:
                with open(out_failed, 'a') as f:
                    f.write('%s\n' % query_url)
                continue


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('--base_url', type=str, default=BASE_URL)
    parser.add_argument('--force_overwrite', action='store_true',
                        help='This argument controls whether or not to '
                             're-download the existing date sets. The default '
                             'is False, which will not re-download the '
                             'existing data sets.')

    args = parser.parse_args()
    main(**vars(args))
