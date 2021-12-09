#!/usr/bin/env python
# This script generates station maps for TOAR-2 data sets downloaded by the
# download_toar2.py script.
#
# Steven Lu
# December 8, 2021

import os
import sys
import json
import numpy as np
from tqdm import tqdm
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


# Define a unique color for each TOAR-2 network
NETWORK_COLORS = {
    'AIRBASE': (230, 25, 75),
    'AIRMAP': (60, 180, 75),
    'AQS': (255, 225, 25),
    'AUSAQN': (0, 130, 200),
    'CAPMON': (245, 130, 48),
    'CASTNET': (145, 30, 180),
    'EANET': (70, 240, 240),
    'EMEP': (240, 50, 230),
    'GAW': (210, 245, 60),
    'ISRAQN': (250, 190, 212),
    'KRAQN': (0, 128, 128),
    'NAPS': (220, 190, 255),
    'NIES': (170, 110, 40),
    'OTHER': (255, 250, 200),
    'RSA': (128, 0, 0),
    'UBA': (170, 255, 195)
}


def main(data_root_dir, out_file, parameter):
    if not os.path.exists:
        print('[ERROR] Data root directory does not exist.')
        sys.exit(1)

    network_names = [f for f in os.listdir(data_root_dir)
                     if os.path.isdir(os.path.join(data_root_dir, f))]
    data_dict = dict()
    for network_name in tqdm(network_names, desc='Visualizing networks'):
        if network_name not in NETWORK_COLORS.keys():
            print('[ERROR] Network name %s is not recognized' % network_name)
            sys.exit(1)

        data_dict.setdefault(network_name, dict())

        station_ids = os.listdir(os.path.join(data_root_dir, network_name))
        for station_id in tqdm(station_ids, desc=network_name):
            data_dir = os.path.join(os.path.abspath(data_root_dir),
                                    '%s/%s' % (network_name, station_id))

            # Get the station lat/lon info from downloaded JSON file
            if parameter is None:
                station_files = [f for f in os.listdir(data_dir)
                                 if f.endswith('.json')]
            else:
                station_files = [f for f in os.listdir(data_dir)
                                 if f.endswith('.json') and parameter in f]

            if len(station_files) == 0:
                continue

            station_file = os.path.join(data_dir, station_files[0])
            with open(station_file, 'r') as f:
                station_data = json.load(f)

            if not isinstance(station_data, dict):
                print('[WARN] Unexpected format for %s. Skipped.' %
                      station_file)
                continue

            if 'metadata' not in station_data.keys():
                print('[WARN] Station data %s does not contain metadata field. '
                      'Skipped.' % station_file)
                continue

            station_metadata = station_data['metadata']
            if 'station_lat' not in station_metadata.keys() or \
                    'station_lon' not in station_metadata.keys():
                print('[WARN] Station metadata %s does not contain station_lat'
                      'or station_lon field. Skipped.' % station_file)
                continue

            station_lon = station_metadata['station_lon']
            station_lat = station_metadata['station_lat']
            data_dict[network_name][station_id] = [station_lon, station_lat]

    # Plot all stations where we have downloaded data from
    fig, ax = plt.subplots(figsize=(18, 9),
                           subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_global()
    ax.coastlines()
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    for network_name, network_data in data_dict.items():
        network_data = np.array(list(network_data.values()))
        if len(network_data.shape) != 2:
            continue

        ax.scatter(network_data[:, 0].astype(float),
                   network_data[:, 1].astype(float),
                   color=np.array(NETWORK_COLORS[network_name]) / 255.0,
                   label='TOAR-2 Network %s' % network_name, marker='o', s=1)

    ax.legend()
    plt.savefig(out_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data_root_dir', type=str)
    parser.add_argument('out_file', type=str)
    parser.add_argument('--parameter', type=str, default=None)

    args = parser.parse_args()
    main(**vars(args))
