#!/usr/bin/env python
# Plot the TOAR-2 data density per month.
#
# Steven Lu
# February 28, 2022

import json
import calendar
import numpy as np
from urllib.parse import quote
from urllib.request import urlopen
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


BASE_URL = 'https://join.fz-juelich.de/services/rest/surfacedata/'


def main(year, month, out_plot, min_lat, max_lat, min_lon, max_lon):
    # Construct TOAR-2 query URL for all stations
    _, n_days = calendar.monthrange(year, month)
    data_after = quote('%d-%02d-01T00:00:00' % (year, month))
    data_before = quote('%d-%02d-%2dT11:59:59' % (year, month, n_days))
    station_query = '%s/search/?parameter_name=o3&boundingbox=[%f,%f,%f,%f]&' \
                    'data_after=%s&data_before=%s&as_dict=True&format=json' % \
                    (BASE_URL, min_lon, min_lat, max_lon, max_lat, data_after,
                     data_before)

    # Get all stations that have o3 measurements.
    response = urlopen(station_query)
    all_station = json.load(response)

    data_range = quote('[%s-%02d-01 00:00,%s-%02d-%02d 11:59]' %
                       (year, month, year, month, n_days))
    station_o3 = list()

    for station in all_station:
        query_url = '%s/stats/?id=%s&sampling=daily&statistics=dma8epa' \
                    '&daterange=%s' % (BASE_URL, station['id'], data_range)
        response = urlopen(query_url)
        station_data = json.load(response)
        station_lon = station_data['metadata']['station_lon']
        station_lat = station_data['metadata']['station_lat']

        if 'dma8epa' not in station_data.keys():
            o3_ratio = -1
        else:
            o3 = station_data['dma8epa']
            o3_valid_counts = np.sum(~np.isnan(o3))
            o3_ratio = o3_valid_counts / n_days

        station_o3.append([o3_ratio, station_lon, station_lat])

    # Generate plot
    station_o3 = np.array(station_o3)
    fig, ax = plt.subplots(figsize=(18, 9),
                           subplot_kw={'projection': ccrs.PlateCarree()})
    plt.scatter(station_o3[:, 1], station_o3[:, 2], c=station_o3[:, 0],
                cmap='jet', s=10)
    plt.clim((-1, 1))
    plt.colorbar()
    ax.coastlines()
    ax.stock_img()
    ax.set_extent([-140, -50, 10, 80], crs=ccrs.PlateCarree())  # NA region
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    plt.title(f'TOAR-2 data density - {month}/{year}')
    plt.savefig(out_plot, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('year', type=int)
    parser.add_argument('month', type=int)
    parser.add_argument('out_plot', type=str)
    parser.add_argument('--min_lat', type=float, default=10.0)
    parser.add_argument('--max_lat', type=float, default=80.0)
    parser.add_argument('--min_lon', type=float, default=-140.0)
    parser.add_argument('--max_lon', type=float, default=-50)

    args = parser.parse_args()
    main(**vars(args))
