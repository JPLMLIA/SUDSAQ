#!/usr/bin/env python3
# Step 1 feature extraction script:
# Extract raw values from toar2 station metadata (in .json format), and then
# save the extracted values in .h5 file.
#
# Steven Lu

import os
import sys
import h5py
import json
import fnmatch


def add_metadata(meta_dict, toar_meta):
    network_name = toar_meta['network_name']
    station_id = toar_meta['station_id']
    id = toar_meta['id']
    meta_id = f'{network_name}.{station_id}.{id}'

    if meta_id in list(meta_dict.keys()):
        return False
    else:
        meta_dict.setdefault(meta_id, toar_meta)

        return True


def main(in_toar_json_dir, out_metadata_h5):
    if not os.path.exists(in_toar_json_dir):
        print(f'[ERROR] Input directory does not exist: '
              f'{os.path.abspath(in_toar_json_dir)}')
        sys.exit(1)

    meta_dict = dict()
    unique_station_counter = 0
    for root, _, filenames in os.walk(in_toar_json_dir):
        for filename in fnmatch.filter(filenames, '*.json'):
            json_file = open(os.path.join(root, filename), 'r')
            print(f'[INFO] Processing file: {os.path.join(root, filename)}')
            toar_json = json.load(json_file)
            toar_meta = toar_json['metadata']

            success = add_metadata(meta_dict, toar_meta)
            if success:
                unique_station_counter += 1
    print(f'[INFO] Successfully processed {unique_station_counter} toar2 files')

    station_lat = [meta_dict[k]['station_lat'] for k in meta_dict.keys()]
    station_lon = [meta_dict[k]['station_lon'] for k in meta_dict.keys()]
    station_type = [meta_dict[k]['station_type'] for k in meta_dict.keys()]
    station_type_of_area = [meta_dict[k]['station_type_of_area'] for k in meta_dict.keys()]
    station_category = [meta_dict[k]['station_category'] for k in meta_dict.keys()]
    station_country = [meta_dict[k]['station_country'] for k in meta_dict.keys()]
    station_state = [meta_dict[k]['station_state'] for k in meta_dict.keys()]
    station_alt = [meta_dict[k]['station_alt'] for k in meta_dict.keys()]
    station_nightlight_5km = [meta_dict[k]['station_nightlight_5km'] for k in meta_dict.keys()]
    station_climatic_zone = [meta_dict[k]['station_climatic_zone'] for k in meta_dict.keys()]
    station_wheat_production = [meta_dict[k]['station_wheat_production'] for k in meta_dict.keys()]
    station_rice_production = [meta_dict[k]['station_rice_production'] for k in meta_dict.keys()]
    station_nox_emissions = [meta_dict[k]['station_nox_emissions'] for k in meta_dict.keys()]
    station_omi_no2_column = [meta_dict[k]['station_omi_no2_column'] for k in meta_dict.keys()]
    station_toar_category = [meta_dict[k]['station_toar_category'] for k in meta_dict.keys()]
    station_htap_region = [meta_dict[k]['station_htap_region'] for k in meta_dict.keys()]
    station_reported_alt = [meta_dict[k]['station_reported_alt'] for k in meta_dict.keys()]
    station_alt_flag = [meta_dict[k]['station_alt_flag'] for k in meta_dict.keys()]
    station_coordinate_status = [meta_dict[k]['station_coordinate_status'] for k in meta_dict.keys()]
    station_google_alt = [meta_dict[k]['station_google_alt'] for k in meta_dict.keys()]
    station_etopo_alt = [meta_dict[k]['station_etopo_alt'] for k in meta_dict.keys()]
    station_etopo_min_alt_5km = [meta_dict[k]['station_etopo_min_alt_5km'] for k in meta_dict.keys()]
    station_etopo_relative_alt = [meta_dict[k]['station_etopo_relative_alt'] for k in meta_dict.keys()]
    station_dominant_landcover = [meta_dict[k]['station_dominant_landcover'] for k in meta_dict.keys()]
    station_max_nightlight_25km = [meta_dict[k]['station_max_nightlight_25km'] for k in meta_dict.keys()]
    station_max_population_density_25km = [meta_dict[k]['station_max_population_density_25km'] for k in meta_dict.keys()]
    station_nightlight_1km = [meta_dict[k]['station_nightlight_1km'] for k in meta_dict.keys()]
    station_population_density = [meta_dict[k]['station_population_density'] for k in meta_dict.keys()]
    google_resolution = [meta_dict[k]['google_resolution'] for k in meta_dict.keys()]
    station_max_population_density_5km = [meta_dict[k]['station_max_population_density_5km'] for k in meta_dict.keys()]

    out_file = h5py.File(out_metadata_h5, 'w')
    out_file.create_dataset('station_lat', data=station_lat)
    out_file.create_dataset('station_lon', data=station_lon)
    out_file.create_dataset('station_type', data=station_type)
    out_file.create_dataset('station_type_of_area', data=station_type_of_area)
    out_file.create_dataset('station_category', data=station_category)
    out_file.create_dataset('station_country', data=station_country)
    out_file.create_dataset('station_state', data=station_state)
    out_file.create_dataset('station_alt', data=station_alt)
    out_file.create_dataset('station_nightlight_5km', data=station_nightlight_5km)
    out_file.create_dataset('station_climatic_zone', data=station_climatic_zone)
    out_file.create_dataset('station_wheat_production', data=station_wheat_production)
    out_file.create_dataset('station_rice_production', data=station_rice_production)
    out_file.create_dataset('station_nox_emissions', data=station_nox_emissions)
    out_file.create_dataset('station_omi_no2_column', data=station_omi_no2_column)
    out_file.create_dataset('station_toar_category', data=station_toar_category)
    out_file.create_dataset('station_htap_region', data=station_htap_region)
    out_file.create_dataset('station_reported_alt', data=station_reported_alt)
    out_file.create_dataset('station_alt_flag', data=station_alt_flag)
    out_file.create_dataset('station_coordinate_status', data=station_coordinate_status)
    out_file.create_dataset('station_google_alt', data=station_google_alt)
    out_file.create_dataset('station_etopo_alt', data=station_etopo_alt)
    out_file.create_dataset('station_etopo_min_alt_5km', data=station_etopo_min_alt_5km)
    out_file.create_dataset('station_etopo_relative_alt', data=station_etopo_relative_alt)
    out_file.create_dataset('station_dominant_landcover', data=station_dominant_landcover)
    out_file.create_dataset('station_max_nightlight_25km', data=station_max_nightlight_25km)
    out_file.create_dataset('station_max_population_density_25km', data=station_max_population_density_25km)
    out_file.create_dataset('station_nightlight_1km', data=station_nightlight_1km)
    out_file.create_dataset('station_population_density', data=station_population_density)
    out_file.create_dataset('google_resolution', data=google_resolution)
    out_file.create_dataset('station_max_population_density_5km', data=station_max_population_density_5km)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('in_toar_json_dir', type=str)
    parser.add_argument('out_metadata_h5', type=str)

    args = parser.parse_args()
    main(**vars(args))
