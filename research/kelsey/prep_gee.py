'''
Preps gee data for monthly analysis, adding time component
(gee datasets fire, modis, pop are yearly), fills nans with 0,
matches longitude of momo, and splits into training, testing sets
'''

import xarray as xr

month = 'jan'
root_dir = '/Users/kelseydoerksen/exp_runs/rf'
gee_dir = '/Users/kelseydoerksen/gee'


def format_gee(gee, momo):
    gee = gee.expand_dims(time=momo['time'])
    gee_t = gee.transpose('lat', 'lon', 'time')
    gee_t.coords['lon'] = gee_t.coords['lon']+180
    gee_final = gee_t.fillna(0)
    return gee_final


fire_data = []
modis_data = []
pop_data = []
years = [2011, 2012, 2013, 2014, 2015]
for year in years:
    pop = xr.open_dataset('{}/pop/2010_population_density_globe.nc'.format(gee_dir))
    modis = xr.open_dataset('{}/modis/{}_modis_globe_mode.nc'.format(gee_dir, year))
    fire = xr.open_dataset('{}/fire/{}_fire_globe_mode.nc'.format(gee_dir, year))
    momo = xr.open_dataset('{}/{}/{}/test.data.nc'.format(root_dir, month, year))
    formatted_pop = format_gee(pop, momo)
    formatted_modis = format_gee(modis, momo)
    formatted_fire = format_gee(fire, momo)
    formatted_pop.to_netcdf('{}/{}/{}/test.pop.nc'.format(root_dir, month, year))
    formatted_modis.to_netcdf('{}/{}/{}/test.modis.nc'.format(root_dir, month, year))
    formatted_fire.to_netcdf('{}/{}/{}/test.fire.nc'.format(root_dir, month, year))
    fire_data.append(formatted_fire)
    modis_data.append(formatted_modis)
    pop_data.append(formatted_pop)

gee_type = {'pop': pop_data, 'modis': modis_data, 'fire': fire_data}
for k, v in gee_type.items():
    train_2011 = v[1].merge(v[2]).merge(v[3]).merge(v[4])
    train_2011.to_netcdf('{}/{}/2011/train.{}.nc'.format(root_dir, month, k))

    train_2012 = v[0].merge(v[2]).merge(v[3]).merge(v[4])
    train_2012.to_netcdf('{}/{}/2012/train.{}.nc'.format(root_dir, month, k))

    train_2013 = v[0].merge(v[1]).merge(v[3]).merge(v[4])
    train_2013.to_netcdf('{}/{}/2013/train.{}.nc'.format(root_dir, month, k))

    train_2014 = v[0].merge(v[1]).merge(v[2]).merge(v[4])
    train_2014.to_netcdf('{}/{}/2014/train.{}.nc'.format(root_dir, month, k))

    train_2015 = v[0].merge(v[1]).merge(v[2]).merge(v[3])
    train_2015.to_netcdf('{}/{}/2015/train.{}.nc'.format(root_dir, month, k))
