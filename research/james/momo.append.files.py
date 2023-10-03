import xarray as xr

from glob import glob
from tqdm import tqdm

from sudsaq.data import save_by_month
from sudsaq.silos.momo.process import extract_tar


def from_files():
    files = glob('/data/MLIA_active_data/data_SUDSAQ/MOMOchem/*.nc')
    vars = {}
    for file in files:
        name = file.split('/')[-1][7:-3]
        name = name.replace('_', '.')
        if name in vars:
            vars[name].append(file)
        else:
            vars[name] = [file]

    dss = []
    lon = None
    for name, files in tqdm(vars.items(), desc='Processing Variables'):
        ds = xr.open_mfdataset(files)
        ds = ds.rename({'var': f'momo.{name}'})
        if lon is None:
            lon = ds.lon.values
            lon[lon > 180] -= 360
        ds['lon'] = lon
        dss.append(ds)

    merged = xr.merge(dss)

    save_by_month(merged, '/data/MLIA_active_data/data_SUDSAQ/data/momo/processing/')
    print('Done')

import xarray as xr
from glob import glob
from sudsaq.data import save_by_month
from sudsaq.silos.momo.process import extract_tar
def from_tars(
    input  = '/data/MLIA_active_data/data_SUDSAQ/data/momo/OH/',
    output = '/data/MLIA_active_data/data_SUDSAQ/data/momo/OH/',
):
    """
    """
    files = glob(f'{input}/*.tar.gz')
    for file in tqdm(files, desc='Processing Tars'):
        ds = extract_tar(file)
        lon = ds.lon.values
        lon[lon > 180] -= 360
        ds['lon'] = lon
        rename = {}
        for var in ds:
            if '/' in var:
                rename[var] = 'momo.' + var.replace('/', '.')
            else:
                rename[var] = f'momo.{var}'
        ds = ds.rename(rename)
        save_by_month(ds, output)


def from_tars(
    input  = '/data/MLIA_active_data/data_SUDSAQ/data/momo/OH/*.tar.gz',
    output = '/data/MLIA_active_data/data_SUDSAQ/data/momo/OH/',
    years  = range(2014, 2015)
):
    """
    """
    years = [str(year) for year in years]
    files = glob(input)
    for file in tqdm(files, desc='Processing Tars'):
        year = file.split('/')[-1][11:15]
        if year in years:
            ds = extract_tar(file)
            #
            lon = ds.lon.values
            lon[lon > 180] -= 360
            #
            ds['lon'] = lon
            #
            rename = {}
            for var in ds:
                if '/' in var:
                    rename[var] = 'momo.' + var.replace('/', '.')
                else:
                    rename[var] = f'momo.{var}'
            #
            ds = ds.rename(rename)
            #
            save_by_month(ds, output)


def merge_years(years=range(2005, 2011)):
    years = [str(year) for year in years]
    for year in tqdm(years, desc='Merging Years'):
        for month in range(1, 12+1):
            try:
                month = f'{month:02}'
                print(f'Processing {year}, {month}')
                print(f'- Loading')
                current = xr.open_dataset(f'/data/MLIA_active_data/data_SUDSAQ/data/momo/{year}/{month}.nc')
                append  = xr.open_dataset(f'/data/MLIA_active_data/data_SUDSAQ/data/momo/processing/{year}/{month}.nc')

                current['lon'] = append['lon']

                print('- Merging')
                merged = xr.merge([current, append])

                print('- Saving')
                save_by_month(merged, '/data/MLIA_active_data/data_SUDSAQ/data/momo/appended/')
            except Exception as e:
                print(f'! Failed to merge: {e}')


def convert_lon(
    years  = range(2011, 2016),
    input  = '/data/MLIA_active_data/data_SUDSAQ/data/momo/lon360/',
    output = '/data/MLIA_active_data/data_SUDSAQ/data/momo/',
):
    """
    """
    years = [str(year) for year in years]
    lon   = None
    for year in tqdm(years, desc='Converting Years'):
        for month in range(1, 12+1):
            month = f'{month:02}'
            print(f'Processing {year}, {month}')
            try:
                print(f'- Loading')
                ds = xr.open_dataset(f'{input}/{year}/{month}.nc')

                if lon is None:
                    lon = ds.lon.values
                    lon[lon > 180] -= 360
                ds['lon'] = lon

                print('- Saving')
                save_by_month(ds, output)
            except Exception as e:
                print(f'! Failed to convert: {e}')


def feature_compare_years(
    input = '/data/MLIA_active_data/data_SUDSAQ/data/momo/',
    years = range(2005, 2011),
    base  = '/data/MLIA_active_data/data_SUDSAQ/data/momo/2011/01.nc'
):
    def report(a, b, msg):
        c = a - b
        if c:
            print(f'{len(c)} {msg}')
            for d in sorted(c):
                print(f'- {d}')
    years = [str(year) for year in years]
    base  = set(xr.open_dataset(base))
    for year in tqdm(years, desc='Comparing Years'):
        for month in range(1, 12+1):
            month = f'{month:02}'
            comp = set(xr.open_dataset(f'{input}/{year}/{month}.nc'))
            report(base, comp, f'features missing from {year}, {month}:')
            report(comp, base, f'new features included in {year}, {month}:')

def convert_lon(
    years  = range(2005, 2016),
    input  = '/projects/mlia-active-data/data_SUDSAQ/data/toar/matched/',
    output = '/scratch_lg_edge/sudsaq/data/toar/',
):
    """
    """
    years = [str(year) for year in years]
    lon   = None
    for year in tqdm(years, desc='Converting Years'):
        for month in range(1, 12+1):
            month = f'{month:02}'
            print(f'Processing {year}, {month}')
            try:
                print(f'- Loading')
                ds = xr.open_dataset(f'{input}/{year}/{month}.nc')
                if lon is None:
                    lon = ds.lon.values
                    lon[lon > 180] -= 360
                ds['lon'] = lon
                print('- Saving')
                save_by_month(ds, output)
            except Exception as e:
                print(f'! Failed to convert: {e}')
