"""
"""
import logging
import datetime as dt
import os
import re
import xarray   as xr

from glob import glob

from sudsaq.config import Config

Logger = logging.getLogger('sudsaq/select.py')

def save_by_month(ds, path):
    """
    Saves an xarray dataset by monthly files
    """
    Logger.info('Saving output by month')
    for year, yds in ds.groupby('time.year'):
        Logger.info(f'{year}: ')
        # Check if directory exists, otherwise create it
        output = f'{path}/{year}'
        if not os.path.exists(output):
            os.mkdir(output, mode=0o771)

        for month, mds in yds.groupby('time.month'):
            Logger.info(f'- {month:02}')
            mds.to_netcdf(f'{output}/{month:02}.nc', engine='netcdf4')

def split_and_stack(ds, config):
    """
    Splits the target from the data and stacks both to be 1 or 2d
    """
    # Target is a variable case
    if config.target in ds:
        Logger.info(f'Target is {config.target}')
        target = ds[config.target]
    # Calculated target case
    else:
        # Only support basic operations: [-, +, /, *]
        v1, op, v2 = re.findall(r'(\S+)', config.target)
        target = eval(f'ds[{v1!r}] {op} ds[{v2!r}]')
        Logger.info(f'Target is {config.target}')

    Logger.info(f'Creating the stacked training and target objects')

    # Create the stacked data
    data = ds[config.train].to_array().stack({'loc': ['lat', 'lon', 'time']})
    data = data.transpose('loc', 'variable')

    # Stack the target data
    target = target.stack({'loc': ['lat', 'lon', 'time']})
    target = target.dropna('loc')

    # Subselect data where there is target data
    data = data.sel(loc=target['loc'])

    Logger.debug(f'Target shape: {list(zip(target.dims, target.shape))}')
    Logger.debug(f'Data   shape: {list(zip(data.dims, data.shape))}')

    return data, target

def daily(ds, config):
    """
    Aligns a dataset to a daily average
    """
    # Select time ranges per config
    time = ds.time.dt.time
    data = []
    for sect, sel in config.input.daily.items():
        Logger.debug(f'- {sect}: Selecting times {sel.time} on variables {sel.vars}')
        if isinstance(sel.time, list):
            mask = (dt.time(sel.time[0]) < time) & (time < dt.time(sel.time[1]))
        else:
            mask = (time == dt.time(sel.time))

        data.append(ds[sel.vars].where(mask, drop=True).resample(time='1D').mean())

    # Merge the selections together
    ds = xr.merge(data)

    # Cast back to custom Dataset (xr.merge creates new)
    ds = Dataset(ds)

    return ds

def load(config, split=False):
    """
    """
    Logger.info('Collecting files')
    files = []

    # TODO: Generalize to allow new datasets not hardcoded
    if config.input.momo:
        momo = glob(config.input.momo.regex)
        Logger.debug(f'Collected {len(momo)} files using regex `{config.input.momo.regex}`')
        files += momo

    if config.input.toar:
        toar = glob(config.input.toar.regex)
        Logger.debug(f'Collected {len(toar)} files using regex `{config.input.toar.regex}`')
        files += toar

    Logger.info('Lazy loading the dataset')
    ds = xr.open_mfdataset(files, parallel=True, engine='netcdf4')

    Logger.info('Casting xarray.Dataset to custom Dataset')
    ds = Dataset(ds)

    if config.input.sel:
        Logger.info('Subselecting data')

        for dim, sel in config.input.sel.items():
            if dim == 'vars':
                Logger.debug(f'Selecting variables: {sel}')
                ds = ds[sel]
            else:
                if isinstance(sel, list):
                    sel = slice(*sel)
                Logger.debug(f'Selecting on dimension {dim} using {sel}')
                ds = ds.sel(**{dim: sel})

    if config.input.daily:
        Logger.info('Aligning to a daily average')
        ds = daily(ds, config)

    Logger.info('Loading data into memory')
    ds.load()

    # Hardcoded by script
    if split:
        Logger.debug('Performing split and stack')
        return split_and_stack(ds, config)

    Logger.debug('Returning dataset')
    return ds

class Dataset(xr.Dataset):
    """
    Small override of xarray.Dataset that enables regex matching
    names in the variables list
    """
    # TODO: Bad keys failing to report which keys are bad: KeyError: 'momo'
    __slots__ = () # Required for subclassing

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError as e:
            if isinstance(key, str):
                keys = [var for var in self.variables.keys() if re.fullmatch(key, var)]
                if keys:
                    Logger.debug(f'Matched {len(keys)} variables with regex {key!r}: {keys}')
                    return super().__getitem__(keys)
            raise e
