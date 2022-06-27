"""
"""
import logging
import datetime as dt
import re
import xarray   as xr

from glob import glob

from sudsaq.config import Config

Logger = logging.getLogger('sudsaq/select.py')

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

    # Create the stacked training data
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
        if isinstance(sel.time, list):
            mask = (dt.time(sel.time[0]) < time) & (time < dt.time(sel.time[1]))
        else:
            mask = (time == dt.time(sel.time))

        data.append(ds[sel.vars].where(mask, drop=True).resample(time='1D').mean())

    # Merge the selections together
    ds = xr.merge(data)

    return ds

def load(config, split=False):
    """
    """
    Logger.info('Collecting files')
    files = []

    if config.input.momo:
        momo = glob(config.input.momo.regex)
        Logger.debug(f'Collected {len(momo)} files using regex `{config.input.momo.regex}`')
        files += momo

    if config.input.toar:
        toar = glob(config.input.toar.regex)
        Logger.debug(f'Collected {len(toar)} files using regex `{config.input.toar.regex}`')
        files += toar

    Logger.info('Lazy loading the dataset')
    ds = xr.open_mfdataset(files, parallel=True, engine='scipy')

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

    # Logger.info('Loading data into memory')
    # ds.load()

    # Hardcoded by script
    if split:
        Logger.debug('Performing split and stack')
        return split_and_stack(ds, config)

    Logger.debug('Returning dataset')
    return ds
