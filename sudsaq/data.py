"""
"""
import logging
import datetime as dt
import h5py
import numpy    as np
import os
import re
import xarray   as xr

from glob                  import glob
from sklearn.preprocessing import StandardScaler
from tqdm                  import tqdm

from sudsaq.config import Config

h5py._errors.silence_errors()

# List of UTC+Offset, (West Lon, East Lon) to apply in daily()
Timezones = [
# offset, ( west, east)
    (  0, (  0.0, 7.5)),
    (  1, (  7.5, 22.5)),
    (  2, ( 22.5, 37.5)),
    (  3, ( 37.5, 52.5)),
    (  4, ( 52.5, 67.5)),
    (  5, ( 67.5, 82.5)),
    (  6, ( 82.5, 97.5)),
    (  7, ( 97.5, 112.5)),
    (  8, (112.5, 127.5)),
    (  9, (127.5, 142.5)),
    ( 10, (142.5, 157.5)),
    ( 11, (157.5, 172.5)),
    ( 12, (172.5, 180.0)),
    (-12, (180.0, 187.5)),
    (-11, (187.5, 202.5)),
    (-10, (202.5, 217.5)),
    ( -9, (217.5, 232.5)),
    ( -8, (232.5, 247.5)),
    ( -7, (247.5, 262.5)),
    ( -6, (262.5, 277.5)),
    ( -5, (277.5, 292.5)),
    ( -4, (292.5, 307.5)),
    ( -3, (307.5, 322.5)),
    ( -2, (322.5, 337.5)),
    ( -1, (337.5, 352.5)),
    (  0, (352.5, 360.0))
]

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

def calc(ds, string):
    """
    Performs simple calculations to create new features at runtime.
    """
    for key in list(ds):
        # Find this key not followed by a digit or word character (eg. prevents momo.no matching to momo.no2)
        string = re.sub(fr'({key})(?!\d|\w)', f"ds['{key}']", string)

    return eval(string)

def split_and_stack(ds, config, lazy=True):
    """
    Splits the target from the data and stacks both to be 1 or 2d
    """
    # Target is a variable case
    if config.target in ds:
        Logger.info(f'Target is {config.target}')
        target = ds[config.target]
    # Calculated target case
    else:
        target = calc(ds, config.target)
        Logger.info(f'Target is {config.target}')

    Logger.info(f'Creating the stacked training and target objects')

    # Save the lat/lon dimensions before dropping na for easy reconstruction later
    config._reindex = ds[['lat', 'lon']]

    # Create the stacked objects
    data = ds[config.train].to_array().flatten()
    data = data.transpose('loc', 'variable')

    # Use the locations valid by this variable only, but this variable may be excluded otherwise
    if config.use_locs_of:
        Logger.debug(f'Using locations from variable: {config.use_locs_of}')
        # mean('time') removes the time dimension so it is ignored
        merged = xr.merge([target, ds[config.use_locs_of].mean('time')])
        merged = Dataset(merged).to_array().flatten()
        # Replace locs in the target with NaNs if the use_locs_of had a NaN
        merged = merged.where(~merged.isel(variable=1).isnull())
        # Extract the target, garbage collect the other
        target = merged.isel(variable=0)
    else:
        target = DataArray(target).flatten()

    Logger.debug(f'Target shape: {list(zip(target.dims, target.shape))}')
    Logger.debug(f'Data   shape: {list(zip(data.dims, data.shape))}')

    if not lazy:
        Logger.info('Loading data into memory')
        data.load()
        target.load()
        Logger.debug(f'Memory footprint in GB:')
        Logger.debug(f'- Data   = {data.nbytes / 2**30:.3f}')
        Logger.debug(f'- Target = {target.nbytes / 2**30:.3f}')

    if config.input.scale:
        Logger.info('Scaling data (X)')
        scaler  = StandardScaler(**config.input.StandardScaler)
        data[:] = scaler.fit_transform(data)

    return data, target

def daily(ds, config):
    """
    Aligns a dataset to a daily average
    """
    def select_times(ds, sel, time):
        """
        Selects timestamps using integer hours (0-23) over all dates
        """
        if isinstance(sel, list):
            mask = (dt.time(sel[0]) <= time) & (time <= dt.time(sel[1]))
            ds   = ds.where(mask, drop=True)

            # Floor the times to the day for the groupby operation
            ds.coords['time'] = ds.time.dt.floor('1D')

            # Now group as daily taking the mean
            ds = ds.groupby('time').mean()
        else:
            mask = (time == dt.time(sel))
            ds   = ds.where(mask, drop=True)

            # Floor the times to the day for the groupby operation
            ds.coords['time'] = ds.time.dt.floor('1D')

        return ds

    # Select time ranges per config
    time = ds.time.dt.time
    data = []
    for sect, sel in config.input.daily.items():
        Logger.debug(f'- {sect}: Selecting times {sel.time} on variables {sel.vars}')
        if sel.local:
            Logger.debug('-- Using local timezones')
            ns    = ds[sel.vars]
            local = []
            for offset, bounds in tqdm(Timezones, desc='Timezones Processed'):
                sub  = ns.sel(lon=slice(*bounds))
                time = ( sub.time + np.timedelta64(offset, 'h') ).dt.time
                sub  = select_times(sub, sel.time, time)

                local.append(sub)

            # Merge these datasets back together to create the full grid
            Logger.debug('-- Merging local averages together (this can take awhile)')
            data.append(xr.merge(local))
        else:
            sub = select_times(ds[sel.vars], sel.time, time)
            data.append(sub)

    # Add variables that don't have a time dimension back in
    timeless = ds.drop_dims('time')
    if len(timeless) > 0:
        Logger.debug(f'- Appending {len(timeless)} timeless variables: {list(timeless)}')
        data.append(timeless)

    # Merge the selections together
    Logger.debug('- Merging all averages together')
    ds = xr.merge(data)

    # Cast back to custom Dataset (xr.merge creates new)
    ds = Dataset(ds)

    return ds

def load(config, split=False, lazy=True):
    """
    """
    Logger.info('Collecting files')
    files = []
    for string in config.input.glob:
        match = glob(string)
        Logger.debug(f'Collected {len(match)} files using "{string}"')
        files += match

    Logger.info('Lazy loading the dataset')
    ds = xr.open_mfdataset(files, engine='netcdf4', lock=False, parallel=config.get('_parallel', True))

    Logger.info('Casting xarray.Dataset to custom Dataset')
    ds = Dataset(ds)

    for key, args in config.input.replace_vals.items():
        left, right = args.bounds
        value       = float(args.value) or np.nan
        Logger.debug(f'Replacing values between ({left}, {right}) with {value} for key {key}')

        ds[key] = ds[key].where(
            (ds[key] < left) | (right < ds[key]),
            value
        )

    if config.input.calc:
        Logger.info('Calculating variables')

        for key, string in config.input.calc.items():
            Logger.debug(f'- {key} = {string}')
            ds[key] = calc(ds, string)

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

    # `split` is hardcoded by the calling script
    # If `lazy` is set in the config use that else use the parameter
    lazy = config.input.get('lazy', lazy)
    if split:
        Logger.debug('Performing split and stack')
        return split_and_stack(ds, config, lazy)

    if not lazy:
        Logger.info('Loading data into memory')
        ds.load()

    Logger.debug('Returning dataset')
    return ds

class Dataset(xr.Dataset):
    """
    Small override of xarray.Dataset that enables regex matching names in the variables
    list
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

    def to_array(self, *args, **kwargs):
        """
        """
        return DataArray(super().to_array(*args, **kwargs))

class DataArray(xr.DataArray):
    """
    """
    def flatten(self, *args, **kwargs):
        dims = ['lat', 'lon']
        if 'time' in self.dims:
            dims.append('time')
        return self.stack({'loc': dims})

    def to_dataset(self, *args, **kwargs):
        return Dataset(super().to_dataset(*args, **kwargs))
