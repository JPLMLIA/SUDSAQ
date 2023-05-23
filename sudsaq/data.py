"""
"""
import logging
import datetime as dt
import h5py
import numpy    as np
import os
import re
import xarray   as xr

from tqdm import tqdm

Logger = logging.getLogger('sudsaq/select.py')

try:
    from wcmatch import glob as _glob
    glob = lambda pattern: _glob.glob(pattern, flags=_glob.BRACE)
    Logger.debug('Using wcmatch for glob')
except:
    from glob import glob
    Logger.debug('Failed to load wcmatch, falling back to builtin glob')

from sudsaq import Config


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


def unstacked(func):
    """
    Unstacks the first parameter of the decorated `func` and restacks it if
    the incoming `data` is already stacked, otherwise do nothing
    """
    def wrapped(data, *args, **kwargs):
        loc = False
        if 'loc' in data.dims:
            Logger.debug(f'Unstacked data for {func.__name__}()')
            loc  = True
            data = data.unstack()

        data = func(data, *args, **kwargs)

        if loc and isinstance(data, (
            xr.core.dataarray.DataArray,
            xr.core.dataarray.Dataset
        )):
            return flatten(data)
        return data

    return wrapped


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

    Logger.debug(f'Attempting to evaluate: {string!r}')
    return eval(string)


def flatten(data):
    """
    """
    if isinstance(data, xr.core.dataarray.Dataset):
        data = data.to_array()

    dims = ['lat', 'lon']
    if 'time' in data.dims:
        dims.append('time')
    return data.stack({'loc': dims})


@unstacked
def resample(data, freq, how='mean'):
    """
    """
    data = data.resample(time=freq)
    data = getattr(data, how)()

    return data


@unstacked
def subsample(data, dim, N):
    """
    Subsamples along a dimension by dropping every N sample

    Parameters
    ----------
    data: xarray
        Data to subsample on
    dim: str
        Name of the dimension to subsample
    N: int
        Every Nth sample is dropped
    """
    # Select every Nth index
    drop = data[dim][N-1::N]
    return data.drop_sel(**{dim: drop})


def scale(x, dims=['loc']):
    """
    The standard score of a sample x is calculated as:
        z = (x - u) / s
    """
    u = x.mean(skipna=True, dim=dims)
    s = x.std(skipna=True, dim=dims)
    z = (x - u) / s
    return z


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
    data = flatten(ds[config.train]).transpose('loc', 'variable')

    # Use the locations valid by this variable only, but this variable may be excluded otherwise
    if config.input.use_locs_of:
        Logger.debug(f'Using locations from variable: {config.input.use_locs_of}')
        # mean('time') removes the time dimension so it is ignored
        merged = flatten(xr.merge([target, ds[config.input.use_locs_of].mean('time')]))
        # Replace locs in the target with NaNs if the use_locs_of had a NaN
        merged = merged.where(~merged.isel(variable=1).isnull())
        # Extract the target, garbage collect the other
        target = merged.isel(variable=0)
    else:
        target = flatten(target)

    Logger.debug(f'Target shape: {list(zip(target.dims, target.shape))}')
    Logger.debug(f'Data   shape: {list(zip(data.dims, data.shape))}')

    if config.input.scale:
        Logger.info('Scaling data (X)')
        data = scale(data)

    if not lazy:
        Logger.info('Loading data into memory')
        data.load()
        target.load()
        Logger.debug(f'Memory footprint in GB:')
        Logger.debug(f'- Data   = {data.nbytes / 2**30:.3f}')
        Logger.debug(f'- Target = {target.nbytes / 2**30:.3f}')

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

    # Convert the Timezones from 0-360 format to -180-180 format if needed
    if ds.lon.min() < 0:
        global Timezones
        for i, (tz, (west, east)) in enumerate(Timezones):
            if west >= 180:
                west -= 360
                east -= 360
            Timezones[i] = (tz, (west, east))

    # Select time ranges per config
    time = ds.time.dt.time
    data = []
    for sect, sel in config.input.daily.items():
        Logger.debug(f'- {sect}: Selecting times {sel.time} on variables {sel.vars}')
        if sel.local:
            Logger.debug('-- Using local timezones')
            ns    = ds[sel.vars].sortby('lon')
            local = []
            for offset, bounds in tqdm(Timezones, desc='Timezones Processed'):
                Logger.debug(f'--- Processing offset {offset} for (west, east) bounds {bounds}')
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
    return Dataset(ds)


def config_sel(ds, sels):
    """
    Performs custom sel operations defined in the config

    Parameters
    ----------
    sels: mlky.Section
        Selections defined by the config
    """
    for dim, sel in sels.items():
        if dim == 'vars':
            Logger.debug(f'Selecting variables: {sel}')
            ds = ds[sel]

        # Special integer month support
        elif dim == 'month':
            Logger.debug(f'Selecting: month=={sel}')
            ds = ds.sel(time=ds['time.month']==sel)

        # Support for special drop operations
        elif dim == 'drop_date':
            Logger.debug(f'Dropping date:')
            mask = np.full(ds.time.shape, True)
            if 'year' in sel:
                Logger.debug(f" - year = {sel['year']}")
                mask &= ds.time.dt.year == sel['year']
            if 'month' in sel:
                Logger.debug(f" - month = {sel['month']}")
                mask &= ds.time.dt.month == sel['month']
            if 'day' in sel:
                Logger.debug(f" - day = {sel['day']}")
                mask &= ds.time.dt.day == sel['day']

            if mask.any():
                ds = ds.where(ds.time[~mask], drop=True)

        elif isinstance(sel, list):
            Logger.debug(f'Selecting: {sel[0]} < {dim} < {sel[1]}')
            ds = ds.sortby(dim)
            # Enables crossing the 0 lon line
            if isinstance(sel, list) and sel[1] < sel[0]:
                ds = ds.where(ds[dim][(sel[0] < ds[dim]) | (ds[dim] < sel[1])])
            else:
                ds = ds.sel(**{dim: slice(*sel)})

        else:
            Logger.debug(f'Selecting: {dim}=={sel}')
            ds = ds.sel(**{dim: sel})

    return Dataset(ds)


def load(config, split=False, lazy=True):
    """
    """
    Logger.info('Collecting files')
    files = []
    for string in config.input.glob:
        match = glob(string)
        Logger.debug(f'Collected {len(match)} files using "{string}"')
        files += match

    if not files:
        Logger.error('No files collected, exiting early')
        return None, None if split else None

    Logger.info('Lazy loading the dataset')
    ds = xr.open_mfdataset(files,
        engine   = config.input.get('engine'  , 'netcdf4'),
        lock     = config.input.get('lock'    , False    ),
        parallel = config.input.get('parallel', False    ),
        chunks   = dict(config.input.chunks)
    )

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

    ds = config_sel(ds, config.input.sel)

    if config.input.daily:
        Logger.info('Aligning to a daily average')
        ds = daily(ds, config)

    if config.input.resample:
        Logger.info('Resampling data')
        ds = resample(ds, **config.input.resample)

    if config.input.subsample:
        Logger.info('Subsampling data')
        ds = subsample(ds, **config.input.subsample)

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
    return ds.sortby('lon')


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
                key  = key.replace('\\\\', '\\')
                keys = [var for var in self.variables.keys() if re.fullmatch(key, var)]
                if keys:
                    Logger.debug(f'Matched {len(keys)} variables with regex {key!r}: {keys}')
                    return super().__getitem__(keys)
            raise e
