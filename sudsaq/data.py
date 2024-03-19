"""
"""
# Builtin
import logging
import datetime as dt
import os
import re

# External
import h5py
import numpy  as np
import xarray as xr

from mlky import Config
from tqdm import tqdm

# Internal


Logger = logging.getLogger('sudsaq/select.py')

try:
    from wcmatch import glob as _glob
    glob = lambda pattern: _glob.glob(pattern, flags=_glob.BRACE)
    Logger.debug('Using wcmatch for glob')
except:
    from glob import glob
    Logger.debug('Failed to load wcmatch, falling back to builtin glob')


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
    Unstacks the first parameter of the decorated function and restacks it if
    the incoming data is already stacked; otherwise, do nothing.

    Parameters
    ----------
    func : function
        The function to be decorated.

    Returns
    -------
    function
        The wrapped function.

    Notes
    -----
    This decorator checks if the first parameter of the decorated function (`func`) is
    stacked along a specific dimension named 'loc'. If it is, it unstacks the data
    before passing it to the decorated function. After the function call, it checks
    if the result is a DataArray or Dataset and flattens it along certain dimensions
    if necessary.
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
    Save an xarray dataset into monthly files.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to be saved.
    path : str
        The path to the directory where the monthly files will be saved.

    Notes
    -----
    The dataset is grouped by year and then by month, and each group is saved as a separate NetCDF file.
    Each file name corresponds to the month in the format 'MM.nc'.
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
    Perform simple calculations to create new features at runtime.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing the variables to be used in the calculations.
    string : str
        The mathematical expression defining the calculation to be performed.
        Variables in the expression should correspond to the keys in the dataset.

    Returns
    -------
    result : xarray.DataArray or xarray.Dataset
        The result of the calculation.

    Notes
    -----
    This function dynamically evaluates the mathematical expression provided in the 'string' parameter.
    Each variable in the expression is replaced with the corresponding dataset key.
    The calculation is performed using Python's built-in eval function.
    """
    for key in list(ds):
        # Find this key not followed by a digit or word character (eg. prevents momo.no matching to momo.no2)
        string = re.sub(fr'({key})(?!\d|\w)', f"ds['{key}']", string)

    Logger.debug(f'Attempting to evaluate: {string!r}')
    return eval(string)


def flatten(data):
    """
    Flatten multi-dimensional xarray data along specified dimensions.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        The input xarray data to be flattened.

    Returns
    -------
    xarray.DataArray
        The flattened data.

    Notes
    -----
    If the input data is a dataset, it will first be converted into a DataArray
    using the to_array method.
    The flattening process stacks the data along latitude ('lat') and longitude ('lon') dimensions.
    If the data contains a 'time' dimension, it will also be stacked along this dimension.
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
    Resample time series data to a specified frequency.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        The input time series data.
    freq : str
        The frequency to which the data will be resampled.
        This should be a string representing the desired time frequency,
        such as 'D' for daily, 'W' for weekly, 'M' for monthly, etc.
    how : str, optional
        The method used to aggregate the data after resampling.
        Default is 'mean'. Other options include 'sum', 'min', 'max', etc.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The resampled time series data.

    Notes
    -----
    This function relies on the resample method of xarray objects.
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
    Scale the input data `x` along specified dimensions to have a mean of 0 and a standard deviation of 1.

    Parameters
    ----------
    x : xarray.DataArray
        The input data array to be scaled.

    dims : list, optional
        The dimensions along which to calculate the mean and standard deviation for scaling.
        Default is ['loc'].

    Returns
    -------
    xarray.DataArray
        A new DataArray containing the scaled values.

    Notes
    -----
    The standard score (z-score) of a sample `x` is calculated as:
        z = (x - u) / s
    where `u` is the mean and `s` is the standard deviation along the specified dimensions.
    This function scales the input data along the specified dimensions to have a mean of 0 and a standard deviation of 1.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> data = xr.DataArray(np.random.rand(10, 10), dims=['lat', 'lon'])
    >>> scaled_data = scale(data, dims=['lat'])
    """
    u = x.mean(skipna=True, dim=dims)
    s = x.std(skipna=True, dim=dims)
    z = (x - u) / s
    return z


def sel_by_latlon_pair(ds: xr.Dataset, pairs: list, remove: bool=False) -> xr.Dataset:
    """
    Selects data from an xarray Dataset based on latitude and longitude pairs.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the data.

    pairs : list
        A list of latitude and longitude pairs used for selection.

    remove : bool, optional
        If True, remove the specified pairs instead of selecting them.
        Default is False.

    Returns
    -------
    xr.Dataset
        A new xarray Dataset containing the selected data based on the specified pairs.

    Notes
    -----
    This function selects data from the input xarray Dataset based on latitude and longitude pairs.
    If `remove` is True, it removes the specified pairs from the selection instead of selecting them.
    """
    if remove:
        pairs = list(set(ds['loc'].data) - set(pairs))

    return ds.sel(loc=pairs)


def split_and_stack(ds, lazy=True):
    """
    Splits the target from the data and stacks both to be 1d

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset containing both data and target.
    lazy : bool, optional
        Whether to lazily load the data into memory. Defaults to True.

    Returns
    -------
    data : xarray.DataArray
        The stacked data.
    target : xarray.DataArray
        The stacked target.

    Notes
    -----
    Config options used in this function:
    - Config.target : The target variable name or calculable.
    - Config.train : The training variables list or regex.
    - Config.input.use_locs_of : The variable to use for selecting locations.
    - Config.input.scale : Whether to scale the input data.
    """
    # Target is a variable case
    if Config.target in ds:
        Logger.info(f'Target is {Config.target}')
        target = ds[Config.target]
    # Calculated target case
    else:
        target = calc(ds, Config.target)
        Logger.info(f'Target is {Config.target}')

    Logger.info(f'Creating the stacked training and target objects')

    # Save the lat/lon dimensions before dropping na for easy reconstruction later
    Config._reindex = ds[['lat', 'lon']]

    # Create the stacked objects
    data = flatten(ds[Config.train]).transpose('loc', 'variable')

    # Use the locations valid by this variable only, but this variable may be excluded otherwise
    if Config.input.use_locs_of:
        Logger.debug(f'Using locations from variable: {Config.input.use_locs_of}')
        # mean('time') removes the time dimension so it is ignored
        merged = flatten(xr.merge([target, ds[Config.input.use_locs_of].mean('time')]))
        # Replace locs in the target with NaNs if the use_locs_of had a NaN
        merged = merged.where(~merged.isel(variable=1).isnull())
        # Extract the target, garbage collect the other
        target = merged.isel(variable=0)
    else:
        target = flatten(target)

    Logger.debug(f'Target shape: {list(zip(target.dims, target.shape))}')
    Logger.debug(f'Data   shape: {list(zip(data.dims, data.shape))}')

    if Config.input.scale:
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


def daily(ds):
    """
    Aligns a dataset to a daily average

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset to align.

    Returns
    -------
    Dataset
        The aligned dataset.
    """
    def select_times(ds, sel, time):
        """
        Selects timestamps using integer hours (0-23) over all dates

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset to select timestamps from.
        sel : int or list of int
            The hour(s) to select.
        time : xarray.DataArray
            The time dimension of the dataset.

        Returns
        -------
        xarray.Dataset
            The dataset with selected timestamps.
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

    # Select time ranges per Config
    time = ds.time.dt.time
    data = []
    for sect, sel in Config.input.daily.items():
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
    Performs custom sel operations defined in the Config

    Parameters
    ----------
    sels: mlky.Section
        Selections defined by the Config
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
                ds = ds.sel(time=ds.time[~mask])

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


def load(split=False, lazy=True):
    """
    Loads and preprocesses data from files.

    Parameters
    ----------
    split : bool, optional
        Whether to split and stack the dataset. Defaults to False.
    lazy : bool, optional
        Whether to lazily load the dataset into memory. Defaults to True.

    Returns
    -------
    Dataset or tuple of Dataset and target
        The loaded and preprocessed dataset or a tuple containing the dataset and target.

    Notes
    -----
    Config options used in this function:
    - Config.input.glob (list): A list of glob patterns used to collect files. If wcmatch is installed, this can be a complex pattern.
    - Config.input.engine (str): The engine used to open the files (default is 'netcdf4').
    - Config.input.lock (bool): Whether to use file locking when opening files.
    - Config.input.parallel (bool): Whether to use parallel reading when opening files.
    - Config.input.chunks (dict): Chunk sizes for parallel reading.
    - Config.input.replace_vals (dict): Replacement values for specified variables.
    - Config.input.calc (dict): Calculations to perform on variables.
    - Config.input.sel (dict): Selection criteria for variables.
    - Config.input.daily (bool): Daily alignment options.
    - Config.input.resample (dict): Resampling options.
    - Config.input.subsample (dict): Subsampling options.
    - Config.input.lazy (bool): Whether to lazily load the dataset into memory.
    """
    Logger.info('Collecting files')
    files = []
    for string in Config.input.glob:
        match = glob(string)
        Logger.debug(f'Collected {len(match)} files using "{string}"')
        files += match

    if not files:
        Logger.error('No files collected, exiting early')
        return None, None if split else None

    Logger.info('Lazy loading the dataset')
    ds = xr.open_mfdataset(files,
        engine   = Config.input.get('engine'  , 'netcdf4'),
        lock     = Config.input.get('lock'    , False    ),
        parallel = Config.input.get('parallel', False    ),
        chunks   = dict(Config.input.chunks)
    )

    Logger.info('Casting xarray.Dataset to custom Dataset')
    ds = Dataset(ds)

    for key, args in Config.input.replace_vals.items():
        left, right = args.bounds
        value       = float(args.value) or np.nan
        Logger.debug(f'Replacing values between ({left}, {right}) with {value} for key {key}')

        ds[key] = ds[key].where(
            (ds[key] < left) | (right < ds[key]),
            value
        )

    if Config.input.calc:
        Logger.info('Calculating variables')

        for key, string in Config.input.calc.items():
            Logger.debug(f'- {key} = {string}')
            ds[key] = calc(ds, string)

    ds = config_sel(ds, Config.input.sel)

    if Config.input.daily:
        Logger.info('Aligning to a daily average')
        ds = daily(ds)

    if Config.input.resample:
        Logger.info('Resampling data')
        ds = resample(ds, **Config.input.resample)

    if Config.input.subsample:
        Logger.info('Subsampling data')
        ds = subsample(ds, **Config.input.subsample)

    # `split` is hardcoded by the calling script
    # If `lazy` is set in the Config use that else use the parameter
    lazy = Config.input.get('lazy', lazy)
    if split:
        Logger.debug('Performing split and stack')
        return split_and_stack(ds, lazy)

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
