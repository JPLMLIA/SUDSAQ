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


def calculate(ds, string, evaluate=False):
    """
    Formats a simple expression by replacing `(key)` with `ds['key']`

    Parameters
    ----------
    ds : xr.Dataset
        The dataset being operated on
    string : str
        Code to be performed on the Dataset object where keys are surrounded by parenthesis
    evaluate : bool, default=False
        Call eval(string) to return instead

    Returns
    -------
    string : str or xr.DataArray
        Formatted string or
    keys : list
        List of keys that were used in the string
    """
    keys = []
    for key in list(ds):
        current = string
        string  = re.sub(f'(\({key}\))', f"ds['{key}']", string)

        if current != string:
            keys.append(key)

    if evaluate:
        return eval(string), keys
    return string, keys


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


def daily(ds, sels):
    """
    Aligns a dataset to a daily average

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset to align.
    sels : mlky.Sect
        {key: sel} where
            key: name of selection
            sel: Sect containing:
                time: int or 2 item list
                    - int: select this exact time
                    - list:
                vars: str, list
                    Variables to operate on

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
    for sect, sel in sels.items():
        Logger.debug(f'- {sect}: Selecting times {sel.time} on variables {sel.vars}')
        try:
            ds[sel.vars]
        except:
            Logger.error(f'Could not select variables: {sel.vars}, skipping')
            continue

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


class Loader:
    def __init__(self, input=None):
        """
        """
        # If a given input section is provided use that, else fallback to default location
        self.C = input or Config.input

        self.target = Config.target

        # Keys needed for calculating the target variable, if needed
        self.tkeys = None

    def glob(self):
        """
        Globs for input files and opens the dataset

        Config Options
        --------------
        .glob: list
            Glob strings
        .open_mfdataset: dict
            kwargs for function
        .lazy: bool
            Lazy load dataset or not
        """
        Logger.info('Collecting files')
        files = []
        for string in self.C.glob:
            match = glob(string)
            Logger.debug(f'Collected {len(match)} files using "{string}"')
            files += match

        if not files:
            Logger.error('No files collected')
            return False

        Logger.info('Opening mfdataset')
        self.ds = xr.open_mfdataset(files, **self.C.open_mfdataset)

        return True

    def select(self):
        """
        Performs custom sel operations defined in the Config

        Config Options
        --------------
        .sels: Sect
            {dim: sel} where
                dim: Dimension to select on
                This can also be:
                    'vars': Select specific variables ds[sel]
                    'month': Select on the time dimension such that time.month == sel
                    'drop_date': Drop a specific date
                        sel may contain:
                            'year': Drop this year
                            'month': Drop this month
                            'day': Drop this day
                        A mask will be constructed using these three keys, ex:
                            sel:
                                year: 2004
                                month: 2
                                day: 29
                        This will drop 2004-02-29
                            sel:
                                year: 2004
                                month: 2
                        This will drop all of February 2004
                if `dim` is an existing dimension, `sel` can be:
                    - 2 item list
                    - A single value
        """
        ds = self.ds

        for dim, sel in self.C.sel.items():

            # Select on an existing dimension
            if dim in ds.coords:
                if isinstance(sel, list):
                    if len(sel) > 2:
                        Logger.error(f'List selection on a dimension must be 2 items in length')
                        continue

                    i, j = sorted(sel)

                    # Discover which way to create the slice
                    a, b = ds[dim][[0, -1]]

                    # Increasing
                    if a < b:
                        bounds = slice(i, j)
                    # Decreasing
                    elif a > b:
                        bounds = slice(j, i)

                    Logger.debug(f'Selecting on {dim}: {bounds}')

                    ds = ds.sel(**{dim: bounds})

                # Select this specific value on the dimension
                else:
                    Logger.debug(f'Selecting: {dim}=={sel}')
                    ds = ds.sel(**{dim: sel})

            # Select specific variables
            if dim == 'vars':
                Logger.debug(f'Selecting variables: {sel}')

                t = None
                if self.target in ds:
                    t = self.target, ds[self.target]
                elif self.tkeys:
                    t = self.tkeys, ds[self.tkeys]

                ds = Dataset(ds)[sel]

                # Ensure the target is retained
                if t is not None:
                    target, t  = t
                    ds[target] = t

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

            else:
                valid = list(ds.coords) + ['vars', 'month', 'drop_date']
                Logger.error(f'Dimension {dim!r} not one of: {valid}')

        self.ds = Dataset(ds)

    def split_and_stack(self):
        """
        Splits the target from the data and stacks both to be 1d
        """
        Logger.info(f'Creating the stacked training and target objects')

        # Extract the target
        target  = self.ds[self.target]
        self.ds = self.ds.drop_vars([self.target])

        # Save the lat/lon dimensions before dropping na for easy reconstruction later down the pipeline
        Config._reindex = self.ds[['lat', 'lon']]

        # Create the stacked objects
        self.data   = flatten(self.ds).transpose('loc', 'variable')
        self.target = flatten(target)

        Logger.debug(f'Data   shape: {self.data.sizes}')
        Logger.debug(f'Target shape: {self.target.sizes}')

        if self.C.scale:
            Logger.info('Scaling X data')
            self.data = scale(self.data)

        # Remove to save space
        del self.ds

        Logger.debug(f'- Data   = {self.data.nbytes / 2**30:.3f}')
        Logger.debug(f'- Target = {self.target.nbytes / 2**30:.3f}')
        return self.data, self.target

    def load(self, split=False):
        """
        Loads and preprocesses data from files.
        """

        if not self.glob():
            return

        # If this is a calculated target, track which variables are relevant so they don't get dropped
        if self.target not in self.ds:
            _, self.tkeys = calculate(self.ds, self.target)

        # Perform subselections
        self.select()

        # Load into memory after subselections to reduce initial memory footprint
        if self.C.lazy:
            Logger.info('Dataset is lazy')
        else:
            Logger.info('Loading full dataset into memory')
            self.ds.load()

        # Replace values for certain keys
        for key, args in self.C.replace_vals.items():
            left, right = args.bounds
            value       = float(args.value) or np.nan

            Logger.debug(f'Replacing values between ({left}, {right}) with {value} for key {key}')

            self.ds[key] = self.ds[key].where(
                (self.ds[key] < left) | (right < self.ds[key]),
                value
            )

        # Calculate new variables per the config
        if self.C.calc:
            Logger.info('Calculating variables')

            for key, string in self.C.calc.items():
                Logger.debug(f'- {key} = {string}')
                ds[key], _ = calculate(ds, string, evaluate=True)

        if self.C.daily:
            Logger.info('Aligning to a daily average')
            self.ds = daily(self.ds, self.C.daily)

        if self.C.resample:
            Logger.info('Resampling data')
            self.ds = resample(self.ds, **self.C.resample)

        if self.C.subsample:
            Logger.info('Subsampling data')
            self.ds = subsample(self.ds, **self.C.subsample)

        # Try to calculate the target if it doesn't exist
        if self.target not in self.ds:
            Logger.debug(f'Target not in ds, attempting to calculate it: {self.target}')
            self.ds['target'], drop = calculate(self.ds, self.target, evaluate=True)
            self.target = 'target'

            Logger.debug(f'Dropping keys as they were used to create the target: {drop}')
            self.ds = self.ds.drop_vars(drop)

        Logger.debug(f'Memory of ds = {self.ds.nbytes / 2**30:.3f}')
        Logger.info('Finished loading')

        if split:
            return self.split_and_stack()

        return self.ds
