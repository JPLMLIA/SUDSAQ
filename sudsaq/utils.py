"""
"""
# Builtin
import logging
import os
import pickle
import shutil
import sys

from pathlib import Path

# External
import dask
# import cf_xarray as cfxr
import xarray as xr

from dask.distributed import Client
from mlky import Config

# Internal
from sudsaq.data import unstacked


Logger = logging.getLogger('sudsaq/utils.py')


def init(args):
    """
    Initializes the root logger with parameters defined by the config.

    Parameters
    ----------
    args

    Notes
    -----
    config keys:
        log:
            level: str
            format: str
            datefmt: str
    """
    # Initialize quietly to access a few options
    Config(args.config, args.patch)

    levels = {
        'critical': logging.CRITICAL,
        'error'   : logging.ERROR,
        'warning' : logging.WARNING,
        'info'    : logging.INFO,
        'debug'   : logging.DEBUG
    }

    handlers = []

    # Create console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(levels.get(Config.log.level or '', logging.INFO))
    handlers.append(sh)

    if Config.log.file:
        # Make sure the directory exists first
        mkdir(Config.log.file)

        if Config.log.reset and os.path.exists(Config.log.file):
            os.remove(Config.log.file)

        # Add the file logging
        fh = logging.FileHandler(Config.log.file)
        fh.setLevel(logging.DEBUG)
        handlers.append(fh)

    logging.basicConfig(
        level    = logging.DEBUG,
        format   = Config.log.format  or '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt  = Config.log.datefmt or '%m-%d %H:%M',
        handlers = handlers,
        # force    = True
    )

    # Reinitialize now that logging is set up
    Config(args.config, args.patch, **Config.mlky)

    Logger.debug(f'Logging initialized using Config({args.config}, {args.patch})')

    if Config.log.config:
        shutil.copy(args.config, Config.log.config)

    # Logger.debug('Instantiating the Dask cluster')
    # dask.config.set(dict(config.dask.config))
    # config.dask_client = Client(**config.dask.client)

    return args, Config


def align_print(iterable, enum=False, delimiter='=', offset=1, prepend='', print=print):
    """
    Pretty prints an iterable in the form {key} = {value} such that the delimiter (=)
    aligns on each line

    Parameters
    ----------
    iterable: iterable
        Any iterable with a .items() function
    enum: bool, default = False
        Whether to include enumeration of the items
    delimiter, default = '='
        The symbol to use between the key and the value
    offset: int, default = 1
        Space between the key and the delimiter: {key}{offset}{delimiter}
        Defaults to 1, eg: "key ="
    prepend: str, default = ''
        Any string to prepend to each line
    print: func, default = print
        The print function to use. Allows using custom function instead of Python's normal print
    """
    # Determine how much padding between the key and delimiter
    pad = max([1, len(max(iterable.keys(), key=len))]) + offset

    # Build the formatted string
    fmt = prepend
    if enum:
        fmt += '- {i:' + f'{len(str(len(iterable)))}' + '}: '
    fmt += '{key:'+ str(pad) + '}' + delimiter + ' {value}'

    # Create the formatted list
    fmt_list = []
    for i, (key, value) in enumerate(iterable.items()):
        string = fmt.format(i=i, key=key, value=value)
        fmt_list.append(string)

    for string in fmt_list:
        print(string)

    return fmt_list


def mkdir(path):
    """
    Attempts to create directories for a given path
    """
    # Make sure this is a directory path
    path, _ = os.path.split(path)

    # Cast to pathlib and use their mkdir
    path = Path(path)

    try:
        path.mkdir(mode=0o775, parents=True, exist_ok=True)
    except:
        Logger.exception(f'Exception raised attempting to create directory: {dir}')
    finally:
        if not path.exists():
            Logger.error(f'Failed to create directory {dir}')


def load_pkl(file):
    """
    Loads data from a pickle

    Parameters
    ----------
    file : str
        Path to a Python pickle file to load

    Returns
    -------
    any
        The data object loaded from the pickle file
    """
    return pickle.load(open(file, 'rb'))


def save_pkl(data, output, **kwargs):
    """
    Saves data to a file via pickle

    Parameters
    ----------
    data: any
        Any pickleable object
    output : str
        Path to a file to dump the data to via pickle
    """
    mkdir(output)
    with open(output, 'wb') as file:
        pickle.dump(data, file)


def encode(data):
    """
    """
    return cfxr.encode_multi_index_as_compress(data, 'loc')


def decode(file):
    """
    """
    return cfxr.decode_compress_to_multi_index(xr.open_dataset(file), 'loc')


@unstacked
def save_netcdf(data, name, output, dataset=False, reindex=None):
    """
    Handles saving NetCDF (.nc) files. Unstacks an object if the `loc` dimension is present.

    Parameters
    ----------
    data: xr.core.dataarray.DataArray or xr.core.dataarray.Dataset
        An xarray object to be saved. If the object is not one of these types, the
        function quietly returns
    name: str
        Name of the object for logging purposes
    output: str
        Path to output to
    dataset: bool, default = False
        If this should be unstacked to a Dataset object instead of a DataArray
    reindex: Dataset or DataArray, default = None
        If provided, reindexes the data object with the given object
        This is primarily used to expand dimensions back to the MOMO grid after having
        shrunk from dropna
    """
    if not isinstance(data, (xr.core.dataarray.DataArray, xr.core.dataarray.Dataset)):
        Logger.error(f'Wrong dtype for object {name!r}, cannot save netcdf. Got: {type(data)}')
        return

    # Names always get set on DataArray objects
    if isinstance(data, xr.core.dataarray.DataArray):
        data.name = name

        # Only DataArrays can be cast back to Datasets
        if dataset:
            data = data.to_dataset('variable')

    # Apply reindexing if its available
    if isinstance(reindex, (xr.core.dataarray.DataArray, xr.core.dataarray.Dataset)):
        try:
            data = data.reindex_like(reindex)
        except:
            Logger.exception('Failed to reindex, leaving as-is')

    # Correct if lon got mixed up as it normally does during the pipeline
    data = data.sortby('lon')

    # if 'loc' in data.dims:
    #     Logger.warning(f'The `loc` dimension required being encoded for saving and decoded after loading from file')
    #     data = encode(data)
    #     output += '.ec'

    Logger.info(f'Saving {name} to {output}')
    data.to_netcdf(output, engine='netcdf4')


def save_objects(output, kind, **others):
    """
    Simplifies saving objects by taking a dictionary of {name: obj}
    where obj is an xarray object to be passed to save_netcdf
    """
    # These objects will be converted to a dataset
    datasets = ['data', 'contributions', 'explanation']

    for name, obj in others.items():
        if Config.output[name]:
            save_netcdf(
                data    = obj,
                name    = name,
                output  = f'{output}/{kind}.{name}.nc',
                reindex = Config._reindex,
                dataset = name in datasets
            )
        else:
            Logger.warning(f'Object {name!r} is not enabled to be saved in the config, skipping')


def load_from_run(path, kind=None, objs=None):
    """
    Loads objects from a given run
    """
    files = {
        'model'      : f'{path}/model.pkl',
        'data'       : f'{path}/{kind}.data.nc',
        'target'     : f'{path}/{kind}.target.nc',
        'predict'    : f'{path}/{kind}.predict.nc',
        'explanation': f'{path}/shap.explanation.nc'
    }
    if not objs:
        objs = files.keys()

    ret = []
    for obj, file in files.items():
        if obj in objs:
            if not os.path.exists(file):
                Logger.error(f'File not found: {file}')

            if file.endswith('.pkl'):
                Logger.info(f'Loading {obj}: {file}')
                ret.append(
                    load_pkl(file)
                )
            elif file.endswith('.nc'):
                Logger.info(f'Loading {obj}: {file}')
                ret.append(
                    xr.open_dataset(file)
                )
            elif file.endswith('.ec'):
                Logger.info(f'Loading {obj}: {file}')
                ret.append(
                    decode(file)
                )
            else:
                Logger.error(f'Invalid option: {obj}')

    return ret


def catch(func):
    """
    Decorator that protects the caller from an exception raised by the called function.

    Parameters
    ----------
    func: function
        The function to wrap

    Returns
    -------
    _wrap: function
        The wrapper function that calls func inside of a try/except block
    """
    def _wrap(*args, **kwargs):
        """
        Wrapper function

        Parameters
        ----------
        *args: list
            List of positional arguments for func
        **kwargs: dict
            Dict of keyword arguments for func
        """
        try:
            func(*args, **kwargs)
        except:
            Logger.exception(f'Function <{func.__name__}> raised an exception')

    # Need to pass the docs on for sphinx to generate properly
    _wrap.__doc__ = func.__doc__
    return _wrap
