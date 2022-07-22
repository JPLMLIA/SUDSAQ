"""
"""
import logging
import os
import pickle
import sys
import xarray as xr

from sudsaq.config import Config

Logger = logging.getLogger('sudsaq/utils.py')

def init(args):
    """
    Initializes the root logger with parameters defined by the config.

    Parameters
    ----------
    config: timefed.config.Config
        MilkyLib configuration object

    Notes
    -----
    config keys:
        log:
            level: str
            format: str
            datefmt: str
    """
    config = Config(args.config, args.section)

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
    sh.setLevel(levels.get(config.log.level or '', logging.INFO))
    handlers.append(sh)

    if config.log.file:
        # Make sure the directory exists first
        mkdir(config.log.file)

        # Add the file logging
        fh = logging.FileHandler(config.log.file)
        fh.setLevel(logging.DEBUG)
        handlers.append(fh)

    logging.basicConfig(
        level    = logging.DEBUG,
        format   = config.log.format  or '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt  = config.log.datefmt or '%m-%d %H:%M',
        handlers = handlers,
        # force    = True
    )

    logging.getLogger().debug(f'Logging initialized using Config({args.config}, {args.section})')

def save_netcdf(data, name, output):
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
    """
    if isinstance(data, (xr.core.dataarray.DataArray, xr.core.dataarray.Dataset)):
        if 'loc' in data.dims:
            Logger.warning(f'Saving {name} must be done unstacked')
            data = data.unstack()

        Logger.info(f'Saving {name} to {output}')
        data.to_netcdf(output, engine='netcdf4')

def save_pkl(file, data):
    """
    Saves data to a file via pickle

    Parameters
    ----------
    file : str
        Path to a file to dump the data to via pickle
    data: any
        Any pickleable object
    """
    with open(file, 'wb') as file:
        pickle.dump(data, file)

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
    print: func or list of func, default = print
        The print function to use. Allows using logging instead of Python's normal print
        May be a list of functions to use, such as [logger.info, file.write]
    """
    # Determine how much padding between the key and delimiter
    pad = max([1, len(max(iterable.keys(), key=len))]) + offset

    # Build the formatted string
    fmt = prepend
    if enum:
        fmt += '- {i:' + f'{len(str(len(iterable)))}' + '}: '
    fmt += '{key:'+ str(pad) + '}' + delimiter + ' {value}'

    for i, (key, value) in enumerate(iterable.items()):
        string = fmt.format(i=i, key=key, value=value)

        if isinstance(print, (list, tuple)):
            for func in print:
                # If this is writing to file, append \n
                if func.__name__ == 'write':
                    func(string + '\n')
                else:
                    func(string)
        else:
            # If this is writing to file, append \n
            if print.__name__ == 'write':
                print(string + '\n')
            else:
                print(string)

def mkdir(path):
    """
    Attempts to create directories for a given path
    """
    # Make sure this is a directory path
    path, _ = os.path.split(path)

    # Split into parts to reconstruct
    split = path.split('/')

    # Now reconstruct the path one step at a time and ensure the directory exists
    for i in range(2, len(split)+1):
        dir = '/'.join(split[:i])
        if not os.path.exists(dir):
            try:
                os.mkdir(dir, mode=0o771)
            except Exception as e:
                Logger.exception(f'Failed to create directory {dir}')
                raise e
