"""
"""
import logging
import pickle
import sys

from sudsaq.config import Config


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
    Pretty prints an iterable in the form {key} = {value}
    offset: int
        Space between the key and the delimiter: {key}{offset}{delimiter}
        Defaults to 1, eg: "key ="
    """
    # Determine how much padding between the
    pad = max([1, len(max(iterable.keys(), key=len)) + offset])

    # Build the formatted string
    fmt = prepend
    if enum:
        fmt += '- {i:' + f'{len(iterable)}' + '}: '
    fmt += '{key:'+ str(pad) + '}' + delimiter + ' {value}'

    for i, (key, value) in enumerate(iterable.items()):
        print(fmt.format(i=i, key=key, value=value))
