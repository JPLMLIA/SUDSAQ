"""
"""
import logging
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

    logging.basicConfig(
        level   = levels.get(config.log.level or '', logging.DEBUG),
        format  = config.log.format  or '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt = config.log.datefmt or '%m-%d %H:%M',
        stream  = sys.stdout,
        # force   = True
    )
