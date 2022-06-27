"""
"""
import argparse
import logging
import xarray as xr

from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)

from sudsaq.config import (
    Config,
    Section
)
from sudsaq.data  import load
from sudsaq.utils import (
    align_print,
    load_pkl
)

Logger = logging.getLogger('sudsaq/ml/analyze.py')

def analyze(model=None, data=None, target=None):
    """
    """
    config = Config()

    if model is None:
        if not os.path.exists(config.input.model):
            Logger.error('No model provided')
            return 1
        load_pkl(config.input.model)

    if data is None:
        data, target = load(config, split=True)

    Logger.info('Predicting')
    predict    = xr.zeros_like(target)
    predict[:] = model.predict(data)

    Logger.info('Calculating scores')
    stats = Section('scores', {
        'mape': mean_absolute_percentage_error(target, predict),
        'mse' : mean_squared_error(target, predict),
        'r2'  : r2_score(target, predict),
    })

    # Log the scores
    align_print(stats, enum=False, prepend=' '*4, print=Logger.info)

    # Attach additional objects
    stats.predict = predict

    return stats
