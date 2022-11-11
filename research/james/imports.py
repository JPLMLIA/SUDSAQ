%load_ext autoreload
%autoreload 2
%matplotlib inline

import numpy  as np
import pandas as pd
import xarray as xr

from scipy.stats import (
    gaussian_kde,
    pearsonr
)
from sklearn.inspection import permutation_importance
from sklearn.metrics    import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)
from treeinterpreter import treeinterpreter as ti
from sudsaq.ml       import treeinterpreter as aqti

from sudsaq.config import (
    Config,
    Section,
    Null
)
from sudsaq.data  import load, Dataset
from sudsaq.ml    import plots
from sudsaq.utils import (
    align_print,
    load_pkl,
    mkdir,
    save_pkl,
    save_netcdf
)

import cartopy.crs       as ccrs
import matplotlib.pyplot as plt
import pandas            as pd
import seaborn           as sns

from sudsaq.config import Config

# Set seaborn styles
sns.set_style('darkgrid')
sns.set_context('talk')

from types import SimpleNamespace

Logger = SimpleNamespace(
    exception = lambda string: print(f'EXCEPTION: {string}'),
    info      = lambda string: print(f'INFO: {string}'),
    error     = lambda string: print(f'ERROR: {string}'),
    debug     = lambda string: print(f'DEBUG: {string}')
)
