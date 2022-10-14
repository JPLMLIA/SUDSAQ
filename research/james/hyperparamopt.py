#%%


#%%

from sudsaq.config import Config
from sudsaq.data import load

config = Config('research/james/create.bias.11-15.8hour_avg.yml', 'jan')

#%%

data, target = load(config, split=True, lazy=True)

#%%

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold
)

kfold  = GroupKFold(n_splits=len(set(data.time.dt.year.values)))
groups = target.time.dt.year.values

#%%
gscv = GridSearchCV(
    RandomForestRegressor(),
    cv          = kfold,
    error_score = 'raise',
    n_jobs      = -1,
    **config.hyperoptimize.GridSearchCV.param_grid
    # **config.hyperoptimize.GridSearchCV
)

#%%

gscv.fit(data, target)


{}

# align_print(gscv.best_params_, enum=False, prepend='  ', print=Logger.info)

#%%
# Create the predictor
model = lambda: RandomForestRegressor(**config.model.params, **gscv.best_params_)
