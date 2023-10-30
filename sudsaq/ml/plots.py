"""
"""
# Builtin
import logging

# External
import cartopy.crs       as ccrs
import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import seaborn           as sns
import xarray            as xr

from mlky  import Config
from scipy import stats

# Internal
from sudsaq.utils import catch

# Set seaborn styles
sns.set_style('darkgrid')
sns.set_context('talk')

# Increase matplotlib's logger to warning to disable the debug spam it makes
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('fiona').setLevel(logging.WARNING)

Logger = logging.getLogger('sudsaq/ml/plots.py')

@catch
def compare_target_predict(target, predict, reindex=None, title=None, save=None):
    """
    """
    def draw(data, ax, title, vmax=None, vmin=None):
        """
        """
        data.plot.pcolormesh(x='lon', y='lat', ax=ax,
            levels = 20,
            cmap   = 'viridis',
            vmax   = vmax,
            vmin   = vmin
        )
        ax.coastlines()
        ax.gridlines(draw_labels=True, color='dimgray', linewidth=0.5)
        ax.set_title(title)

        # Remove the top and right axis tick labels
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

    Logger.info('Generating plot: compare_target_predict')
    fig = plt.figure(figsize=(20*3, 12*2))
    if title:
        fig.suptitle(title, fontsize=44)

    if isinstance(reindex, (xr.core.dataarray.DataArray, xr.core.dataarray.Dataset)):
        target  = target.reindex_like(reindex).sortby('lon')
        predict = predict.reindex_like(reindex).sortby('lon')

    # Target
    draw(
        target,
        plt.subplot(231, projection=ccrs.PlateCarree()),
        'Truth'
    )
    # Predicted
    draw(
        predict,
        plt.subplot(232, projection=ccrs.PlateCarree()),
        'Predicted'
    )
    # Residuals
    residual = target - predict
    draw(
        residual,
        plt.subplot(233, projection=ccrs.PlateCarree()),
        'Residuals'
    )
    ## Standardize the colorbar range
    vmax = max([target.max(), predict.max(), residual.max()])
    vmin = min([target.min(), predict.min(), residual.min()])
    # Target
    draw(
        target,
        plt.subplot(234, projection=ccrs.PlateCarree()),
        'Standardized Truth',
        vmax, vmin
    )
    # Predicted
    draw(
        predict,
        plt.subplot(235, projection=ccrs.PlateCarree()),
        'Standardized Predicted',
        vmax, vmin
    )
    # Residuals
    draw(
        residual,
        plt.subplot(236, projection=ccrs.PlateCarree()),
        'Standardized Residuals',
        vmax, vmin
    )

    plt.tight_layout()
    if save:
        Logger.info(f'Saving compare_target_predict plot to {save}')
        plt.savefig(save)

@catch
def truth_vs_predicted(target, predict, label=None, save=None):
    """
    """
    Logger.info('Generating plot: truth_vs_predicted')
    fig, ax = plt.subplots(figsize=(10, 10))

    # Retrieve the limits and expand them by 5% so everything fits into a square grid
    limits = min([target.min(), predict.min()]), max([target.max(), predict.max()])
    limits = limits[0] - np.abs(limits[0] * .05), limits[1] + np.abs(limits[1] * .05)
    ax.set_ylim(limits)
    ax.set_xlim(limits)

    # Create the horizontal line for reference
    ax.plot((limits[0], limits[1]), (limits[0], limits[1]), '--', color='r')

    # Create the density values
    kernel  = stats.gaussian_kde([target, predict])
    density = kernel([target, predict])

    plot = ax.scatter(target, predict, c=density, cmap='viridis', label=label, s=5)

    # Create the colorbar without ticks
    cbar = fig.colorbar(plot, ax=ax)
    cbar.set_ticks([])

    # Set labels
    cbar.set_label('Density')
    ax.set_xlabel('Truth')
    ax.set_ylabel('Predicted')
    ax.set_title('Truth vs Predicted')

    if label:
        legend = ax.legend(handlelength=0, handletextpad=0, loc='upper left', prop={'family': 'monospace'})
        legend.legendHandles[0].set_visible(False)

    plt.axis('equal')
    plt.tight_layout()
    if save:
        Logger.info(f'Saving truth_vs_predicted plot to {save}')
        plt.savefig(save)

@catch
def importance(df, pdf=None, save=None):
    """
    """
    Logger.info('Generating plot: importance')

    # Retrieve the config for this plot type
    config = Config.plots.importances

    perm = False
    if isinstance(pdf, pd.core.frame.DataFrame):
        perm = True

        # Sort by df
        pdf = pdf[df.columns]

    # Normalize the scores, defaults to True
    if config.get('normalize', True): # Defaults to True
        df = df / df.max(axis=1).importance

        if perm:
            pdf = pdf / pdf.max(axis=1).importance

    # Control how many features are plotted
    if config.count:
        Logger.info(f'Plotting only the top {config.count} features')
        df = df[df.columns[:config.count]]

        if perm:
            pdf = pdf[df.columns[:config.count]]

    # Retrieve the figsize from the config if provided, otherwise attempt to calculate a good shape
    if config.figsize:
        figsize = config.figsize
    else:
        size = config.get('count', df.shape[1])
        figsize = (size*3, size)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        x          = range(df.shape[1]),
        height     = df.loc['importance'],
        yerr       = df.loc['stddev'],
        label      = 'Model',
        align      = 'center',
        linewidth  = 0,
        width      = .5,
        tick_label = df.columns
    )

    if perm:
        ax.bar(
            x         = ax.get_xticks() + .1,
            height    = pdf.loc['importance'],
            yerr      = pdf.loc['stddev'],
            label     = 'Permutation',
            align     = 'edge',
            linewidth = 0,
            width     = .5,
            alpha     = .9
        )

        # Shift the labels slightly to align better
        ax.set_xticks(ax.get_xticks() + .15)

        # Only enable the legend if permutations are drawn
        ax.legend()

    ax.tick_params(axis='x', labelrotation=config.get('labelrotation', 0))

    # Disable vertical axis lines and set labels
    ax.grid(False, axis='x')
    ax.set_title('Feature Importance')
    ax.set_ylabel('Score')

    plt.tight_layout()
    if save:
        Logger.info(f'Saving importances plot to {save}')
        plt.savefig(save)
