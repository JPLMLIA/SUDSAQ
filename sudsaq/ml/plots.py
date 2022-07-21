"""
"""
import logging
import cartopy.crs       as ccrs
import matplotlib.pyplot as plt
import pandas            as pd
import seaborn           as sns

from sudsaq.config import Config

# Set seaborn styles
sns.set_style('darkgrid')
sns.set_context('talk')

# Increase matplotlib's logger to warning to disable the debug spam it makes
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('fiona').setLevel(logging.WARNING)

Logger = logging.getLogger('sudsaq/ml/plots.py')

def compare_target_predict(target, predict, title=None, save=None):
    """
    """
    def draw(data, ax, title, vmax=None, vmin=None):
        """
        """
        data.plot.pcolormesh(x='lon', y='lat', ax=ax,
            levels = 10,
            cmap   = 'viridis',
            vmax   = vmax,
            vmin   = vmin
        )
        ax.coastlines()
        ax.gridlines(draw_labels=True, color='dimgray', linewidth=0.5)
        ax.set_title(title)

    fig = plt.figure(figsize=(10*3, 6*2))
    if title:
        fig.suptitle(title, fontsize=44)

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

    if save:
        Logger.info(f'Saving compare_target_predict plot to {save}')
        plt.savefig(save)

def truth_vs_predicted(target, predict, save=None):
    """
    """
    def scatter(ax):
        """
        """
        # Scatter truth vs predicted
        ax.scatter(x=target, y=predict)

        # Normalize the axis limits
        ax.set_ylim(limits)
        ax.set_xlim(limits)

        # Create the horizontal line for reference
        ax.plot((limits[0], limits[1]), (limits[0], limits[1]), 'r')

        ax.set_xlabel('Truth')
        ax.set_ylabel('Predicted')
        ax.set_title('Scatter')

    def density(ax):
        """
        """
        ax.hist2d(target, predict, bins=(100, 100), range=[limits, limits])

        # Set labels
        ax.set_xlabel('Truth')
        ax.set_title('Density')

        # Create the horizontal line for reference
        ax.plot((limits[0], limits[1]), (limits[0], limits[1]), 'r')

    # Create the axis
    fig = plt.figure(figsize=(7*2, 7))
    fig.suptitle('Truth vs Predicted', fontsize=44)

    # Retrieve the limits for plotting
    limits = min([target.min(), predict.min()]) - 5, max([target.max(), predict.max()]) + 5

    # Generate plots
    scatter(plt.subplot(121))
    density(plt.subplot(122))

    if save:
        Logger.info(f'Saving truth_vs_predicted plot to {save}')
        plt.savefig(save)

def importances(df, pdf=None, save=None):
    def draw(df, ax, title):
        """
        Takes in a DataFrame of the form DataFrame(columns=variables, index=['importance', 'stddev'])

        Parameters
        ----------
        df: pandas.DataFrame
            The feature importances to plot
        """
        if config.count:
            Logger.debug(f'Plotting only the top {config.count} features')
            df[df.columns[:config.count]]

        ax.bar(x=range(df.shape[1]), height=df.loc['importance'], yerr=df.loc['stddev'], align='center', tick_label=df.columns)

        ax.set_title(title)

        if config.labelrotation:
            ax.tick_params(axis='x', labelrotation=config.labelrotation)

    # Retrieve the config for this plot type
    config = Config().plots.importances

    # Check if pdf was provided
    perms = isinstance(pdf, pd.core.frame.DataFrame)

    # Retrieve the figsize from the config if provided, otherwise attempt to calculate a good shape
    if config.figsize:
        figsize = config.figsize
    else:
        height  = 2 if perms else 1
        width   = config.get('count', df.shape[1])
        figsize = (5 * width, 5 * height)

    # If the permutation importance is provided, draw both, otherwise just the normal
    fig = plt.figure(figsize=figsize)
    if perms:
        fig.suptitle('Feature Importance', fontsize=44)

        # Generate plots (2 rows, 1 column)
        draw(df ,
            ax    = plt.subplot(211),
            title = 'Model Importance'
        )
        draw(pdf,
            ax    = plt.subplot(212),
            title = 'Permutation Importance'
        )
    else:
        draw(df, plt.subplot(111), 'Feature Importance')

    # Fix suptitle clipping when perms is drawn
    if perms:
        fig.tight_layout(rect=[0, 0.03, 1, 0.93])

    if save:
        Logger.info(f'Saving importances plot to {save}')
        plt.savefig(save)
