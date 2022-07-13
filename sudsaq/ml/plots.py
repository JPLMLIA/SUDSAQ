"""
"""
import logging
import cartopy.crs       as ccrs
import matplotlib.pyplot as plt

Logger = logging.getLogger('sudsaq/ml/plots.py')

def compare_target_predict(target, predict, title=None):
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
