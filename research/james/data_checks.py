import os

import click
import matplotlib.pyplot as plt
import numpy  as np
import xarray as xr

def plot(da, res='h', output='.', figsize=(7, 10)):
    """
    """
    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)

    gaps = (da.time - da.time.shift(time=1)) / np.timedelta64(1, res)
    gaps.plot.scatter(ax=axes[0])
    axes[0].set_title(da.name)

    metrics = ['mean', 'max', 'min', 'std']
    for i, metric in enumerate(metrics):
        ax = axes[i+1]
        data = getattr(da, metric)(['lat', 'lon'])
        data.plot.scatter(ax=ax, alpha=.5)
        ax.set_ylabel(metric)

    if output:
        plt.savefig(f'{output}/{da.name}.metrics.png')

@click.command()
@click.argument('file')
@click.option('-r', '--resolution', default='h', type=click.Choice(['h', 'D']), help='Resolution to report the time gaps in')
@click.option('-o', '--output', default='./plots')
@click.option('-f', '--figsize', default=(14, 14))
def main(file, resolution, output):
    """\
    Creates simple plots for each variable of a given netcdf file
    """
    ds = xr.open_mfdataset([file])

    if not os.path.exists(output):
        os.mkdir(output)

    if not os.path.exists(f'{output}/{file}'):
        os.mkdir(f'{output}/{file}')

    for key, da in ds.items():
        click.echo(f'Processing: {key}')
        plot(da=da, res=resolution, output=f'{output}/{file}', figsize=figsize)

#%%
if __name__ == '__main__':
    main()

#%%

for key, da in ds.items():
    break
da

np.timedelta64(1, 'D')

ds

plot(da)

help(da.plot.scatter)

da.plt
