import logging
import matplotlib.pyplot as plt
import seaborn as sns

from sudsaq.config import Config

# Set seaborn context and style for plot generation
sns.set_context('talk')
sns.set_style('darkgrid')

Logger = logging.getLogger('sudsaq/silos/toar/plots.py')


def data_counts_by_month(df):
    """
    """
    Logger.debug(f'Plotting data_counts_by_month')
    config = Config().plots

    nf = df.reset_index()
    nf.index = pd.to_datetime(nf.datetime)

    gf = nf.groupby(pd.Grouper(freq='M'))
    gf = gf.count()
    gf = gf.sum(axis=1)
    ax = gf.plot(kind='bar', figsize=(50, 10))

    plt.tight_layout()
    if config.outdir:
        output = f'{config.outdir}/data_counts_by_month.png'
        Logger.debug(f'Writing to: {output}')
        plt.savefig(output)
    else:
        plt.show()
