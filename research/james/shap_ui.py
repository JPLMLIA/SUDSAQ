"""
"""
import cartopy.crs       as ccrs
import matplotlib.pyplot as plt
import seaborn           as sns
import panel as pn
import param
import shap
import xarray as xr

from matplotlib import pyplot as plt

from sudsaq.ml.explain import (
    Dataset,
    Explanation
)

# Set seaborn styles
sns.set_style('darkgrid')
sns.set_context('talk')


def load(file):
    """
    Loads a shap explanation object from a sudsaq netcdf dataset
    """
    ds = Dataset(xr.open_dataset(file).load())
    ns = ds.stack(loc=['lat', 'lon', 'time']).transpose()
    return ns.to_explanation()

ex = load('.local/data/shap/v4/jul/2011/test.explanation.nc')

class Summary(param.Parameterized):
    """
    """
    # max_display = param.Integer(default=20, bounds=(1, 40))
    # max_display = param.ListSelector(default=20, objects=list(range(1, 41)))
    max_display = param.ObjectSelector(default=20, objects=list(range(1, 41)))
    plot_type   = param.ObjectSelector(default='dot', objects=['dot', 'bar', 'violin', 'compact_dot'])

    cache = {}

    @param.depends('max_display', 'plot_type')
    def view(self):
        """
        """
        key = (self.max_display, self.plot_type)
        if key in self.cache:
            return self.cache[key]

        plt.close('all')
        shap.summary_plot(ex,
            show        = False,
            max_display = self.max_display,
            plot_type   = self.plot_type
        )
        fig = pn.pane.Matplotlib(plt.gcf(), dpi=144)

        self.cache[key] = fig
        # return pn.pane.Str(f'max_display: {self.max_display}\nplot_type: {self.plot_type}')
        return fig

    def panel(self):
        return pn.Row(self.param, self.view).servable()

# summary = Summary(name='Summary')
# summary.panel()
# server = summary.panel().show(threaded=True)

#%%

class Dependence(param.Parameterized):
    """
    """
    cache = {}

    feature = param.ObjectSelector(default=ex.feature_names[0], objects=ex.feature_names)
    interaction_index = param.ObjectSelector(default='auto', objects=['auto'] + ex.feature_names)

    @param.depends('feature', 'interaction_index')
    def view(self):
        """
        """
        key = (self.feature, self.interaction_index)
        if key in self.cache:
            return self.cache[key]

        fig = plt.Figure()
        ax  = fig.subplots()
        shap.dependence_plot(
            self.feature,
            shap_values        = ex.values,
            features           = ex.data,
            feature_names      = ex.feature_names,
            display_features   = None,
            interaction_index  = self.interaction_index,
            color              = '#1E88E5',
            axis_color         = '#333333',
            cmap               = None,
            dot_size           = 16,
            x_jitter           = 0,
            alpha              = 1,
            title              = None,
            xmin               = None,
            xmax               = None,
            ax                 = ax,
            show               = False
        )
        fig = pn.pane.Matplotlib(fig, dpi=144)
        self.cache[key] = fig

        return fig

    def panel(self):
        return pn.Row(self.param, self.view).servable()

dep = Dependence(name="Dependence")
# dep1.panel()
# dep2 = Dependence(name="Dependence2")
# pn.Tabs(dep1.panel(), dep2.panel())

#%%
#
# ds = ex.to_dataset()
#
# ns = ds['values'].to_dataset('variable')
# ns = ns.unstack('loc')
# ms = ns.mean('time')
#
# ms['momo.t'].plot()
#
# #%%
#
# ds = ex.to_dataset()['values'].to_dataset('variable').unstack('loc')
# ds
# hj
# #%%

ds = ex.to_dataset()['values'].to_dataset('variable').unstack('loc')

class Spatial(param.Parameterized):
    """
    """
    cache   = {}
    feature = param.ObjectSelector(
        default = list(ds)[0],
        objects = list(ds)
    )
    timestamp = param.ObjectSelector(
        default = 'mean',
        objects = ['mean'] + list(ds.time.values)
    )
    # def __init__(self, explanation, **kwargs):
    #     """
    #     """
    #
    #
    #     super().__init__(**kwargs)

    @param.depends('feature', 'timestamp')
    def view(self):
        """
        """
        key = (self.feature, self.timestamp)
        if key in self.cache:
            return self.cache[key]

        data = ds[self.feature]
        if self.timestamp == 'mean':
            data = data.mean('time')
        else:
            data = data.sel(time=self.timestamp)

        fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': ccrs.PlateCarree()})

        data.plot.pcolormesh(x='lon', y='lat', ax=ax, levels=20, cmap='bwr')
        ax.coastlines()
        ax.gridlines(draw_labels=True, color='dimgray', linewidth=0.5)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        gl = ax.gridlines(draw_labels=['left', 'bottom'], color='dimgray', linewidth=1)
        fig = pn.pane.Matplotlib(fig, dpi=144)
        self.cache[key] = fig

        return fig

    def panel(self):
        return pn.Row(self.param, self.view).servable()

spatial = Spatial(name="Spatial")
pn.Tabs(
    ('Dependence', dep.panel()),
    ('Spatial', spatial.panel())
)#.servable()

#%%
import logging
import matplotlib.pyplot as plt
import panel as pn
import shap
import xarray as xr

logging.basicConfig(level=logging.DEBUG)

from sudsaq.ml.explain import (
    Dataset,
    Explanation
)


class SUDSAQ_UI:
    file = '~/projects/suds-air-quality/.local/data/shap/v4/jul/2015/test.explanation.nc'
    ds = None
    ex = None
    cache = {}

    plotting = pn.indicators.LoadingSpinner(value=True, name='Generating Plot')

    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)

    def load(self, file):
        """
        """
        self.log.info(f'Loading file: {file}')
        try:
            self.ds = Dataset(xr.open_dataset(file).load())
            ns = self.ds.stack(loc=['lat', 'lon', 'time']).transpose()
            self.ex = ns.to_explanation()
            self.file = file
        except:
            self.log.exception(f'Failed to load file: {file}')
        self.log.debug('Finished load()')

    def summary(self):
        """
        """
        def bar():
            plt.close('all')
            shap.plots.bar(self.ex, max_display=None, show=False)
            plt.title('Average Absolute SHAP Value')
            plt.tight_layout()
            return pn.pane.Matplotlib(plt.gcf())

        def beeswarm():
            plt.close('all')
            shap.summary_plot(self.ex, max_display=None, show=False)
            # plt.title('Average Absolute SHAP Value')
            plt.tight_layout()
            return pn.pane.Matplotlib(plt.gcf())

        if self.ex is None:
            self.load(self.file)

        col = pn.Column(
            pn.pane.Markdown(f'## Using file: {self.file}'),
            pn.Row(bar(), beeswarm()),
            sizing_mode = 'stretch_width'
        )
        return col

    def dependence(self):
        return pn.pane.Markdown('# Dependence')

    def spatial(self):
        return pn.pane.Markdown('# Spatial')

    def panel(self):
        return pn.Column(
            pn.pane.Markdown('# SUDS AQ SHAP Explorer', width=800),
            pn.Tabs(
                ('Summary', self.summary()),
                ('Dependence', self.dependence()),
                ('Spatial', self.spatial())
            )
        ).servable()

ui = SUDSAQ_UI()
ui.panel().show()

#%%

help(finp.param.watch)



#%%
pn.pane.Matplotlib

#%%
import logging
import matplotlib.pyplot as plt
import panel as pn
import shap
import xarray as xr

logging.basicConfig(level=logging.DEBUG)

from mlky import Section

from sudsaq.ml.explain import (
    Dataset,
    Explanation
)

self = Section({})
self.file = '~/projects/suds-air-quality/.local/data/shap/v4/jul/2013/test.explanation.nc'
self.ds = None
self.ex = None
self.log = logging.getLogger('UI')

ds = xr.open_dataset(self.file)
ds = ds.load()
Dataset(ds)

#%%
self.ds = Dataset(xr.open_dataset(self.file).load())
ns = self.ds.stack(loc=['lat', 'lon', 'time']).transpose()
self.ex = ns.to_explanation()
self.file = file


#%%
def load(file):
    """
    """
    self.log.info(f'Loading file: {file}')
    try:
        self.ds = Dataset(xr.open_dataset(file).load())
        ns = self.ds.stack(loc=['lat', 'lon', 'time']).transpose()
        self.ex = ns.to_explanation()
        self.file = file
    except:
        self.log.exception(f'Failed to load file: {file}')
    self.log.debug('Finished load()')

load(self.file)

#%%
# Load a file if user changes
# finp = pn.widgets.FileInput(accept='.nc', multiple=False)
# pn.bind(load, finp.param.filename, watch=True)

summary = pn.Column(
    pn.pane.Markdown(f'Processing file: {self.file}'),
    pn.pane.Markdown('# Summary')
)

def dependence():
    return pn.pane.Markdown('# Dependence')

def spatial():
    return pn.pane.Markdown('# Spatial')

pn.Column(
    pn.pane.Markdown('# SUDS AQ SHAP Explorer', width=800),
    pn.Tabs(
        ('Summary', summary),
        ('Dependence', dependence()),
        ('Spatial', spatial())
    )
).servable().show()

#%%



#%%
