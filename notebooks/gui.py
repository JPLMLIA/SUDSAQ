import logging
import numpy  as np
import pandas as pd
import panel  as pn
import plotly.express as px
import shap
import xarray as xr

from glob    import glob
from pathlib import Path

from matplotlib import pyplot as plt

import hvplot.xarray

logging.basicConfig(level=logging.WARN)

from mlky import Sect

Sect._opts.convertListTypes = False
Sect._opts.convertItems     = False

from sudsaq.ml.explain import (
    Dataset,
    Explanation
)
pn.config.theme = 'dark'

pn.extension('terminal', 'plotly', loading_spinner='dots', loading_color='#00aa41')
pn.param.ParamMethod.loading_indicator = True


def init_logger(terminal):
    logger = logging.getLogger("terminal")
    logger.setLevel(logging.DEBUG)

    stream = logging.StreamHandler(terminal)
    stream.terminator = "  \n"
    formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")

    stream.setFormatter(formatter)
    stream.setLevel(logging.DEBUG)
    logger.addHandler(stream)

    return logger

def cont_to_ex(dir, predicts=True):
    """
    Converts SUDSAQ TreeInterpreter outputs to SHAP Explanation objects
    """
    bias = xr.open_mfdataset(f'{dir}/test.bias.nc').bias.data
    data = xr.open_mfdataset(f'{dir}/test.data.nc').to_array('variable').transpose('lat', 'lon', 'time', 'variable').data
    cont = xr.open_mfdataset(f'{dir}/test.contributions.nc').to_array('variable').transpose('lat', 'lon', 'time', 'variable')

    ds = Dataset(cont.to_dataset(name='values'))
    ds['data'] = (('lat', 'lon', 'time', 'variable'), data)
    ds['base_values'] = (('lat', 'lon', 'time'), bias)
    ex = ds.stack(loc=['lat', 'lon', 'time']).transpose().to_explanation()

    if predicts:
        p = xr.open_mfdataset(f'{dir}/test.predict.nc')
        ds['predict'] = p.predict

    return ds, ex

#%%

class Waterfalls:
    ex = None
    ds = None

    def __init__(self, log, ex=None, ds=None, display=20):
        """
        """
        self.log = log

        self.display = display
        self.widgets = pn.Column(*([None]*2))

        self.initialize(ex, ds)

        self.default = pn.widgets.StaticText(
            name  = 'Waterfall',
            value = 'Load data to generate plot'
        )

    def initialize(self, ex, ds):
        """
        """
        if ex is None:
            return

        if ds is None:
            return

        self.ex = ex

        self.log.debug('[Waterfall] Initializing')

        # grid = ds[['lat', 'lon', 'predict']]
        # grid = grid.stack(point=['time', 'lat', 'lon'])

        # grid = pd.DataFrame({
        #     'time': grid['time'],
        #     'lat': grid['lat'],
        #     'lon': grid['lon'],
        #     'predict': grid['predict']
        # })
        grid = ds[['lat', 'lon']]
        grid = grid.to_dataframe().reset_index()

        # Create plotly map using this grid
        fig = px.scatter_mapbox(grid,
            lat     = 'lat',
            lon     = 'lon',
            opacity = .3,
            zoom    = 1,
            # color   = "predict",
            mapbox_style = 'open-street-map',
            color_continuous_scale = px.colors.cyclical.IceFire
        )
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        fig.layout.autosize = True

        plot = pn.pane.Plotly(fig)
        opts = pn.Row(plot, sizing_mode='stretch_width', height=300)

        # Clicking on the grid will call addPoint
        pn.bind(self.addPoint, plot.param.click_data, watch=True)
        pn.bind(self.aggregatePoints, plot.param.selected_data, watch=True)

        self.tabs = pn.Tabs(closable=True, tabs_location='left')

        png = pn.pane.PNG('howtoSHAPwaterfalls.png', height=500)
        self.tabs.append(
            ('How To', png)
        )

        self.widgets[0] = opts
        self.widgets[1] = self.tabs

        self.log.debug('[Waterfall] Done')

    def __call__(self, ex=None, ds=None, display=None):
        """
        """
        if isinstance(display, int):
            self.display = display

        self.initialize(ex, ds)

        if self.ex is None:
            self.log.debug('[Waterfall] No data available')
            self.widgets[0] = self.default
            self.widgets[1] = None

        return self.widgets

    def plot(self, data):
        return shap.plots.waterfall(data, max_display=self.display, show=False)

    def aggregatePoints(self, event):
        """
        """
        if not event:
            return

        # Remove the generated plot from mem
        plt.close('all')

        points = [point['pointIndex'] for point in event['points']]
        self.log.debug(f'[Waterfall] Averaging points {points}')

        data = self.ex[points].mean(axis=0)
        axes = self.plot(data)
        plot = pn.pane.Matplotlib(plt.gcf(),
            height = 2500,
            tight  = True,
            format = 'svg',
            sizing_mode = 'stretch_width'
        )

        sel = list(event)[-1]
        loc = event[sel]['mapbox']

        self.tabs.append(
            ('aggregate', plot)
        )
        tab = len(self.tabs) - 1
        self.tabs.active = tab

        self.log.debug(f'[Waterfall] Done, setting active tab: {tab}')

    def addPoint(self, event):
        """
        """
        if not event:
            return

        # Remove the generated plot from mem
        plt.close('all')

        data  = event['points'][0]
        point = data['pointIndex']
        self.log.debug(f'[Waterfall] Plotting point {point}')

        print(self.ex.shape)

        axes = self.plot(self.ex[point])
        plot = pn.pane.Matplotlib(plt.gcf(),
            height = 2500,
            tight  = True,
            format = 'svg',
            sizing_mode = 'stretch_width'
        )

        loc = str((data['lat'], data['lon']))
        # tag = pn.widgets.TextInput(name='Tag')
        # tab = pn.Row(loc, tag)

        self.tabs.append(
            (loc, plot)
        )
        tab = len(self.tabs) - 1
        self.tabs.active = tab

        self.log.debug(f'[Waterfall] Done, setting active tab: {tab}')


MonthOrder = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

def _retrieve_models(path):
    """
    """
    valid = []
    for models in path.glob('*/'):
        validated = False
        for fold in models.glob('*/'):
            if any([
                (fold/'test.contributions.nc').exists(),
                (fold/'test.explanation.nc').exists()
            ]):
                valid.append(models)
                validated = True
                break

            if validated:
                break

    ordered = sorted(valid, key=lambda month: MonthOrder.index(month.name)) # SUDSAQ specific

    return ordered


class GUI:
    def __init__(self):
        """
        """
        self.data = None

        self.paths   = Sect()
        self.widgets = Sect()

        self.app = pn.template.MaterialTemplate(
            title   = 'SUDS AQ GUI',
            sidebar = self.sidebar(),
            main    = self.main(),
            sidebar_width = 1024
        )

        self.ds = None
        self.ex = None

    # SIDEBAR

    def _select_model(self, value):
        """
        """
        # No operation on reset
        if not value:
            return

        self.log.debug(f'Selected model: {value}')
        self.paths.model = self.paths.base / value

        self.log.debug('Verifying folds')

        folds = []
        for fold in self.paths.model.glob('*/'):
            if any([
                (fold/'test.contributions.nc').exists(),
                (fold/'test.explanation.nc').exists()
            ]):
                folds.append(fold)

        strings = [str(f) for f in folds]
        names   = [f.name for f in folds]
        names   = sorted(names, key=lambda name: int(name)) # SUDSAQ specific
        self.log.debug('Discovered valid paths:\n'+'\n'.join(strings))
        self._toggle_select_widget('fold', names)

    def _select_fold(self, value):
        """
        """
        # No operation on reset
        if not value:
            return

        self.log.debug(f'Selected fold: {value}')
        self.paths.fold = self.paths.model / value

        self.log.debug('Verifying explanation files')
        files = []
        for kind in ('test.contributions.nc', 'test.explanation.nc'):
            if (expl := self.paths.fold/kind).exists():
                files.append(expl)

        strings = [str(f) for f in files]
        names   = [f.name for f in files]
        self.log.debug('Discovered valid paths:\n'+'\n'.join(strings))
        self._toggle_select_widget('expl', names)

    def _select_ex_file(self, value):
        """
        """
        # No operation on reset
        if not value:
            return

        self.log.debug(f'Selected explanation type: {value}')
        self.paths.expl = self.paths.fold / value

        self.log.debug(f'Working on file: {self.paths.expl}')
        self.widgets.working.value = str(self.paths.expl)

    def _verify_base_directory(self, value):
        """
        """
        # No operation on reset
        if not value:
            return

        self.log.debug(f'Verifying input path: {value}')

        self.paths.base = path = Path(value)
        if not path.exists():
            self.log.error(f'Path does not exist: {path}')

            # Reset the follow-up widgets
            for kind in ('model', 'fold', 'expl'):
                self._toggle_select_widget(kind, [])
            return

        paths   = _retrieve_models(path)
        strings = [str(p) for p in paths]
        names   = [p.name for p in paths]
        self.log.debug('Discovered valid paths:\n'+'\n'.join(strings))
        self._toggle_select_widget('model', names)

        return paths

    def _toggle_select_widget(self, widget, options):
        """
        """
        if isinstance(widget, str):
            widget = self.widgets[widget]

        if not options:
            widget.disabled = True
            widget.value    = ''
            widget.options  = []
        else:
            # Resets the value if it was already initialized to trigger downstream bind functions
            widget.value    = ''
            widget.value    = options[0]
            widget.options  = options
            widget.disabled = False

    def _load_working_file(self, value):
        """
        """
        # No operation on reset
        if value == "":
            return

        if self.ds:
            self.log.debug('Closing previous dataset')
            self.ds.close()
            del self.ex

        self.paths.expl = Path(value)
        if self.paths.expl.exists():
            self.log.debug(f'Loading working file: {self.paths.expl}')

            if self.paths.expl.name.endswith('.contributions.nc'):
                self.ds, self.ex = cont_to_ex(self.paths.expl.parent)
            else:
                self.ds = xr.open_mfdataset([self.paths.expl], mode='r', lock=False)
                self.ds = Dataset(self.ds)
                self.ex = self.ds.to_explanation(auto=True)

            self.log.debug(self.ds)

            # Entire dataset is the data initially, may subset later
            self.data = self.ds

            # Update plotting tabs
            self.wf(ex=self.ex, ds=self.ds, display=self.ds['variable'].size)
        else:
            self.log.error(f'Working file does not exist: {self.paths.expl}')

    def sidebar(self):
        """
        """
        terminal = pn.widgets.Terminal(
            "SUDSAQ Logs\n",
            options     = {"cursorBlink": False},
            height      = 650,
            sizing_mode = 'stretch_width'
        )
        self.log = init_logger(terminal)

        # Run directory widget
        self.widgets.base = pn.widgets.TextInput(
            name        = 'Run Directory',
            placeholder = '/path/to/a/run/',
            sizing_mode = 'stretch_width'
        )
        pn.bind(self._verify_base_directory, self.widgets.base, watch=True)

        # Model from run directory widget
        self.widgets.model = pn.widgets.Select(
            name     = 'Select Model',
            options  = [],
            disabled = True,
            sizing_mode = 'stretch_width'
        )
        pn.bind(self._select_model, self.widgets.model, watch=True)

        # Model fold
        self.widgets.fold = pn.widgets.Select(
            name     = 'Select Fold',
            options  = [],
            disabled = True,
            sizing_mode = 'stretch_width'
        )
        pn.bind(self._select_fold, self.widgets.fold, watch=True)

        # Data file to work with
        self.widgets.expl = pn.widgets.Select(
            name        = 'Select Explanation File',
            options     = [],
            disabled    = True,
            sizing_mode = 'stretch_width'
        )
        pn.bind(self._select_ex_file, self.widgets.expl, watch=True)

        # Working file name
        self.widgets.working = pn.widgets.TextInput(
            name        = 'Working File',
            value       = '',
            sizing_mode = 'stretch_width'
        )
        pn.bind(self._load_working_file, self.widgets.working, watch=True)

        return [
            self.widgets.base,
            pn.Row(
                self.widgets.model,
                self.widgets.fold,
                self.widgets.expl,
            ),
            pn.Spacer(height=5),
            self.widgets.working,
            pn.Spacer(height=5),
            terminal
        ]

    # MAIN

    def main(self):
        """
        """
        self.wf = Waterfalls(self.log)

        tabs = pn.Tabs(
            # ('Instructions', pn.widgets.StaticText(name='TODO', value='Instructions')),
            # ('Contributions', pn.widgets.StaticText(name='Contributions', value='Plot')),
            ('Waterfall', self.wf()),
            # ('WF', self._waterfall()),
            sizing_mode='stretch_width'
        )
        return tabs

    # PLOTS

    def _nearest(self, value, key, widget):
        """
        """
        nearest = float(self.data[key].sel({key: value}, method='nearest'))

        self.widgets[key] = nearest
        print(widget)
        widget.value = nearest

        self.log.debug(f'Nearest {value} = {nearest}')

    def _waterfall(self):
        """
        """
        if not (c := self.widgets.waterfall):
            c = [None] * 10
            c = self.widgets.waterfall = pn.Column(*c, sizing_mode='stretch_width')

        if self.data is not None:
            self.log.debug('[Waterfall] Loading options')

            point = pn.widgets.TextInput(
                name     = 'Selected point (lat, lon)',
                disabled = True,
            )

            btn = pn.widgets.Button(name='Generate', button_type='primary')

            grid = self.data[['lat', 'lon']]
            grid = grid.to_dataframe().reset_index()
            fig = px.scatter_mapbox(grid,
                lat='lat',
                lon='lon',
                opacity=.2,
                # color_discrete_sequence=[''],
                zoom=1,
                mapbox_style='open-street-map',
                # height=200,
                # width=500
            )
            fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
            fig.layout.autosize = True
            # c[3] = pn.pane.Plotly(fig)
            plot = pn.pane.Plotly(fig)

            opts = pn.Column(point, btn)
            c[0] = pn.Row(opts, plot, sizing_mode='stretch_width', height=200)


            def plotLine(event):
                print(event)
                if not event:
                    return

                data = event['points'][0]
                lat, lon = data['lat'], data['lon']

                data = self.data['momo.t'].sel(lat=lat, lon=lon)

                # fig = px.line(x=data.time, y=data, markers=True)
                # fig.layout.autosize = True

                # line.data[0].x = data.time
                line.data[0].y = data

            # line = px.line(y=[None]*self.data.time.size, x=self.data.time, markers=True)
            # line.layout.autosize = True
            #
            # pn.bind(plotLine, plot.param.click_data, watch=True)
            #
            # c[1] = pn.pane.Plotly(line)

            def waterfall(event):
                if not event:
                    return

                plt.close('all')

                point = event['points'][0]['pointIndex']
                fig = shap.plots.waterfall(self.ex[point], max_display=self.ds['variable'].size, show=False)

                self.log.debug(f'[Waterfall] Plotting point {point}')

                return pn.pane.Matplotlib(fig, height=2500, tight=True, format='svg', sizing_mode='stretch_width')

            c[1] = pn.bind(waterfall, plot.param.click_data)

        else:
            self.log.debug('[Waterfall] No data available')
            c[0] = pn.widgets.StaticText(name='Watefall', value='Load data to generate plot')

        return c


if __name__ == '__main__':
    gui = GUI()

    gui.app.servable()

# /Volumes/MLIA_active_data/data_SUDSAQ/models/bias/gattaca.v4.bias-median.Europe-v1
# /Volumes/MLIA_active_data/data_SUDSAQ/models/bias/gattaca.v4.bias-median
# /Users/jamesmo/projects/sudsaq/dev/.local/data/bias/v4.1
# /Users/jamesmo/projects/sudsaq/dev/bias-v6
