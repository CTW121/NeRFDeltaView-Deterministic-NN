import sys

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.widgets import RectangleSelector
from matplotlib.path import Path
from matplotlib.figure import Figure

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

# from scipy.stats import kde
from scipy import stats

import seaborn as sns

class ScatterPlot(QVBoxLayout):
    """
    Scatter plot widget.

    Parameters:
    -----------
    widget : QWidget
        The parent widget to contain the scatter plot.
    selectInd : function
        Callback function to handle selected indices.
    data : numpy.ndarray
        Data array for the scatter plot.
    """
    def __init__(self, widget, selectInd, data):
        super().__init__()

        self.data = data
        # self.data = np.random.rand(10, 2)
        self._main = QVBoxLayout(widget)

        # layout = QVBoxLayout(self._main)
        self.canvas = FigureCanvas(Figure(figsize=(10,10)))
        self._main.addWidget(self.canvas)

        subplot_kw = dict(xlim=(0, 1), ylim=(-1.0, 15.0), autoscale_on=False)
        self.fig = self.canvas.figure
        self.ax = self.fig.subplots(subplot_kw=subplot_kw)
        # self.heatmap_ax = self.fig.subplots()
        self.ax.set_title("Uncertainty distribution")
        self.ax.set_xlabel('Uncertainty value')

        self.selectInd = selectInd

        self.create_density_plot()
        self.create_scatterplot()

    def create_scatterplot(self):
        """
        Create scatter plot.
        """
        self.selected_ind = []
        self.scatter_pts = self.ax.scatter(self.data[:, 0], self.data[:, 1], color='magenta', s=20, edgecolor='none', alpha=0.3)
        
    def create_density_plot(self):
        """
        Create density plot.
        """
        sns_kde = sns.kdeplot(self.data, ax=self.ax, color='blue', fill=True, legend=False)

    def onselect(self, eclick, erelease):
        """
        Handle selection event on the scatter plot.

        Parameters:
        -----------
        eclick : matplotlib event
            Event object for the initial click.
        erelease : matplotlib event
            Event object for the release after selection.
        """
        # self.ax.patches[0].set_facecolor('g')
        # self.ax.patches[0].linestyle('-')

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        selected_indices = np.where((self.data[:, 0] >= min(x1, x2)) &
                                    (self.data[:, 0] <= max(x1, x2)) &
                                    (self.data[:, 1] >= min(y1, y2)) &
                                    (self.data[:, 1] <= max(y1, y2)))[0]
        self.selectInd(selected_indices)
        
        if len(selected_indices) > 0:
            self.alphas = np.where(np.isin(np.arange(len(self.data)), selected_indices), 1.0, 0.001)
        else:
            self.alphas = np.full(len(self.data), 0.3)

        self.scatter_pts.set_alpha(self.alphas)

        self.canvas.draw_idle()
        self.canvas.flush_events()
        self.canvas.draw()
        self.canvas.flush_events()