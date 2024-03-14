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
    def __init__(self, widget, selectInd, data):
        super().__init__()

        self.data = data
        # self.data = np.random.rand(10, 2)
        self._main = QVBoxLayout(widget)    # _main ?

        # layout = QVBoxLayout(self._main)
        self.canvas = FigureCanvas(Figure(figsize=(10,10)))
        self._main.addWidget(self.canvas)

        subplot_kw = dict(xlim=(0, 1), ylim=(-1.0, 15.0), autoscale_on=False)
        # subplot_kw = dict(xlim=(-0.05, 1), ylim=(-1.0, 15.0), autoscale_on=False)
        # subplot_kw = dict(autoscale_on=True)
        self.fig = self.canvas.figure
        self.ax = self.fig.subplots(subplot_kw=subplot_kw)
        # self.ax = self.fig.subplots()
        # self.heatmap_ax = self.fig.subplots()
        self.ax.set_title("Uncertainty distribution")
        self.ax.set_xlabel('Uncertainty value')
        # self.ax.set_ylabel('Uncertainty')
        # self.ax.set_ylim(-0.25, 1.00)
        # self.ax.legend().set_visible(False)

        self.selectInd = selectInd

        self.create_density_plot()
        self.create_scatterplot()

    def create_scatterplot(self):
        self.selected_ind = []
        # self.scatter_pts = self.ax.scatter(self.data[:, 0], self.data[:, 1], color='red', s=20, edgecolor='none')
        self.scatter_pts = self.ax.scatter(self.data[:, 0], self.data[:, 1], color='magenta', s=20, edgecolor='none', alpha=0.3)
        
    def create_density_plot(self):
        # sns_kde = sns.kdeplot(self.data, ax=self.ax, color='green', fill=True, legend=False)
        sns_kde = sns.kdeplot(self.data, ax=self.ax, color='blue', fill=True, legend=False)
        # sns.histplot(self.data, ax=self.ax, color='green', kde=True, bins=100)
        # nbins = 300
        # # data = self.data
        # # k = stats.gaussian_kde([data[:, 0], data[:, 1]])
        # k = stats.gaussian_kde([self.data[:, 0], self.data[:, 1]])
        # xi, yi = np.mgrid[0.000:1.000:nbins*1j, 0.000:1.000:nbins*1j]
        # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        
        # # ax = self.heatmap_ax
        # density_plot = self.ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.Greens)

        # https://stackoverflow.com/questions/32462881/add-colorbar-to-existing-axis

    def onselect(self, eclick, erelease):
        # self.ax.patches[0].set_facecolor('g')
        # self.ax.patches[0].linestyle('-')

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        # print("x1: {}   | y1: {}".format(x1, y1))
        # print("x2: {}   | y2: {}".format(x2, y2))

        selected_indices = np.where((self.data[:, 0] >= min(x1, x2)) &
                                    (self.data[:, 0] <= max(x1, x2)) &
                                    (self.data[:, 1] >= min(y1, y2)) &
                                    (self.data[:, 1] <= max(y1, y2)))[0]
        # print("selected_indices: \n", selected_indices)
        self.selectInd(selected_indices)
        
        if len(selected_indices) > 0:
            # self.alphas = [1.0 if i in selected_indices else 0.001 for i in range(len(self.data))]
            self.alphas = np.where(np.isin(np.arange(len(self.data)), selected_indices), 1.0, 0.001)
        else:
        #     alphas = [0.3 for i in range(len(self.data))]
            self.alphas = np.full(len(self.data), 0.3)
        # print("unique :", np.unique(np.array(alphas)))

        self.scatter_pts.set_alpha(self.alphas)

        self.canvas.draw_idle()
        self.canvas.flush_events()
        self.canvas.draw()
        self.canvas.flush_events()