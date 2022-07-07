import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

register_matplotlib_converters()


def plot_series_abs(series):
    fig = plt.figure(figsize=[10, 5])
    ax = fig.add_subplot(111)
    ax.set_title("Price history of " + str(series.name))
    ax.plot(series)
    ax.set_ylabel("USD")
    plt.grid()
    plt.show()


def plot_series_rel(series):
    fig = plt.figure(figsize=[10, 5])
    ax = fig.add_subplot(111)
    ax.set_title("Daily returns of " + str(series.name))
    ax.plot(series.pct_change())
    ax.set_ylabel("daily percentage change")
    plt.grid()
    plt.show()
