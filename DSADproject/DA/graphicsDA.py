import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.colors as color
import numpy as np
import scipy.cluster.hierarchy as hclust
import statsmodels.graphics.mosaicplot as smosaic
import pandas as pd


def plot_scatter_groups(x, y, g, labels, x1, y1, g1, labels1,
                        title='Plot observations in the discriminant axes',
                        lx='z1', ly='z2'):
    '''
    Plot discriminant scores and centers
    '''

    q = len(labels1)
    f = plt.figure(figsize=(10, 7))
    ax = f.add_subplot(1, 1, 1)
    ax.set_title(title, fontsize=14, color='k')
    ax.set_xlabel(lx, fontsize=12, color='k')
    ax.set_ylabel(ly, fontsize=12, color='k')
    sb.scatterplot(x=x, y=y, hue=g, ax=ax, hue_order=g1)
    sb.scatterplot(x=x1, y=y1, hue=g1, ax=ax, legend=False, marker='s',
                   s=200)
    for i in range(len(labels)):
        ax.text(x[i], y[i], labels[i])
    for i in range(len(labels1)):
        ax.text(x1[i], y1[i], labels1[i], fontsize=16)


def plot_distribution(z, y, g, title=""):
    '''
    Plots the probability distribution for groups
    z, y - expect numpy.ndarray
    g - number of groups
    '''

    f = plt.figure(figsize=(10, 7))
    assert isinstance(f, plt.Figure)
    ax = f.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title(title, fontsize=14, color='k')
    for v in g:
        sb.kdeplot(data=z[y == v], fill=True,
                   warn_singular=False, ax=ax, label=v)


def show():
    plt.show()
