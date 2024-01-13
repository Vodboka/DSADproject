import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd


# create a correlogram as a heatmap for a correlation matrix
def correlogram(matrix=None, dec=1, title='Correlogram', valmin=-1, valmax=1):
    plt.figure(title, figsize=(15, 11))
    plt.title(title, fontsize=16, color='k', verticalalignment='bottom')
    sb.heatmap(data=np.round(matrix, dec), vmin=valmin, vmax=valmax, cmap='bwr', annot=True)


# creates the graphical table based on the intensity of link between
# values on X and Y axes
def intensity_map(matrix=None, dec=1, title='Intensity Map',
                  valmin=None, valmax=None,):
    plt.figure(title, figsize=(15, 11))
    plt.title(title, fontsize=16, color='k', verticalalignment='bottom')
    sb.heatmap(data=np.round(matrix, dec), vmin=valmin, vmax=valmax,
                # cmap = 'Oranges', annot = True)
               # cmap='Purples', annot=True)
                # cmap = 'Blues', annot = True)
               cmap='Reds', annot=True)


def correlCircle(matrix=None, V1=0, V2=1, dec=1,
                 XLabel=None, YLabel=None, minVal=-1, maxVal=1, title='Correlation Circle'):
    plt.figure(title, figsize=(8, 8))
    plt.title(title, fontsize=14, color='k', verticalalignment='bottom')
    T = [t for t in np.arange(0, np.pi*2, 0.01)]
    X = [np.cos(t) for t in T]
    Y = [np.sin(t) for t in T]
    plt.plot(X, Y)
    plt.axhline(y=0, color='g')
    plt.axvline(x=0, color='g')
    if XLabel==None or YLabel==None:
        if isinstance(matrix, pd.DataFrame):
            plt.xlabel(matrix.columns[V1], fontsize=14, color='k', verticalalignment='top')
            plt.ylabel(matrix.columns[V2], fontsize=14, color='k', verticalalignment='bottom')
        else:
            plt.xlabel('Var '+str(V1+1), fontsize=14, color='k', verticalalignment='top')
            plt.ylabel('Var '+str(V2+1), fontsize=14, color='k', verticalalignment='bottom')
    else:
        plt.xlabel(XLabel, fontsize=14, color='k', verticalalignment='top')
        plt.ylabel(YLabel, fontsize=14, color='k', verticalalignment='bottom')

    if isinstance(matrix, np.ndarray):
        plt.scatter(x=matrix[:, V1], y=matrix[:, V2], c='r', vmin=minVal, vmax=maxVal)
        for i in range(matrix.shape[0]):
            plt.text(x=matrix[i, V1], y=matrix[i, V2], s='(' +
                    str(np.round(matrix[i, V1], dec))
                     + ', ' + str(np.round(matrix[i, V2], dec)) + ')')

    if isinstance(matrix, pd.DataFrame):
        # plt.text(x=0.5, y=0.5, s='we have a pandas.DatFrame')
        plt.scatter(x=matrix.iloc[:, V1], y=matrix.iloc[:, V2], c='b', vmin=minVal, vmax=maxVal)
        for i in range(matrix.values.shape[0]):
            plt.text(x=matrix.iloc[i, V1], y=matrix.iloc[i, V2], s='(' +
                    str(np.round(matrix.iloc[i, V1], dec))
                     + ', ' + str(np.round(matrix.iloc[i, V2], dec)) + ')')


def principalComponents(eigenvalues=None, XLabel='Principal components', YLabel='Eigenvalues (variance)',
                        title='Explained variance by the principal components'):
    plt.figure(title, figsize=(13, 8))
    plt.title(title, fontsize=14, color='k', verticalalignment='bottom')
    plt.xlabel(XLabel, fontsize=14, color='k', verticalalignment='top')
    plt.ylabel(YLabel, fontsize=14, color='k', verticalalignment='bottom')
    # f(x) = y
    # create labels for the X axis: C1, C2, C3, ...
    components = ['C'+str(j+1) for j in range(eigenvalues.shape[0])]
    plt.plot(components, eigenvalues, 'bo-')
    plt.axhline(y=1, color='r')


def show():
    plt.show()