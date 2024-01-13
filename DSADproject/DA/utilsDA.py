import numpy as np
import pandas as pd
import scipy.stats as sts
import pandas.api.types as pdt
import sklearn.preprocessing as pp
import collections as co
import scipy.linalg as lin
import sklearn.discriminant_analysis as disc
import graphicsDA as graphics
import scipy.stats as sstats


def dispersion(x, y):
    '''
    Calculation of centers, group frequencies,
    group labels and scatter matrix
    x - data tables
    y - grouping variable
    x, y - numpy.ndarray
    '''

    n, m = np.shape(x)
    means = np.mean(x, axis=0)
    counter = co.Counter(y)
    g = np.array([i for i in counter.keys()])  # retrieve labels
    ng = np.array([i for i in counter.values()])  # retrieve frequencies
    q = len(g)
    xg = np.ndarray(shape=(q, m))
    for k, i in zip(g, range(len(g))):
        xg[i, :] = np.mean(x[y == k, :], axis=0)
    xg_med = xg - means
    sst = n * np.cov(x, rowvar=False, bias=True)
    ssb = np.transpose(xg_med) @ np.diag(ng) @ xg_med
    ssw = sst - ssb
    return g, ng, xg, sst, ssb, ssw


def regularise(t, y=None):
    '''
    Eigenvector regularisation
    t - table of eigenvectors,
    expect either numpy.ndarray or pandas.DataFrame
    '''

    # if type(t) is pd.DataFrame:
    if isinstance(t, pd.DataFrame):
        for c in t.columns:
            minim = t[c].min()
            maxim = t[c].max()
            if abs(minim) > abs(maxim):
                t[c] = -t[c]
                if y is not None:
                    # determine column index
                    k = t.columns.get_loc(c)
                    y[:, k] = -y[:, k]
    if isinstance(t, np.ndarray):
        for i in range(np.shape(t)[1]):
            minim = np.min(t[:, i])
            maxim = np.max(t[:, i])
            if np.abs(minim) > np.abs(maxim):
                t[:, i] = -t[:, i]
    return None


def replace_na_df(t):
    '''
    replace missing values by
    mean/mode
    t - pandas.DataFrame
    '''

    for c in t.columns:
        if pdt.is_numeric_dtype(t[c]):
            if t[c].isna().any():
                avg = t[c].mean()
                t[c] = t[c].fillna(avg)
        else:
            if t[c].isna().any():
                mode = t[c].mode()
                t[c] = t[c].fillna(mode[0])
    return None


def replace_na(X):
    '''
     replace missing values by mean
     t - numpy.ndarray
     '''
    means = np.nanmean(X, axis=0)
    k_nan = np.where(np.isnan(X))
    X[k_nan] = means[k_nan[1]]
    return None


def tabulation(X, varLabel=None, obsLabel=None, fileName=None):
    '''
    X - numpy.ndarray
    '''

    X_tab = pd.DataFrame(X)
    if varLabel is not None:
        X_tab.columns = varLabel
    if obsLabel is not None:
        X_tab.index = obsLabel
    if fileName is None:
        X_tab.to_csv("Tabulation.csv")
    else:
        X_tab.to_csv(fileName)
    return X_tab


def codify(t, vars):
    '''
    codification of categorical variables
    '''

    for v in vars:
        t[v] = pd.Categorical(t[v]).codes
    return None


def lda(sst, ssb, n, q):
    '''
    Linear Discriminant Analysis (LDA)
    '''

    m = len(sst)
    cov_inv = np.linalg.inv(sst)
    h = cov_inv @ ssb
    if np.allclose(h, np.transpose(h)):
        eigenvalues, eigenvectors = np.linalg.eig(h)
    else:
        c = lin.sqrtm(ssb)
        h = np.transpose(c) @ cov_inv @ c
        eigenvalues, eigenvectors_ = np.linalg.eig(h)
        eigenvectors = cov_inv @ c @ eigenvectors_
    k_inv = np.flipud(np.argsort(eigenvalues))
    r = min(m, q - 1)
    alpha = np.real(eigenvalues[k_inv[:r]])
    u = np.real(eigenvectors[:, k_inv[:r]])
    regularise(u)
    l = alpha * (n - q) / ((1 - alpha) * (q - 1))
    return alpha, l, u


def classification_functions(x, xg, cov, ng):
    '''
    Calculation of classification functions
    for LDA and for Bayes
    '''

    n = np.shape(x)[0]
    q = np.shape(xg)[0]
    cov_inv = np.linalg.inv(cov / n)
    f = xg @ cov_inv
    f0 = np.empty(shape=(q,))
    for i in range(q):
        f0[i] = -0.5 * f[i, :] @ xg[i, :]
    f0_b = f0 + np.log(ng / n)  # free terms for Bayes
    return f, f0, f0_b


def classification_functions_z(z, zg, ng):
    '''
    Calculation of classification functions
    on discriminated variables
    '''

    n = np.shape(z)[0]
    q = np.shape(zg)[0]
    cov_inv = np.diag(1.0 / np.var(z, axis=0))
    f = zg @ cov_inv
    f0 = np.empty(shape=(q,))
    for i in range(q):
        f0[i] = -0.5 * f[i, :] @ zg[i, :]
    f0_b = f0 + np.log(ng / n)  # free terms for Bayes
    return f, f0, f0_b


def predict_bayes(x, xg, cov, ng, g):
    '''
    Prediction based on Bayesian classification scores
    '''

    n, m = np.shape(x)
    cov = cov / n
    q = len(g)
    dist = np.linalg.inv(cov)
    classif = np.empty(shape=(n,), dtype=np.int64)
    s = np.empty(shape=(n, q))
    log_p_apriori = 2 * np.log(ng / n)
    for i in range(n):
        for k in range(q):
            d = (x[i, :] - xg[k, :]) @ dist @ (x[i, :] - xg[k, :])
            s[i, k] = log_p_apriori[k] - d
        classif[i] = np.argmax(s[i, :])
    return g[classif]


def predict(x, f, f0, g):
    '''
    Prediction based on classification functions
    '''

    n, m = np.shape(x)
    classif = np.empty(shape=(n,), dtype=np.int64)
    for i in range(n):
        rez = f @ x[i, :] + f0
        classif[i] = np.argmax(rez)
    return g[classif]


def discrim_accuracy(y, classif, g):
    q = len(g)
    n = len(y)
    mat_c = pd.DataFrame(data=np.zeros((q, q)), index=g, columns=g)
    for i in range(n):
        mat_c.loc[y[i], classif[i]] += 1
    groups_accuracy = np.diag(mat_c) * 100 / np.sum(mat_c, axis=1)
    mat_c['accuracy'] = groups_accuracy
    return mat_c


def discrim_power(ssb, ssw, n, q):
    r = (n - q) / (q - 1)
    f = r * np.diag(ssb) / np.diag(ssw)
    p_value = 1 - sts.f.cdf(f, q - 1, n - q)
    return f, p_value

