import numpy as np


def standardize(X):  # we assume to receive a numpy.ndarray
    means = np.mean(a=X, axis=0)  # we compute the means on the columns
    print(means.shape)
    stds = np.std(a=X , axis=0)  # we have the variables on columns
    return (X - means) / stds


    return None
