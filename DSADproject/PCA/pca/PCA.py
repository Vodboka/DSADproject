'''
A class for implementing PCA (Principal Components Analysis)
'''
import numpy as np


class PCA:
    def __init__(self, X):  # assume X is a standardized numpy.array
        self.X = X
        # compute variance-covariance matrix of X
        self.Cov = np.cov(m=X, rowvar=False)  # we have the variables on the columns
        print(self.Cov.shape)
        # extract the eigenvalues and the eigenvectors for variance-covariance matrix of X
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(a=self.Cov)
        print(self.eigenvalues, self.eigenvalues.shape)
        print(self.eigenvectors.shape)
        # we need to sort the eigenvalues in descending order, along with the eigenvectors
        k_desc = [k for k in reversed(np.argsort(self.eigenvalues))]
        print(k_desc, type(k_desc))
        self.alpha = self.eigenvalues[k_desc]
        self.A = self.eigenvectors[:, k_desc]

        # regularisation of eigenvectors
        for j in range(self.A.shape[1]):
            minCol = np.min(a=self.A[:, j], axis=0)  # we have the variables on the columns
            maxCol = np.max(a=self.A[:, j], axis=0)
            if np.abs(minCol) > np.abs(maxCol):
                # multiplying an eigenvector with a scalar does not change the nature of the eigenvector
                self.A[:, j] = (-1) * self.A[:, j]
                
        # compute the principal components
        # self.C = np.matmul(self.X, self.A)
        self.C = self.X @ self.A

        # compute the correlation between the observed variables and principal components
        # aka factor loadings
        self.Rxc = self.A * np.sqrt(self.alpha)

        self.C2 = self.C * self.C


    def getEigenValues(self):
        # return self.eigenvalues
        return self.alpha

    def getEigenVectors(self):
        # return self.eigenvectors
        return self.A

    def getPrinComp(self):
        return self.C

    def getFactorLoadings(self):
        return self.Rxc

    def getScores(self):
        return self.C / np.sqrt(self.alpha)

    def getQualObs(self):
        SL = np.sum(a=self.C2, axis=1)  # sums on the lines
        return np.transpose(self.C2.T / SL)

    def getContribObs(self):
        return self.C2 / (self.X.shape[0] * self.alpha)

    def getCommun(self):
        Rxc2 = np.square(self.Rxc)
        return np.cumsum(a=Rxc2, axis=1)  # the cumulative sums on the lines
