#!/usr/bin/env python

"""
Clustering with K-Medoids and an analogous of K-Means++ initialization.

Guilherme S. Franca <guifranca@gmail.com>
08/12/2016

"""


from __future__ import division

import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import datasets

def pdf(D):
    """Make a probability vector based on D. Just normalize it."""
    return D/D.sum()

def discrete_rv(p):
    """Return an integer according to probability function p."""
    u = np.random.uniform()
    cdf = np.cumsum(p)
    j = np.searchsorted(cdf, u)
    return j

def kplus(K, D):
    n = D.shape[0]
    M = [np.random.randint(0, n)]
    d = np.zeros(n)
    for k in range(1, K):
        for i in range(n):
            d[i] = np.min(D[i,M])
        p = pdf(d)
        j = discrete_rv(p)
        M.append(j)
    return np.array(M)

def euclidean(X):
    """Compute squared Euclidean distance matrix of data points."""
    V = scipy.spatial.distance.pdist(X, 'sqeuclidean')
    return scipy.spatial.distance.squareform(V)

def kmedoids(K, X, maxiter=50):
    """K-medoids algorithm."""
    D = euclidean(X) # squared euclidean distance matrix
    # initialize centers in a k-means++ manner
    # M contain the indices of the data points (medoids)
    M = kplus(K, D)
    M.sort()
    
    converged = False
    count = 0
    while not converged and count < maxiter:
        converged = True
        
        # determine clusters
        # J has the same dimension as X and tells to each cluster
        # each data point is assigned to, in order
        J = np.argmin(D[:,M], axis=1)
        
        # update medoids
        for k in range(K):
            old_Mk = M[k]
            xs = np.where(J==k)[0]
            j = np.argmin(D[np.ix_(xs, xs)].mean(axis=1))
            M[k] = xs[j]
            if not np.array_equal(old_Mk, M[k]):
                converged = False
        M.sort()
        count += 1

    return J, M


##############################################################################
if __name__ == '__main__':
    import sys

    np.random.seed(18)
    
    mean = np.array([0, 0])
    cov = np.array([[4, 0], [0, 1]])
    data1 = np.random.multivariate_normal(mean, cov, 200)

    mean = np.array([3, 5])
    cov = np.array([[1, 0.8], [0.8, 2]])
    data2 = np.random.multivariate_normal(mean, cov, 200)

    mean = np.array([-2, 3])
    cov = np.array([[0.5, 0], [0, 0.5]])
    data3 = np.random.multivariate_normal(mean, cov, 200)

    # data has contains data generated from 3 clusters
    data = np.concatenate((data1, data2, data3))

    # applying kmedoids algorithm
    K = 3
    J, M = kmedoids(K, data)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = getattr(cm, 'spectral')(np.linspace(0, 1, K))
    for k in range(K):
        xs = data[:,0][np.where(J==k)]
        ys = data[:,1][np.where(J==k)]
        ax.scatter(xs, ys, color=colors[k], alpha=.6)
    plt.show()

    
