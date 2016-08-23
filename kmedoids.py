#!/usr/bin/env python

"""
Clustering with K-Medoids and an analogous of K-Means++ initialization.

Guilherme S. Franca <guifranca@gmail.com>
08/18/2016

"""


from __future__ import division
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import datasets
from sklearn import metrics
from sklearn.cluster import KMeans
import itertools

from kmeans import kmeans


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
    """Initialization based on k-means++."""
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

def kmedoids(K, D, maxiter=100):
    """K-medoids algorithm."""
    M = kplus(K, D)
    M.sort()
    
    converged = False
    count = 0
    while not converged and count < maxiter:
        converged = True
        
        # determine clusters
        # J has the same dimension as X and tells the cluster
        # each data point is assigned to
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

