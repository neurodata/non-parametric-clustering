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
from distance import procrustes


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

def kmedoids_(K, D, maxiter=100):
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

def kmedoids(K, D, maxiter=50, numtimes=10):
    """Wrapper on kmedoids. We run several times an pick the best answer."""
    J = []; M = []; f = np.inf;
    for i in range(numtimes):
        curr_J, curr_M = kmedoids_(K, D, maxiter)
        curr_f = 0
        for k in range(K):
            ix = np.where(curr_J==k)[0]
            curr_f += D[ix, curr_M[k]].sum()
        if curr_f < f:
            f = curr_f
            J = curr_J
            M = curr_M
    return J, M

if __name__ == '__main__':
#    mean = np.array([0, 0])
#    cov = np.array([[4, 0], [0, 1]])
#    data1 = np.random.multivariate_normal(mean, cov, 200)
#
#    mean = np.array([3, 5])
#    cov = np.array([[1, 0.8], [0.8, 2]])
#    data2 = np.random.multivariate_normal(mean, cov, 200)
#
#    mean = np.array([-2, 3])
#    cov = np.array([[0.5, 0], [0, 0.5]])
#    data3 = np.random.multivariate_normal(mean, cov, 200)
#
#    # data has contains data generated from 3 clusters
#    data = np.concatenate((data1, data2, data3))
#
#    K = 3
#    D = euclidean(data)
#    print kmedoids(K, D)
    
    digits = datasets.load_digits()
    images = digits.images
    D = procrustes_distance(images[:100])
    print D
