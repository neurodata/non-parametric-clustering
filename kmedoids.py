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

def missclassification_error(true_labels, pred_labels):
    """Clustering missclassification error. Gives the percentage
    of correct points clustered correctly.
    
    """
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(pred_labels)
    min_wrong = np.inf
    for permutation in itertools.permutations(unique_pred):
        f = {a:b for a, b in zip(unique_true, permutation)}
        wrong = 0
        for i in range(len(true_labels)):
            if f[true_labels[i]] != pred_labels[i]:
                wrong += 1
        if wrong < min_wrong:
            min_wrong = wrong
    return min_wrong/len(true_labels)

def mnist_eval(distance_func, metric_func, 
               numbers=[1,2,3], nrange=[10,100,10], num_avg=100):
    """Return metric evaluation on MNIST dataset."""
    
    digits = datasets.load_digits()
    images = digits.images
    kmedoids_metric = []
    kmeans_metric = []
    
    for n in nrange:
        # generate true labels
        labels = np.concatenate([[m]*n for m in numbers])

        data = np.concatenate([
                images[np.where(digits.target==i)][:n].reshape((n, 64)) 
                for i in numbers
        ])

        m1 = []
        m2 = []
        for i in range(num_avg):
            j1, _ = kmedoids(len(numbers), distance_func(data))
            j2, _ = kmeans(len(numbers), data)
            a = metric_func(labels, j1)
            b = metric_func(labels, j2)
            m1.append(a)
            m2.append(b)

        kmedoids_metric.append(np.array(m1).mean())
        kmeans_metric.append(np.array(m2).mean())
    
    return kmedoids_metric, kmeans_metric

def gauss_eval(distance_func, metric_func, nrange=[10,100,10], num_avg=100):
    """Return metric evaluation on MNIST dataset."""
    
    kmedoids_metric = []
    kmeans_metric = []
    
    for n in nrange:
        data = np.concatenate((
            np.random.multivariate_normal([0, 0], [[4,0], [0,1]], n),
            np.random.multivariate_normal([3, 5], [[1,0.8], [0.8,2]], n),
            np.random.multivariate_normal([-2, 3], [[0.5,0], [0,0.5]], n))
        )
        
        labels = np.concatenate([[m]*n for m in range(3)])

        m1 = []
        m2 = []
        m3 = []
        for i in range(num_avg):
            j1, _ = kmedoids(3, distance_func(data))
            j2, _ = kmeans(3, data)
            a = metric_func(labels, j1)
            b = metric_func(labels, j2)
            m1.append(a)
            m2.append(b)

        kmedoids_metric.append(np.array(m1).mean())
        kmeans_metric.append(np.array(m2).mean())
    
    return kmedoids_metric, kmeans_metric

def mnist_eval_sklearn(metric_func, numbers=[1,2,3], 
                       nrange=[10, 100, 10]):
    digits = datasets.load_digits()
    images = digits.images
    metric = []
    for n in nrange:
        labels = np.concatenate([[m]*n for m in numbers])
        data = np.concatenate([
                images[np.where(digits.target==i)][:n].reshape((n, 64)) 
                for i in numbers
        ])
        km = KMeans(len(numbers))
        r = km.fit(data)
        metric.append(metric_func(labels, r.labels_))

    return metric


###############################################################################
if __name__ == '__main__':
    pass
    #numbers = [1, 2, 3]
    #nrange = range(10,100,10)
    #dfunc = euclidean
    #mfunc = metrics.normalized_mutual_info_score
    #m1, m2 = mnist_eval(dfunc, mfunc, numbers, nrange)
    #m1, m2 = gauss_eval(dfunc, mfunc, nrange=[10, 30, 10], num_avg=20)
