#!/usr/bin/env python

"""
Clustering with K-Medoids and an analogous of K-Means++ initialization.

"""


from __future__ import division

import numpy as np

import distance


def pdf(D):
    """Make a probability vector based on D. Just normalize it."""
    return D/D.sum()

def discrete_rv(p):
    """Return an integer according to probability function p."""
    u = np.random.uniform()
    cdf = np.cumsum(p)
    j = np.searchsorted(cdf, u)
    return j

def kpp(K, D):
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

def kmedoids_single(K, D, maxiter=50):
    """K-medoids algorithm.
    
    Input
    -----
    K: number of clusters
    D: distance matrix
    maxiter: optional. Number of iterations

    Output
    ------
    J: labels
    C: centroids
    F: sum of intra-cluster distances
    
    """
    ok = False
    while not ok:
        M = kpp(K, D)
        if len(np.unique(M)) == K:
            ok = True
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
            try:
                j = np.argmin(D[np.ix_(xs, xs)].mean(axis=1))
            except:
                print "Wrong Clustering: No elements in cluster"
                raise
            M[k] = xs[j]
            if not np.array_equal(old_Mk, M[k]):
                converged = False
        M.sort()

        # compute objective function
        F = np.array([D[np.where(J==k)[0], M[k]].sum() for k in range(K)]).sum()
        
        count += 1

    return J, M, F

def kmedoids(K, D, maxiter=50, numtimes=5):
    """Wrapper on kmedoids. We run the algorithm several times an pick 
    the best answer.
    
    """
    F = np.inf
    J = np.empty(D.shape[0])
    M = np.empty(K)
    for i in range(numtimes):
        cJ, cM, cF = kmedoids_single(K, D, maxiter)
        if cF < F:
            F = cF
            J = cJ
            M = cM
    return J, M


if __name__ == '__main__':
    #import matplotlib.pyplot as plt
    #import matplotlib.cm as cm

    #mean = np.array([0, 0])
    #cov = np.array([[4, 0], [0, 1]])
    #data1 = np.random.multivariate_normal(mean, cov, 300)

    #mean = np.array([3, 5])
    #cov = np.array([[1, 0.8], [0.8, 2]])
    #data2 = np.random.multivariate_normal(mean, cov, 300)

    #mean = np.array([-2, 3])
    #cov = np.array([[0.5, 0], [0, 0.5]])
    #data3 = np.random.multivariate_normal(mean, cov, 300)

    #data = np.concatenate((data1, data2, data3))

    #K = 3
    #D = distance.euclidean_matrix(data)
    #J, M = kmedoids(K, D)

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #colors = getattr(cm, 'spectral')(np.linspace(0, 1, K))
    #for k in range(K):
    #    xs = data[:,0][np.where(J==k)]
    #    ys = data[:,1][np.where(J==k)]
    #    ax.scatter(xs, ys, color=colors[k], alpha=.6)
    #plt.show()
 
    from sklearn import datasets
    import sys

    digits = datasets.load_digits()
    images = digits.images

    a = images[np.where(digits.target == 1)][np.random.choice(range(173), 100)]
    b = images[np.where(digits.target == 2)][np.random.choice(range(173), 100)]
    c = images[np.where(digits.target == 6)][np.random.choice(range(173), 100)]
    data = np.concatenate([a, b, c])

    #D = distance.euclidean_matrix(data.reshape(len(data), 64))
    D = distance.procrustes_matrix(data)
    J, M = kmedoids(3, D, numtimes=10)
    print J

