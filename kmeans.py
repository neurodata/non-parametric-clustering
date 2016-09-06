#!/usr/bin/env python

"""
Clustering with K-Means and initialization with K-Means++.

Here we implement a different version of K-Means++ to work with arbitrary
distances and data sets that can be matrices intead of vectors.

"""


from __future__ import division

import numpy as np


def forgy(K, X):
    """Forgy's method, just pick k random elements of the data set."""
    N = X.shape[0]
    return X[np.random.choice(range(N), size=K)]

def discrete_rv(p):
    """Return an integer according to probability function p."""
    u = np.random.uniform()
    cdf = np.cumsum(p)
    j = np.searchsorted(cdf, u)
    return j

def kpp(K, X, distance):
    """This is the k-means++ initialization proposed by Arthur and
    Vassilvitskii (2007).
    
    Inputs
    ------
    K: number of clusters
    X: data set
    distance: function that accepts two arguments and return the distance
        between them; distance(A, B) -> real x

    Output
    ------
    array with each element being a centroid

    
    """
    N = X.shape[0]
    C = [X[np.random.randint(0, N)]] # centers
    D = np.zeros(N)                  # distances
    for k in range(1, K):
        for n in range(N):
            D[n] = np.array([distance(X[n], c) for c in C]).min()
        p = D/D.sum()
        j = discrete_rv(p)
        C.append(X[j])
    return np.array(C)
    
def kmeans_(K, X, distance, max_iter=50):
    """K-means, Loyd's algorithm. We use K-means++ for initialization.

    Input
    -----
    K: number of clusters
    X: data set
    distance: function which accepts two objects and returns the distance
        between them; distance(A, B) -> real x
    max_iter: optional parameters to control the number of iterations

    Output
    ------
    labels: 1D array with labels for points in X, in order 
    centroids: array containing the centers of the clusters
    
    """
    N = X.shape[0]
    mus = kpp(K, X, distance)
    labels = np.empty(N, dtype=np.int8)
    
    converged = False
    count = 0
    while not converged and count <= max_iter:
        
        converged = True
        
        # for each vector label it according to the closest centroid
        for n in range(N):
            for k in range(K):
                D = np.array([distance(X[n], mus[k]) for k in range(K)])
            labels[n] = np.argmin(D)
        
        # update the centroids based on the vectors in the cluster
        for k in range(K):
            old_mu = mus[k]
            new_mu = np.mean(X[np.where(labels==k)], axis=0)
            if not np.array_equal(new_mu, old_mu):
                mus[k] = new_mu
                converged = False

        # compute objective
        J = sum([0.5*distance(x, mus[k]) for k in range(K) 
                                         for x in X[np.where(labels==k)]])
        
        count += 1
    
    return labels, mus, J

def kmeans(K, X, distance, max_iter=50, numtimes=5):
    """Wrapper around kmeans_single. We run the algorithm few times
    and pick the best answer.
    
    """
    J = np.inf
    for i in range(numtimes):
        cZ, cM, cJ = kmeans_(K, X, distance, max_iter)
        if cJ < J:
            J = cJ
            Z = cZ
            M = cM
    return Z, M


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    mean = np.array([0, 0])
    cov = np.array([[4, 0], [0, 1]])
    data1 = np.random.multivariate_normal(mean, cov, 200)
                                                                                
    mean = np.array([3, 5])
    cov = np.array([[1, 0.8], [0.8, 2]])
    data2 = np.random.multivariate_normal(mean, cov, 200)

    mean = np.array([-2, 3])
    cov = np.array([[0.5, 0], [0, 0.5]])
    data3 = np.random.multivariate_normal(mean, cov, 200)
    
    data = np.concatenate((data1, data2, data3))
    
    K = 3
    J, M = kmeans(K, data, lambda a, b: ((a-b)**2).sum())
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = getattr(cm, 'spectral')(np.linspace(0, 1, K))
    for k in range(K):
        xs = data[:,0][np.where(J==k)]
        ys = data[:,1][np.where(J==k)]
        ax.scatter(xs, ys, color=colors[k], alpha=.6)
    plt.show()
