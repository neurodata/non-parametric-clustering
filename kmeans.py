"""
Kmeans++ clustering algorithm

Using Euclidean Distance. Later we can replace by other distance.

"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt


def shortest_distance(x, C):
    """Shortest distance between vector x and all the vectors in C."""
    return ((x - np.array(C))**2).sum(axis=1).min()

def pdf(D):
    """Make a probability vector based on D. Just normalize it."""
    return D/D.sum()

def discrete_rv(p):
    """Return an integer according to probability function p."""
    u = np.random.uniform()
    cdf = np.cumsum(p)
    j = np.searchsorted(cdf, u)
    return j

def kplus(k, X):
    """This is the k-means++ initialization"""
    C = [X[np.random.randint(0, len(X))]] # centers
    D = np.zeros(len(X))                  # distances squared
    for n in range(1, k):
        for i in range(len(X)):
            D[i] = shortest_distance(X[i], C)
        p = pdf(D)
        j = discrete_rv(p)
        C.append(X[j])
    return np.array(C)
    
def kmeans(k, X, max_iter=50):
    """K-means algorithm."""
    centroids = kplus(k, X)
    labels = np.empty(len(X), dtype=np.int8)
    converged = False
    count = 0
    while not converged and count <= max_iter:
        converged = True
        
        # for each vector label it according to the closest centroid
        for i, x in enumerate(X):
            distances = ((centroids - x)**2).sum(axis=1)
            labels[i] = np.argmin(distances)
        
        # update the centroids based on the vectors in the cluster
        for j in range(k):
            old_centroid = centroids[j]
            new_centroid = X[np.where(labels==j)].mean(axis=0)
            if not np.array_equal(new_centroid, old_centroid):
                centroids[j] = new_centroid
                converged = False
        
        count += 1
    
    return labels, centroids, count


