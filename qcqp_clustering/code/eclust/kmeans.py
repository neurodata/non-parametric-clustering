"""K-means++ clustering."""

from __future__ import division

import numpy as np
import itertools
import scipy.optimize

import kmeanspp


def euclidean(a, b):
    return np.linalg.norm(a-b)**2

def kmeans(K, X, distance=euclidean, max_iter=50):
    """K is the number of clusters. X is the dataset, which is a matrix
    with one data point per row. distance is a function
    which accepts two data points and returns a real number.
    
    """
    N = X.shape[0] # number of points
    mus = kmeanspp.kpp(K, X, ret='centers')
    labels = np.empty(N, dtype=np.int8)
    
    converged = False
    count = 0
    while not converged and count < max_iter:
        
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
        
        count += 1

    if count >= max_iter:
        print "Warning: K-means didn't converge after %i iterations." % count
    
    return labels
    #return labels, mus, J, count

