"""Kmeans++ initialization"""

# Guilherme Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata

import numpy as np


def energy_kernel(x, y):
    return np.linalg.norm(x) + np.linalg.norm(y) - np.linalg.norm(x-y)

def euclidean_distance(x, y):
    return np.linalg.norm(x-y)**2

def kpp(k, X):
    """This is the k-means++ initialization proposed by Arthur and
    Vassilvitskii (2007). We label the points according to closest
    distance to the centers.
    
    """
    K = k
    N = X.shape[0]
    distance = euclidean_distance
    C = [X[np.random.randint(0, N)]] # centers
    D = np.zeros(N)                  # distances
    for k in range(1, K):
        for n in range(N):
            D[n] = np.array([distance(X[n], c) for c in C]).min()
        p = D/D.sum()
        j = discrete_rv(p)
        C.append(X[j])
    mus = np.array(C)
    
    labels = np.empty(N, dtype=np.int8)
    for n in range(N):
        for k in range(K):
            D = np.array([distance(X[n], mus[k]) for k in range(K)])
        labels[n] = np.argmin(D)
    return labels

def discrete_rv(p):
    """Return an integer according to probability function p."""
    u = np.random.uniform()
    cdf = np.cumsum(p)
    j = np.searchsorted(cdf, u)
    return j

