#!/usr/bin/env python

"""This module contains different distance functions to be used on
shapes.

"""

from __future__ import division

import numpy as np
import scipy.spatial.distance


def procrustes_matrix(X):
    """Given data points X, computes the distance matrix using procrustes
    distance.
    
    """
    n = X.shape[0]
    D = np.empty(shape=(n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = procrustes(X[i], X[j])
            D[i, j] = D[j, i] = dist
    return D

def euclidean(X, Y):
    """Computes squared euclidean distance between X and Y."""
    return ((X-Y)**2).sum()

def euclidean_matrix(X):
    """Compute squared Euclidean distance matrix of data points."""
    V = scipy.spatial.distance.pdist(X, 'sqeuclidean')
    return scipy.spatial.distance.squareform(V)

def norm(X, Y, o='fro'):
    return np.linalg.norm(X-Y, ord=o)

def procrustes(X, Y):
    """Procrustes distance between X and Y."""
    Xbar = X.mean(axis=0)
    Ybar = Y.mean(axis=0)

    A = X - Xbar
    B = Y - Ybar
    
    sA = np.sqrt((A**2).sum())
    sB = np.sqrt((B**2).sum())

    A = A/sA
    B = B/sB

    C = (A.T).dot(B)
    U, S, Vt = np.linalg.svd(C)

    v = np.ones(C.shape[0])
    v[-1] = np.linalg.det(U.dot(Vt))

    R = U.dot(np.diag(v)).dot(Vt)

    return np.linalg.norm(A.dot(R) - B, ord='fro')

def procrustes2(X, Y):
    """Procrustes distance between X and Y."""
    muX = X.mean(axis=0)
    muY = X.mean(axis=0)

    A = X - muX
    B = Y - muY
    
    sA = np.sqrt((A**2).sum())
    sB = np.sqrt((B**2).sum())

    A = A/sA
    B = B/sB

    Z = (A.T).dot(B)
    U, S, Vt = np.linalg.svd(Z)
    V = Vt.T

    #v = np.ones(Z.shape[0])
    #v[-1] = np.linalg.det(U.dot(Vt))

    #R = V.dot(np.diag(v)).dot(U.T)
    R = V.dot(U.T)

    return np.linalg.norm(A.dot(R) - B, ord='fro')



if __name__ == '__main__':
    import parse_ellipses
    x1 = parse_ellipses.parse_ellipse(30, 20, 2, 'data/cluster1.dat')
    x2 = parse_ellipses.parse_ellipse(30, 20, 2, 'data/cluster2.dat')
    x3 = parse_ellipses.parse_ellipse(30, 20, 2, 'data/cluster3.dat')

    X = x1[0]
    Y = x1[10]
    
    print procrustes2(X, Y)
    
