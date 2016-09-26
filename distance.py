#!/usr/bin/env python

"""This module contains different distance functions to be used on
shapes.

"""

from __future__ import division

import numpy as np
import scipy.spatial.distance

import matplotlib.pyplot as plt

import ellipse


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

def sort_points2D(P):
    """Sort the points in matrix P according to the angle in x-y plane."""
    cp = np.array([np.complex(P[0,i], P[1,i]) for i in range(len(P[0]))])
    thetas = np.angle(cp)
    idx = np.argsort(thetas)
    return P[:,idx]

def align(P, Q):
    """Align P into Q by rotation. Use SVD."""
    Z = P.dot(Q.T)
    U, sigma, Vt = np.linalg.svd(Z)
    R = Vt.T.dot(U)
    det = np.linalg.det(R)
    if det < 0:
        d = np.ones(k)
        d[-1] = -1
        R = Vt.T.dot(np.diag(d)).dot(U)
    Qhat = R.dot(P)
    dist = np.linalg.norm(Qhat - Q)
    return Qhat, dist

def best_alignment(P, Q, tol=10**-15):
    """Cycle the points in P and align to Q. Pick the smallest distance."""
    k, n = P.shape
    finalQhat, finaldist = align(P, Q)
    if finaldist <= tol:
        return finalQhat, finaldist
    for i in range(n):
        js = range(-1, n-1)
        P = P[:,js]
        Qhat, dist = align(P, Q)
        if dist < finaldist:
            finaldist = dist
            finalQhat = Qhat
            if finaldist <= tol:
                break
    return finalQhat, finaldist

def procrustes(X, Y, transpose=True):
    """Procrustes distance between X and Y.
    We assume that X and Y have the same dimension and in the form

    [[x1, y1, z1, ...],
     [x2, y2, z2, ...]
        ...
     [xn, yn, zn, ...]]

    i.e. a (n,k) matrix where n is the number of points and k is the
    dimension of each point.

    """
    assert X.shape == Y.shape
    
    if transpose:
        P = X.T 
        Q = Y.T
    else:
        P = X
        Q = X
    k, n = P.shape
    
    # eliminate translation
    pbar = P.mean(axis=1)
    qbar = Q.mean(axis=1)
    Ptilde = P - pbar.reshape((k,1))
    Qtilde = Q - qbar.reshape((k,1))
    
    # rescale
    Ptilde = Ptilde/np.linalg.norm(Ptilde)
    Qtilde = Qtilde/np.linalg.norm(Qtilde)
    
    # find rotation or reflection
    Qtildehat, dist = best_alignment(Ptilde, Qtilde)
    
    #return Qtildehat, Qtilde, dist
    return dist


###############################################################################
if __name__ == '__main__':
    x1 = ellipse.parse_ellipse(30, 20, 2, 'data/cluster1.dat')
    x2 = ellipse.parse_ellipse(30, 20, 2, 'data/cluster2.dat')
    x3 = ellipse.parse_ellipse(30, 20, 2, 'data/cluster3.dat')
    X = x1[1]
    Y = x2[15]
    
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)
    ellipse.plot_data([X.T, Y.T], ax1)
    ax1.set_xlim([-20,20])
    ax1.set_ylim([-20,20])
    
    Qh, Q, d = procrustes(X, Y)

    ax2 = fig.add_subplot(122)
    ellipse.plot_data([Qh, Q], ax2)

    fig.savefig('ellipse.png')

