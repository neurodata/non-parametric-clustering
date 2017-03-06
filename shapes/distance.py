#!/usr/bin/env python

"""Different distance functions to be used on shapes."""

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
        d = np.ones(U.shape[0])
        d[-1] = -1
        R = Vt.T.dot(np.diag(d)).dot(U)
    Qhat = R.dot(P)
    dist = np.linalg.norm(Qhat - Q)
    return Qhat, dist

def best_alignment(P, Q, cycle=False, tol=1e-3):
    """Cycle the points in P and align to Q. 
    Pick the smallest distance if cycle is True.
    
    """
    k, n = P.shape
    finalQhat, finaldist = align(P, Q)
    if finaldist <= tol or not cycle:
        return finalQhat, finaldist

    # cycle the points and compute alignment each time
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

def procrustes(X, Y, transpose=True, fullout=False, cycle=False):
    """Procrustes distance between X and Y.
    We assume that X and Y have the same dimension and in the form

    [[x1, y1, z1, ...],
     [x2, y2, z2, ...]
        ...
     [xn, yn, zn, ...]]

    i.e. a (n,k) matrix where n is the number of points and k is the
    dimension of each point.

    If fullout=True it will print the final \hat{Y} which is the transformed
    X. If cycle=True it will 

    """
    assert X.shape == Y.shape
    
    n, k = X.shape

    if transpose:
        P = X.T 
        Q = Y.T
    else:
        P = X
        Q = Y
    
    # eliminate translation
    pbar = P.mean(axis=1)
    qbar = Q.mean(axis=1)
    Ptilde = P - pbar.reshape((k,1))
    Qtilde = Q - qbar.reshape((k,1))
    
    # rescale
    Ptilde = Ptilde/np.linalg.norm(Ptilde)
    Qtilde = Qtilde/np.linalg.norm(Qtilde)
    
    # find rotation or reflection
    Qtildehat, dist = best_alignment(Ptilde, Qtilde, cycle=cycle)
    
    if not fullout:
        return dist
    else:
        return Qtildehat.T, Qtilde.T, dist

def raw_dist(P, Q):
    return np.linalg.norm(P - Q)


###############################################################################
if __name__ == '__main__':

    import shapes
    import matplotlib.pyplot as plt
    import mnistshape as mshape

    import cPickle, gzip, sys
    import matplotlib.pyplot as plt

    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    images, labels = train_set
    
    """
    # ploting shapes and alignment for digits
    a, b, c = 3, 5, 7
    pairs = [[a,a], [b,b], [c,c], [a,b], [a,c], [b,c]]
    fig, axes = plt.subplots(nrows=len(pairs), ncols=4,
                                figsize=(15, 23))
    for (a, b), row in zip(pairs, axes):
        
        ax1, ax2, ax3, ax4 = row

        i1 = np.where(labels==a)
        i2 = np.where(labels==b)
        im1 = images[i1][np.random.randint(0, len(i1[0]))].reshape((28,28))
        im2 = images[i2][np.random.randint(0, len(i2[0]))].reshape((28,28))
        X = mshape.get_shape(im1, n=30, s=5.0)
        Y = mshape.get_shape(im2, n=30, s=5.0)

        ax1.plot(X[:,0], X[:,1], 'o-k')
        ax1.set_xlim([0, 28])
        ax1.set_ylim([0, 28])
        ax2.plot(Y[:,0], Y[:,1], 'o-k')
        ax2.set_xlim([0, 28])
        ax2.set_ylim([0, 28])
    
        Qh, Q, d = procrustes(X, Y, cycle=False, fullout=True)
        ax3.plot(Qh[:,0], Qh[:,1], 'o-b', alpha=.7)
        ax3.fill(Qh[:,0], Qh[:,1], 'b', alpha=.3)
        ax3.plot(Q[:,0], Q[:,1], 'o-r', alpha=.7)
        ax3.fill(Q[:,0], Q[:,1], 'r', alpha=.3)
        ax3.set_title(r'$D(X,Y)=%f$ (no cycle)'%d)
        ax3.set_xlim([-.35,.35])
        ax3.set_ylim([-.35,.35])
    
        Qh, Q, d = procrustes(X, Y, cycle=True, fullout=True)
        ax4.plot(Qh[:,0], Qh[:,1], 'o-b', alpha=.7)
        ax4.fill(Qh[:,0], Qh[:,1], 'b', alpha=.3)
        ax4.plot(Q[:,0], Q[:,1], 'o-r', alpha=.7)
        ax4.fill(Q[:,0], Q[:,1], 'r', alpha=.3)
        ax4.set_title(r'$D(X,Y)=%f$ (cycle)'%d)
        ax4.set_xlim([-.35,.35])
        ax4.set_ylim([-.35,.35])
        
    fig.savefig('alignment_digits.pdf')
    """

    # ploting shapes and alignment for digits
    a, b = 5, 7
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(3*4, 2*4))
    
    ax1, ax2, ax3 = axes[0]
    i1 = np.where(labels==a)
    i2 = np.where(labels==b)
    im1 = images[i1][np.random.randint(0, len(i1[0]))].reshape((28,28))
    im2 = images[i2][np.random.randint(0, len(i2[0]))].reshape((28,28))
    ax1.imshow(im1, cmap=plt.cm.gray)
    ax1.axis('off')
    ax2.imshow(im2, cmap=plt.cm.gray)
    ax2.axis('off')
    ax3.axis('off')
    
    ax1, ax2, ax3 = axes[1]
    X = mshape.get_shape(im1, n=30, s=5.0)
    Y = mshape.get_shape(im2, n=30, s=5.0)

    ax1.plot(X[:,0], X[:,1], 'o-k')
    ax1.set_xlim([0, 28])
    ax1.set_ylim([0, 28])
    ax2.plot(Y[:,0], Y[:,1], 'o-k')
    ax2.set_xlim([0, 28])
    ax2.set_ylim([0, 28])
    
    Qh, Q, d = procrustes(X, Y, cycle=False, fullout=True)
    ax3.plot(Qh[:,0], Qh[:,1], 'o-b', alpha=.7)
    ax3.fill(Qh[:,0], Qh[:,1], 'b', alpha=.3)
    ax3.plot(Q[:,0], Q[:,1], 'o-r', alpha=.7)
    ax3.fill(Q[:,0], Q[:,1], 'r', alpha=.3)
    ax3.set_title(r'$D(X,Y)=%f$'%d)
    ax3.set_xlim([-.35,.35])
    ax3.set_ylim([-.35,.35])
    
    fig.savefig('rep_shape_distance.png')
