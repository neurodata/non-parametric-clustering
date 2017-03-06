#!/usr/bin/env python

"""This module computes Procrustes distance using only the outside contour
for alignment, however we use all contours which has a matching between
two shapes to compute the final distance.

"""

from __future__ import division

import numpy as np
import scipy.spatial.distance


def align(P, Q):
    """Align P into Q by rotation. Use SVD. Return the rotated P and 
    the rotation matrix. Q is unchanged.
    
    """
    Z = P.dot(Q.T)
    U, sigma, Vt = np.linalg.svd(Z)
    R = Vt.T.dot(U)
    det = np.linalg.det(R)
    if det < 0:
        d = np.ones(U.shape[0])
        d[-1] = -1
        R = Vt.T.dot(np.diag(d)).dot(U)
    Qhat = R.dot(P)
    return Qhat, R
    
def procrustes(Xs, Ys, transpose=True, fullout=False):
    """Xs and Ys is a list of shapes or contours, 
    each contour is a Nx2 matrix. 
    
    """

    # use only outside contour for alignment
    X, Y = Xs[0], Ys[0] 
    assert X.shape == Y.shape
    n, k = X.shape
    if transpose:
        P = X.T 
        Q = Y.T
        newXs = [P]
        newYs = [Q]
        for i in range(1, len(Xs)):
            newXs.append(Xs[i].T)
        for i in range(1, len(Ys)):
            newYs.append(Ys[i].T)
        Xs = newXs
        Ys = newYs
    else:
        P = X
        Q = Y
    
    # translation
    pbar = P.mean(axis=1).reshape((k,1))
    qbar = Q.mean(axis=1).reshape((k,1))
    Ptilde = P - pbar
    Qtilde = Q - qbar
    
    # scaling
    sp = np.linalg.norm(Ptilde)
    sq = np.linalg.norm(Qtilde)
    Ptilde = Ptilde/sp
    Qtilde = Qtilde/sq
    
    # rotation/reflection
    Qtildehat, R = align(Ptilde, Qtilde)

    # apply transformation to all contours
    Xtilde = [Qtildehat]
    Ytilde = [Qtilde]
    for i in range(1, len(Xs)):
        xt = Xs[i] - pbar
        xt = xt/sp
        xt = R.dot(xt)
        Xtilde.append(xt)
    for i in range(1, len(Ys)):
        yt = (Ys[i] - qbar)/sq
        Ytilde.append(yt) 
        
    num_contours = min(len(Xtilde), len(Ytilde))
    distance = 0
    for i in range(num_contours):
        distance += np.linalg.norm(Xtilde[i] - Ytilde[i])

    if fullout:
        if transpose:
            finalXs = []
            finalYs = []
            for i in range(len(Xtilde)):
                finalXs.append(Xtilde[i].T)
            for i in range(len(Ytilde)):
                finalYs.append(Ytilde[i].T)
        return finalXs, finalYs, distance
    else:
        return distance


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
    
    # ploting shapes and alignment for digits
    a, b, c = 1, 6, 8
    pairs = [[a,a], [b,b], [c,c], [a,b], [a,c], [b,c]]
    fig, axes = plt.subplots(nrows=len(pairs), ncols=3,
                                figsize=(3.2*3, 3.2*len(pairs)))
    for (a, b), row in zip(pairs, axes):
        ax1, ax2, ax3 = row
        i1 = np.where(labels==a)[0]
        i2 = np.where(labels==b)[0]
        im1 = images[np.random.choice(i1)].reshape((28,28))
        im2 = images[np.random.choice(i2)].reshape((28,28))
        Xs = mshape.get_shape2(im1, n=30, s=5.0)
        Ys = mshape.get_shape2(im2, n=30, s=5.0)
        ax1.imshow(im1, cmap=plt.cm.gray)
        R = np.array([[0,1], [-1,0]])
        for X in Xs:
            X = X.dot(R)
            ax1.plot(X[:,1], X[:,0]+28, 'o-y')
        #ax1.set_xlim([0, 28])
        #ax1.set_ylim([0, 28])
        ax1.axis('off')
        ax2.imshow(im2, cmap=plt.cm.gray)
        for Y in Ys:
            Y = Y.dot(R)
            ax2.plot(Y[:,1], Y[:,0]+28, 'o-y')
        #ax2.set_xlim([0, 28])
        #ax2.set_ylim([0, 28])
        ax2.axis('off')
    
        Qsh, Qs, d = procrustes(Xs, Ys, fullout=True)
        for i, Qh in enumerate(Qsh):
            ax3.plot(Qh[:,0], Qh[:,1], 'o-b', alpha=.7)
        for i, Q in enumerate(Qs):
            ax3.plot(Q[:,0], Q[:,1], 'o-r', alpha=.7)
        ax3.set_title(r'$D(X,Y)=%f$'%d)
        ax3.set_xlim([-.35,.35])
        ax3.set_ylim([-.35,.35])

        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        ax3.set_aspect('equal')
    
    fig.savefig('alignment_multi_contours.pdf')

    
    """
    i1 = np.where(labels==6)[0]
    i2 = np.where(labels==8)[0]
    im1 = images[np.random.choice(i1)].reshape((28,28))
    im2 = images[np.random.choice(i2)].reshape((28,28))
    X = mshape.get_shape2(im1, n=30, s=5.0)
    Y = mshape.get_shape2(im2, n=30, s=5.0)
    A, B, d = procrustes(X, Y, fullout=True)
    """

