
"""Functions for procrustes alignment."""


from __future__ import division

import numpy as np
import scipy.spatial


def align(P, Q, rotation=False):
    """Align P into Q by rotation or reflection. If rotation is True
    return the rotation matrix.
    
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
    dist = np.linalg.norm(Qhat - Q)
    if rotation:
        return Qhat, dist, R
    else:
        return Qhat, dist

def best_alignment(P, Q, rotation=False, cycle=False, tol=1e-5):
    """Cycle the points in P and align to Q. 
    Pick the smallest distance if cycle is True.
    
    """
    k, n = P.shape
    if rotation:
        finalQhat, finaldist, finalR = align(P, Q, rotation=True)
    else:
        finalQhat, finaldist = align(P, Q)
        
    if finaldist <= tol or not cycle:
        if rotation:
            return finalQhat, finaldist, finalR
        else:
            return finalQhat, finaldist
    
    # cycle the points and compute alignment each time
    for i in range(n):
        js = range(-1, n-1)
        P = P[:,js]
        if rotation:
            Qhat, dist, R = align(P, Q, rotation=True)
        else:
            Qhat, dist = align(P, Q)
            
        if dist < finaldist:
            finaldist = dist
            finalQhat = Qhat
            if rotation:
                finalR = R
            if finaldist <= tol:
                break
    if rotation:
        return finalQhat, finaldist, finalR
    else:
        return finalQhat, finaldist

def fix_dimensions(X, Y):
    """Generate new matrix to fix dimensions between X and Y.
    Assume they are skinny matrices. X always contains the matrix
    with zeros.
    
    """
    n, k = X.shape
    m, q = Y.shape
    assert k == q
    if n > m:
        Z = np.zeros((n, k))
        Z[:m,:k] = Y
        #X, Y = Z, X
        X, Y = X, Z
    elif n < m:
        Z = np.zeros((m, k))
        Z[:n,:k] = X
        #X, Y = Z, Y
        X, Y = Z, Y
    return X, Y

def procrustes(X, Y, fullout=False, cycle=False):
    """Compute procrustes alignment between X and Y. It aligns X onto Y.
    Both are matrices and don't need to have the same number or coordinate
    points. Assume X and Y are (N,k) matrix where N is the number of data
    points (skiny matrices).
    
    """
    X, Y = fix_dimensions(X, Y)
    n, k = X.shape
    P, Q = X.T, Y.T 
    # translation
    pbar = P.mean(axis=1)
    qbar = Q.mean(axis=1)
    P = P - pbar.reshape((k,1))
    Q = Q - qbar.reshape((k,1))
    # rescale
    P = P/np.linalg.norm(P)
    Q = Q/np.linalg.norm(Q)
    # rotation or reflection
    Qh, d = best_alignment(P, Q, cycle=cycle)
    if fullout:
        return Qh.T, Q.T, d
    else:
        return d

def procrustes2(X, Y, fullout=False):
    """This uses the scipy library for procrustes distance."""
    X, Y = fix_dimensions(X, Y)
    # this tries to align X into Y
    Y_rescaled, Y_hat, d = scipy.spatial.procrustes(Y, X)
    if fullout:
        return Y_hat, Y_rescaled, d
    else:
        return d
    
def procrustes3(X, Y, numpoints, fullout=False, cycle=False):
    X, Y = fix_dimensions(X, Y)
    n, k = X.shape
    
    # we will just use the first n points for alignment
    P, Q = X[:numpoints,:].T, Y[:numpoints,:].T
    X, Y = X.T, Y.T
    
    # translation
    pbar = P.mean(axis=1)
    qbar = Q.mean(axis=1)
    P = P - pbar.reshape((k,1))
    X = X - pbar.reshape((k,1))
    Q = Q - qbar.reshape((k,1))
    Y = Y - qbar.reshape((k,1))
    
    # rescale
    sp = np.linalg.norm(P)
    sq = np.linalg.norm(Q)
    P = P/sp
    X = X/sp
    Q = Q/sq
    Y = Y/sq
    
    # rotation or reflection
    Qh, d, R = best_alignment(P, Q, cycle=cycle, rotation=True)
    Yh = R.dot(X)
    d = np.linalg.norm(Yh - Y)
    if fullout:
        return Yh.T, Y.T, d
    else:
        return d


