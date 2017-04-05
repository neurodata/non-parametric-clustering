
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt


def fast_mean(X, Y):
    """We can compute the mean in O(n) in 1D for sorted X and Y."""
    sx = 0
    sy = 0
    i = 1
    j = 1
    nx = len(X)
    ny = len(Y)

    while i <= nx and j <= ny:
        if X[i-1] <= Y[j-1]:
            sx += ((j-1)-(ny-(j-1)))/ny*X[i-1]
            i += 1
        else:
            sy += ((i-1)-(nx-(i-1)))/nx*Y[j-1]
            j += 1
    if i > nx:
        sy += Y[j-1:ny].sum()
    else:
        sx += X[i-1:nx].sum()
    return (sx/nx) + (sy/ny)
            
def fast_energy(X, Y):
    #X = np.sort(A)
    #Y = np.sort(B)
    return 2*fast_mean(X,Y) - fast_mean(X,X) - fast_mean(Y,Y)

def fastT(X, Y):
    nx = len(X)
    ny = len(Y)
    return nx*ny/(nx+ny)*fast_energy(X, Y)

def mean(X, Y):
    """Compute mean of X and Y. This is O(n^2)."""
    return sum([np.linalg.norm(x-y) for x in X for y in Y])/len(X)/len(Y)

def energy(X, Y):
    return 2*mean(X,Y) - mean(X,X) - mean(Y,Y)

def T(X, Y):
    nx = len(X)
    ny = len(Y)
    return nx*ny/(nx+ny)*energy(X, Y)


if __name__ == "__main__":
    X = np.random.normal(0,1,100)
    Y = np.random.normal(2,1,100)
    print T(X, Y)
    print fastT(X, Y)
