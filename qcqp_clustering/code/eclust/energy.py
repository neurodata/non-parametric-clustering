"""Energy Statistics Functions"""

# Guilherme Franca <guifranca@gmail.com>
# Johns Hopkinks University, Neurodata


from __future__ import division

import numpy as np
import numbers


def mean1D(X, Y):
    """We can compute the mean in O(n) in 1D for sorted X and Y.
    We are actually sorting the data inside this function which gives
    O(n \log n). 
    TODO: Be carefull to remove this in a more specific implementation.

    Input: one dimensional X and Y
    Output: sample mean between the two sets

    """
    sx = 0
    sy = 0
    i = 1
    j = 1
    nx = len(X)
    ny = len(Y)
    
    X = np.sort(X)
    Y = np.sort(Y)

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

def mean(X, Y):
    """Compute mean of X and Y. This is O(n^2).
    Input: two sets of data X and Y
    Output: mean between samples
    
    """
    return sum([np.linalg.norm(x-y) for x in X for y in Y])/len(X)/len(Y)
            
def energy(X, Y):
    """Compute energy function in 1D it is  O(n log n) in D >= 2 is O(n^2).
    Input: lists of data X and Y
    Output: energy distance betwee both sets
    
    """

    if isinstance(X[0], numbers.Number): # 1D
        G = mean1D
    else:
        G = mean
    return 2*G(X,Y) - G(X,X) - G(Y,Y)

def between_sample(A):
    """Compute between-sample energy statistics between k sets.
    Input: A = [A_1, A_2, ..., A_k] where each A_i is a sample.
    Output: between sample energy statistics (S)

    """
    k = len(A)
    n = [len(a) for a in A]
    N = sum(n)
    s = 0
    for i in range(k):
        for j in range(i+1, k):
            c= n[i]*n[j]/2/N
            s += c*energy(A[i], A[j])
    return s

def within_sample(A):
    """Compute within-sample dispersion statistics between k sets.
    Input: A[A_1, A_2, ..., A_k] where each A_i is a sample
    Output: between sample energy statistics (S)

    """
    if isinstance(X[0], numbers.Number): # 1D
        return sum([len(a)/2*mean1D(a,a)  for a in A])
    else:
        return sum([len(a)/2*mean(a,a)  for a in A])

def total_dispersion(A):
    """Compute the total dispersion of the sample.
    Input: A[A_1, A_2, ..., A_k] where each A_i is a sample.
    Output: total dispersion = between_sample + within_sampe
    
    """
    X = np.concatenate(A)
    if isinstance(X[0], numbers.Number): # 1D
        return len(X)/2*mean1D(X, X)
    else:
        return len(X)/2*mean(X, X)

def energy_kernel(x, y, alpha=1, cutoff=0):
    w = np.power(np.linalg.norm(x), alpha) + \
        np.power(np.linalg.norm(y), alpha) - \
        np.power(np.linalg.norm(x-y), alpha)
    if w <= cutoff:
        w = 0
    return w


###############################################################################
if __name__ == "__main__":
   
    import matplotlib.pyplot as plt
    import data

    """
    # testing the above formulas, 1 D first
    print "Testing 1D"
    X = np.random.normal(0,1,200)
    Y = np.random.normal(2,1,200)


    print "Mean:", mean(X, Y)
    print "Mean 1D:", mean1D(np.sort(X),np.sort(Y))
    print "Energy:", energy(X, Y)
    b = between_sample([X,Y])
    print "Between Sample:", b
    w = within_sample([X,Y])
    print "Within Sample:",  w
    print "Between + Within Samples:", b + w
    print "Total Dispersion:", total_dispersion([X,Y])
    print 
    """
    
    """
    # testing the above formulas, 2 D
    print "Testing 2D"
    X = np.random.multivariate_normal([0,0], np.eye(2), 200)
    Y = np.random.multivariate_normal([2,0], np.eye(2), 200)

    print "Mean:", mean(X, Y)
    print "Energy:", energy(X, Y)
    b = between_sample([X,Y])
    print "Between Sample:", b
    w = within_sample([X,Y])
    print "Within Sample:",  w
    print "Between + Within Samples:", b + w
    print "Total Dispersion:", total_dispersion([X,Y])
    print
    """

    # testing the above formulas when we shuffle the points
    X = np.random.normal(0, 1, 200)
    Y = np.random.normal(3, 1, 200)
    #X = np.random.multivariate_normal([0, 0], np.eye(2), 200)
    #Y = np.random.multivariate_normal([1, 0], [[1,0], [0,20]], 200)

    ns = range(0, 100, 2)
    dat = np.zeros((len(ns), 5))
    i = 0;
    for n in ns:
        Xp, Yp = data.mix_data([X, Y], n)
        dat[i][0] = between_sample([Xp, Yp])
        dat[i][1] = within_sample([Xp, Yp])
        dat[i][2] = total_dispersion([Xp, Yp])
        #dat[i][4] = kmeans_func([Xp, Yp])
        i += 1
    fig = plt.figure(figsize=(3*3, 3*3))
    ax = fig.add_subplot(131)
    ax.plot(ns, dat[:, 0], 'bo')
    ax = fig.add_subplot(132)
    ax.plot(ns, dat[:, 1], 'ro')
    ax = fig.add_subplot(133)
    ax.plot(ns, dat[:, 2], 'go')
    #ax = fig.add_subplot(154)
    #ax.plot(ns, dat[:, 3], 'yo')
    #ax = fig.add_subplot(155)
    #ax.plot(ns, dat[:, 4], 'co')
    plt.show()

