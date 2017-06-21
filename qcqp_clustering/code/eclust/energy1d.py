"""Energy Statistics Clustering in 1D"""

# Guilherme Franca <guifranca@gmail.com>
# Johns Hopkinks University, Neurodata


from __future__ import division

import numpy as np


def mean1D(X, Y):
    """Assume X and Y are both sorted lists.
    Compute the energy mean in O(n) time.
    
    """
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

def within_sample(A):
    """Compute within-sample dispersion statistics between k sets.
    Input: A[A_1, A_2, ..., A_k] where each A_i is a sample
    Output: between sample energy statistics (S)

    """
    #return sum([len(a)/2*mean1D(a,a)  for a in A])
    return sum([len(a)/2*mean2(a)  for a in A])

def mean2(A):
    """Another version of the above. Assume A is sorted."""
    n = len(A)
    return (2./n**2)*sum([A[i]*(2.*i +1 - n) for i in range(n)])

def two_clusters1D(X):
    """Optimize energy statistics for two clusters in 1D."""
    sorted_indices = np.argsort(X)
    Y = X[sorted_indices]
    best_cost = np.inf
    best_split = 1
    for i in range(1, len(Y)):
        A = Y[:i]
        B = Y[i:]
        cost = within_sample([A, B])
        if cost < best_cost:
            best_cost = cost
            best_split = i
    sorted_labels = np.zeros(len(Y), dtype=int)
    sorted_labels[best_split:] = 1
    final_labels = np.zeros(len(Y), dtype=int)
    for k, i in zip(sorted_labels, sorted_indices):
        final_labels[i] = k
    return final_labels, best_cost


###############################################################################
if __name__ == "__main__":
   
    import data
    from metric import accuracy
    from sklearn.cluster import KMeans
    from sklearn.mixture import GMM
    import matplotlib.pyplot as plt
    import sys
   
    #X1 = np.random.normal(0, 1, 100)
    #X1 = np.random.gamma(2, 2, 1000)
    #X1 = np.random.uniform(0, 1, 1000)
    
    #X2 = np.random.uniform(0.5, 1.5, 100)
    #X2 = np.random.normal(2, 1, 100)
    #X2 = np.random.gamma(2, 2, 2000)
    
    #X, z = data.shuffle_data([X1, X2])

    #X, z = data.univariate_lognormal([0, -1.5], [0.25, 1], [4000, 4000])
    #X, z = data.univariate_normal([0, 5], [1, 2], [4000, 4000])
    X, z = data.univariate_lognormal([0, -1.5], [0.3, 1.5], [1000, 1000])
    #data.histogram(X, z, fname='plot.pdf')
    #plt.hist(X[np.where(z==0)], 80, facecolor='blue', alpha=0.6, normed=1)
    #plt.hist(X[np.where(z==1)], 80, facecolor='red', alpha=0.6, normed=1)
    #plt.show()

    zh, cost = two_clusters1D(X)
    print accuracy(z, zh)

    Y = np.array([[x] for x in X])
    km = KMeans(2)
    zh = km.fit_predict(Y)
    print accuracy(z, zh)

    gmm = GMM(2)
    gmm.fit(Y)
    zh = gmm.predict(Y)
    print accuracy(z, zh)
    
