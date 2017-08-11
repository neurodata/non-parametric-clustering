"""Energy Statistics Clustering in 1D.

Use exact formula with no initialization to perform clustering in 1D
from energy statistics.

"""

# Guilherme Franca <guifranca@gmail.com>
# Johns Hopkinks University, Neurodata


from __future__ import division

import numpy as np


def mean1D(X, Y):
    """Assume X and Y are both sorted lists.
    Compute the ``energy mean'' in O(n) time.
    This is the function g(X, Y) from the paper.
    
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

    Input: A = [A_1, A_2, ..., A_k] where each A_i is a sample
    Output: between sample energy statistics (S)

    """
    #return sum([len(a)/2*mean1D(a,a)  for a in A])
    return sum([len(a)/2*mean2(a)  for a in A])

def mean2(A):
    """Another version of the above, i.e. compute function g(A_i,A_i). 
    Assume A is sorted."""
    n = len(A)
    return (2./n**2)*sum([A[i]*(2.*i +1 - n) for i in range(n)])

def two_clusters1D(X):
    """Optimize within energy statistics for two clusters in 1D.

    This algorithm performs 2-class-clustering in 1D. 
    
    Input: X = data matrix
    Output: labels, objective function value
    
    """
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
   
    from sklearn.cluster import KMeans
    from sklearn.mixture import GMM
    from scipy.stats import sem
    import sys
    
    import data
    from metric import accuracy
    from gmm import em_gmm_vect as gmm_em
    import qcqp as ke
    from kmeanspp import kpp
   
    #X1 = np.random.normal(0, 1, 100)
    #X1 = np.random.gamma(2, 2, 1000)
    #X1 = np.random.uniform(0, 1, 1000)
    X1 = np.random.lognormal(0, 0.3, 1000)
    X2 = np.random.lognormal(-1, 1, 1000)
    
    #X2 = np.random.uniform(0.5, 1.5, 100)
    #X2 = np.random.normal(2, 1, 100)
    #X2 = np.random.gamma(2, 2, 2000)
    
    X, z = data.shuffle_data([X1, X2])
    Y = np.array([[x] for x in X])
    
    z0 = kpp(2, Y, ret='labels')
    Z0 = ke.ztoZ(z0)

    zh, cost = two_clusters1D(X)
    print accuracy(z, zh)
        
    km = KMeans(2)
    zh = km.fit_predict(Y)
    print accuracy(z, zh)
        
    zh = gmm_em(2, Y)
    print accuracy(z, zh)

    sys.exit()

    #X, z = data.univariate_lognormal([0, -1.5], [0.25, 1], [4000, 4000])
    #X, z = data.univariate_lognormal([0, -1.5], [0.3, 1.5], [1000, 1000])
    #data.histogram(X, z, fname='plot.pdf')
    #plt.hist(X[np.where(z==0)], 80, facecolor='blue', alpha=0.6, normed=1)
    #plt.hist(X[np.where(z==1)], 80, facecolor='red', alpha=0.6, normed=1)
    #plt.show()
    
    num_experiments = 3
    table = np.zeros((num_experiments, 4))
    for i in range(num_experiments):
        #X, z = data.univariate_normal([0, 2], [1, 1], [500, 500])
        X, z = data.univariate_lognormal([0, 0.5], [0.8, 0.05], [100, 100])
        Y = np.array([[x] for x in X])

        z0 = kpp(2, Y, ret='labels')
        Z0 = ke.ztoZ(z0)
        
        zh, cost = two_clusters1D(X)
        table[i,0] = accuracy(z, zh)
        
        #rho = lambda x, y: np.linalg.norm(x-y)
        #G = ke.kernel_matrix(Y, rho)
        #zh = ke.minw(2, G, Z0)
        #table[i,1] = accuracy(z, zh)
        table[i,1] = 0
    
        km = KMeans(2)
        zh = km.fit_predict(Y)
        table[i, 2] = accuracy(z, zh)

        #gmm = GMM(2)
        #gmm.fit(Y)
        #zh = gmm.predict(Y)
        zh = gmm_em(2, Y)
        table[i, 3] = accuracy(z, zh)

    print "Energy1D:", table[:,0].mean(), sem(table[:,0])
    print "Energy:", table[:,1].mean(), sem(table[:,0])
    print "k-means:", table[:,2].mean(), sem(table[:,1])
    print "GMM:", table[:,3].mean(), sem(table[:,2])
    
