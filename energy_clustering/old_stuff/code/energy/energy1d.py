"""Energy Statistics Clustering in 1D.

Use exact formula with no random initialization to perform clustering in 1D
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
    from beautifultable import BeautifulTable
    
    import data
    from metric import accuracy
    import eclust
    import initialization
    
    num_experiments = 10
    table = np.zeros((num_experiments, 5))
    for i in range(num_experiments):
        X, z = data.univariate_lognormal([0, -1.5], [0.3, 1.5], [100, 100])
        #X, z = data.univariate_normal([0, 5], [1, 22], [15, 15])
        Y = np.array([[x] for x in X])
        k = 2

        # 1D energy clustering
        zh, cost = two_clusters1D(X)
        table[i,0] = accuracy(z, zh)
       
        # initialization
        z0 = initialization.kmeanspp(k, Y, ret='labels')
        Z0 = eclust.ztoZ(z0)
        rho = lambda x, y: np.linalg.norm(x-y)
        G = eclust.kernel_matrix(Y, rho)
        z1 = initialization.spectral(k, G)
        Z1 = eclust.ztoZ(z1)
        
        # Hartigan's method
        zh = eclust.energy_hartigan(k, G, Z0)
        table[i,1] = accuracy(z, zh)
        
        zh = eclust.energy_hartigan(k, G, Z1)
        table[i,2] = accuracy(z, zh)
    
        # standard k-means
        km = KMeans(2)
        zh = km.fit_predict(Y)
        table[i, 3] = accuracy(z, zh)

        # GMM
        gmm = GMM(2)
        gmm.fit(Y)
        zh = gmm.predict(Y)
        table[i, 4] = accuracy(z, zh)
    
    t = BeautifulTable()
    t.column_headers = ["Method", "Mean Accuracy", "Std Error"]
    t.append_row(["Energy1D", table[:,0].mean(), sem(table[:,0])])
    t.append_row(["H-Energy++:", table[:,1].mean(), sem(table[:,1])])
    t.append_row(["H-Energy-top:", table[:,2].mean(), sem(table[:,2])])
    t.append_row(["k-means:", table[:,3].mean(), sem(table[:,3])])
    t.append_row(["GMM:", table[:,4].mean(), sem(table[:,4])])
    print t
    
