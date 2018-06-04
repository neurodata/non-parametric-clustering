"""Initialization methods."""

# Guilherme Franca <guifranca@gmail.com>
# Johns Hopkins University

from __future__ import division

import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

from sklearn.metrics.pairwise import pairwise_distances


def euclidean_distance(x, y):
    return np.linalg.norm(x-y)**2

def kmeans_plus(k, X):
    """This is the k-means++ initialization proposed by Arthur and
    Vassilvitskii (2007). We label the points according to closest
    distance to the centers.
    
    """
    K = k
    N = X.shape[0]
    distance = euclidean_distance
    C = [X[np.random.randint(0, N)]] # centers
    D = np.zeros(N)                  # distances
    for k in range(1, K):
        for n in range(N):
            D[n] = np.array([distance(X[n], c) for c in C]).min()
        p = D/D.sum()
        j = discrete_rv(p)
        C.append(X[j])
    mus = np.array(C)
    
    labels = np.empty(N, dtype=np.int8)
    for n in range(N):
        for k in range(K):
            D = np.array([distance(X[n], mus[k]) for k in range(K)])
            labels[n] = np.argmin(D)
    return labels

def kmeans_plus2(k, X):
    """This is the k-means++ initialization proposed by Arthur and
    Vassilvitskii (2007). We label the points according to closest
    distance to the centers.

    Same as above but return labels and means.
    
    """
    K = k
    N = X.shape[0]
    distance = euclidean_distance
    C = [X[np.random.randint(0, N)]] # centers
    D = np.zeros(N)                  # distances
    for k in range(1, K):
        for n in range(N):
            D[n] = np.array([distance(X[n], c) for c in C]).min()
        p = D/D.sum()
        j = discrete_rv(p)
        C.append(X[j])
    mus = np.array(C)
    
    labels = np.empty(N, dtype=np.int8)
    for n in range(N):
        for k in range(K):
            D = np.array([distance(X[n], mus[k]) for k in range(K)])
            labels[n] = np.argmin(D)
    return labels, mus

def discrete_rv(p):
    """Return an integer according to probability function p."""
    u = np.random.uniform()
    cdf = np.cumsum(p)
    j = np.searchsorted(cdf, u)
    return j

def topeigen(k, G, W, run_times=1, init='k-means++'):
    """This is similar to the spectral clustering proposed by
    Ng, Jordan, and Weiss, however numerically it seems to be a little better
    and more stable.
    In this case we are effectivelly solving the eigenvalue problem
    for the matrix G D^{-1}, where G has diagonals set to zero.
    
    """
    K = np.copy(G)
    n, _ = K.shape
    W2 = np.sqrt(W)
    K = W2.dot(K.dot(W2))
    for i in range(n):
        K[i,i] = 0
    D = np.diag(K.sum(axis=1))
    
    eigvals, Y = eigh(K, D, eigvals=(n-k,n-1))
    Yt = np.array([D.dot(Y[:,i]) for i in range(Y.shape[1])]).T
    
    for i in range(k):
        Yt[i] = Yt[i]/np.linalg.norm(Yt[i])
    
    km = KMeans(k, init=init, n_init=run_times)
    labels = km.fit_predict(Yt)
    return labels

def topeigen2(k, G, W, run_times=1, init='k-means++'):
    """This is similar to the spectral clustering proposed by
    Ng, Jordan, and Weiss.
    In this case we are effectivelly solving the eigenvalue problem
    for the matrix D^{-1}G, where G has diagonals set to zero.
    
    """
    K = np.copy(G)
    n, _ = K.shape
    W2 = np.sqrt(W)
    K = W2.dot(K.dot(W2))
    for i in range(n):
        K[i,i] = 0
    D = np.diag(K.sum(axis=1))
    
    eigvals, Yt = eigh(K, D, eigvals=(n-k,n-1))
    
    for i in range(k):
        Yt[i] = Yt[i]/np.linalg.norm(Yt[i])
    
    km = KMeans(k, init=init, n_init=run_times)
    labels = km.fit_predict(Yt)
    return labels

def spectralNg(k, G, W, run_times=1, init='k-means++'):
    """This is spectral clustering proposed by
    Ng, Jordan, and Weiss.
    It considers the eigenvalue problem
    for the matrix D^{-1/2} G D^{-1/2}, where G has diagonals set to zero.
    
    """
    K = np.copy(G)
    n, _ = K.shape
    W2 = np.sqrt(W)
    K = W2.dot(K.dot(W2))
    for i in range(n):
        K[i,i] = 0
    D = np.diag(K.sum(axis=1))
    D2 = np.power(D, 0.5)
    
    eigvals, Y = eigh(K, D, eigvals=(n-k,n-1))
    Yt = np.array([D2.dot(Y[:,i]) for i in range(Y.shape[1])]).T
    
    for i in range(k):
        Yt[i] = Yt[i]/np.linalg.norm(Yt[i])
    
    km = KMeans(k, init=init, n_init=run_times)
    labels = km.fit_predict(Yt)
    return labels

def spectral(k, G, W):
    """Spectral clustering from standard sklearn library."""
    n = G.shape[0]
    W2 = np.sqrt(W)
    Gtilde = W2.dot(G.dot(W2))
    sc = SpectralClustering(k, affinity='precomputed')
    labels = sc.fit_predict(Gtilde)
    return labels
    

###############################################################################
if __name__ == "__main__":

    # testing the above initializations, i.e. kmeans++ and spectral clustering
    from scipy.stats import sem
    from prettytable import PrettyTable 
    
    import data
    import eclust
    import metric

    table = []
    for i in range(10):
        
        # generate data ##############
        D = 2
        n1 = 100
        n2 = 100
        m1 = 0.5*np.ones(D)
        s1 = np.eye(D)
        m2 = 2*np.ones(D)
        s2 = 1.2*np.eye(D)
        X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])
        k = 2
        G = eclust.kernel_matrix(X, 
                        lambda x, y: np.power(np.linalg.norm(x-y),1))
        W = np.eye(n1+n2)
        
        results = []
        
        zh = kmeans_plus(k, X)
        results.append(metric.accuracy(z, zh))
        zh = spectral(k, G, W)
        results.append(metric.accuracy(z, zh))
        zh = topeigen(k, G, W, run_times=10, init='k-means++')
        results.append(metric.accuracy(z, zh))
        zh = topeigen2(k, G, W, run_times=10, init='k-means++')
        results.append(metric.accuracy(z, zh))
        zh = spectralNg(k, G, W, init='k-means++', run_times=10)
        results.append(metric.accuracy(z, zh))

        table.append(results)
    
    table = np.array(table)

    t = PrettyTable(['Method', 'Mean Accuracy', 'Std Error'])
    t.add_row(["k-means++", table[:,0].mean(), sem(table[:,0])])
    t.add_row(["Spectral Clustering (sklearn [Shi, Malik])", 
                    table[:,1].mean(), sem(table[:,1])])
    t.add_row(["** Top Eigenvalues (mine, G D^{-1})", table[:,2].mean(), 
                                                        sem(table[:,2])])
    t.add_row(["Top Eigenvalues (mine, D^{-1} G)", table[:,3].mean(),
                                                        sem(table[:,3])])
    t.add_row(["Spectral Clustering Ng ( D^{-1/2} G D^{-1/2})", 
                                        table[:,4].mean(), sem(table[:,4])])
    print t
    
