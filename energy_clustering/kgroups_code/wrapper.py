"""Wrappers to run clustering algorithms."""

# Guilhere Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata

from __future__ import division

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import SpectralClustering

import init
import metric
import eclust

def initialize(method, k, G, X, W):
    if method == "spectral":
        z0 = init.topeigen(k, G, W)
    elif method == "k-means++":
        z0 = init.kmeans_plus(k, X)
    else:
        z0 = np.random.randint(0, k, len(X))
    Z0 = eclust.ztoZ(z0)
    return Z0

def kernel_kmeans(k, X, G, W=None, run_times=5, ini="k-means++"):
    if type(W) == type(None):
        W = np.eye(len(X))
    best_score = -np.inf
    for _ in range(run_times):
        Z0 = initialize(ini, k, G, X, W)
        zh = eclust.kernel_kmeans(k, G, Z0, W, max_iter=300)
        Zh = eclust.ztoZ(zh)
        score = eclust.objective(Zh, G, W)
        if score > best_score:
            best_score = score
            best_z = zh
    return best_z

def kernel_kgroups(k, X, G, W=None, run_times=5, ini="k-means++"):
    if type(W) == type(None):
        W = np.eye(len(X))
    best_score = -np.inf
    for _ in range(run_times):
        Z0 = initialize(ini, k, G, X, W)
        zh = eclust.kernel_kgroups(k, G, Z0, W, max_iter=300)
        Zh = eclust.ztoZ(zh)
        score = eclust.objective(Zh, G, W)
        if score > best_score:
            best_score = score
            best_z = zh
    return best_z

def spectral(k, X, G, W=None, run_times=5):
    if type(W) == type(None):
        W = np.eye(len(X))
    best_score = -np.inf
    for _ in range(run_times):
        zh = init.topeigen(k, G, W, run_times=run_times)
        Zh = eclust.ztoZ(zh)
        score = eclust.objective(Zh, G, W)
        if score > best_score:
            best_score = score
            best_z = zh
    return best_z

def kmeans(k, X, run_times=5):
    km = KMeans(k, n_init=run_times)
    km.fit(X)
    zh = km.labels_
    mu = km.cluster_centers_
    return zh, mu

def gmm(k, X, run_times=5):
    gm = GMM(k, n_init=run_times, init_params='kmeans')
    #gm = GMM(k)
    gm.fit(X)
    zh = gm.predict(X)
    mu = gm.means_
    cov = gm.covariances_
    return zh, mu, cov

def spectral_clustering(k, X, G, W=None, run_times=5):
    if type(W) == type(None):
        W = np.eye(len(X))
    W2 = np.sqrt(W)
    Gtilde = W2.dot(G.dot(W2))
    sc = SpectralClustering(k, affinity='precomputed', n_init=run_times)
    zh = sc.fit_predict(Gtilde)
    return zh


###############################################################################
if __name__ == "__main__":
    
    import data
    import metric
    from prettytable import PrettyTable
    import sys

    n = 400
    d = 10
    n1, n2 = np.random.multinomial(n, [1/2, 1/2])
    m1 = np.zeros(d)
    m2 = 0.7*np.ones(d)
    s1 = s2 = np.eye(d)
    X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])

    G = eclust.kernel_matrix(X, lambda x, y: np.linalg.norm(x-y))
    W = np.eye(n)
    k = 2

    t = PrettyTable(["Method", "Accuracy"])
    
    zh = kernel_kmeans(k, X, G, W, run_times=5, ini="k-means++")
    a = metric.accuracy(z, zh)
    t.add_row(["Kernel k-means", a])
    
    zh = kernel_kgroups(k, X, G, W, run_times=5, ini="k-means++")
    a = metric.accuracy(z, zh)
    t.add_row(["Kernel k-groups", a])
    
    zh = spectral(k, X, G, W, run_times=5)
    a = metric.accuracy(z, zh)
    t.add_row(["Spectral", a])
    
    zh = kmeans(k, X, run_times=5)
    a = metric.accuracy(z, zh)
    t.add_row(["k-means", a])
    
    zh = gmm(k, X, run_times=5)
    a = metric.accuracy(z, zh)
    t.add_row(["gmm", a])
    
    zh = spectral_clustering(k, X, G, W, run_times=5)
    a = metric.accuracy(z, zh)
    t.add_row(["Spectral Clustering", a])

    print t

