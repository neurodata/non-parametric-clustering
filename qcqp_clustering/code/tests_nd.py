"""
Several tests and comparison for clustering.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GMM

from eclust.kkmeans import KernelKMeans
from eclust.metric import accuracy
import eclust.data as data
from eclust.withingraph import EClust
from eclust.energy1d import two_clusters1D
from eclust.energy import energy_kernel
from eclust.objectives import kernel_score


def kernel_energy(k, X, alpha=1, cutoff=0, num_times=10, truth=None):
    best_score = 0
    for i in range(num_times):
        if truth is not None:
            km = KernelKMeans(n_clusters=2, max_iter=300, kernel=energy_kernel,
                      kernel_params={'alpha':alpha, 'cutoff':cutoff},
                      labels=truth)
        else:
            km = KernelKMeans(n_clusters=2, max_iter=300, kernel=energy_kernel,
                      kernel_params={'alpha':alpha, 'cutoff':cutoff})
        zh = km.fit_predict(X)
        score = kernel_score(km.kernel_matrix_, zh)
        if score > best_score:
            best_score = score
            best_z = zh
    return best_z

def spectral_energy(k, X, alpha=1, cutoff=0, n_neighbors=10, num_times=10):
    best_score = 0
    for i in range(num_times):
        sc = SpectralClustering(n_clusters=k, affinity=energy_kernel,
                        assign_labels='kmeans', n_init=20, 
                        kernel_params={'alpha':1, 'cutoff':0},
                        n_neighbors=n_neighbors)
        sc.fit(X)
        zh = sc.labels_
        score = kernel_score(sc.affinity_matrix_, zh)
        if score > best_score:
            best_z = zh
            best_score = score
    return best_z


###############################################################################
if __name__ == '__main__':
    
    m1 = np.array([0, 0, 0, 0])
    s1 = np.array([
        [0.6, .2, .2, .2], 
        [.2, 0.6, 0, .2],
        [.2, 0, 0.6, .2],
        [.2, .2, .2, 0.6],
    ])
    #s1 = np.eye(4)
    n1 = 300
    
    m2 = np.array([1.5, 0, 0, 0])
    s2 = np.array([
        [0.5, 0, 0, 0], 
        [0, 0.5, 0, 0],
        [0, 0, 0.5, 0],
        [0, 0, 0, 0.5],
    ])
    n2 = 300
    
    k = 2
    
    X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])
    #X, z = data.multivariate_lognormal([m1, m2], [s1, s2], [n1, n2])
    #X, z = data.circles([1, .2], [0.1, 0.2], [200, 200])
    
    X, z = data.spirals([1, -1], [200, 200])
    #data.plot(X, z, 'spirals.pdf')
    
    # clustering with different algorithms
    
    zh = kernel_energy(k, X, alpha=.5, cutoff=0, num_times=5, truth=z)
    print "Kernel/EnergyT:", accuracy(z, zh)
    
    zh = kernel_energy(k, X, alpha=.5, cutoff=0, num_times=5)
    print "Kernel/Energy:", accuracy(z, zh)

    #ec = EClust(n_clusters=k, max_iter=100, labels=zh)
    #zh = ec.fit_predict(X)
    #print "Within Graph Energy:", accuracy(z, zh)
    #print accuracy(z, zh)
    
    #zh = spectral_energy(k, X, alpha=.5, cutoff=0, num_times=10, 
    #                        n_neighbors=10)
    #print "Spectral/Energy:", accuracy(z, zh)
    
    #G = spectral.gram_matrix(X, spectral.energy_kernel)
    #sc = SpectralClustering(n_clusters=2, affinity='precomputed')
    #zh = sc.fit_predict(G)
    #print "Spectral Clustering /Precomputed Energy:", accuracy(z, zh)
    
    gmm = GMM(n_components=k)
    gmm.fit(X)
    zh = gmm.predict(X)
    print "GMM:", accuracy(z, zh)

    km = KMeans(n_clusters=k)
    zh = km.fit_predict(X)
    print "k-means:", accuracy(z, zh)

