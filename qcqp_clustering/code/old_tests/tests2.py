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


def kernel_energy(k, X, alpha=.5, cutoff=0, num_times=8):
    best_score = 0
    for i in range(num_times):
        km = KernelKMeans(n_clusters=2, max_iter=300, kernel=energy_kernel,
                      kernel_params={'alpha':alpha, 'cutoff':cutoff})
        zh = km.fit_predict(X)
        score = kernel_score(km.kernel_matrix_, zh)
        if score > best_score:
            best_score = score
            best_z = zh
    return best_z

def spectral_energy(k, X, alpha=.5, cutoff=0, n_neighbors=12, num_times=8):
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

def kmeans(k, X):
    km = KMeans(n_clusters=k)
    zh = km.fit_predict(X)
    return zh

def gmm(k, X):
    gmm = GMM(n_components=k)
    gmm.fit(X)
    zh = gmm.predict(X)
    return zh

def cluster_many(k, X, z, clustering_function, num_times):
    accuracies = []
    for i in range(num_times):
        zh = clustering_function(k, X)
        a = accuracy(z, zh)
        accuracies.append(a)
    return np.array(accuracies)

def compare_algos(k, X, z, num_times=10):
    funcs = [kernel_energy, spectral_energy, kmeans, gmm]
    N = len(X)
    ns = range(10, N, 20)
    table = np.zeros((len(ns), len(funcs)*3 + 1))
    indices = np.arange(N)
    for i, n in enumerate(ns):
        idx = np.random.choice(indices, n)
        data = X[idx]
        labels = z[idx]
        
        table[i,0] = n
        j = 1
        for f in funcs:
            accs = cluster_many(k, data, labels, f, num_times)
            table[i,j] = accs.mean()
            table[i,j+1] = accs.max()
            table[i,j+2] = accs.min()
            j += 3
    return table

def plot_results(table, fname='test.pdf'):
    ns = table[:, 0]
    N, M = table.shape
    fig = plt.figure()
    ax = fig.add_subplot(111)
    labels = iter(['kernel-energy', 'spectral-energy', 'k-means', 'gmm'])
    colors = iter(plt.cm.rainbow(np.linspace(0,1,4)))
    symbols = iter(['o', 'D', '^', 'v'])
    j = 1
    while j < M:
        err = table[:,j+1] - table[:,j+2]
        ax.errorbar(ns, table[:,j], yerr=err, fmt=next(symbols), 
            label=next(labels))
        j += 3
    ax.legend(loc='best', shadow=False)
    fig.savefig(fname)
    


###############################################################################
if __name__ == '__main__':
    
    k = 2
    m1 = np.array([0,0])
    s1 = np.eye(2)
    n1 = 200
    m2 = np.array([2,0])
    s2 = np.eye(2)
    n2 = 200
    X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])
    #X = np.array([[x] for x in X])

    X, z = data.spirals([1,-1], [200, 200])
    
    table = compare_algos(k, X, z, 5)
    
    plot_results(table, fname='test_spiral.pdf')
