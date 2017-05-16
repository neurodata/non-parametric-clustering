
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GMM

from kkmeans import KernelKMeans
from metric import accuracy
from eclust.objectives import kernel_score
from eclust.energy import energy_kernel


def kernel_energy(k, X, alpha=.5, cutoff=0, num_times=8):
    best_score = 0
    for i in range(num_times):
        km = KernelKMeans(n_clusters=k, max_iter=300, kernel=energy_kernel,
                      kernel_params={'alpha':alpha, 'cutoff':cutoff})
        zh = km.fit_predict(X)
        score = kernel_score(km.kernel_matrix_, zh)
        if score > best_score:
            best_score = score
            best_z = zh
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

def compare_algos(k, X, z, funcs=[kernel_energy, kmeans, gmm], num_times=10):
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

def plot_results(table, fname='comparison.pdf'):
    ns = table[:, 0]
    N, M = table.shape
    fig = plt.figure()
    ax = fig.add_subplot(111)
    labels = iter(['kernel-energy', 'k-means', 'gmm'])
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

