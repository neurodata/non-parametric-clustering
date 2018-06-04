"""Clustering using the dermatology dataset:

https://archive.ics.uci.edu/ml/datasets/dermatology

"""

# Guilhere Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata

from __future__ import division

import numpy as np
import pandas as pd
import sklearn

from prettytable import PrettyTable

import run_clustering as cluster
import energy


# some semmimetric functions
rho = lambda x, y: np.power(np.linalg.norm(x-y), 0.5)
rho_gauss = lambda x, y: 2-2*np.exp(-np.linalg.norm(x-y)**2/(2*(3.5)**2))
rho_exp = lambda x, y: 2-2*np.exp(-np.linalg.norm(x-y)/(2*3.5))


# get dermatology data, last column are labels
df = pd.read_csv('./data/dermatology.data', sep=',', header=None)
data = df.values[:,:-1]
true_labels = df.values[:,-1] - 1 # 6 classes, starting with 0

# delete missing entries
delete_missing = np.where(data=='?')[0]
data = np.delete(data, delete_missing, axis=0)
data = np.array(data, dtype=float)
true_labels = np.delete(true_labels, delete_missing, axis=0)

# normalize data
data = (data - data.mean(axis=0))/data.std(axis=0)

G = energy.eclust.kernel_matrix(data, rho)
#G = energy.eclust.kernel_matrix(data, rho_gauss)
#G = energy.eclust.kernel_matrix(data, rho_exp)

kmeans_labels = cluster.kmeans(6, data, run_times=10, init="k-means++")
gmm_labels = cluster.gmm(6, data, run_times=10, init="kmeans")
spectral_labels = cluster.spectral(6, data, G, run_times=10)
energy_spectral_labels = cluster.energy_spectral(6, data, G, run_times=10)
lloyd_labels = cluster.energy_lloyd(6, data, G, run_times=10, init="spectral")
hart_labels = cluster.energy_hartigan(6,data,G,run_times=10,init="spectral")

t = PrettyTable(['Algorithm', 'Accuracy', 'A-Rand', 'Mutual Info', 'V-Measure', 
                'Fowlkes-Mallows'])

algos = ['kmeans', 'GMM', 'spectral', 'energy_spectral', 'energy_lloyd',
         'energy_hartigan']
pred_labels = [kmeans_labels, gmm_labels, spectral_labels,
               energy_spectral_labels, lloyd_labels, hart_labels]

for algo, pred_label in zip(algos, pred_labels):
    t.add_row([algo, 
        energy.metric.accuracy(true_labels, pred_label),
        sklearn.metrics.adjusted_rand_score(true_labels, pred_label),
        sklearn.metrics.adjusted_mutual_info_score(true_labels, pred_label),
        sklearn.metrics.v_measure_score(true_labels, pred_label),
        sklearn.metrics.fowlkes_mallows_score(true_labels, pred_label)]
    )

print t
