
# Guilhere Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata

from __future__ import division

import numpy as np
import pandas as pd
import sklearn

from prettytable import PrettyTable

import wrapper
import eclust
import metric

import sys

# some semmimetric functions
rho = lambda x, y: np.power(np.linalg.norm(x-y), 0.5)
rho1 = lambda x, y: np.linalg.norm(x-y)
rho_exp = lambda x, y: 2-2*np.exp(-np.linalg.norm(x-y)/(2*sigma))
rho_gauss = lambda x, y: 2-2*np.exp(-np.linalg.norm(x-y)**2/(2*(1)**2))

#df = pd.read_csv('data/wdbc.data', sep=',', header=None)
#df = pd.read_csv('data/iris.data', sep=',', header=None)
#classes = {
#    'Iris-setosa': 0,
#    'Iris-versicolor': 1,
#    'Iris-virginica': 2
#}

z = np.array([classes[v] for v in df[4].values])
df = df.drop(4, axis=1)
data = df.values
data = (data - data.mean(axis=0))/data.std(axis=0)

G = eclust.kernel_matrix(data, rho_gauss)

k = 3
nt = 5
r = []
r.append(wrapper.kmeans(k, data, run_times=nt))
r.append(wrapper.gmm(k, data, run_times=nt))
r.append(wrapper.spectral_clustering(k, data, G, run_times=nt))
r.append(wrapper.spectral(k, data, G, run_times=nt))
r.append(wrapper.kernel_kmeans(k, data, G, run_times=nt, ini='spectral'))
r.append(wrapper.kernel_kgroups(k,data,G,run_times=nt, ini='spectral'))

t = PrettyTable(['Algorithm', 'Accuracy', 'A-Rand'])
algos = ['kmeans', 'GMM', 'spectral clustering', 'spectral', 
         'kernel k-means', 'kernel k-groups']

for algo, zh in zip(algos, r):
    t.add_row([algo, 
        metric.accuracy(z, zh),
        sklearn.metrics.adjusted_rand_score(z, zh)
    ])

print t

