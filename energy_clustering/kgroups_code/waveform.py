"""Clustering using the waveform dataset:

https://archive.ics.uci.edu/ml/datasets/Waveform+Database+Generator+%28Version+1%29

"""

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
rho = lambda x, y: np.power(np.linalg.norm(x-y), 1)
rho1 = lambda x, y: np.linalg.norm(x-y)


# get dermatology data, last column are labels
df = pd.read_csv('data/waveform-+noise.data', sep=',', header=None)

data = df.values[:,:-1]
z = df.values[:,-1]  # 3 classes, starting with 0

idx = np.random.choice(range(len(data)), 2000)
data = data[idx]
z = z[idx]
data = (data - data.mean(axis=0))/data.std(axis=0)

sigma2 = sum([np.linalg.norm(x-y)**2 
                for x in data for y in data])/(len(data)**2)
sigma = np.sqrt(sigma2)

rho_exp = lambda x, y: 2-2*np.exp(-np.linalg.norm(x-y)/(2*sigma))
rho_gauss = lambda x, y: 2-2*np.exp(-np.linalg.norm(x-y)**2/(2*(sigma)**2))

# normalize data

G = eclust.kernel_matrix(data, rho_gauss)
#G = energy.eclust.kernel_matrix(data, rho_gauss)
#G = energy.eclust.kernel_matrix(data, rho_exp)

r = []
r.append(wrapper.kmeans(3, data, run_times=5))
r.append(wrapper.gmm(3, data, run_times=5))
r.append(wrapper.spectral_clustering(3, data, G, run_times=5))
r.append(wrapper.spectral(3, data, G, run_times=5))
#r.append(wrapper.kernel_kmeans(3, data, G, run_times=5, ini='random'))
#r.append(wrapper.kernel_kmeans(3, data, G, run_times=5, ini='k-means++'))
r.append(wrapper.kernel_kmeans(3, data, G, run_times=5, ini='spectral'))
#r.append(wrapper.kernel_kgroups(3,data,G,run_times=5, ini='random'))
#r.append(wrapper.kernel_kgroups(3,data,G,run_times=5, ini='k-means++'))
r.append(wrapper.kernel_kgroups(3,data,G,run_times=5, ini='spectral'))

t = PrettyTable(['Algorithm', 'Accuracy', 'A-Rand'])
algos = ['kmeans', 'GMM', 'spectral clustering', 'spectral', 
         'kernel k-means', 'kernel k-groups']

for algo, zh in zip(algos, r):
    t.add_row([algo, 
        metric.accuracy(z, zh),
        sklearn.metrics.adjusted_rand_score(z, zh)
    ])

print t

