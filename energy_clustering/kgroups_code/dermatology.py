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

import wrapper
import eclust
import metric


# some semmimetric functions
rho = lambda x, y: np.power(np.linalg.norm(x-y), 0.5)
rho1 = lambda x, y: np.linalg.norm(x-y)
rho_gauss = lambda x, y: 2-2*np.exp(-np.linalg.norm(x-y)**2/(2*(3.5)**2))
rho_exp = lambda x, y: 2-2*np.exp(-np.linalg.norm(x-y)/(2*3.7))


# get dermatology data, last column are labels
df = pd.read_csv('data/dermatology_data.csv', sep=',', header=None)

data = df.values[:,:-1]
z = df.values[:,-1] - 1 # 6 classes, starting with 0

# replace missing entries in age by the mean
#missing_id = np.where(data=='?')
#not_missing_id = [i for i in range(len(data)) if i not in missing_id[0]]
#mean_age = np.array(data[not_missing_id,33], dtype=float).mean()
#data[missing_id] = mean_age

#data = np.array(data, dtype=float)

# delete missing entries
delete_missing = np.where(data=='?')[0]
data = np.delete(data, delete_missing, axis=0)
data = np.array(data, dtype=float)
z = np.delete(z, delete_missing, axis=0)

# normalize data
data = (data - data.mean(axis=0))/data.std(axis=0)

G = eclust.kernel_matrix(data, rho)
#G = energy.eclust.kernel_matrix(data, rho_gauss)
#G = energy.eclust.kernel_matrix(data, rho_exp)

r = []
r.append(wrapper.kmeans(6, data, run_times=10))
r.append(wrapper.gmm(6, data, run_times=10))
r.append(wrapper.spectral_clustering(6, data, G, run_times=10))

r.append(wrapper.spectral(6, data, G, run_times=10))

#r.append(wrapper.kernel_kmeans(6, data, G, run_times=10, ini='random'))
r.append(wrapper.kernel_kmeans(6, data, G, run_times=10, ini='k-means++'))
#r.append(wrapper.kernel_kmeans(6, data, G, run_times=10, ini='spectral'))

#r.append(wrapper.kernel_kgroups(6,data,G,run_times=10, ini='random'))
r.append(wrapper.kernel_kgroups(6,data,G,run_times=10, ini='k-means++'))
#r.append(wrapper.kernel_kgroups(6,data,G,run_times=10, ini='spectral'))

t = PrettyTable(['Algorithm', 'Accuracy', 'A-Rand'])
algos = ['kmeans', 'GMM', 'spectral clustering', 'spectral', 
         'kernel k-means', 'kernel k-groups']

for algo, zh in zip(algos, r):
    t.add_row([algo, 
        metric.accuracy(z, zh),
        sklearn.metrics.adjusted_rand_score(z, zh)
    ])

print t

Z = np.array(eclust.ztoZ(z), dtype=int)
Zh = np.array(eclust.ztoZ(zh), dtype=int)

df = pd.DataFrame(Z)
df.to_csv('data/dermatology_true_label_matrix.csv', index=False, header=None)

df = pd.DataFrame(Zh)
df.to_csv('data/dermatology_pred_label_matrix.csv', index=False, header=None)

