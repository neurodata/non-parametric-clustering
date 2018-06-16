"""Experiment for Cigars, Circles."""

# author: Guilherme S. Franca <guifranca@gmail.com>
# Johns Hopkins University

from __future__ import division

import numpy as np
from prettytable import PrettyTable
from scipy.stats import sem

import data
import eclust
import metric
import wrapper

k = 2
total_points = 800
num_experiments = 30

m1 = np.zeros(2)
m2 = np.array([6.5, 0])
s1 = np.array([[1,0], [0,20]])
s2 = np.array([[1,0], [0,20]])

r1 = 1
r2 = 3
eps = 0.2

r = []
for _ in range(num_experiments):
    n1, n2 = np.random.multinomial(total_points, [0.5, 0.5])
    #X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])
    X, z = data.circles([r1, r2], [eps, eps], [n1, n2])
    
    #G = eclust.kernel_matrix(X, 
    #        lambda x, y: 2 - 2*np.exp(-np.linalg.norm(x-y)/2/2))
    G = eclust.kernel_matrix(X, 
            lambda x, y: 2 - 2*np.exp(-np.linalg.norm(x-y)**2/2/1))
    
    row = []
    zh = wrapper.kmeans(k, X)
    a = metric.accuracy(z, zh)
    row.append(a)
    
    zh = wrapper.gmm(k, X)
    a = metric.accuracy(z, zh)
    row.append(a)
    
    zh = wrapper.spectral_clustering(k, X, G)
    a = metric.accuracy(z, zh)
    row.append(a)
    
    zh = wrapper.kernel_kmeans(k, X, G, ini='random')
    a = metric.accuracy(z, zh)
    row.append(a)
    
    
    zh = wrapper.kernel_kgroups(k, X, G, ini='random')
    a = metric.accuracy(z, zh)
    row.append(a)
    
    r.append(row)
r = np.array(r)

t = PrettyTable(['Method', 'Accuracy', 'Std'])
for i, m in enumerate(['k-means', 'gmm', 'spectral clustering', 
            'kernel k-means', 'kernel k-groups']):
    t.add_row([m, r[:,i].mean(), sem(r[:,i])])

print t
