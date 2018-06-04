"""Experiment unbalanced Gaussians."""

# author: Guilherme S. Franca <guifranca@gmail.com>
# Johns Hopkins University

from __future__ import division

import numpy as np
import multiprocessing as mp
import pandas as pd

import argparse

import data
import eclust
import metric
import wrapper

parser = argparse.ArgumentParser(description="Unbalanced Gaussians.")
parser.add_argument('-o', dest='output', required=True,
                    type=str, action='store', help="file name for output")
parser.add_argument('-n', type=int, required=True,
                    dest='num_experiments', action='store', 
                    help="number of experiments")

args = parser.parse_args()

num_points = range(0, 250, 10)
k = 2
N = 300
D = 4
d = 2
output = args.output
num_experiments = args.num_experiments

def generate_data(m):
    m1 = np.zeros(D)
    s1 = np.eye(D)
    m2 = np.concatenate((1.5*np.ones(d), np.zeros(D-d)))
    s2 = np.diag(np.concatenate((.5*np.ones(d), np.ones(D-d))))
    pi1 = (N-m)/N/2
    pi2 = (N+m)/N/2
    n1, n2 = np.random.multinomial(N, [pi1, pi2])
    X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])
    return X, z

r = []
for _ in range(num_experiments):
    for m in num_points:
        X, z = generate_data(m)
        G = eclust.kernel_matrix(X, lambda x, y: np.linalg.norm(x-y))
        
        zh = wrapper.kmeans(k, X)
        a = metric.accuracy(z, zh)
        r.append(['k-means', m, a])
        
        zh = wrapper.gmm(k, X)
        a = metric.accuracy(z, zh)
        r.append(['gmm', m, a])
        
        zh = wrapper.spectral_clustering(k, X, G)
        a = metric.accuracy(z, zh)
        r.append(['spectral clustering', m, a])
        
        zh = wrapper.kernel_kmeans(k, X, G)
        a = metric.accuracy(z, zh)
        r.append(['kernel k-means', m, a])
        
        
        zh = wrapper.kernel_kgroups(k, X, G)
        a = metric.accuracy(z, zh)
        r.append(['kernel k-groups', m, a])

df = pd.DataFrame(np.array(r), columns=['method', 'points', 'accuracy'])
df.to_csv(output, index=False)
