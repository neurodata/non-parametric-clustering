"""Experiment for high dimensional Gaussians, here we change the covariance."""

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

parser = argparse.ArgumentParser(description="High dimensional Gaussian"\
                                             "changing the covariance.")
parser.add_argument('-o', dest='output', required=True,
                    type=str, action='store', help="file name for output")
parser.add_argument('-n', type=int, required=True,
                    dest='num_experiments', action='store', 
                    help="number of experiments")

args = parser.parse_args()

dimensions = range(10, 725, 25)
k = 2
total_points = 200
output = args.output
num_experiments = args.num_experiments

def generate_data(D):
    d = 10
    m1 = np.zeros(D)
    s1 = np.eye(D)
    m2 = np.concatenate((np.ones(d), np.zeros(D-d)))
    s2_1 = np.array([1.367,  3.175,  3.247,  4.403,  1.249,
                     1.969, 4.035,   4.237,  2.813,  3.637])
    s2 = np.diag(np.concatenate((s2_1, np.ones(D-d))))
    n1, n2 = np.random.multinomial(total_points, [0.5, 0.5])
    X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])
    return X, z

r = []
for _ in range(num_experiments):
    for dim in dimensions:
        X, z = generate_data(dim)
        G = eclust.kernel_matrix(X, lambda x, y: np.linalg.norm(x-y))
        
        zh = wrapper.kmeans(k, X)
        a = metric.accuracy(z, zh)
        r.append(['k-means', dim, a])
        
        zh = wrapper.gmm(k, X)
        a = metric.accuracy(z, zh)
        r.append(['gmm', dim, a])
        
        zh = wrapper.spectral_clustering(k, X, G)
        a = metric.accuracy(z, zh)
        r.append(['spectral clustering', dim, a])
        
        zh = wrapper.kernel_kmeans(k, X, G)
        a = metric.accuracy(z, zh)
        r.append(['kernel k-means', dim, a])
        
        
        zh = wrapper.kernel_kgroups(k, X, G)
        a = metric.accuracy(z, zh)
        r.append(['kernel k-groups', dim, a])

df = pd.DataFrame(np.array(r), columns=['method', 'dimension', 'accuracy'])
df.to_csv(output, index=False)
