"""Experiment for high dimensional Gaussians and LogGaussians
with varying number of point and different kernels.

"""

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

parser = argparse.ArgumentParser(description="High dimensional Gaussian/"\
            "LogGaussian with varying number of points and different kernels.")
parser.add_argument('-o', dest='output', required=True,
                    type=str, action='store', help="file name for output")
parser.add_argument('-t', type=str, required=True,
                    dest='type', action='store', 
                    choices=["normal", "lognormal"],
                    help="normal or lognormal distribution")
parser.add_argument('-n', type=int, required=True,
                    dest='num_experiments', action='store', 
                    help="number of experiments")

args = parser.parse_args()

num_points = range(10, 410, 10)
k = 2
D = 20
d = 5
output = args.output
num_experiments = args.num_experiments
distr_type = args.type

def generate_data(n):
    m1 = np.zeros(D)
    s1 = 0.5*np.eye(D)
    m2 = 0.5*np.concatenate((np.ones(d), np.zeros(D-d)))
    s2 = np.eye(D)
    n1, n2 = np.random.multinomial(n, [0.5, 0.5])
    if distr_type == 'normal':
        X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])
    elif distr_type == 'lognormal':
        X, z = data.multivariate_lognormal([m1, m2], [s1, s2], [n1, n2])
    return X, z

r = []
for _ in range(num_experiments):
    for n in num_points:
        X, z = generate_data(n)
        G1 = eclust.kernel_matrix(X, lambda x, y: np.linalg.norm(x-y))
        G2 = eclust.kernel_matrix(X, 
                lambda x, y: np.power(np.linalg.norm(x-y), 0.5))
        G3 = eclust.kernel_matrix(X, 
                lambda x, y: 2-2*np.exp(-np.linalg.norm(x-y)/2))
        
        zh = wrapper.kmeans(k, X)
        a = metric.accuracy(z, zh)
        r.append(['k-means', n, a])
        
        zh = wrapper.gmm(k, X)
        a = metric.accuracy(z, zh)
        r.append(['gmm', n, a])
        
        zh = wrapper.spectral_clustering(k, X, G3)
        a = metric.accuracy(z, zh)
        r.append([r'spectral clustering $\widetilde{\rho}_1$', n, a])
        
        zh = wrapper.kernel_kgroups(k, X, G1)
        a = metric.accuracy(z, zh)
        r.append([r'kernel k-groups $\rho_{1}$', n, a])
        
        zh = wrapper.kernel_kgroups(k, X, G2)
        a = metric.accuracy(z, zh)
        r.append([r'kernel k-groups $\rho_{1/2}$', n, a])
        
        zh = wrapper.kernel_kgroups(k, X, G3)
        a = metric.accuracy(z, zh)
        r.append([r'kernel k-groups $\widetilde{\rho}_{1}$', n, a])

df = pd.DataFrame(np.array(r), columns=['method', 'points', 'accuracy'])
df.to_csv(output, index=False)
