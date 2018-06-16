"""Experiments for one dimensional data."""

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

parser = argparse.ArgumentParser(description="One dimensional data.")
parser.add_argument('-o', dest='output', required=True,
                    type=str, action='store', help="file name for output")
parser.add_argument('-t', dest='type', required=True, 
                    choices=['gauss', 'loggauss'],
                    type=str, action='store', help="can be gauss or loggauss")
parser.add_argument('-n', type=int, required=True,
                    dest='num_experiments', action='store', 
                    help="number of experiments")

args = parser.parse_args()

number_points = range(10, 800, 20)
k = 2
output = args.output
num_experiments = args.num_experiments
type_dist = args.type

def generate_data(total_points):
    m1 = 0
    s1 = 1.5
    m2 = 1.5
    s2 = 0.3
    n1, n2 = np.random.multinomial(total_points, [0.5, 0.5])
    if type_dist == 'gauss':
        X, z = data.univariate_normal([m1, m2], [s1, s2], [n1, n2])
    elif type_dist == 'loggauss':
        X, z = data.univariate_lognormal([m1, m2], [s1, s2], [n1, n2])
    Y = np.array([[x] for x in X])
    return Y, z

r = []
for _ in range(num_experiments):
    for n in number_points:
        X, z = generate_data(n)
        G = eclust.kernel_matrix(X, lambda x, y: np.linalg.norm(x-y))
        
        zh = wrapper.kmeans(k, X)
        a = metric.accuracy(z, zh)
        r.append(['k-means', n, a])
        
        zh = wrapper.gmm(k, X)
        a = metric.accuracy(z, zh)
        r.append(['gmm', n, a])
        
        zh = wrapper.kernel_kgroups(k, X, G)
        a = metric.accuracy(z, zh)
        r.append(['kernel k-groups', n, a])

df = pd.DataFrame(np.array(r), columns=['method', 'num_points', 'accuracy'])
df.to_csv(output, index=False)

