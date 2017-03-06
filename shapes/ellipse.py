#!/usr/bin/env python

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt


def parse_ellipse(N, n, k, fname):
    """Parse data generated from mathematica. Each data point is a
    matrix with n rows and k columns. There are N data points.
    
    """
    f = open(fname)
    data = f.read().strip().split('}')
    data = [d.strip('{').strip('\t{').strip('\n{').split(', ') for d in data]
    data = [[float(d[0]), float(d[1])] for d in data if d != ['']]
    x = np.array(data)
    x = x.reshape((N, n, k))
    return x

def plot_data(data, ax, color=['b', 'r'], s=['o', 'd']):
    for i, P in enumerate(data):
        xs = P[0,:]
        ys = P[1,:]
        ax.plot(xs, ys, s[i], color=color[i], markersize=10, alpha=.5)
        ax.plot(xs[0], ys[0], 's', color=color[i], markersize=10, alpha=.5)


###############################################################################
if __name__ == '__main__':
    x1 = parse_ellipse(30, 20, 2, "data/cluster1.dat")
    x2 = parse_ellipse(30, 20, 2, "data/cluster2.dat")
    x3 = parse_ellipse(30, 20, 2, "data/cluster3.dat")
    print x1.shape
    print x2.shape
    print x3.shape

