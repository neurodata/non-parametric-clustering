#!/usr/bin/env python

import numpy as np

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

if __name__ == '__main__':
    x1 = parse_ellipse(30, 20, 2, "data/cluster1.dat")
    x2 = parse_ellipse(30, 20, 2, "data/cluster2.dat")
    x3 = parse_ellipse(30, 20, 2, "data/cluster3.dat")
    print x1.shape
    print x2.shape
    print x3.shape
