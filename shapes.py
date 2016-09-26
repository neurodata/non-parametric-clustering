#!/usr/bin/env python

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt


def ellipse(a, b, x0, y0, alpha, s, n):
    """Generate ellipse.
    a, b are the axis
    alpha is the rotated angle
    s is a rescaling
    n it the number of landmark points
    
    """
    angles = [2*np.pi*i/n for i in range(n)]
    points = np.array([[a*np.cos(t), b*np.sin(t)] for t in angles])
    R = np.array([
        [np.cos(alpha), np.sin(alpha)], 
        [-np.sin(alpha),  np.cos(alpha)]
    ])
    points = s*points.dot(R) + np.array([[x0, y0]])
    return points

def rectangle(a, b, x0, y0, alpha, s, n):
    """Plot rectangle.
    a, b are the axis
    alpha is the rotated angle
    s is a rescaling
    n it the number of landmark points
    
    """
    b += 5/(n/4)
    side1 = np.array([[a*i/(n/4),0] for i in range(int(n/4))])
    side2 = np.array([[a,b*i/(n/4)] for i in range(int(n/4))])
    side3 = side1[::-1] + np.array([[0, side2[-1,1]]])
    side4 = side2 + np.array([[-a, 0]])
    points = np.concatenate((side1, side2, side3, side4))
    R = np.array([
        [np.cos(alpha), np.sin(alpha)], 
        [-np.sin(alpha),  np.cos(alpha)]
    ])
    b -= 5/(n/4)
    points = s*points.dot(R) + np.array([[x0-a/2, y0-b/2]])
    return points

def cluster_shape(a, b, n, m, shape_func):
    """Plot a cluster of ellipses by translating, rotating and 
    rescaling. n is the number of data points, m the number of points
    on each ellipse.
    
    """
    t = 50*np.random.rand(n, 2)
    s = 80*np.random.rand(n)
    alpha = np.random.uniform(0, 2*np.pi, n)
    data = np.empty((n, m, 2))
    for i in range(n):
        data[i] = shape_func(a, b, t[i,0], t[i,1], alpha[i], s[i], m)
    return data

def plot_shape(data, ax):
    ax.plot(data[:,0], data[:,1], '-o')

def plot_cluster_shapes(data, ax):
    for d in data:
        plot_shape(d, ax)


###############################################################################
if __name__ == '__main__':
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    data = cluster_shape(1, 2, 30, 20, ellipse)
    plot_cluster_shapes(data, ax)
    ax = fig.add_subplot(122)
    data = cluster_shape(1, 2, 30, 20, rectangle)
    plot_cluster_shapes(data, ax)
    plt.show()
    
