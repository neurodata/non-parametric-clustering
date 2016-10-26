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

def ellipse_random(a, b, x0, y0, alpha, s, n):
    """Add random noise to the points of the ellipse."""
    angles = [2*np.pi*i/n for i in range(n)]
    points = np.array([[a*np.cos(t)+0.08*np.random.randn(),
                        b*np.sin(t)+0.08*np.random.randn()] for t in angles])
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

def plot_shape(data, ax, title=''):
    ax.plot(data[:,0], data[:,1], '-o')
    ax.set_title(title)

def plot_cluster_shapes(data, ax):
    for d in data:
        plot_shape(d, ax)


###############################################################################
if __name__ == '__main__':

    from distance import procrustes

    """
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    data = cluster_shape(1, 2, 30, 20, ellipse)
    plot_cluster_shapes(data, ax)
    ax = fig.add_subplot(122)
    data = cluster_shape(1, 2, 30, 20, rectangle)
    plot_cluster_shapes(data, ax)
    plt.show()
    """

    ###########################################################################
    # showing how it works for ellipses
    ###########################################################################
    pairs = [[1,3], [2,5]]
    
    data = []
    labels = []
    abnoise = []
    z = 0
    for a, b in pairs:
        for i in range(100):
            #a = a + 0.05*np.random.uniform(-1, 1)
            #b = b + 0.05*np.random.uniform(-1, 1)
            x0, y0 = 2*np.random.randn(2)
            alpha = np.random.uniform(0, 2*np.pi)
            s = 2*np.random.randn()
            n = 25
            data.append(ellipse_random(a, b, x0, y0, alpha, s, n))
            abnoise.append([a,b])
            labels.append(z)
        z += 1
    data = np.array(data)
    abnoise = np.array(abnoise)
    labels = np.array(labels)
    idx = range(len(labels))
    np.random.shuffle(idx)
    data = data[idx]
    abnoise = abnoise[idx]
    labels = labels[idx]

    pairss = [[0,0], [1,1], [0,1]]
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,12))
    for (a, b), row in zip(pairss, axes):
        
        ax1, ax2, ax3 = row

        i = np.random.choice(np.where(labels==a)[0])
        j = np.random.choice(np.where(labels==b)[0])
        X = data[i]
        Y = data[j]
        
        ax1.plot(X[:,0], X[:,1], 'o-b', alpha=.7)
        ax1.plot(Y[:,0], Y[:,1], 'o-r', alpha=.7)
        r1 = pairs[a][0]/pairs[a][1]
        r11 = abnoise[i][0]/abnoise[i][1]
        r2 = pairs[b][0]/pairs[b][1]
        r21 = abnoise[j][0]/abnoise[j][1]
        ax1.set_title(r'$(%.3f, %.3f) \to (%.3f, %.3f)$'%(r1, r2, r11, r21))

        Qh, Q, d = procrustes(X, Y, fullout=True, cycle=False)
        ax2.plot(Qh[:,0], Qh[:,1], 'o-b', alpha=.7)
        ax2.fill(Qh[:,0], Qh[:,1], 'b', alpha=.3)
        ax2.plot(Q[:,0], Q[:,1], 's-r', alpha=.7)
        ax2.fill(Q[:,0], Q[:,1], 'r', alpha=.3)
        ax2.set_title(r'$D(X,Y)=%f$ (no cycle)'%d)
        ax2.set_xlim([-0.4,0.4])
        ax2.set_ylim([-0.4,0.4])
        
        Qh, Q, d = procrustes(X, Y, fullout=True, cycle=True)
        ax3.plot(Qh[:,0], Qh[:,1], 'o-b', alpha=.7)
        ax3.fill(Qh[:,0], Qh[:,1], 'b', alpha=.3)
        ax3.plot(Q[:,0], Q[:,1], 's-r', alpha=.7)
        ax3.fill(Q[:,0], Q[:,1], 'r', alpha=.3)
        ax3.set_title(r'$D(X,Y)=%f$ (cycle)'%d)
        ax3.set_xlim([-0.4,0.4])
        ax3.set_ylim([-0.4,0.4])

    fig.savefig('ellipse_noise2.pdf')

