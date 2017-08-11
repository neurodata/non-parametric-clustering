"""
Generate Data
=============

Some usefull functions to generate or manipulate data.
For instance, generate data from multivariate normal distribution,
circles, lognormal, etc. Also some functions to plot data for
visualization.

"""

# Guilherme Franca <guifranca@gmail.com>
# Johns Hopkins Unversity, Neurodata


from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import numbers
import pylab

from sklearn.decomposition import PCA

from pplotcust import *


def multivariate_normal(means, sigmas, ns):
    """Generate several normal distributed data points.
    
    Input: 
        means = [m1, m2, ...]
        sigmas = [s1, s2, ...]
        ns = [n1, n2, ...]
    
    Output: data X and labels z.

    Each parameter in these lists are associated to one multivariate
    normal distribution. The data is shuffled so points are not ordered.
    
    """
    n = len(means)
    X = []
    z = []
    k = 0
    for mu, sigma, n in zip(means, sigmas, ns):
        X.append(np.random.multivariate_normal(mu, sigma, n))
        z.append([k]*n)
        k += 1
    X = np.concatenate(X)
    z = np.concatenate(z)
    idx = np.random.permutation(sum(ns))
    return X[idx], z[idx]

def univariate_normal(means, sigmas, ns):
    """Generate several univariate normal distributed data points.
    
    Input: 
        means = [m1, m2, ...]
        sigmas = [s1, s2, ...]
        ns = [n1, n2, ...]
    
    Output: data X and labels z.
    
    """
    n = len(means)
    X = []
    z = []
    k = 0
    for mu, sigma, n in zip(means, sigmas, ns):
        X.append(np.random.normal(mu, sigma, n))
        z.append([k]*n)
        k += 1
    X = np.concatenate(X)
    z = np.concatenate(z)
    idx = np.random.permutation(sum(ns))
    return X[idx], z[idx]

def univariate_lognormal(means, sigmas, ns):
    """Return data according to log normal distribtion.

    Input: means = [mu1, mu2, ...], 
           sigmas = [sigma1, sigma2, ...], 
           ns = [n1, n2, ...]
    
    Ouput: Data matrix X with corresponding labels z
    
    """
    x, z = univariate_normal(means, sigmas, ns)
    return np.exp(x), z
    
def multivariate_lognormal(means, sigmas, ns):
    """Return data according to log normal distribtion.

    Input: means = [mu1, mu2, ...], 
           sigmas = [sigma1, sigma2, ...], 
           ns=[n1, n2, ...]
    
    Ouput: Data matrix X with corresponding labels z
    
    """
    x, z = multivariate_normal(means, sigmas, ns)
    return np.exp(x), z

def circles(rs, ms, eps, ns):
    """Generate concentric circles in 2D with gaussian noise.
    
    Input: rs = [r1, r2, ...] radius
           ms = [m_1, m_2, ...] center of the circles
           eps = [epsilon1, epsilon2, ...] perturbation
           ns = [n1, n2, ...] number of points
    
    Output: data matrix X with labels z
    
    """
    k = 0
    X = []
    z = []
    for r, m, e, n in zip(rs, ms, eps, ns):
        x = np.array([(r*np.cos(a)+m[0], r*np.sin(a)+m[1])
                        for a in np.random.uniform(0, 2*np.pi, n)])
        x += e*np.random.multivariate_normal([0,0],np.eye(2), n)
        X.append(x)
        z.append([k]*n)
        k += 1
    X = np.concatenate(X)
    z = np.concatenate(z)
    idx = np.random.permutation(sum(ns))
    return X[idx], z[idx]

def spirals(rs, ms, ns, noise=0.2):
    """Generate spirals in 2D
    
    Input: rs = [r1, r2, ...] radius
           ms = [m_1, m_2, ...] centers
           ns = [n1, n2, ...] number of points
           noise = size of guassian noise
    
    Output: data matrix X with labels z
    
    """
    X = []
    z = []
    for k, (r, m, n) in enumerate(zip(rs, ms, ns)):
        ts = np.linspace(0, 4*np.pi, n) 
        x = np.array([[r*t*np.cos(t)+noise*np.random.normal(0,1)+m[0],
                       r*t*np.sin(t)+noise*np.random.normal(0,1)+m[1]] 
                       for t in ts])
        X.append(x)
        z.append([k]*n)
    X = np.concatenate(X)
    z = np.concatenate(z)
    idx = np.random.permutation(sum(ns))
    return X[idx], z[idx]

def plot(X, z, fname='plot.pdf'):
    """Plot data according to labels in z.
    Use PCA to project in 2D in case data is higher dimensional.

    """

    z_unique = np.unique(z)

    if len(X[0]) > 2:
        pca = PCA(n_components=2)
        X_new = pca.fit_transform(X)
    else:
        X_new = X

    fig = plt.figure()
    ax = fig.add_subplot(111)

    colors = iter(plt.cm.brg(np.linspace(0,1,5)))
    
    for k in z_unique:
        idx = np.where(z==k)
        x = X_new[idx][:,0]
        y = X_new[idx][:,1]
        ax.plot(x, y, 'bo', markersize=4, alpha=.7, color=next(colors))
    
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axes().set_aspect('equal', 'datalim')
    fig.tight_layout()
    fig.savefig(fname)

def histogram(X, z, colors=['#1F77B4', '#FF7F0E'], fname='plot.pdf',
              xlim=None, bins=500):
    """Plot histograms of 1-dimensional data.
    
    Input: X = data matrix
           z = class labels
           color = color for each class
           fname = output file

    Output: None

    """
    z_unique = np.unique(z)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = iter(colors)
    for k in z_unique:
        idx = np.where(z==k)[0]
        x = X[idx]
        ax.hist(x, bins=bins, facecolor=next(colors), 
                histtype='stepfilled',
                alpha=.8, normed=1, linewidth=0.5)
    ax.set_xlabel(r'$x$')
    if xlim:
        ax.set_xlim(xlim)
    plt.tick_params(top='off', bottom='on', left='off', right='off',
            labelleft='off', labelbottom='on')
    for i, spine in enumerate(plt.gca().spines.values()):
        if i !=2:
            spine.set_visible(False)
    frame = plt.gca()
    frame.axes.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(fname)

def histogram_gauss_loggauss(xlim=None, ylim=None, fname='plot.pdf'):
    
    def gauss_func(x, mu, sigma):
        return 1/(np.sqrt(2*np.pi*sigma**2))*np.exp(-(x-mu)**2/(2*sigma**2))

    def loggauss_func(x, mu, sigma):
        return 1/(np.sqrt(2*np.pi*sigma**2)*x)*np.exp(
                    -(np.log(x)-mu)**2/(2*sigma**2))

    colors = iter(plt.cm.brg(np.linspace(0,1,6)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    xs = np.arange(-5, 15, 0.001)
    #xs = np.arange(0.0001, 5, 0.001)
    y1s = np.array([gauss_func(x, 0, 1) for x in xs])
    y2s = np.array([gauss_func(x, 5, 2) for x in xs])
    #y1s = np.array([loggauss_func(x, 0, 0.3) for x in xs])
    #y2s = np.array([loggauss_func(x, -1.5, 1.5) for x in xs])
    c = next(colors)
    ax.plot(xs, y1s, '-', color=c, linewidth=1)
    ax.fill_between(xs, 0, y1s, color=c, alpha=0.5)
    c = next(colors)
    ax.plot(xs, y2s, '-', color=c, linewidth=1)
    ax.fill_between(xs, 0, y2s, color=c, alpha=0.5)

    ax.set_xlabel(r'$x$')
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    plt.tick_params(top='off', bottom='on', left='off', right='off',
            labelleft='off', labelbottom='on')
    for i, spine in enumerate(plt.gca().spines.values()):
        if i !=2:
            spine.set_visible(False)
    frame = plt.gca()
    frame.axes.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(fname)

def shuffle_data(X):
    """Shuffle data and return corresponding labels.

    Input: X = [X1, X2, ...]
    Output: data matrix X and labels z
    
    """
    ns = []
    zs = []
    for i, x in enumerate(X):
        n = len(x)
        ns.append(n)
        zs.append([i]*n)
    z = np.concatenate(zs)
    Y = np.concatenate(X)
    idx = np.random.permutation(sum(ns))
    return Y[idx], z[idx]

def mix_data(A, m):
    """Mix data between subsets of A.

    Input: A = [A_1, A_2, ..., A_n] where each A_i is a subset of data
            m is an integer
    Output: Mix m points from each class and returns a new data matrix X
    
    """
    indices = [np.random.choice(range(len(a)), m*(len(A)-1), replace=False) 
                for a in A]
    X = np.copy(A)
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            X[j][indices[j][i*m:(i+1)*m]] = A[i][indices[i][(j-1)*m:j*m]]
            X[i][indices[i][(j-1)*m:j*m]] = A[j][indices[j][i*m:(i+1)*m]]
    return X

def from_label_to_sets(X, z):
    """Given a data matrix X with corresponding labels z return a list
    [A_1, A_2, ...] where each set correspond to one label (cluster).
    
    """
    A = []
    for label in np.unique(z):
        A.append(X[np.where(z==label)])
    return A

def from_sets_to_labels(A):
    """Given A = [A_1, A_2, ...] return a data matrix with labels: X, z."""
    return shuffle_data(A)



###############################################################################
if __name__ == '__main__':

    #X, z = univariate_normal([0, 5], [1, 2], [80000,80000])
    #X, z = univariate_lognormal([0, -1.5], [0.3, 1.5], [80000, 80000])
    #X, z = univariate_normal([0, 5], [1,1], [80000,80000])
    #X, z = univariate_lognormal([0, 0.5], [0.8,0.05], [80000,80000])
    #histogram(X, z, colors=['#434B9E', '#AC393D'], fname='hist_normal.pdf')
    #histogram(X, z, colors=['g', 'k'], fname='hist_normal.pdf', bins=100)
    #histogram(X, z, colors=['#434B9E', '#AC393D'], fname='hist_lognormal.pdf',
    #          xlim=[0, 4.5])
    #histogram(X, z, colors=['g', 'k'], fname='hist_lognormal.pdf', bins=300,
    #    xlim=[0, 5])
    #histogram_gauss_loggauss(xlim=[-5,15], fname='hist_normal.pdf')
    #histogram_gauss_loggauss(xlim=[0,3], ylim=[0,3.7], 
    #                         fname='hist_lognormal.pdf')

    # cigars
    #m1 = np.zeros(2)
    #m2 = np.array([6.5,0])
    #s1 = s2 = np.array([[1,0],[0,20]])
    #X, z = multivariate_normal([m1, m2], [s1, s2], [200, 200])
    #plot(X, z, fname='./2cigars.pdf')
    
    # 2 circles
    X, z = circles([1, 3], [[0,0], [0,0]], [0.2, 0.2], [400, 400])
    plot(X, z, fname='./2circles.pdf')
    
    # 3 circles
    X, z = circles([1, 3, 5], [[0,0], [0,0], [0,0]], [0.2, 0.2, 0.2], 
                    [400, 400, 400])
    plot(X, z, fname='./3circles.pdf')

    #X, z = spirals([1,-1], [[0.2,0.0], [-0.2,-0.0]], [400,400], noise=0.2)
    #plot(X, z, fname='./2spiral.pdf')
    
