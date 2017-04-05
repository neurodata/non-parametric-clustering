"""
We test clustering using 1D random projections.

"""


from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from sklearn import random_projection
from sklearn.cluster import KMeans
from sklearn import decomposition
from sklearn.metrics import silhouette_score

from evaluation import accuracy
import energy

import sys


def kmeans(X):
    kmeans = KMeans(n_clusters=2).fit(X)
    z = kmeans.labels_
    #mu = kmeans.cluster_centers_
    #J = -kmeans.score(X)
    return z

def kmeans_function(X, z):
    J = 0
    for k in np.unique(z):
        idx = np.where(z==k)
        Xks = X[idx]
        muk = Xks.mean(axis=0)
        J += (((Xks - muk)**2).sum())
    n = X.shape[0]
    d = muk.shape[0]
    return J

def two_gaussians(m1, s1, m2, s2, n):
    """"Generate data from two Gaussians."""

    X1 = np.random.multivariate_normal(m1, s1, n)
    z1 = [0]*n

    X2 = np.random.multivariate_normal(m2, s2, n)
    z2 = [1]*n

    X = np.concatenate((X1,X2))
    z = np.concatenate((z1,z2))

    idx = np.random.permutation(2*n)
    X = X[idx]
    z = z[idx]

    return X, z

def rand_projection(X):
    transf = random_projection.GaussianRandomProjection(n_components=1)
    return transf.fit_transform(X)

def pca_projection(X):
    pca = decomposition.PCA(n_components=1)
    pca.fit(X)
    return pca.transform(X)

def kmeans_multi_random(X, n=10):
    bestJ = np.inf
    for i in range(n):
        Y = rand_projection(X)
        zh = kmeans(Y)
        J = kmeans_function(X, z)
        if J < bestJ:
            bestz = zh
            bestJ = J
    return bestz

def energy_multi_random(X, n=10):
    bestT = 0
    for i in range(n):
        Y = rand_projection(X).flatten()
        idx = np.argsort(Y)
        n = int(len(idx)/2)
        zh = np.zeros(len(Y), dtype=np.int)
        zh[idx[n:]] = 1
        A = Y[idx[:n]]
        B = Y[idx[n:]]
        T = energy.fastT(A, B)
        #T = energy.T(X[idx[:n]], X[idx[n:]])
        if T > bestT:
            bestT = T
            bestz = zh
    return bestz

def plot(X, z, fname='plot.pdf'):
    n = len(X[0])
    if n > 2:
        pca = decomposition.PCA(n_components=2)
        pca.fit(X)
        Xp = pca.transform(X)
    else:
        Xp = X
    id1 = np.where(z==0)
    id2 = np.where(z==1)
    x1 = Xp[id1]
    x2 = Xp[id2]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x1[:,0], x1[:,1], 'bo', alpha=.6)
    ax.plot(x2[:,0], x2[:,1], 'ro', alpha=.6)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axes().set_aspect('equal', 'datalim')
    fig.savefig(fname)


###############################################################################
if __name__ == '__main__':

    """
    n = 2000

    m1 = np.array([0,0])
    s1 = np.array([[.5,-.8],[-.8,15]])

    m2 = np.array([2,1])
    s2 = np.array([[15,1],[1,1]])
    """

    """
    d = 30
    n = 1000
    m1 = np.zeros(d)
    m1[range(0,d,2)] = 1
    s1 = np.eye(d)
    m2 = np.zeros(d)
    m2[range(0,d,2)] = -1
    s2 = np.eye(d)

    X, z = two_gaussians(m1,s1,m2,s2,n)

    plot(X, z, '1000d_gauss.pdf')
    """

    """
    zh = kmeans(X)
    a = accuracy(z, zh)
    print "k-means++:", a

    Y = pca_projection(X)
    zh = kmeans(Y)
    a = accuracy(z, zh)
    print "PCA/k-means++:", a

    zh = kmeans_multi_random(X, z, 30)
    a = accuracy(z, zh)
    print "Random / k-means++:", a

    zh = energy_multi_random(X, z, 30)
    a = accuracy(z, zh)
    print "Random / Energy:", a
    """

    for d in [5,10,15,20,25,30,50,100,200,300,500,1000,2000,5000]:
        n = 1000
        m1 = np.zeros(d)
        m1[range(0,d,2)] = 1
        s1 = np.eye(d)
        m2 = np.zeros(d)
        m2[range(0,d,2)] = -1
        s2 = np.eye(d)
        X, z = two_gaussians(m1,s1,m2,s2,n)

        zh = kmeans(X)
        a_kmeans = accuracy(z, zh)

        Y = pca_projection(X)
        zh = kmeans(Y)
        a_pca = accuracy(z, zh)

        zh = kmeans_multi_random(X, z, 100)
        a_krandom = accuracy(z, zh)

        zh = energy_multi_random(X, z, 100)
        a_erandom = accuracy(z, zh)

        print "%i & %f & %f & %f & %f \\\\" % (d, a_kmeans, a_pca,
                                             a_krandom, a_erandom)
