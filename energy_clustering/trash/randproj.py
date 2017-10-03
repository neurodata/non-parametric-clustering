"""We test clustering using 1D random projections. Only 2 clusters."""

# Guilherme Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata


from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from sklearn import random_projection
from sklearn.cluster import KMeans
from sklearn import decomposition
from sklearn.metrics import silhouette_score

from metric import accuracy
import energy1d
import data


def kmeans(X):
    kmeans = KMeans(n_clusters=2).fit(X)
    z = kmeans.labels_
    score = -kmeans.score(X)
    return z, score

def rand_proj(X):
    transf = random_projection.GaussianRandomProjection(n_components=1)
    return transf.fit_transform(X)

def pca_proj(X):
    pca = decomposition.PCA(n_components=1)
    pca.fit(X)
    return pca.transform(X)

def kmeans_random(X, n=50):
    best_score = np.inf
    for i in range(n):
        Y = rand_proj(X)
        zh, score = kmeans(Y)
        if score < best_score:
            best_z = zh
            best_score = score
    return best_z

def energy_random(X, n=50):
    best_score = np.inf
    for i in range(n):
        Y = rand_proj(X).flatten()
        zh, score = energy1d.two_clusters1D(Y)
        if score < best_score:
            best_z = zh
            best_score = score
    return best_z
    
def test(X):
    zh, score = kmeans(X)
    a = accuracy(z, zh)
    print "k-means original space:", a

    Y = pca_proj(X)
    zh, score = kmeans(Y)
    a = accuracy(z, zh)
    print "k-means/1D PCA:", a

    zh = kmeans_random(X, 20)
    a = accuracy(z, zh)
    print "k-means/random projection 1D:", a

    zh = energy_random(X, 20)
    a = accuracy(z, zh)
    print "energy/random projection 1D", a


###############################################################################
if __name__ == '__main__':

    m1 = np.array([0,0])
    s1 = np.array([[.5,-.8],[-.8,15]])
    n1 = 300
    m2 = np.array([2,1])
    s2 = np.array([[15,1],[1,1]])
    n2 = 300
    X, z = data.multivariate_normal([m1,m2], [s1,s2], [n1,n2])

    test(X)

    
    """
    d = 30
    n = 1000
    m1 = np.zeros(d)
    m1[range(0,d,2)] = 1
    s1 = np.eye(d)
    m2 = np.zeros(d)
    m2[range(0,d,2)] = -1
    s2 = np.eye(d)
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
    """
