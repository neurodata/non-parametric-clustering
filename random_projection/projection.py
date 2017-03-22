
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from sklearn import random_projection
from sklearn.cluster import KMeans
from sklearn import decomposition

from evaluation import accuracy

import sys


def kmeans(X):
    kmeans = KMeans(n_clusters=2).fit(X)
    mu = kmeans.cluster_centers_
    z = kmeans.labels_
    J = -kmeans.score(X)
    return z, J

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

def energy1D(A, B):
    """Assume A and B are a sorted list of elements. Compute the energy
    distance in 1D in O(N).
    
    """
    nA = len(A)
    nB = len(B)
    sumA = 0
    sumB = 0
    i = j = 0
    while i < nA and j < nB:
        if A[i] <= B[j]:
            sumA += A[i]*(2*j-nB)/nB
            i += 1
        else:
            sumB += B[j]*(2*i-nA)/nA
            j += 1
    if i >= nA:
        sumB += B[j:nB].sum()
    else:
        sumA += A[i:nA].sum()
    return sumA/nA + sumB/nB

def energy(A, B):
    """Compute energy distance from standard formula."""
    sum = 0
    for x in A:
        for y in B:
           sum += np.linalg.norm(x-y)
    return sum/(len(A)*len(B))

def two_gaussians(d):
    """"Generate data from two Gaussians."""
    
    n1 = n2 = 1000

    m1 = np.zeros(d)
    s1 = np.eye(d)
    X1 = np.random.multivariate_normal(m1, s1, n1)
    z1 = [0]*n1
    
    m2 = m1
    m2[0] = 2
    s2 = s1
    X2 = np.random.multivariate_normal(m2, s2, n2)
    z2 = [1]*n2

    X = np.concatenate((X1,X2))
    z = np.concatenate((z1,z2))

    idx = np.random.permutation(n1+n2)
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

def kmeans_multi_random(X, z, n=500):
    besta= 0
    for i in range(n):
        Y = rand_projection(X)
        zh, J = kmeans(Y)
        a = accuracy(z, zh)
        if a > besta:
            bestz = zh
            besta = a
    return bestz, besta 
    
def plot_pca(X, z):
    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    Xpca = pca.transform(X)
    id1 = np.where(z==0)
    id2 = np.where(z==1)
    x1 = Xpca[id1]
    x2 = Xpca[id2]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x1[:,0], x1[:,1], 'bo', alpha=.6)
    ax.plot(x2[:,0], x2[:,1], 'ro', alpha=.6)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axes().set_aspect('equal', 'datalim')
    fig.savefig('30d_gauss.pdf')
    


###############################################################################
if __name__ == '__main__':
    
    n = 100
    A = np.random.normal(0,1,n)
    zA = [0]*n
    B = np.random.normal(2,1,n)
    zB = [1]*n
    
    X = np.concatenate((A,B))
    z = np.concatenate((zA,zB))

    idx = np.random.permutation(n)
    X = X[idx]
    z = z[idx]


    A.sort()
    B.sort()
    
    print energy1D(A, B)

    X.sort()
    print energy1D(X[:int(n/2)], X[int(n/2):])

    sys.exit()



    X, z = two_gaussians(5000)

    zh, J = kmeans(X)
    a = accuracy(z, zh)
    print a
    
    Y = pca_projection(X)
    zh, J = kmeans(Y)
    a = accuracy(z, zh)
    print a
    
    zh, a = kmeans_multi_random(X, z)
    print a
    
