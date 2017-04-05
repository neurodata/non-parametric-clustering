"""
Several tests and comparison for clustering.

"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GMM

from kkmeans import KernelKMeans
from evaluation import accuracy
import data
import spectral


def poly(x, y):
    return np.square(x.dot(y)+.1)

def gauss(x, y):
    return np.exp(-np.linalg.norm(x-y)**2)

def energy_kernel(x, y):
    return np.linalg.norm(x) + np.linalg.norm(y) - np.linalg.norm(x-y)


if __name__ == '__main__':
    
    m1 = np.array([0,0])
    s1 = np.array([[1,0],[0,1]])
    n1 = 500
    
    m2 = np.array([4,0])
    s2 = np.array([[1,0],[0,20]])
    n2 = 500
    
    m3 = np.array([0,6])
    s3 = np.array([[2,1],[1,2]])
    n3 = 200

    #X, z = data.multi_gaussians([m1,m2,m3], [s1,s2,s3], [n1,n2,n3])
    #X, z = data.multi_gaussians([m1,m2], [s1,s2], [n1,n2])
    X, z = data.circles([2, 0.5], [0.2, 0.1], [400, 400])
    
    k = 2

    #data.plot(X, z, 'blob_cigar.pdf')
    data.plot(X, z, 'circles.pdf')

    km = KernelKMeans(n_clusters=k, max_iter=300, kernel=energy_kernel, 
                        init='randproj')
    zh = km.fit_predict(X)
    print "Kernel k-means/Energy:", accuracy(z, zh)
    
    #sc = SpectralClustering(n_clusters=k, affinity=energy_kernel,
    #                        assign_labels='kmeans', n_init=20)
    #zh = sc.fit_predict(X)
    #print "Spectral Clustering/Energy:", accuracy(z, zh)
    
    km = KernelKMeans(n_clusters=k, max_iter=300, kernel=gauss)
    zh = km.fit_predict(X)
    print "Kernel k-means/Gauss:", accuracy(z, zh)
    
    km = KernelKMeans(n_clusters=k, max_iter=300, kernel=poly)
    zh = km.fit_predict(X)
    print "Kernel k-means/Polynomial:", accuracy(z, zh)
   
    #G = spectral.gram_matrix(X, spectral.energy_kernel)
    #sc = SpectralClustering(n_clusters=2, affinity='precomputed')
    #zh = sc.fit_predict(G)
    #print "Spectral Clustering /Precomputed Energy:", accuracy(z, zh)
    
    gmm = GMM(n_components=k)
    gmm.fit(X)
    zh = gmm.predict(X)
    print "GMM:", accuracy(z, zh)

    km = KMeans(n_clusters=k)
    zh = km.fit_predict(X)
    print "k-means:", accuracy(z, zh)

