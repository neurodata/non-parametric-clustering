from __future__ import division

import numpy as np
from numpy import linalg as LA
from sklearn.cluster import KMeans


class EnergySpectral:

    def __init__(self, n_clusters, kernel=None):
        self.k = n_clusters

    def fit_predict(self, X):
        self.X = X
        self.n = len(X)
        self.G = self.gram_matrix()
        self.U = self.eigen_matrix()
        self.labels = self.k_means()
        return self.labels

    def energy_kernel(self, x, y):
        """Energy stats kernel."""
        return np.linalg.norm(x) + np.linalg.norm(y) - np.linalg.norm(x-y)

    def gram_matrix(self):
        """Generate Gram matrix from kernel function."""
        G = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i+1, self.n):
                G[i,j] = G[j,i] = self.energy_kernel(self.X[i], self.X[j])
        return G

    def eigen_matrix(self):
        """Form a normalized matrix with first k eigenvector of Gram matrix."""
        eigenvals, eigenvecs = LA.eigh(self.G)
        # V is a matrix with first k eigenvectors as columns
        V = eigenvecs.T[:,:k]
        # U contains the normalized rows of V
        C = np.diag(1./np.sqrt((V**2).sum(axis=1)))
        U = C.dot(V)
        return U

    def k_means(self):
        """Apply k-means to matrix of eigenvectors."""
        km = KMeans(n_clusters=self.k)
        z = km.fit_predict(self.U)
        return z


if __name__ == "__main__":
    import data
    from evaluation import accuracy
    
    m1 = np.array([0,0])
    s1 = np.array([[1,0],[0,1]])
    n1 = 300

    m2 = np.array([4,0])
    s2 = np.array([[4,1],[1,2]])
    n2 = 300

    X, z = data.multi_gaussians([m1,m2], [s1,s2], [n1,n2])

    k = 2

    ec = EnergySpectral(n_clusters=k)
    zh = ec.fit_predict(X)
    print "Energy Clustering:", accuracy(z, zh)

