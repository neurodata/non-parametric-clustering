"""Iterative algorithm to optimize QCQP for energy statistics clustering."""

# Guilherme Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata


from __future__ import division

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state

import kmeanspp


def ztoZ(z):
    """Convert label vector to label matrix."""
    n = z.shape[0]
    k = np.unique(z).shape[0]
    Z = np.zeros((n,k))
    for i in range(n):
        Z[i, int(z[i])] = 1
    return Z

def Ztoz(Z):
    """Convert label matrix to label vector."""
    n, k = Z.shape
    z = np.zeros(n)
    points, labels = np.where(Z==1)
    for i in range(len(points)):
        z[points[i]] = labels[i]
    return z

def objective(Z, G):
    """Compute objective function."""
    Zt = Z.T
    D = np.dot(Zt, Z)
    Dinv = np.linalg.inv(D)
    return np.trace(Dinv.dot(Zt.dot(G.dot(Z))))

def euclidean_rho(x, y, alpha=2):
    return np.power(np.linalg.norm(x-y), alpha)

def kernel_function(x, y, rho, x0=None):
    """Return kernel function based on rho."""
    if x0 == None:
        x0 = np.zeros(x.shape)
    return 0.5*(rho(x,x0) + rho(y,x0) - rho(x,y))

def kernel_matrix(X, rho, x0=None, diag=True):
    """Compute Kernel matrix based on kernel function K(x,y)."""
    kfunc = lambda x, y: kernel_function(x, y, rho, x0)
    G = pairwise_distances(X, metric=kfunc, n_jobs=4)
    if not diag:
        i = range(len(X))
        G[i,i] = 0
    return G

def kenergy_brute(k, G, Z0, max_iter=50):
    """Optimize energy QCQP in a brute force way. Kind of expensive,
    but actually works slightly better than Kernel Kmeans, at least
    it seems like that.
    """
    
    n = G.shape[0]
    Z = Z0
    count = 0
    converged = False
    while not converged and count < max_iter:
        converged = True
        for l in range(n):
            curr_label = np.where(Z[l]==1)[0][0]
            Ws = []
            for j in range(k):
                # for each point, we remove from its original partition
                # then append it to a new partition
                # and compute the objective function, later we
                # pick the maximum
                Znew = np.copy(Z)
                Znew[l, curr_label] = 0
                Znew[l, j] = 1
                Wnew = objective(Znew, G)
                Ws.append(Wnew)
            new_label = np.argmax(Ws)
            Z[l] = np.zeros(k)
            Z[l, new_label] = 1
            if new_label != curr_label:
                converged = False
        count += 1

    if count >= max_iter:
        print "Warning: k-energy didn't converge after %i iterations." % count
    
    return Ztoz(Z)

def kenergy(k, G, Z0, max_iter=50):
    """Optimize energy QCQP. Not working correctly!"""
    
    n = G.shape[0]
    Z = Z0
    count = 0
    converged = False
    while not converged and count < max_iter:
        converged = True
        for l in range(n):
            curr_partition= np.where(Z[l]==1)[0][0]
            Ws = []
            for j in range(k):
                
                # keep copies and change the partition of the point
                val = Z[l, curr_partition]
                Z[l, curr_partition] = 0
                val2 = Z[l, j]
                Z[l, j] = 1
                
                # number of points in the modified partition
                nj = Z[:,j].sum()
                
                # compute contribution to objective function
                H = G.dot(Z)[l,j]/nj
                Ws.append(H)
                
                # restore previous configuration
                Z[l, curr_partition] = val
                Z[l, j] = val2

            new_partition= np.argmax(Ws)
            Z[l] = np.zeros(k)
            Z[l, new_partition] = 1
            
            if new_partition != curr_partition:
                converged = False
        count += 1

    if count >= max_iter:
        print "Warning: k-energy didn't converge after %i iterations." % count
    
    return Ztoz(Z)


class KernelEnergy(BaseEstimator, ClusterMixin):
    """Kernel k-means with Energy Statistics."""

    def __init__(self, n_clusters, kernel_matrix, labels, 
                 max_iter=50, tol=1e-3, random_state=None, 
                 gamma=None, degree=3, coef0=1, verbose=0):
        self.n_clusters = n_clusters
        self.kernel = kernel_matrix
        self.labels_ = labels
        
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.kernel = kernel_matrix
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.verbose = verbose
        
    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self):
        K = self.kernel
        self.kernel_matrix_ = K
        return K

    def fit(self, X, y=None, sample_weight=None):
        n_samples = X.shape[0]

        K = self._get_kernel()
        
        sw = sample_weight if sample_weight else np.ones(n_samples)
        self.sample_weight_ = sw

        dist = np.zeros((n_samples, self.n_clusters))
        self.within_distances_ = np.zeros(self.n_clusters)

        for it in xrange(self.max_iter):
            dist.fill(0)
            self._compute_dist(K, dist, self.within_distances_,
                               update_within=True)
            labels_old = self.labels_
            self.labels_ = dist.argmin(axis=1)

            # Compute the number of samples whose cluster did not change 
            # since last iteration.
            n_same = np.sum((self.labels_ - labels_old) == 0)
            if 1 - float(n_same) / n_samples < self.tol:
                if self.verbose:
                    print "Converged at iteration", it + 1
                break

        self.X_fit_ = X

        return self

    def _compute_dist(self, K, dist, within_distances, update_within):
        """Compute a n_samples x n_clusters distance matrix using the 
        kernel trick."""
        sw = self.sample_weight_

        for j in xrange(self.n_clusters):
            mask = self.labels_ == j

            if np.sum(mask) == 0:
                raise ValueError("Empty cluster found, try smaller n_cluster.")

            denom = sw[mask].sum()
            denomsq = denom * denom

            if update_within:
                KK = K[mask][:, mask]  # K[mask, mask] does not work.
                dist_j = np.sum(np.outer(sw[mask], sw[mask]) * KK / denomsq)
                within_distances[j] = dist_j
                dist[:, j] += dist_j
            else:
                dist[:, j] += within_distances[j]

            dist[:, j] -= 2 * np.sum(sw[mask] * K[:, mask], axis=1) / denom

    def predict(self, X):
        K = self._get_kernel()
        n_samples = X.shape[0]
        dist = np.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist, self.within_distances_,
                           update_within=False)
        return dist.argmin(axis=1)



###############################################################################
if __name__ == '__main__':
    import data
    import metric
    from kmeans import kmeans
    
    for it in range(10):
        print 'Experiment', it

        X, z = data.multivariate_normal(
            [[0,0], [5,0]],
            [np.array([[1,0],[0,15]]), np.array([[1,0],[0,15]])],
            [100, 100]
        )
    
        alpha = 0.8
        rho = lambda x, y: euclidean_rho(x, y, alpha=alpha)
        G = kernel_matrix(X, rho)
    
        # initialization
        mu0, z0 = kmeanspp.kpp(2, X, ret='both')
        Z0 = ztoZ(z0)

        zh = kenergy_brute(2, G, Z0, 100)
        Zh = ztoZ(zh)
        print metric.accuracy(z, zh), objective(Zh, G)
    
        #zh = kenergy(2, G, Z0, 100)
        #Zh = ztoZ(zh)
        #print metric.accuracy(z, zh), objective(Zh, G)
    
        km = KernelEnergy(2, G, z0, max_iter=100)
        zh = km.fit_predict(X)
        Zh = ztoZ(zh)
        print metric.accuracy(z, zh), objective(Zh, G)

        zh = kmeans(2, X, labels_=z0, mus_=mu0)
        Zh = ztoZ(zh)
        print metric.accuracy(z, zh), objective(Zh, G)

        print
        
