"""Kernel k-means for energy statistics."""

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans

import kmeanspp
from energy import energy_kernel


class KernelEnergy(BaseEstimator, ClusterMixin):
    """Kernel k-means with Energy Statistics."""

    def __init__(self, n_clusters, max_iter=50, tol=1e-3, random_state=None,
                 kernel=energy_kernel, gamma=None, degree=3, coef0=1,
                 kernel_params={'alpha': 1}, verbose=0, 
                 init='kmeans++', labels=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.verbose = verbose
        self.init = init
        self.labels_ = labels
        
    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        n_samples = X.shape[0]
        K = pairwise_kernels(X, Y, metric=self.kernel,
                             filter_params=True, **params)
        self.kernel_matrix_ = K
        return K

    def fit(self, X, y=None, sample_weight=None):
        n_samples = X.shape[0]

        K = self._get_kernel(X)
        
        sw = sample_weight if sample_weight else np.ones(n_samples)
        self.sample_weight_ = sw

        # initialization
        if self.labels_ is not None:
            pass
        elif self.init == 'kmeans++':
            self.labels_ = kmeanspp.kpp(self.n_clusters, X)
        else:
            rs = check_random_state(self.random_state)
            self.labels_ = rs.randint(self.n_clusters, size=n_samples)

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
        K = self._get_kernel(X, self.X_fit_)
        n_samples = X.shape[0]
        dist = np.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist, self.within_distances_,
                           update_within=False)
        return dist.argmin(axis=1)


def qcqp_tilde(k, X):
    # compute G ....
    z = kmeanspp.kpp(k, X)
    n = X.shape[0]
    
    t = 0
    converged = False
    while not converged:
    
        for i in range(n):
            
            for j in range(k):
                
            z[i] = 
    
    

###############################################################################
if __name__ == '__main__':
    import energy
    import data
    from metric import accuracy
    from sklearn.cluster import KMeans

    X, z = data.multivariate_normal(
        [[0,0], [2,0]], 
        [np.eye(2), np.eye(2)],
        [100, 100]
    )

    kernel = energy.energy_kernel
    km = KernelEnergy(n_clusters=2, max_iter=100, verbose=1, 
                      kernel_params={'alpha':.8})
    zh = km.fit_predict(X)
    print accuracy(z, zh)
    
    km = KMeans(n_clusters=2)
    zh = km.fit_predict(X)
    print accuracy(z, zh)

