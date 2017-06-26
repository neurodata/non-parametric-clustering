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

def kernel_function(x, y, rho, x0=None):
    """Return kernel function based on rho."""
    if x0 == None:
        x0 = np.zeros(x.shape)
    val = 0.5*(rho(x,x0) + rho(y,x0) - rho(x,y))
    return val

def kernel_matrix(X, rho, x0=None):
    """Compute Kernel matrix based on kernel function K(x,y)."""
    kfunc = lambda x, y: kernel_function(x, y, rho, x0)
    G = pairwise_distances(X, metric=kfunc, n_jobs=4)
    return G

def minw_brute(k, G, Z0, max_iter=50, tol=1e-4, verbose=False):
    """Optimize energy QCQP in a brute force way. Kind of expensive,
    but works. We move points to other partitions and compute the change
    in the cost function.

    """
    
    n = G.shape[0]
    Z = np.copy(Z0)
    count = 0
    converged = False
    while not converged and count < max_iter:
        
        n_changed = 0
        
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
                n_changed += 1
        
        if n_changed/n < tol:
            converged = True
        else:
            count += 1

    if verbose:
        if count >= max_iter:
            print "\tBrute force QCQP didn't converge after %i iterations."\
                        % count
        else:
            print "\tBrute force QCQP converged in %i iterations." % count
    
    return Ztoz(Z)

def minw(k, G, Z0, max_iter=100, tol=1e-4, verbose=False):
    """Optimize the W objective function by considering moving points
    to different partitions. Compute the change in the cost function by
    moving a point then decide the best partition to optimize the cost
    function. It's the same as above but in a more efficient way.
    
    """

    # initial configuration
    n = G.shape[0]
    Z = np.copy(Z0)
    Zt = Z.T
    D = Zt.dot(Z)
    Q_matrix = Zt.dot(G.dot(Z))
    m = D.dot(np.ones(k)) # number of points
    Q = np.array([Q_matrix[i,i] for i in range(k)]) # cost per partition
    count = 0
    converged = False

    while not converged and count < max_iter:
        
        n_changed = 0

        for i in range(n): # for each point
        
            j = np.where(Z[i]==1)[0][0]
            nj = m[j]
            Qj = Q[j]
            Qj_x = G[i,:].dot(Z[:,j])
        
            Aj = (1.0/(nj*(nj-1)))*Qj - (2.0/(nj-1))*Qj_x

            costs = np.zeros(k)
            Qplus = np.zeros(k)
            for l in range(k):
                
                if l == j:
                    costs[l] = -np.inf
                    continue
                
                nl = m[l]
                Ql = Q[l]
                Ql_plus_x = G[i,:].dot(Z[:,l]) + G[i,i]
                Qplus[l] = Ql_plus_x

                Al = (1.0/(nl*(nl+1)))*Ql - (2.0/(nl+1))*Ql_plus_x

                costs[l] = Aj - Al
                
            j_star = np.argmax(costs)
            if costs[j_star] > 0:
                Z[i,j] = 0
                Z[i,j_star] = 1
                m[j] -= 1
                m[j_star] += 1
                Q[j] -= 2*Qj_x
                Q[j_star] += 2*Qplus[j_star]
                n_changed += 1

        if n_changed/n < tol:
            converged = True
        else:
            count += 1

    if verbose:
        if count >= max_iter:
            print "\tQCQP didn't converge after %i iterations." % count
        else:
            print "\tQCQP converged in %i iterations." % count
    
    return Ztoz(Z)

def kernel_kmeans(k, G, Z0, max_iter=100, tol=1e-4, verbose=False):
    """Optimize QCQP through a kernel k-means approach."""

    # initial configuration
    n = G.shape[0]
    Z = np.copy(Z0)
    Zt = Z.T
    D = Zt.dot(Z)
    m = D.dot(np.ones(k)) 
    G_mat = (Z.T).dot(G.dot(Z))
    Q = np.array([G_mat[l,l] for l in range(k)])
    count = 0
    converged = False

    while not converged and count < max_iter:
        
        n_changed = 0

        for i in range(n): # for each point
        
            j = np.where(Z[i]==1)[0][0]
            
            costs = np.zeros(k)
            qls = np.zeros(k)
            for l in range(k):
                Ql_x = G[i,:].dot(Z[:,l])
                qls[l] = Ql_x
                cost = -2*Ql_x/m[l] + Q[l]/(m[l]**2)
                costs[l] = cost
            j_star = np.argmin(costs)
            
            if j_star != j:
                Z[i,j] = 0
                Z[i,j_star] = 1
                m[j] -= 1
                m[j_star] +=1
                Q[j] -= 2*qls[j]
                Q[j_star] += 2*qls[j_star]
                n_changed += 1

        if n_changed/n < tol:
            converged = True
        else:
            count += 1

    if verbose:
        if count >= max_iter:
            print "\tKernel k-means didn't converge after %i iterations." % \
                                                                        count
        else:
            print "\tKernel k-means converged in %i iterations." % count
    
    return Ztoz(Z)

class KernelEnergy(BaseEstimator, ClusterMixin):
    """Kernel k-means with Energy Statistics. Based on a code
    from https://gist.github.com/mblondel/6230787
    This agrees with my above implementation of kernel k-means.
    
    """

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
    from gmm import em_gmm_vect as gmm
    import energy1d
    from scipy.spatial.distance import cityblock
    
    from timeit import default_timer as timer
    
    from beautifultable import BeautifulTable

    # generate data
    """
    X, z = data.multivariate_normal(
        [[0,0,0], [4,0,0]],
        [
            np.array([[1,0,0],[0,1,0],[0,0,1]]), 
            np.array([[1,0,0],[0,19,0],[0,0,1]])
        ],
        [200, 200]
    )
    """
    """
    X, z = data.multivariate_lognormal(
        [np.zeros(20), 0.5*np.concatenate([np.ones(5), np.zeros(15)])],
        [0.5*np.eye(20), np.eye(20)],
        [100, 100]
    )
    """
    X, z = data.circles([1, 3, 5], [0.1, 0.1, 0.1], [200,200, 200])
    #X, z = data.spirals([1,-1], [300,300], noise=0.2)

    # number of clusters
    k = 3

    # semimetric
    def rho(x,y):
        norm = np.power(np.linalg.norm(x-y), 1)
        return norm
    
    def rho_half(x,y):
        norm = np.power(np.linalg.norm(x-y), 0.5)
        return norm

    def rho_gauss(x,y):
        #norm = np.power(np.linalg.norm(x-y), 2)
        #return 2 - 2*np.exp(-norm/2)
        return 2-2*np.exp(-np.linalg.norm(x-y) - 0.5*np.linalg.norm(x-y)**2)
    
    def rho_per(x,y):
        norm = np.power(np.linalg.norm(x-y), 1)
        return np.power(norm, 2)*0.9*np.sin(norm/0.9)
    
    def rho_poly(x,y):
        p = 2
        return np.power(-1+x.dot(x), p) \
                +np.power(-1+y.dot(y), p) \
                -2*np.power(-1+y.dot(x), p)


    # compute Gram matrix
    G = kernel_matrix(X, rho)
    G1 = kernel_matrix(X, rho_half)
    G2 = kernel_matrix(X, rho_gauss)

    # initialization
    mu0, z0 = kmeanspp.kpp(k, X, ret='both')
    Z0 = ztoZ(z0)

    t = BeautifulTable()
    t.column_headers = ["Method", "Accuracy", "Objective", "Exec Time"]
    
    start = timer()
    zh = minw(k, G, Z0, 100, tol=1e-5, verbose=False)
    end = timer()
    Zh = ztoZ(zh)
    t.append_row(["Energy Standard", metric.accuracy(z, zh), 
                  objective(Zh, G), end-start])
    
    start = timer()
    zh = kernel_kmeans(k, G, Z0, 100, tol=1e-5, verbose=False)
    end = timer()
    Zh = ztoZ(zh)
    t.append_row(["Kernel k-means", metric.accuracy(z, zh), 
                  objective(Zh, G), end-start])

    start = timer()
    zh = minw(k, G1, Z0, 100, tol=1e-5, verbose=False)
    end = timer()
    Zh = ztoZ(zh)
    t.append_row(["Energy Half", metric.accuracy(z, zh), 
                  objective(Zh, G1), end-start])
    
    start = timer()
    zh = kernel_kmeans(k, G1, Z0, 100, tol=1e-5, verbose=False)
    end = timer()
    Zh = ztoZ(zh)
    t.append_row(["Kernel k-means half", metric.accuracy(z, zh), 
                  objective(Zh, G1), end-start])
    
    start = timer()
    zh = minw(k, G2, Z0, 100, tol=1e-5, verbose=False)
    end = timer()
    Zh = ztoZ(zh)
    t.append_row(["Energy Gauss", metric.accuracy(z, zh), 
                  objective(Zh, G2), end-start])
    
    start = timer()
    zh = kernel_kmeans(k, G2, Z0, 100, tol=1e-5, verbose=False)
    end = timer()
    Zh = ztoZ(zh)
    t.append_row(["Kernel k-means gauss", metric.accuracy(z, zh), 
                  objective(Zh, G2), end-start])
    
    start = timer()
    zh = kmeans(k, X, labels_=z0, mus_=mu0)
    end = timer()
    Zh = ztoZ(zh)
    t.append_row(["k-means", metric.accuracy(z, zh), 
                  '-', end-start])
    
    try:
        start = timer()
        zh = gmm(k, X, tol=1e-5, max_iter=200)
        end = timer()
        Zh = ztoZ(zh)
        t.append_row(["GMM", metric.accuracy(z, zh), 
                    '-', end-start])
    except:
        t.append_row(["GMM", '-', '-', '-'])
    
    print t
