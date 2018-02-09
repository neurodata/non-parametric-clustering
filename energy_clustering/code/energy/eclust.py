"""Iterative algorithms to optimize QCQP for energy statistics clustering."""

# Guilherme Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata


from __future__ import division

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state


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

def energy_clustering_brute(k, G, Z0, max_iter=100, tol=1e-4, verbose=False):
    """Optimize energy QCQP in a brute force way. This is expensive,
    but it works. We move points to other partitions and compute the change
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

def energy_hartigan(k, G, Z0, max_iter=100, tol=1e-4, verbose=False,
                    return_Z=False):
    """Optimize the W objective function by considering moving points
    to different partitions. Compute the change in the cost function by
    moving a point then decide the best partition to optimize the cost
    function. It's the same as above by brute force but in a more efficient way.
    
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

            if nj <= 1:
                count += 1
                continue
        
            Aj = (1.0/(nj-1))*(Qj/nj - 2*Qj_x + G[i,i])

            costs = np.zeros(k)
            Qplus = np.zeros(k)
            for l in range(k):
                
                if l == j:
                    costs[l] = -np.inf
                    continue
                
                nl = m[l]
                Ql = Q[l]
                Ql_x = G[i,:].dot(Z[:,l])
                Qplus[l] = Ql_x

                Al = (1.0/(nl+1))*(Ql/nl - 2*Ql_x - G[i,i])

                costs[l] = Aj - Al
                
            j_star = np.argmax(costs)
            if costs[j_star] > 0:
                Z[i,j] = 0
                Z[i,j_star] = 1
                m[j] -= 1
                m[j_star] += 1
                Q[j] = Q[j] - 2*Qj_x + G[i,i]
                Q[j_star] = Q[j_star] + 2*Qplus[j_star] + G[i,i]
                n_changed += 1

        if n_changed/n < tol:
            converged = True
        else:
            count += 1

    if verbose:
        if count >= max_iter:
            print "\tE-clustering didn't converge after %i iterations." % count
        else:
            print "\tE-clustering converged in %i iterations." % count

    if return_Z:
        return Z
    else:
        return Ztoz(Z)

def energy_lloyd(k, G, Z0, max_iter=100, tol=1e-4, verbose=False,
                    return_Z=False):
    """Optimize QCQP through a kernel k-means approach, which is based
    on Lloyd's heuristic.
    
    """

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
            print "\tE-L-clustering didn't converge after %i iterations." % \
                                                                        count
        else:
            print "\tE-L-clustering converged in %i iterations." % count
    
    if return_Z:
        return Z
    else:
        return Ztoz(Z)


###############################################################################
if __name__ == '__main__':
    
    # Generate data and cluster using different algorithms to compare
    
    from sklearn.mixture import GMM
    from sklearn.cluster import KMeans
    from scipy.stats import sem
    from timeit import default_timer as timer
    from beautifultable import BeautifulTable
    
    import data
    import metric
    import initialization
    
    D = 100
    d = 10
    n1 = 200
    n2 = 200
    m1 = np.zeros(D)
    s1 = np.eye(D)
    m2 = np.concatenate((0.7*np.ones(d), np.zeros(D-d)))
    s2 = np.eye(D)
    X, z = data.multivariate_lognormal([m1, m2], [s1, s2], [n1, n2])
    k = 2

    def rho(x,y):
        norm = np.power(np.linalg.norm(x-y), 1)
        return norm
    
    G = kernel_matrix(X, rho)
    
    # initialization
    mu0, z0 = initialization.kmeanspp(k, X, ret='both')
    Z0 = ztoZ(z0)
    z1 = initialization.spectral(k, G)
    Z1 = ztoZ(z1)

    t = BeautifulTable()
    t.column_headers = ["Method", "Accuracy", "Objective", "Exec Time"]
    
    start = timer()
    zh = energy_clustering_brute(k, G, Z0)
    end = timer()
    Zh = ztoZ(zh)
    t.append_row(["E-clustering brute", metric.accuracy(z, zh), 
                  objective(Zh, G), end-start])
    
    start = timer()
    zh = energy_hartigan(k, G, Z0)
    end = timer()
    Zh = ztoZ(zh)
    t.append_row(["E-H-clustering++", metric.accuracy(z, zh), 
                  objective(Zh, G), end-start])
    
    t.append_row(['Spectral Clustering:', metric.accuracy(z, z1),
                  objective(Z1,G), '-'])

    start = timer()
    zh = energy_hartigan(k, G, Z1)
    end = timer()
    Zh = ztoZ(zh)
    t.append_row(["E-H-clustering-top", metric.accuracy(z, zh), 
                  objective(Zh, G), end-start])
    
    start = timer()
    zh = energy_lloyd(k, G, Z1)
    end = timer()
    Zh = ztoZ(zh)
    t.append_row(["E-L-clustering", metric.accuracy(z, zh), 
                  objective(Zh, G), end-start])
    
    start = timer()
    gmm = GMM(k)
    gmm.fit(X)
    zh = gmm.predict(X)
    end = timer()
    t.append_row(["GMM", metric.accuracy(z, zh), 
                  objective(Zh, G), end-start])
    
    start = timer()
    km = KMeans(k)
    zh = km.fit_predict(X)
    end = timer()
    t.append_row(["k-means", metric.accuracy(z, zh), 
                  objective(Zh, G), end-start])

    print t

