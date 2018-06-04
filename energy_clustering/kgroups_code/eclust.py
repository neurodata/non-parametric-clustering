"""Iterative algorithms to optimize QCQP for energy statistics clustering."""

# Guilherme Franca <guifranca@gmail.com>
# Johns Hopkins University


from __future__ import division

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

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

def objective(Z, G, W):
    """Compute objective function."""
    W_half = np.sqrt(W)
    W_half_Z = W_half.dot(Z) # This is the same as Z with 1 -> w_i^{1/2} 
    Sum_weights = W_half_Z.T.dot(W_half_Z) # diagonal matrix with s_i 
    Y = Z.dot(np.linalg.inv(np.sqrt(Sum_weights)))
    G_tilde = W.dot(G.dot(W))
    return np.trace(Y.T.dot(G_tilde.dot(Y)))

def kernel_function(x, y, rho, x0=None):
    """Return kernel function based on rho."""
    if type(x0) == type(None):
        x0 = np.zeros(x.shape)
    val = 0.5*(rho(x,x0) + rho(y,x0) - rho(x,y))
    return val

def kernel_matrix(X, rho, x0=None):
    """Compute Kernel matrix based on kernel function K(x,y)."""
    kfunc = lambda x, y: kernel_function(x, y, rho, x0)
    #G = pairwise_distances(X, metric=kfunc, n_jobs=4)
    G = pairwise_distances(X, metric=kfunc)
    return G

def kernel_kgroups(k, G, Z0, W, max_iter=100, tol=1e-4, verbose=False,
                   return_Z=False):
    """Optimize the W objective function by considering moving points
    to different partitions. Compute the change in the cost function by
    moving a point then decide the best partition to optimize the cost
    function. 
    
    """
    n = G.shape[0]
    Z = np.copy(Z0)
    Zt = Z.T
    Gtilde = W.dot(G.dot(W)) # absorb weights into a new matrix
    Q_matrix = Zt.dot(Gtilde.dot(Z))  
    w = W.dot(np.ones(n)) # vector containing weights
    s = list(Zt.dot(w)) # vector of s_i's, sum of weights in each cluster
    q = list(np.diag(Q_matrix)) # vector with costs of each cluster
    
    count = 0
    converged = False
    while not converged and count < max_iter:
        
        n_changed = 0

        for i in range(n): # for each data point
        
            j = np.where(Z[i]==1)[0][0] # current cluster
            Qj_xi = Gtilde[i,:].dot(Z[:,j]) # cost of x_i with C_j
            
            if s[j] <= 1:
                count += 1
                continue
        
            Aj = (1.0/(s[j]-w[i]))*(w[i]*q[j]/s[j] - 2*Qj_xi + Gtilde[i,i])

            delta_q = np.zeros(k) 
            Qplus = np.zeros(k) # store point costs to avoid recomputing below
            for l in range(k):
                
                if l == j:
                    delta_q[l] = -np.inf
                    continue
                
                Ql_xi = Gtilde[i,:].dot(Z[:,l]) # cost of x_i with C_l
                Qplus[l] = Ql_xi

                Al = (1.0/(s[l]+w[i]))*(w[i]*q[l]/s[l] - 2*Ql_xi - Gtilde[i,i])

                delta_q[l] = Aj - Al
                
            j_star = np.argmax(delta_q)
            if delta_q[j_star] > 0:
                Z[i,j] = 0
                Z[i,j_star] = 1
                s[j] -= w[i]
                s[j_star] += w[i]
                q[j] = q[j] - 2*Qj_xi + Gtilde[i,i]
                q[j_star] = q[j_star] + 2*Qplus[j_star] + Gtilde[i,i]
                n_changed += 1

        if n_changed/n < tol:
            converged = True
        else:
            count += 1

    if verbose:
        if count >= max_iter:
            print "\tKernel k-groups didn't in %i iterations." % count
        else:
            print "\tKernel k-groups converged in %i iterations." % count

    if return_Z:
        return Z
    else:
        return Ztoz(Z)

def kernel_kmeans(k, G, Z0, W, max_iter=100, tol=1e-4, verbose=False,
                    return_Z=False):
    """Optimize QCQP through a kernel k-means approach, which is based
    on Lloyd's heuristic.
    
    """
    n = G.shape[0]
    Z = np.copy(Z0)
    Zt = Z.T
    Gtilde = W.dot(G.dot(W)) # absorb weights into a new matrix
    Q_matrix = Zt.dot(Gtilde.dot(Z))  
    w = W.dot(np.ones(n)) # vector containing weights
    s = list(Zt.dot(w)) # vector of s_i's, sum of weights in each cluster
    q = list(np.diag(Q_matrix)) # vector with costs of each cluster

    count = 0
    converged = False
    while not converged and count < max_iter:
        
        n_changed = 0

        for i in range(n): # for each data point
        
            j = np.where(Z[i]==1)[0][0] # current cluster
            costs = np.zeros(k)
            qxs = np.zeros(k)
            for l in range(k):
                Ql_xi = Gtilde[i,:].dot(Z[:,l])
                qxs[l] = Ql_xi
                Jl_xi = q[l]/(s[l]**2) - 2*Ql_xi/s[l]
                costs[l] = Jl_xi
            
            j_star = np.argmin(costs)
            
            if j_star != j:
                Z[i,j] = 0
                Z[i,j_star] = 1
                s[j] -= w[i]
                s[j_star] += w[i]
                q[j] = q[j] - 2*qxs[j]
                q[j_star] = q[j_star] + 2*qxs[j_star]
                n_changed += 1

        if n_changed/n < tol:
            converged = True
        else:
            count += 1

    if verbose:
        if count >= max_iter:
            print "\tKernel k-means didn't converge in %i iterations." % \
                                                                        count
        else:
            print "\tKernel k-means didn't converge in %i iterations." % count
    
    if return_Z:
        return Z
    else:
        return Ztoz(Z)


###############################################################################
if __name__ == '__main__':
    
    from sklearn.mixture import GaussianMixture as GMM
    from sklearn.cluster import KMeans
    from timeit import default_timer as timer
    from prettytable import PrettyTable 
    
    import data
    import metric
    import init
    
    D = 20
    d = 10
    n1 = 200
    n2 = 200
    m1 = np.zeros(D)
    s1 = np.eye(D)
    m2 = np.concatenate((0.7*np.ones(d), np.zeros(D-d)))
    s2 = np.eye(D)
    X, z = data.multivariate_lognormal([m1, m2], [s1, s2], [n1, n2])
    k = 2
    W = np.diag(np.random.normal(1, 0.01, n1+n2))

    rho = lambda x,y: np.power(np.linalg.norm(x-y), 1)
    G = kernel_matrix(X, rho)
    
    # initialization
    z0, mu0 = init.kmeans_plus2(k, X)
    Z0 = ztoZ(z0)
    z1 = init.spectral(k, G, W)
    Z1 = ztoZ(z1)

    t = PrettyTable(["Method", "Accuracy", "Objective", "Exec Time"])
    
    start = timer()
    zh = kernel_kgroups(k, G, Z0, W)
    end = timer()
    Zh = ztoZ(zh)
    t.add_row(["kernel k-groups (k-means++)", metric.accuracy(z, zh), 
                  objective(Zh, G, W), end-start])
    
    start = timer()
    zh = kernel_kgroups(k, G, Z1, W)
    end = timer()
    Zh = ztoZ(zh)
    t.add_row(["kernel k-groups (spectral)", metric.accuracy(z, zh), 
                  objective(Zh, G, W), end-start])
    
    start = timer()
    zh = kernel_kmeans(k, G, Z0, W)
    end = timer()
    Zh = ztoZ(zh)
    t.add_row(["kernel k-means (k-means++)", metric.accuracy(z, zh), 
                  objective(Zh, G, W), end-start])
    
    start = timer()
    zh = kernel_kmeans(k, G, Z1, W)
    end = timer()
    Zh = ztoZ(zh)
    t.add_row(["kernel k-means (spectral)", metric.accuracy(z, zh), 
                  objective(Zh, G, W), end-start])
    
    start = timer()
    gmm = GMM(k)
    gmm.fit(X)
    zh = gmm.predict(X)
    end = timer()
    t.add_row(["GMM", metric.accuracy(z, zh), "-", end-start])
    
    start = timer()
    km = KMeans(k)
    zh = km.fit_predict(X)
    end = timer()
    t.add_row(["k-means", metric.accuracy(z, zh), "-", end-start])

    print t

