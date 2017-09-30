"""Functions to run clustering algorithms. Just need to call these functions
in the tests.
"""

# Guilhere Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GMM

import energy.initialization as initialization
import energy.metric as metric
import energy.eclust as eclust 
import energy.energy1d as energy1d

def initialize(init, k, G, X):
        if init == "spectral":
            z0 = initialization.topeigen(k, G)
            Z0 = eclust.ztoZ(z0)
        elif init == "kmeans++":
            z0 = initialization.kmeanspp(k, X)
            Z0 = eclust.ztoZ(z0)
        elif init == "random":
            z0 = np.random.randint(0, k, len(X))
            Z0 = eclust.ztoZ(z0)
        else:
            raise ValueError("No initialization method provided")
        return Z0

def energy_lloyd(k, X, G, z, run_times=1, init="spectral"):
    """Run few times and pick the best objective function value."""
    best_score = -np.inf
    for rt in range(run_times):
        Z0 = initialize(init, k, G, X)
        zh = eclust.energy_lloyd(k, G, Z0, max_iter=300)
        Zh = eclust.ztoZ(zh)
        score = eclust.objective(Zh, G)
        
        if score > best_score:
            best_score = score
            best_z = zh

    return metric.accuracy(z, best_z)

def energy_hartigan(k, X, G, z, run_times=1, init="spectral"):
    """Run few times and pick the best objective function value."""
    best_score = -np.inf
    for rt in range(run_times):
        Z0 = initialize(init, k, G, X)
        zh = eclust.energy_hartigan(k, G, Z0, max_iter=300)
        Zh = eclust.ztoZ(zh)
        score = eclust.objective(Zh, G)
        
        if score > best_score:
            best_score = score
            best_z = zh

    return metric.accuracy(z, best_z)

def energy1D(X, z):
    """Energy clustering in 1 dimension. No need to run multiple times since
    this is exact.
    
    """
    zh, cost = energy1d.two_clusters1D(X)
    return metric.accuracy(z, zh)

def kmeans(k, X, z):
    """We will not run several times because probably sklearn library
    already do something to make it stable.
    
    """
    km = KMeans(k)
    km.fit(X)
    zh = km.labels_
    return metric.accuracy(z, zh)

def gmm(k, X, z):
    """We will not run several times because sklearn library is probably
    taken care of that already.
    
    """
    gm = GMM(k)
    gm.fit(X)
    zh = gm.predict(X)
    return metric.accuracy(z, zh)

##############################################################################
if __name__ == "__main__":
    import energy.data as data
    
    D = 1
    k = 2
    m1 = np.zeros(D)
    m2 = -1.5*np.ones(D)
    s1 = 0.3*np.eye(D)
    s2 = 1.5*np.eye(D)
    X, z = data.multivariate_lognormal([m1,m2], [s1,s2], [100,100])
    G = eclust.kernel_matrix(X, lambda x, y: np.linalg.norm(x-y))
    
    print "Energy-spectral:", metric.accuracy(initialization.topeigen(k, G),z)
    print "Energy-Lloyd:", energy_lloyd(k, X, G, z, init="random",
                                        run_times=1)
    print "Energy-Hartigan:", energy_hartigan(k, X, G, z, init="random",
                                        run_times=1)
    print "k-means:", kmeans(k, X, z)
    print  "GMM:", gmm(k, X, z)
