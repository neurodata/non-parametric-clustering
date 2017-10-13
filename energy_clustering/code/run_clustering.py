"""Functions to run clustering algorithms. Just need to call these functions
in the tests.
"""

# Guilhere Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.cluster import SpectralClustering

import energy.initialization as initialization
import energy.metric as metric
import energy.eclust as eclust 
import energy.energy1d as energy1d

def initialize(init, k, G, X):
    """Initializaton function to use in energy Lloyd and Hartigan."""
    if init == "spectral":
        z0 = initialization.topeigen(k, G)
        Z0 = eclust.ztoZ(z0)
    elif init == "k-means++":
        z0 = initialization.kmeanspp(k, X)
        Z0 = eclust.ztoZ(z0)
    elif init == "random":
        z0 = np.random.randint(0, k, len(X))
        Z0 = eclust.ztoZ(z0)
    else:
        raise ValueError("No initialization method provided")
    return Z0

def energy_lloyd(k, X, G, z, run_times=10, init="spectral"):
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

def energy_hartigan(k, X, G, z, run_times=10, init="spectral"):
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

def energy_spectral(k, X, G, z, run_times=10, init="random"):
    """Run few times and pick the best objective function value.
    Choose the initializatio for k-means, which can be k-means++ or random.
    
    """
    best_score = -np.inf
    for rt in range(run_times):
        zh = initialization.topeigen(k, G, run_times=run_times, init="random")
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

def kmeans(k, X, z, run_times=10, init='k-means++'):
    """run_times is the number of times the algorithm is gonna run.
    init = {'k-means++', 'random'}
    """
    km = KMeans(k, n_init=run_times, init=init)
    km.fit(X)
    zh = km.labels_
    return metric.accuracy(z, zh)

def gmm(k, X, z, run_times=10, init='kmeans'):
    """GMM from sklearn library. init = {'kmeans', 'random'}, run_times
    is the number of times the algorithm is gonna run with different
    initializations.
    
    """
    gm = GMM(k, n_init=run_times, init_params=init)
    gm.fit(X)
    zh = gm.predict(X)
    return metric.accuracy(z, zh)

def spectral(k, X, G, z, run_times=10):
    """Spectral clustering from sklearn library. 
    run_times is the number of times the algorithm is gonna run with different
    initializations.
    
    """
    sc = SpectralClustering(k, affinity='precomputed', n_init=run_times)
    zh = sc.fit_predict(G)
    return metric.accuracy(z, zh)


##############################################################################
if __name__ == "__main__":
    import energy.data as data
    
    """
    D = 100
    d = 10
    n1 = n2 = 100
    m1 = np.zeros(D)
    s1 = np.eye(D)
    m2 = np.concatenate((0.7*np.ones(d), np.zeros(D-d)))
    s2 = np.eye(D)
    X, z = data.multivariate_normal([m1,m2], [s1, s2], [n1, n2])

    D = 10 
    d = 10
    q = 2
    n1 = n2 = 100
    m1 = np.zeros(D)
    m2 = np.concatenate((np.ones(d), np.zeros(D-d)))
    s1 = np.eye(D)
    #s1 = np.diag(np.concatenate((np.random.uniform(1, 5, 10), np.ones(D-d))))
    #s2 = np.eye(D)
    #s2_1 = np.random.uniform(1, 5, 10)
    s2_1 = np.array([1.367,  3.175,  3.247,  4.403,  1.249,  
            1.969, 4.035,   4.237,  2.813,  3.637])
    s2 = np.diag(np.concatenate((s2_1, np.ones(D-d))))
    #for a in range(d):
    #    s1[a,a] = np.power(1/(a+1), q)
    #for a in range(d):
    #    s2[a,a] = np.power(a+1, q)
    X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])
    
    k = 2
    G = eclust.kernel_matrix(X, lambda x, y: np.linalg.norm(x-y))
    
    k = 2
    D = 20
    d = 5
    m1 = np.zeros(D)
    s1 = 0.5*np.eye(D)
    n1 = n2 = 15
    m2 = 0.5*np.concatenate((np.ones(d), np.zeros(D-d)))
    s2 = np.eye(D)
    X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])

    X, z = data.circles([1, 3, 5], [0.2, 0.2, 0.2], [400, 400, 400])
    k = 3
    
    rho = lambda x, y: 2-2*np.exp(-np.linalg.norm(x-y)**2/2/4)
    G = eclust.kernel_matrix(X, rho)
    """

    n = 400
    n1, n2 = np.random.multinomial(n, [0.5, 0.5])
    m1 = 1.5
    s1 = 0.3
    m2 = 0
    s2 = 1.5
    X, z = data.univariate_normal([m1, m2], [s1, s2], [n1, n2])
    #X, z = data.univariate_lognormal([m1, m2], [s1, s2], [n1, n2])
    Y = np.array([[x] for x in X])
    G = eclust.kernel_matrix(Y, lambda x, y: np.linalg.norm(x-y))
    
    k = 2
    init="k-means++"
    #init="random"

    print "Energy-spectral:", energy_spectral(k, Y, G, z, init=init,
                                                run_times=5)
    print "Spectral Clustering:", spectral(k, Y, G, z, run_times=5)
    print "Energy-Lloyd:", energy_lloyd(k, Y, G, z, init=init,
                                        run_times=5)
    print "Energy-Hartigan:", energy_hartigan(k, Y, G, z, init=init,
                                        run_times=5)
    print "k-means:", kmeans(k, Y, z, run_times=5, init=init)
    print  "GMM:", gmm(k, Y, z, run_times=5, init="kmeans")

