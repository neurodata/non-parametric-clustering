"""Functions to run clustering algorithms. Just need to call these functions
in the tests.
"""

# Guilhere Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import SpectralClustering

import energy.initialization as initialization
import energy.metric as metric
import energy.eclust as eclust 


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
    elif init == "gmm":
        z0 = gmm(k, X, run_times=1, init='kmeans')
        Z0 = eclust.ztoZ(z0)
    else:
        raise ValueError("No initialization method provided")
    return Z0

def energy_lloyd(k, X, G, run_times=10, init="spectral"):
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

    return best_z

def energy_hartigan(k, X, G, run_times=10, init="spectral"):
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

    return best_z

def energy_spectral(k, X, G, run_times=10, init="random"):
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

    return best_z

def kmeans(k, X, run_times=10, init='k-means++'):
    """run_times is the number of times the algorithm is gonna run.
    init = {'k-means++', 'random'}
    """
    km = KMeans(k, n_init=run_times, init=init)
    km.fit(X)
    zh = km.labels_
    return zh

def gmm(k, X, run_times=10, init='kmeans'):
    """GMM from sklearn library. init = {'kmeans', 'random'}, run_times
    is the number of times the algorithm is gonna run with different
    initializations.
    
    """
    gm = GMM(k, n_init=run_times, init_params=init)
    gm.fit(X)
    zh = gm.predict(X)
    return zh

def spectral(k, X, G, run_times=10):
    """Spectral clustering from sklearn library. 
    run_times is the number of times the algorithm is gonna run with different
    initializations.
    
    """
    sc = SpectralClustering(k, affinity='precomputed', n_init=run_times)
    zh = sc.fit_predict(G)
    return zh


