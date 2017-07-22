"""
Tests and comparison between Energy, K-means, and GMM to access number
of clusters.

"""

# Guilhere Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata

from __future__ import division

import numpy as np

import eclust.data as data
import eclust.kmeans as km
import eclust.gmm as gm
import eclust.qcqp as ke
import eclust.kmeanspp as kpp
from eclust import objectives


def energy(k, X, kernel_matrix, run_times=5):
    """Run few times and pick the best objective function value."""
    G = kernel_matrix
    best_score = -np.inf
    for rt in range(run_times):
        
        z0 = kpp.kpp(k, X, ret='labels')
        Z0 = ke.ztoZ(z0)
        
        zh = ke.minw(k, G, Z0, max_iter=300)
        Zh = ke.ztoZ(zh)
        score = ke.objective(Zh, G)
        if score > best_score:
            best_score = score
            best_z = zh
    
    return best_z, best_score

def kmeans(k, X, run_times=5):
    """Run k-means couple times and pick the best answer."""
    best_score = np.inf
    for rt in range(run_times):
        
        mu0, z0 = kpp.kpp(k, X, ret='both')
        
        zh = km.kmeans(k, X, labels_=z0, mus_=mu0, max_iter=300)
        score = objectives.kmeans(data.from_label_to_sets(X, zh))
        
        if score < best_score:
            best_score = score
            best_z = zh
    
    return best_z, best_score

def gmm(k, X, run_times=5):
    """Run gmm couple times and pick the best answer."""
    best_score = -np.inf
    best_z = None
    for rt in range(run_times):
        try:
            zh = gm.em_gmm_vect(k, X)
            score = objectives.loglikelihood(data.from_label_to_sets(X, zh))
            if score > best_score:
                best_score = score
                best_z = zh
        #except np.linalg.LinAlgError:
        except:
            pass
    if best_z is not None:
        return best_z, best_score
    else:
        return 0

def detect_clusters(num_points, num_permutations):
    """Check if algorithms are detecting clusters, compared to chance."""

    # generate data from uniform
    Y = np.random.uniform(0, 10, num_points)
    X = np.array([[x] for x in Y])
    
    # cluster with k=2 and pick objectives values
    rho = lambda x, y: np.linalg.norm(x-y)
    G = ke.kernel_matrix(X, rho)
    k=2
    z_energy, J_energy = energy(k, X, G, run_times=5)
    z_kmeans, J_kmeans = kmeans(k, X, run_times=5)
    z_gmm, J_gmm = gmm(k, X, run_times=5)

    # random permute labels and compute objectives
    times_energy = 0
    times_kmeans = 0
    times_gmm = 0
    for i in range(num_permutations):

        #fake_z = np.random.randint(0, 2, num_points)
        #fake_Z = ke.ztoZ(fake_z)
        fake_z_energy = np.random.choice(z_energy, len(z_energy))
        fake_Z_energy = ke.ztoZ(fake_z_energy)

        fake_z_kmeans = np.random.choice(z_kmeans, len(z_kmeans))
        fake_z_gmm = np.random.choice(z_gmm, len(z_gmm))
        
        JJ_energy = ke.objective(fake_Z_energy, G)
        JJ_kmeans = objectives.kmeans(
                        data.from_label_to_sets(X, fake_z_kmeans))
        JJ_gmm = objectives.loglikelihood(
                        data.from_label_to_sets(X, fake_z_gmm))
    
        if JJ_energy > J_energy:
            times_energy += 1
        if JJ_kmeans < J_kmeans:
            times_kmeans += 1
        if JJ_gmm > J_gmm:
            times_gmm += 1

    return times_energy, times_kmeans, times_gmm

if __name__ == '__main__':
    print detect_clusters(100, 1000)
