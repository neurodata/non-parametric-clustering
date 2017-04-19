"""Other objective functions to compare with energy."""

# Guilherme Franca <guifranca@gmail.com>
# Johns Hopkinks University, Neurodata


from __future__ import division

import numpy as np
from scipy.stats import multivariate_normal


def kmeans(A):
    """Compute k-means function.
    Input: A = [A1, A2, ..., Ak] where each Ai is a sample from a distribution
    Output: k-means objective function
    """
    return sum([((a - a.mean(axis=0))**2).sum() for a in A])

def loglikelihood(A):
    """Compute log likelihood function of GMM.
    Assume hard assignments, i.e. A = [A_1, A_2, ...] so each set A_1
    corresponds to one cluster.
    """
    return sum([multivariate_normal.logpdf(a, mean=a.mean(axis=0), 
                cov=np.cov(a.T)).sum() for a in A])

def kernel_score(K, z):
    # Make Z matrix
    unique_labels = np.unique(z)
    Z = np.zeros((len(z), len(unique_labels)), dtype=int)
    for k in unique_labels:
        Z[np.where(z==k), k] = 1
    D = Z.T.dot(Z)
    Dm = np.linalg.inv(D**(.5))
    H = Z.dot(Dm)
    A = H.T.dot(K).dot(H)
    return np.trace(A)
    

###############################################################################
if __name__ == "__main__":
   
    import matplotlib.pyplot as plt
    import data

    X = np.random.multivariate_normal([0,0], np.eye(2), 200)
    Y = np.random.multivariate_normal([2,0], np.eye(2), 200)
    Z = np.random.multivariate_normal([5,0], 0.5*np.eye(2), 200)

    print kmeans([X, Y, Z])
    print loglikelihood([X, Y, Z])
