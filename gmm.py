"""
EM algorithm for GMMM

1. Unitialize \mu_k, \Sigma_k, and \pi_k. Evaluate log likelihood
2. E Step: evaluate responsabilities using the current values of parameters.
3. M Step: re-estimate \mu_k, \Sigma_k, and \pi_k using current
   responsabilities.
4. Evaluate log likelihood and check for convergence. Repeat from step 2
   if not converged.


"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def gaussian(x, mu, sigma):
    """Return multidimensional gaussian at point x."""
    return multivariate_normal.pdf(x, mu, sigma)

def log_likelihood(x, mu, sigma, pi):
    """Return the log likelihood function based on data and current
    parameters.
    
    """
    K = len(pi)
    log_lh = 0
    for xn in x:
        log_lh += np.array([pi[k]*gaussian(xn, mu[k], sigma[k]) 
                            for k in range(K)]).sum()
    return log_lh

def gmm_mle(x, mu, sigma, pi, tol=0.001):
    """Estimate a GMM using maximum likelihood estimation through EM
    algorithm. mu, sigma, pi are the initial values.
    TODO: vectorize things.
    
    """
    K = len(pi)
    N = len(x)
    converged = False
    gamma = np.zeros((N, K), dtype=float)
    old_log_lh = log_likelihood(x, mu, sigma, pi)

    while not converged:
        
        # compute responsabilities, E-step
        for n in range(N):
            total = sum([pi[j]*gaussian(x[n], mu[j], sigma[j]) 
                            for j in range(K)])
            for k in range(K):
                gamma[n, k] = pi[k]*gaussian(x[n], mu[k], sigma[k])/sum_piN

        # compute parameters, M-step
        for k in range(K):
            Nk = gamma[:,k].sum()
            mu[k] = sum([gamma[n,k]*x[n]  for n in range(N)])/Nk
            # TODO: check gaussian collapse under a point
            sigma[k] = sum([gamma[n,k]*(x[n]-mu[k]).outer(x[n]-mu[k]) 
                            for n in range(N)])/Nk
            pi[k] = Nk/N

        # compute log likelihood and check convergence
        new_log_lh = log_likelihood(x, K, mu, sigma, pi)
        if abs(new_log_lh - old_log_lh) <= tol:
            converged = True
        else:
            old_log_lh = new_log_lh

    return mu, sigma, pi


