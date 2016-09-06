#!/usr/bin/env python

"""
EM algorithm for GMMM

We implement the Expectation Maximization (EM) algorithm for Gaussian
Mixture Models (GMM). The algorithm is described as follows:

1. Initialize \mu_k, \Sigma_k, and \pi_k. Evaluate log likelihood function.

2. E Step: evaluate responsabilities using the current values of parameters.

3. M Step: re-estimate \mu_k, \Sigma_k, and \pi_k using the updated
   responsabilities.

4. Evaluate log likelihood function and check for convergence. 
   Repeat from step 2 if not converged.


"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse


def gaussian(x, mu, sigma):
    return multivariate_normal.pdf(x, mu, sigma)


class GMM:
    """Fit a GMM using EM algorithm."""

    def __init__(self, data, means, covs, pi, tol=0.001):
        self.x = data
        self.mu = means
        self.sigma = covs
        self.pi = pi
        self.N = len(data)
        self.K = len(pi)
        self.tol = tol
        self.gamma = np.zeros((self.N, self.K), dtype=float) # respons.
    
    def gauss(self, n, k):
        """Return multidimensional gaussian at point x_n for component k, 
        i.e. N(x_n | \mu_k, \sigma_k).

        """
        return multivariate_normal.pdf(self.x[n], self.mu[k], self.sigma[k])

    def log_likelihood(self):
        """Compute log likelihood function."""
        f = 0
        for n in range(self.N):
            f += np.array([self.pi[k]*self.gauss(n, k) 
                                for k in range(self.K)]).sum()
        return f

    def xxT(self, n, k):
        x = self.x[n] - self.mu[k]
        return np.outer(x, x)

    def gmm_em(self):
        """Implementation of EM algorithm."""
        converged = False
        old_loglh = self.log_likelihood()

        while not converged:
            
            # compute responsabilities, E-step
            for n in range(self.N):
                total = sum([self.pi[j]*self.gauss(n, j) 
                                        for j in range(self.K)])
                for k in range(self.K):
                    self.gamma[n, k] = self.pi[k]*self.gauss(n, k)/total
            
            # compute parameters, M-step
            for k in range(self.K):
                Nk = self.gamma[:,k].sum()
                self.mu[k] = sum([self.gamma[n, k]*self.x[n] 
                                                for n in range(self.N)])/Nk
                # TODO: check gaussian collapse under a point
                self.sigma[k] = sum([self.gamma[n, k]*self.xxT(n, k)
                                                for n in range(self.N)])/Nk
                self.pi[k] = Nk/self.N

            # compute log likelihood and check convergence
            new_loglh = self.log_likelihood()
            if abs(new_loglh - old_loglh) <= self.tol:
                converged = True
            else:
                old_loglh = new_loglh

    def fit(self):
        self.gmm_em()

    def mean(self):
        return self.mu

    def cov(self):
        return self.sigma

    def responsabilities(self):
        return self.gamma


