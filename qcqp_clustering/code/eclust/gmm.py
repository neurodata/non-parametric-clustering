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

import kmeanspp


class GMM:
    """Fit a GMM using EM algorithm."""

    def __init__(self, n_clusters, tol=0.001):
        self.K = n_clusters
        self.tol = tol
    
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

    def initialize(self):
        self.mu, z = kmeanspp.kpp(self.K, self.x, ret='both')
        self.sigma = np.array([np.eye(len(self.mu[0])) for k in range(self.K)])
        self.pi = np.array([len(np.where(z==k)[0])/self.N 
                                for k in np.unique(z)])
        self.gamma = np.zeros((self.N, self.K), dtype=float) # respons.

    def fit(self, X):
        self.x = X
        self.N = len(X)
        self.initialize()
        self.gmm_em()
        self.make_labels()

    def make_labels(self):
        self.labels = self.gamma.argmax(axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels



from scipy.stats import multivariate_normal as mvn
from numpy.core.umath_tests import matrix_multiply as mm

def em_gmm_vect(k, xs, tol=0.001, max_iter=200):
    """Vectorized GMM, faster."""
    
    n, p = xs.shape
    
    mus, z = kmeanspp.kpp(k, xs, ret='both')
    pis = np.array([len(np.where(z==i)[0])/n for i in np.unique(z)])
    #sigma = np.array([np.eye(len(self.mus[0])) for i in range(k)])
    sigmas = np.array([np.eye(p)] * k)

    ll_old = 0
    for i in range(max_iter):
        exp_A = []
        exp_B = []
        ll_new = 0

        # E-step
        ws = np.zeros((k, n))
        for j in range(k):
            ws[j, :] = pis[j] * mvn(mus[j], sigmas[j]).pdf(xs)
        ws /= ws.sum(0)

        # M-step
        pis = ws.sum(axis=1)
        pis /= n

        mus = np.dot(ws, xs)
        mus /= ws.sum(1)[:, None]

        sigmas = np.zeros((k, p, p))
        for j in range(k):
            ys = xs - mus[j, :]
            sigmas[j] = (ws[j,:,None,None]*mm(ys[:,:,None], ys[:,None,:])).\
                        sum(axis=0)
        sigmas /= ws.sum(axis=1)[:,None,None]

        # update complete log likelihoood
        ll_new = 0
        for pi, mu, sigma in zip(pis, mus, sigmas):
            ll_new += pi*mvn(mu, sigma).pdf(xs)
        ll_new = np.log(ll_new).sum()

        if np.abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new

    z = ws.T
    labels = np.argmax(z, axis=1)

    return labels
    #return ll_new, pis, mus, sigmas, ws
