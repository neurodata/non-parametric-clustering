"""Two rough implementations of GMM/EM algorithm

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
from numpy.core.umath_tests import matrix_multiply

import initialization


class GMM:
    """Fit a GMM using EM algorithm."""

    def __init__(self, n_clusters, tol=1e-6):
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
        self.mu, z = initialization.kmeanspp(self.K, self.x, ret='both')
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


def gmm(k, xs, tol=1e-6, max_iter=200):
    """Vectorized version of GMM. Faster than above but still rough."""
    
    n, p = xs.shape
    
    mus, z = initialization.kmeanspp(k, xs, ret='both')
    pis = np.array([len(np.where(z==i)[0])/n for i in np.unique(z)])
    sigmas = np.array([np.eye(p)]*k)

    ll_old = 0
    for i in range(max_iter):
        exp_A = []
        exp_B = []
        ll_new = 0

        # E-step, ws are responsabilities
        ws = np.zeros((k, n))
        for j in range(k):
            ws[j, :] = pis[j]*multivariate_normal(mus[j], sigmas[j]).pdf(xs)
        ws /= ws.sum(0)
            
        # M-step
        pis = ws.sum(axis=1)
        pis /= n

        mus = np.dot(ws, xs)
        mus /= ws.sum(1)[:, None]

        sigmas = np.zeros((k, p, p))
        for j in range(k):
            ys = xs - mus[j, :]
            sigmas[j] = (ws[j,:,None,None]*\
                       matrix_multiply(ys[:,:,None], ys[:,None,:])).sum(axis=0)
        sigmas /= ws.sum(axis=1)[:,None,None]

        # update complete log likelihoood
        ll_new = 0
        for pi, mu, sigma in zip(pis, mus, sigmas):
            ll_new += pi*multivariate_normal(mu, sigma).pdf(xs)
        ll_new = np.log(ll_new).sum()

        # convergence test
        if np.abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new

    z = ws.T
    labels = np.argmax(z, axis=1)

    return labels
    #return ll_new, pis, mus, sigmas, ws


###############################################################################
if __name__ == "__main__":

    from sklearn.mixture import GMM as sk_GMM
    
    import data
    import metric

    np.random.seed(12)

    D = 10
    m1 = np.zeros(D)
    s1 = np.eye(D)
    m2 = np.ones(D)
    s2 = 2*np.eye(D)
    X, z = data.multivariate_normal([m1, m2], [s1, s2], [100, 100])
    k = 2

    # scikit-learn library has a better procedure to estimate the covariance
    # matrix.

    g = GMM(k)
    zh  = g.fit_predict(X)
    print "GMM class:", metric.accuracy(z, zh)

    zh = gmm(k, X)
    print "GMM func:", metric.accuracy(z, zh)
    
    sg = sk_GMM(k)
    sg.fit(X)
    zh = sg.predict(X)
    print "GMM sklearn:", metric.accuracy(z, zh)


