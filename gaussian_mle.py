"""
MLE estimator for a single gaussian.

"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt


def gaussian_mle_estimator(x):
    """Return the mean and covariance matrix based on an MLE estimation for
    a single gaussian.
    
    """
    D = x[0].shape[0]
    mean_mle = np.array([x[:, i].mean() for i in range(D)])
    cov_mle = np.array(sum([np.outer(x[i] - mean_mle, x[i] - mean_mle) 
                        for i in range(len(x))]))
    cov_mle = cov_mle / (len(x) - 1)
    return mean_mle, cov_mle

def gaussianD(x, mu, sigma):
    D = len(x)
    c = np.sqrt(((2*np.pi)**D)*np.linalg.det(sigma))
    c = 1/c
    inv_sigma = np.linalg.inv(sigma)
    return c*np.exp(-(1/2)*(x-mu).dot(inv_sigma.dot(x-mu)))

def gaussian1D(x, mu, sigma):
    c = np.sqrt(2*np.pi)*sigma
    c = 1/c
    return c*np.exp(-(1/2)*((x-mu)**2)/(sigma**2))


def histogram(x, mean_mle, sigma_mle):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ['b', 'r', 'g']
    for i in range(len(mean_mle)):
        ax.hist(x[:,i], color=colors[i], bins=80, normed=True, alpha=.3)
        xs = np.linspace(x[:,i].min(), x[:,i].max(), 200)
        ys = [gaussian1D(u, mean_mle[i], sigma_mle[i,i]) for u in xs]
        ax.plot(xs, ys, '-', color=colors[i], linewidth=2)
    plt.savefig('hist.png')


##############################################################################
if __name__ == '__main__':
    
    mean = np.array([2, 4, 1])
    sigma = np.array([
                [1, 0, 0], 
                [0, 2, 0], 
                [0, 0, 5]
            ])

    data = np.random.multivariate_normal(mean, sigma, 5000)
    mu, cov = gaussian_mle_estimator(data)
    print "Mean:", mu
    print "Covariance:", cov
    #histogram(data, mu, cov)

