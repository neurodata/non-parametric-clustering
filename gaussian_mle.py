"""
MLE estimator for a single gaussian.

"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def gaussian_mle_estimator(x):
    """Return the mean and covariance matrix based on an MLE estimation for
    a single gaussian.
    
    """
    D = x[0].shape[0]
    mean_mle = np.array([x[:, i].mean() for i in range(D)])
    cov_mle = np.array(sum([np.outer(x[i] - mean_mle, x[i] - mean_mle) 
                        for i in range(len(x))]))
    cov_mle = cov_mle/(len(x)-1)
    return mean_mle, cov_mle

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

