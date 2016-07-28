"""
MLE estimator for a single gaussian.

"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse


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

def scatter_ellipse(data, mu, cov):
    """Plot confidence interval (95%) of the data as an ellipse representing
    the gaussian together with the data points.
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='auto')

    # plot confidence interval as an ellipse
    vals, vecs = np.linalg.eigh(cov)
    idx = vals.argsort()[::-1]
    vals = vals[idx]
    vecs = vecs[:,idx]
    alfa = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    a, b = 2*np.sqrt(5.991*vals)
    ellipse = Ellipse(xy=mu, width=a, height=b, angle=alfa, 
                      facecolor='yellow', alpha=.4, edgecolor='yellow',
                      linestyle='-', linewidth=3, zorder=1)
    ax.add_artist(ellipse)
    
    ax.scatter(data[:,0], data[:,1], color='blue', alpha=.5, zorder=2)
    #plt.savefig('gauss_ellipse.pdf')


