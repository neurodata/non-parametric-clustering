from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def multi_gaussians(means, sigmas, ns):
    """Generate several normal distributed data points.
    
    Parameters: 
        means = [m1, m2, ...]
        sigmas = [s1, s2, ...]
        ns = [n1, n2, ...]

        Each parameter in these lists are associated to one multivariate
        normal distribution. The data is shuffled so points are not ordered.
    
    Returns: data X and labels z.
    
    """
    n = len(means)
    X = []
    z = []
    k = 0
    for mu, sigma, n in zip(means, sigmas, ns):
        X.append(np.random.multivariate_normal(mu, sigma, n))
        z.append([k]*n)
        k += 1
    X = np.concatenate(X)
    z = np.concatenate(z)
    idx = np.random.permutation(sum(ns))
    return X[idx], z[idx]

def circles(rs, eps, ns):
    """Generate concentric circles with radius in rs and gaussian noise
    in eps.
    
    """
    k = 0
    X = []
    z = []
    for r, e, n in zip(rs, eps, ns):
        x = np.array([[r*np.cos(a), r*np.sin(a)] 
                            for a in np.random.uniform(0, 2*np.pi, n)])
        x += e*np.random.multivariate_normal([0,0],np.eye(2), n)
        X.append(x)
        z.append([k]*n)
        k += 1
    X = np.concatenate(X)
    z = np.concatenate(z)
    idx = np.random.permutation(sum(ns))
    return X[idx], z[idx]

def plot(X, z, fname='plot.pdf'):
    """Plot clusters according to labels in z.
    If data is multidimensional, pick the first two principal components.
    
    """

    z_unique = np.unique(z)

    if len(X[0]) > 2:
        pca = PCA(n_components=2)
        X_new = pca.fit_transform(X)
    else:
        X_new = X

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = iter(plt.cm.rainbow(np.linspace(0,1,len(z_unique))))
    for k in z_unique:
        idx = np.where(z==k)
        x = X_new[idx][:,0]
        y = X_new[idx][:,1]
        ax.plot(x, y, 'bo', alpha=.6, color=next(colors))
    
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axes().set_aspect('equal', 'datalim')
    fig.savefig(fname)


if __name__ == '__main__':
    X, z = multi_gaussians(
        [[0,0,0], [10,0,0], [1,10,10]], 
        [np.eye(3), np.eye(3), np.eye(3)], 
        [100, 100, 100]
    )
    #X, z = circles([1, 3, 5], [0.1, 0.2, 0.3], [200, 200, 500])
    plot(X, z)

