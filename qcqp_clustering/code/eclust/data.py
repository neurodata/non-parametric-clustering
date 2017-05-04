"""Functions used to generate or manipulate data."""

# Guilherme Franca <guifranca@gmail.com>
# Johns Hopkins Unversity, Neurodata


from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import numbers

from sklearn.decomposition import PCA


def multivariate_normal(means, sigmas, ns):
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

def univariate_normal(means, sigmas, ns):
    """Generate several univariate normal distributed data points.
    
    Parameters: 
        means = [m1, m2, ...]
        sigmas = [s1, s2, ...]
        ns = [n1, n2, ...]
    
    Returns: data X and labels z.
    
    """
    n = len(means)
    X = []
    z = []
    k = 0
    for mu, sigma, n in zip(means, sigmas, ns):
        X.append(np.random.normal(mu, sigma, n))
        z.append([k]*n)
        k += 1
    X = np.concatenate(X)
    z = np.concatenate(z)
    idx = np.random.permutation(sum(ns))
    return X[idx], z[idx]

def univariate_lognormal(means, sigmas, ns):
    """Return data according to log normal distribtion.
    Input: means = [mu1, mu2, ...], 
           sigmas = [sigma1, sigma2, ...], 
           ns = [n1, n2, ...]
    Ouput: Data matrix X with corresponding labels z
    """
    x, z = univariate_normal(means, sigmas, ns)
    return np.exp(x), z
    
def multivariate_lognormal(means, sigmas, ns):
    """Return data according to log normal distribtion.
    Input: means = [mu1, mu2, ...], 
           sigmas = [sigma1, sigma2, ...], 
           ns=[n1, n2, ...]
    Ouput: Data matrix X with corresponding labels z
    """
    x, z = multivariate_normal(means, sigmas, ns)
    return np.exp(x), z

def circles(rs, eps, ns):
    """Generate concentric circles with radius in rs and gaussian noise
    in eps.
    Input: rs = [r1, r2, ...] radius
           eps = [epsilon1, epsilon2, ...] perturbation
           ns = [n1, n2, ...] number of points
    Output: data matrix X with labels z
    
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

def spirals(rs, ns, noise=0.2):
    """Generate spirals with radiuses in rs=[...] and number of points
    in ns=[...]
    
    """
    X = []
    z = []
    for k, (r, n) in enumerate(zip(rs, ns)):
        ts = np.linspace(0, 4*np.pi, n) 
        x = np.array([[r*t*np.cos(t)+noise*np.random.normal(0,1),
                       r*t*np.sin(t)+noise*np.random.normal(0,1)] for t in ts])
        X.append(x)
        z.append([k]*n)
    X = np.concatenate(X)
    z = np.concatenate(z)
    idx = np.random.permutation(sum(ns))
    return X[idx], z[idx]

def plot(X, z, fname='plot.pdf'):
    """Plot data according to labels in z.
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

def hist(X, z, fname='plot.pdf'):
    """Plot histograms of 1-dimensional data."""
    z_unique = np.unique(z)
    fig = plt.figure()
    bins=np.arange(min(X), max(X) + 0.2, 0.2)
    ax = fig.add_subplot(111)
    colors = iter(plt.cm.rainbow(np.linspace(0,1,len(z_unique))))
    for k in z_unique:
        idx = np.where(z==k)[0]
        x = X[idx]
        ax.hist(x, bins=bins, facecolor=next(colors), alpha=.6, normed=1)
    fig.savefig(fname)

def shuffle_data(X):
    """Shuffle data and return corresponding labels.
    Input: X = [X1, X2, ...]
    Output: data matrix X and labels z
    
    """
    ns = []
    zs = []
    for i, x in enumerate(X):
        n = len(x)
        ns.append(n)
        zs.append([i]*n)
    z = np.concatenate(zs)
    Y = np.concatenate(X)
    idx = np.random.permutation(sum(ns))
    return Y[idx], z[idx]

def mix_data(A, m):
    """Mix data between subsets of A.
    Input: A = [A_1, A_2, ..., A_n] where each A_i is a subset of data
            m is an integer
    Output: Mix m points from each class and returns a new data matrix X
    
    """
    indices = [np.random.choice(range(len(a)), m*(len(A)-1), replace=False) 
                for a in A]
    X = np.copy(A)
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            X[j][indices[j][i*m:(i+1)*m]] = A[i][indices[i][(j-1)*m:j*m]]
            X[i][indices[i][(j-1)*m:j*m]] = A[j][indices[j][i*m:(i+1)*m]]
    return X

def from_label_to_sets(X, z):
    """Given a data matrix X with corresponding labels z return a list
    [A_1, A_2, ...] where each set correspond to one label (cluster).
    
    """
    A = []
    for label in np.unique(z):
        A.append(X[np.where(z==label)])
    return A

def from_sets_to_labels(A):
    """Given A = [A_1, A_2, ...] return a data matrix with labels: X, z."""
    return shuffle_data(A)


###############################################################################
if __name__ == '__main__':
    """
    X, z = multi_gaussians(
        [[0,0,0], [10,0,0], [1,10,10]], 
        [np.eye(3), np.eye(3), np.eye(3)], 
        [100, 100, 100]
    )
    #X, z = circles([1, 3, 5], [0.1, 0.2, 0.3], [200, 200, 500])
    plot(X, z)
    """

    """
    a = np.random.randint(0, 9, 10)
    b = np.random.randint(10, 19, 10)
    c = np.random.randint(20, 29, 10)
    print a, b, c
    x, y, z = mix_data([a, b, c], 1)
    print x, y, z
    """

    """
    X, z = multivariate_gaussians(
        [[0,0,0], [10,0,0], [1,10,10]], 
        [np.eye(3), np.eye(3), np.eye(3)], 
        [5, 10, 15])
    print X
    print z
    print from_label_to_sets(X, z)

    """

    X, z = spirals([1,-1], [200,200], noise=0.2)
    plot(X, z)

