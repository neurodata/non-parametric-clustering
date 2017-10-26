"""Gap statistic from Tibshirani, Walther, Hastie.
We implement this for any function that cluster operating on the data,
and also for any kernel method that operates on a gram matrix, with kernel
generating the semimetric from energy statistics.

"""

# Guilherme Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata

from __future__ import division

import numpy as np
from scipy.linalg import eigvalsh
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
import pandas as pd

import energy.initialization as initialization
import energy.eclust as eclust


def draw_uniform(X):
    """Draw data uniformly over the range of the columns of X."""
    n, p = X.shape
    maxr = X.max(axis=0)
    minr = X.min(axis=0)
    Z = np.zeros((n, p))
    for i in range(p):
        Z[:,i] = np.random.uniform(low=minr[i], high=maxr[i], size=n)
    return Z

def draw_svd(X):
    """Draw reference data by centering and using principal components
    of the data. Then generate uniformly over the range.
    
    """
    Xp = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xp)
    Xpp = X.dot(Vt.T)
    Zp = draw_uniform(Xpp)
    Z = Zp.dot(Vt)
    return Z

def kmeans(k, X):
    km = KMeans(k)
    km.fit(X)
    obj_value = np.abs(km.score(X))
    return obj_value

def energy_hartigan(k, X, G, run_times=3):
    best_score = -np.inf
    for rt in range(run_times):
        z0 = initialization.kmeanspp(k, X)
        Z0 = eclust.ztoZ(z0)
        zh = eclust.energy_hartigan(k, G, Z0, max_iter=300)
        Zh = eclust.ztoZ(zh)
        score = eclust.objective(Zh, G)
        if score > best_score:
            best_score = score
    return best_score

def gap_statistics(X, B, K, cluster_func, type_ref='svd'):
    """Implement gap statistics from Tibshrani using any clustering method
    that operates on the original data.
    
    Parameters:
        
        X: data set
        B: number of reference samples
        K: maximum number of clusters 
        cluster_func: function used for clusterin, accept k and X, returns 
            objective function value
        type_ref: reference distribution method {"uniform", "svd"}
    
    """
    K = K+1
    Wks = [] # will contain objective function values for each k
    gaps = [] # will contain the gaps
    sks = [] # will contain the variances
    for k in range(1, K):
    
        # cluster original data
        Wk = cluster_func(k, X)
        Wks.append(Wk)
        
        # generate reference data and cluster this data
        Wkbs = []
        for b in range(B):
            # generate reference distribution
            if type_ref == 'svd':
                Z = draw_svd(X)
            else:
                Z = draw_uniform(X)
            # cluster reference set
            Wkb = cluster_func(k, Z)
            Wkbs.append(Wkb)

        log_Wkbs = np.log(Wkbs)
        l_bar = np.mean(log_Wkbs)

        # gap statistic
        gap = l_bar - np.log(Wk)
        gaps.append(gap)

        # compute variance
        sdk = np.std(log_Wkbs)
        sk = sdk*np.sqrt(1+1/B)
        sks.append(sk)

    # find number of clusters
    for i in range(len(gaps)-1):
        delta = gaps[i] - gaps[i+1] + sks[i+1]
        if delta >= 0:
            break
    k_hat = i+1

    # collect data, for ploting reasons
    d = {'score': np.array(Wks), 'gap': np.array(gaps), 'var': np.array(sks)}
    df = pd.DataFrame(data=d, index=range(1,K))
    gap2 = df['gap'][:-1].values - df['gap'][1:].values + df['var'][1:].values
    df.drop(df.tail(1).index, inplace=True)
    df['gap2'] = pd.Series(gap2, index=df.index)
    return k_hat, df

def elbow(X, G, K, cluster_func):
    """Elbow method."""
    K = K+1
    Wks = [] # will contain objective function values for each k
    ks = range(1, K)
    for k in ks:
        Wk = cluster_func(k, X, G)
        Wks.append(Wk)
    return np.array(Wks)

def eigenvalues(G):
    K = np.copy(G)
    n, _ = K.shape
    for i in range(n):
        K[i,i] = 0
    D = np.diag(K.sum(axis=1))
    lambdas = eigvalsh(K, D)
    return np.flip(lambdas, axis=0)


################################################################################
if __name__ == '__main__':
    import energy.data  as data
    import matplotlib.pyplot as plt
    from customize_plots import *
    import sys

    """
    m1 = np.zeros(10)
    m2 = 1*np.ones(10)
    m3 = 2*np.ones(10)
    s1 = np.eye(10)
    s2 = np.eye(10)
    s3 = np.eye(10)
    n1 = n2 = n3 = 100
    X, z = data.multivariate_lognormal([m1,m2,m3], [s1,s2,s3], [n1,n2,n3])
    """

    df = pd.read_csv("synapse_data/synapse_features.csv")
    X = df.values
    #n, p = X.shape
    #idx = np.random.choice(range(n), 3000)
    #X = Y[idx]
    K = 20

    k_kmeans, df_kmeans = gap_statistics(X, 10, K, kmeans, type_ref="svd")
    print k_kmeans

    rho = lambda a, b: 2 - 2*np.exp(-np.linalg.norm(a-b)/2/16)
    #rho = lambda a, b: np.power(np.linalg.norm(a-b), 0.5)
    G = eclust.kernel_matrix(X, rho)
    energy_scores = elbow(X, G, K, energy_hartigan)
    eig = eigenvalues(G)

    fig = plt.figure(figsize=(4*fig_width, 1*fig_height))
    
    ax = fig.add_subplot(141)
    ys = df_kmeans["gap2"].values
    xs = range(len(ys))
    ax.plot(xs, ys, '-oy')

    ax = fig.add_subplot(142)
    ys = energy_scores[:K]
    xs = range(len(ys))
    ax.plot(xs, ys, '-ob')
    
    ax = fig.add_subplot(143)
    ys = eig[:K]
    xs = range(len(ys))
    ax.plot(xs, ys, '-or')
    
    ax = fig.add_subplot(144)
    ys = np.exp((eig[:K] - eig[1:K+1]))
    xs = range(len(ys))
    ax.plot(xs, ys, '-og')
    
    fig.savefig('synapse_gap_test.pdf')

