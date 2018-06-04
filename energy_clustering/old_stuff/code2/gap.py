"""Methods do find number of clusters."""

# Guilherme Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata

from __future__ import division

import numpy as np
from scipy.linalg import eigvalsh
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
import pandas as pd
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from sklearn.decomposition import PCA

import seaborn.apionly as sns

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

def energy_hartigan(k, X, G, run_times=5, return_labels=False):
    best_score = -np.inf
    for rt in range(run_times):
        z0 = initialization.kmeanspp(k, X)
        Z0 = eclust.ztoZ(z0)
        zh = eclust.energy_hartigan(k, G, Z0, max_iter=300)
        Zh = eclust.ztoZ(zh)
        score = eclust.objective(Zh, G)
        if score > best_score:
            best_score = score
            best_zh = zh
    if return_labels:
        return best_zh
    else:
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
    log_Wkbs_sem = [] # standard error from the mean
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
        log_Wkbs_sem.append(stats.sem(log_Wkbs))
        l_bar = np.mean(log_Wkbs)

        # gap statistic
        gap = l_bar - np.log(Wk)
        gaps.append(gap)

        # compute variance
        sdk = np.std(log_Wkbs)
        sk = sdk*np.sqrt(1+1/B)
        sks.append(sk)

    # find number of clusters
    k_hat = 0
    for k in range(len(gaps)-1):
        if gaps[k] >= gaps[k+1] - sks[k+1]:
            k_hat = k
            break
    k_hat = k_hat+1

    # collect data, for ploting purposes 
    d = {
            'score': np.array(Wks), 
            'gap': np.array(gaps), 
            'var': np.array(sks),
            'sem': np.array(log_Wkbs_sem)
    }
    df = pd.DataFrame(data=d, index=range(1,K))
    gap_k = df['gap'][:-1].values
    gap_k_plus_1 = df['gap'][1:].values
    sigma_k_plus_1 = df['var'][1:].values
    gap2 = gap_k - (gap_k_plus_1 - sigma_k_plus_1)
    df.drop(df.tail(1).index, inplace=True)
    df['gap2'] = pd.Series(gap2, index=df.index)
    return k_hat, df

def kernel_gap_statistics(X, B, K, cluster_func, rho, type_ref='svd'):
    """Implement gap statistics from Tibshrani using any clustering method
    that operates on a kernel matrix defined by a semi metric.

    ================================================================
    NOTE: for some reason this method does not work well.
    There is a huge variability when clustering the reference data,
    specially when using the SVD approach.
    
    DON'T USE THIS METHOD!
    ================================================================
    
    Parameters:
        
        X: data set
        B: number of reference samples
        K: maximum number of clusters 
        cluster_func: function used for clusterin, accept k and X and G, 
            returns objective function value
        rho: the semimetric to generate pairwise kernel matrix
        type_ref: reference distribution method {"uniform", "svd"}
    
    """
    K = K+1
    Wks = [] # will contain objective function values for each k
    gaps = [] # will contain the gaps
    sks = [] # will contain the variances
    log_Wkbs_sem = [] # standard error from the mean
    
    G = eclust.kernel_matrix(X, rho) # generate kernel on data

    for k in range(1, K):
    
        # cluster original data
        Wk = cluster_func(k, X, G)
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
            G_ref = eclust.kernel_matrix(Z, rho) 
            Wkb = cluster_func(k, Z, G_ref)
            Wkbs.append(Wkb)
        log_Wkbs = np.log(Wkbs)
        l_bar = np.mean(log_Wkbs)
        log_Wkbs_sem.append(stats.sem(log_Wkbs))

        # gap statistic
        gap = l_bar - np.log(Wk)
        gaps.append(gap)

        # compute variance
        sdk = np.std(log_Wkbs)
        sk = sdk*np.sqrt(1+1/B)
        sks.append(sk)

    # find number of clusters
    #for i in range(len(gaps)-1):
    #    delta = gaps[i] - gaps[i+1] - sks[i+1]
    #    if delta <= 0:
    #        break
    #k_hat = i+1

    # collect data, for ploting purposes 
    d = {
            'score': np.array(Wks), 
            'gap': np.array(gaps), 
            'var': np.array(sks),
            'sem': np.array(log_Wkbs_sem)
    }
    df = pd.DataFrame(data=d, index=range(1,K))
    gap_k = df['gap'][:-1].values
    gap_k_plus_1 = df['gap'][1:].values
    sigma_k_plus_1 = df['var'][1:].values
    gap2 = gap_k - (gap_k_plus_1 - sigma_k_plus_1)
    df.drop(df.tail(1).index, inplace=True)
    df['gap2'] = pd.Series(gap2, index=df.index)
    return df

def elbow_kernel(X, G, K, cluster_func):
    """Elbow method for a kernel method that operates on kernel matrix G.
    K is the maximum number of clusters.
    
    """
    K = K
    Wks = [] 
    ks = range(1, K)
    for k in ks:
        Wk = cluster_func(k, X, G)
        Wks.append(Wk)
    return np.array(Wks)

def eigenvalues(G, num):
    """Compute eigenvalues of kernel matrix G. Return the gaps."""
    K = np.copy(G)
    n, _ = K.shape
    #for i in range(n):
    #    K[i,i] = 0
    D = np.diag(K.sum(axis=1))
    #lambdas = eigvalsh(K, D)
    lambdas = eigvalsh(D-K, D)
    #lambdas = np.flip(lambdas, axis=0) # largest eigenvalues first
    deltas = np.abs(lambdas[1:]-lambdas[:-1])
    return deltas[:num]
    #return lambdas[:num]
   
def plot_gap(infile="experiments_data2/energy_synapse_gap.csv", 
             output="gap.pdf", 
             xlabel="$k$", 
             ylabel1=r"$g_k-\left( g_{k+1} - \sigma_{k+1} \right)$", 
             ylabel2=r"$J_k$"):
    df = pd.read_csv(infile, dtype=float)
    fig = plt.figure(figsize=(fig_width, fig_height))
    # plot gaps
    ax = fig.add_subplot(111)
    xs = range(1,len(df["gap"].values)+1)
    #ax.errorbar(xs, df["gap"].values, yerr=df["var"].values, color="b",
    #            linestyle='-', marker="o", markersize=5, elinewidth=.5,
    #            capthick=0.5, linewidth=1.5, barsabove=False)
    ax.plot(xs, df["gap2"].values, color="b", linestyle="-", linewidth=1.5,
                marker="o", markersize=5)
    ax.plot(xs, [0]*len(xs), linestyle='--', color='k')
    ax.set_xlim([1, len(xs)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel1)
    #with sns.axes_style("whitegrid"):
    axins = inset_axes(ax, width="50%", height="50%", loc=5, 
                        borderpad=1)
    axins.plot(xs[4:10], df["gap2"].values[4:10], 
                   color='b', linestyle='-', 
                   marker='o', linewidth=1.5, markersize=5, 
                   alpha=0.7)
    axins.set_xticks([5,6,7,8,9])
    axins.set_yticks([])
    fig.savefig(output, bbox_inches='tight')

def plot_eigenvalue_gaps(
        gaps, 
        xlabel=r"$k$", 
        ylabel=r"$\big|\lambda_{k+1}-\lambda_k\big|$", 
        #ylabel=r"$\lambda_{k}$", 
        output="eigen_gap.pdf",
        colors=['b', 'r', 'g', 'c'],
        legend=[r'$\rho_{1}$', r'$\widehat{\rho}_{2}$'],
        marker=['o', '^', 'v']):
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)
    for i, g in enumerate(gaps):
        xs = range(1, len(g)+1)
        ax.plot(xs, g, color=colors[i], linestyle='-', marker=marker[i],
                linewidth=1.5, markersize=5, label=legend[i], alpha=0.7)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xlim([1, len(xs)])
    #ax.legend(loc=(0.18,0.6), framealpha=.5, ncol=1)
    ax.legend(loc=0, framealpha=.5, ncol=1)

    #with sns.axes_style("whitegrid"):
    axins = inset_axes(ax, width="50%", height="50%", loc=7, 
                        borderpad=1)
    for i, g in enumerate(gaps):
        axins.plot(xs[2:10], g[2:10], color=colors[i], linestyle='-', 
                       marker=marker[i], linewidth=1.5, markersize=5, 
                       alpha=0.7)
    axins.set_xlim([4,8])
    axins.set_xticks([4,5,6,7,8])
    #axins.set_yticks([])
    #axins.set_ylim([0,0.006])

    fig.savefig(output, bbox_inches='tight')

def plot_elbow_kernel(
        values, 
        xlabel=r"$k$", 
        #ylabel=r"$\textnormal{Tr}(Y^\top G \, Y)$", 
        ylabel=r"$\log Q_{k+1} - \log Q_{k}$", 
        output="elbow.pdf",
        colors=['b', 'r', 'g'],
        legend=[r'$\widehat{\rho}_{\sqrt{7}}$', r'$\rho_{1}$'],
        marker=['o', 's']):
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)
    for i, g in enumerate(values):
        xs = range(1, len(g)+1)
        ax.plot(xs, g, color=colors[i], linestyle='-', marker=marker[i],
                linewidth=1.5, markersize=5, label=legend[i], alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([1, 12])
    ax.legend(loc=2, framealpha=.5, ncol=1)
    
    with sns.axes_style("whitegrid"):
        axins = inset_axes(ax, width="50%", height="50%", loc=1)
        for i, g in enumerate(values):
            axins.plot(xs[2:10], g[2:10], color=colors[i], linestyle='-', 
                    marker=marker[i], linewidth=1.5, markersize=5, alpha=0.7)
    #axins.set_xlim([5,12])
    #axins.set_ylim([19.9,20])
    #axins.set_xticks([])
    #axins.set_yticks([])
    fig.savefig(output, bbox_inches='tight')

def cluster_synapse(k, X, G):
    zh = energy_hartigan(k, X, G, run_times=10, return_labels=True)
    means = []
    for j in range(k):
        idx = np.where(zh==j)
        Y = X[idx]
        means.append(Y.mean(axis=0))
    return zh, np.array(means)

def plot_synapse_2d(X, zh, output='synapse_cluster_2d.pdf'):
    pca = PCA(n_components=2)
    X_new = pca.fit_transform(X)
    d = {
            #r'$x_1$': X_new[:,0]/(10**5), 
            r'$x_1$': X_new[:,0], 
            #r'$x_2$': X_new[:,1]/(10**6),
            r'$x_2$': X_new[:,1],
            r'$\mathcal{C}_j$': [int(z+1) for z in zh]
    }
    df = pd.DataFrame(data=d)
    g = sns.lmplot(r'$x_1$', r'$x_2$', 
                   data=df, hue=r'$\mathcal{C}_j$', 
                   fit_reg=False, scatter=True, scatter_kws={"s":4})
    g.set(xlabel=r'$x_1$')
    g.set(ylabel=r'$x_2$')
    #g.set(xlabel=r'$x_1$~$(\times 10^6)$')
    #g.set(ylabel=r'$x_2$~$(\times 10^5)$')
    g.savefig(output)

def plot_synapse_pairs(X, zh, output='synapse_pairs.pdf'):
    sns.set_style({"font.size": 10, "axes.labelsize": 30})
    d = {'z': zh}
    n, p = X.shape
    for i in range(p):
        d[r'$x_{%i}$'%(i+1)] = X[:,i]
        #d[r'$x_{%i}$'%(i+1)] = X[:,i]/(10**5)
    df = pd.DataFrame(data=d)
    g = sns.PairGrid(df, hue="z",vars=[r'$x_{%i}$'%(i+1) for i in range(p)])

    def scatter_fake_diag(x, y, *a, **kw):
        if x.equals(y):
            kw["color"] = (0, 0, 0, 0)
        plt.scatter(x, y, s=3, *a, **kw)

    g.map(scatter_fake_diag)
    g.map_diag(plt.hist)
    g.savefig(output)


################################################################################
if __name__ == '__main__':
    import energy.data  as data
    import matplotlib.pyplot as plt
    #sns.set_style("ticks", {"xtick.direction":"in", "ytick.direction": "in"})
    from customize_plots import *
    import sys

    # syntetic data for testing
    m1 = np.zeros(10)
    m2 = 2*np.ones(10)
    m3 = 4*np.ones(10)
    m4 = 5*np.ones(10)
    s1 = np.eye(10)
    s2 = np.eye(10)
    s3 = np.eye(10)
    s4 = np.eye(10)
    n1 = n2 = n3 = n4 = 30
    X, z = data.multivariate_normal([m1,m2,m3,m4], [s1,s2,s3,s4], [n1,n2,n3,n4])

    # max number of clusters
    K = 10
    
    # gap statistic with k-means
    k_hat, df_kmeans = gap_statistics(X, B=50, K=K, cluster_func=kmeans, 
                                         type_ref="svd")
    print k_hat

    rho = lambda x, y: np.power(np.linalg.norm(x-y), 1)
    G = eclust.kernel_matrix(X, rho)
    gaps = eigenvalues(G, K)
    
    rho2 = lambda x, y: 2-2*np.exp(-np.power(np.linalg.norm(x-y),2)/2/((2)**2))
    G2 = eclust.kernel_matrix(X, rho2)
    gaps2 = eigenvalues(G2, K)
    
    plot_eigenvalue_gaps([gaps, gaps2], output="eigen_gap.pdf")

    #elb2 = elbow_kernel(X, G2, K+2, cluster_func=energy_hartigan)
    #elb2 = np.log(elb2)
    #elb2 = elb2[1:] - elb2[:-1]
    #elb3 = elbow_kernel(X, G3, K+2, cluster_func=energy_hartigan)
    #elb3 = np.log(elb3)
    #elb3 = elb3[1:] - elb3[:-1]
    #elb4 = elbow_kernel(X, G4, K+2, cluster_func=energy_hartigan)
    #elb4 = np.log(elb4)
    #elb4 = elb4[1:] - elb4[:-1]
    #plot_elbow_kernel([elb2, elb3], output='elbow.pdf')

    
