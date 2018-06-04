"""Density estimation for the algorithms in 1D."""

# Guilherme Franca <guifranca@gmail.com>
# Neurodata, Johns Hopkins University

import numpy as np
import seaborn.apionly as sns
#import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GMM

import energy.energy1d as energy1d
import energy.data as data
import energy.metric as metric

from customize_plots import *


def kmeans(k, X, run_times=5):
    km = KMeans(k, n_init=run_times)
    km.fit(X)
    return km.labels_

def gmm(k, X, run_times=5):
    gm = GMM(k, n_init=run_times)
    gm.fit(X)
    zh = gm.predict(X)
    return zh

### generate data
n = 1000
n1, n2 = np.random.multinomial(n, [0.5, 0.5])
m1 = 1.5
s1 = 0.3
m2 = 0
s2 = 1.5
#X, z = data.univariate_normal([m1, m2], [s1, s2], [n1, n2])
X, z = data.univariate_lognormal([m1, m2], [s1, s2], [n1, n2])
Y = np.array([[x] for x in X])

### clustering
zh_energy, _ = energy1d.two_clusters1D(X)
zh_kmeans = kmeans(2, Y)
zh_gmm = gmm(2, Y)

### estimated classes
x1_true = X[np.where(z==0)]
x2_true = X[np.where(z==1)]
x1_energy = X[np.where(zh_energy==0)]
x2_energy = X[np.where(zh_energy==1)]
x1_kmeans = X[np.where(zh_kmeans==0)]
x2_kmeans = X[np.where(zh_kmeans==1)]
x1_gmm = X[np.where(zh_gmm==0)]
x2_gmm = X[np.where(zh_gmm==1)]

### doing density estimation and ploting
ax = sns.kdeplot(x1_true, shade=True, linewidth=1, linestyle='-',
                label=r"truth", color='k')
sns.kdeplot(x2_true, shade=True, linewidth=1, linestyle='-', 
                ax=ax, color='k')

sns.kdeplot(x2_energy, shade=False, linewidth=1.5, ax=ax, 
            linestyle='-', color='b',
            label=r"$\mathcal{E}^{1D}$")
sns.kdeplot(x1_energy, shade=False, linewidth=1.5, ax=ax, 
            linestyle='-', color='b')

sns.kdeplot(x2_kmeans, shade=False, linewidth=1.5, ax=ax, 
            linestyle='-', color='r',
            label=r"$k$-means")
sns.kdeplot(x1_kmeans, shade=False, linewidth=1.5, ax=ax, 
            linestyle='-', color='r')

sns.kdeplot(x2_gmm, shade=False, linewidth=1.5, ax=ax, 
            linestyle='-', color='g',
            label=r"GMM")
sns.kdeplot(x1_gmm, shade=False, linewidth=1.5, ax=ax, 
            linestyle='-', color='g')

ax.legend(loc=0, framealpha=.5, handlelength=1)
ax.set_xlabel(r'$x$')
#ax.set_xlim([-6,6])
ax.set_xlim([-2,15])
ax.set_ylim([0,0.7])

#sns.plt.savefig("normal_density.pdf", bbox_inches='tight')
sns.plt.savefig("lognormal_density.pdf", bbox_inches='tight')

