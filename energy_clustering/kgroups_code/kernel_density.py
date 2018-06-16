"""Density estimation for the algorithms in 1D."""

# Guilherme Franca <guifranca@gmail.com>
# Johns Hopkins University

import numpy as np
#import seaborn.apionly as sns
import seaborn as sns
import matplotlib.pyplot as plt
from prettytable import PrettyTable

import wrapper
import data
import eclust
import metric


### generate data
k = 2
n = 2000
n1, n2 = np.random.multinomial(n, [0.5, 0.5])
m1 = 0
s1 = 1.5
m2 = 1.5
s2 = 0.3
#X, z = data.univariate_normal([m1, m2], [s1, s2], [n1, n2])
X, z = data.univariate_lognormal([m1, m2], [s1, s2], [n1, n2])
Y = np.array([[x] for x in X])

### clustering
t = PrettyTable(['Method', 'Accuracy'])
G = eclust.kernel_matrix(Y, lambda x, y: np.linalg.norm(x-y))
zh_kmeans = wrapper.kmeans(k, Y)
t.add_row(['k-means', metric.accuracy(z, zh_kmeans)])
zh_gmm = wrapper.gmm(k, Y)
t.add_row(['gmm', metric.accuracy(z, zh_gmm)])
zh_kgroups = wrapper.kernel_kgroups(k, Y, G)
t.add_row(['kernel k-groups', metric.accuracy(z, zh_kgroups)])
print t

### estimated classes
x1_true = X[np.where(z==0)]
x2_true = X[np.where(z==1)]

x1_kmeans = X[np.where(zh_kmeans==0)]
x2_kmeans = X[np.where(zh_kmeans==1)]

x1_gmm = X[np.where(zh_gmm==0)]
x2_gmm = X[np.where(zh_gmm==1)]

x1_kgroups = X[np.where(zh_kgroups==0)]
x2_kgroups = X[np.where(zh_kgroups==1)]

#c = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
c = ['#1F77B4', '#FF7F0E', '#2CA02C']
s = ['^', 'v', 'o']
me = 10

### doing density estimation and ploting
ax = sns.kdeplot(x1_true, shade=True, label=r"truth", color='k')
sns.kdeplot(x2_true, shade=True, ax=ax, color='k')

sns.kdeplot(x1_kmeans, shade=False, ax=ax, label=r'k-means', color=c[0],
            linewidth=2)
sns.kdeplot(x2_kmeans, shade=False, ax=ax, color=c[0], linewidth=2)

sns.kdeplot(x1_gmm, shade=False, ax=ax, label=r'GMM', color=c[1], linewidth=2)
sns.kdeplot(x2_gmm, shade=False, ax=ax, color=c[1], linewidth=2)

sns.kdeplot(x1_kgroups, shade=False, ax=ax, label=r'kernel k-groups',
            color=c[2], linewidth=2)
sns.kdeplot(x2_kgroups, shade=False, ax=ax, color=c[2], linewidth=2)

#ax.legend(loc=0, framealpha=.5, handlelength=1)
ax.legend()
ax.set_xlabel(r'$x$')
#ax.set_xlim([-6,6])
ax.set_xlim([-2,20])
#ax.set_ylim([0,1.4])
ax.set_ylim([0,0.72])

#plt.savefig("normal_density.pdf", bbox_inches='tight')
plt.savefig("lognormal_density.pdf", bbox_inches='tight')

