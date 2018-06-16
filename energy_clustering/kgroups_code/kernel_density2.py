"""Density estimation for the algorithms in 1D."""

# Guilherme Franca <guifranca@gmail.com>
# Johns Hopkins University

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from prettytable import PrettyTable

import sys

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
bw = 0.5 # bandwidth
num_points = 1500 # number points for linspace
#low = -6
#high = 6
low = -2
high = 20 

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

X_plot = np.linspace(low,high,num_points)[:,np.newaxis]

### kernel density estimation
x1_true = X[np.where(z==0)][:, np.newaxis]
x2_true = X[np.where(z==1)][:, np.newaxis]

fig = plt.figure()
ax = fig.add_subplot(111)

kde1 = KernelDensity(kernel='gaussian', bandwidth=bw).fit(x1_true)
log_dens1 = kde1.score_samples(X_plot)
kde2 = KernelDensity(kernel='gaussian', bandwidth=bw).fit(x2_true)
log_dens2 = kde2.score_samples(X_plot)
ax.fill_between(X_plot[:,0], np.exp(log_dens1), alpha=.3, color='k')
ax.fill_between(X_plot[:,0], np.exp(log_dens2), alpha=.3, color='k')

r = []
x1_kmeans = X[np.where(zh_kmeans==0)][:, np.newaxis]
x2_kmeans = X[np.where(zh_kmeans==1)][:, np.newaxis]
r.append([x1_kmeans, x2_kmeans])
x1_gmm = X[np.where(zh_gmm==0)][:, np.newaxis]
x2_gmm = X[np.where(zh_gmm==1)][:, np.newaxis]
r.append([x1_gmm, x2_gmm])
x1_kgroups = X[np.where(zh_kgroups==0)][:, np.newaxis]
x2_kgroups = X[np.where(zh_kgroups==1)][:, np.newaxis]
r.append([x1_kgroups, x2_kgroups])

methods = ['k-means', 'GMM', 'kernel k-groups']
#c = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
colors = ['#1F77B4', '#FF7F0E', '#2CA02C']
s = ['^', 'v', 'o']

for i, (d, c, m) in enumerate(zip(r, colors, methods)):
    kde1 = KernelDensity(kernel='gaussian', bandwidth=bw).fit(d[0])
    log_dens1 = kde1.score_samples(X_plot)
    kde2 = KernelDensity(kernel='gaussian', bandwidth=bw).fit(d[1])
    log_dens2 = kde2.score_samples(X_plot)
    ax.plot(X_plot[:,0], np.exp(log_dens1), label='%s class 1'%m)#, color=c)
    ax.plot(X_plot[:,0], np.exp(log_dens2), label='%s class 2'%m)#, color=c)

ax.legend()
ax.set_xlabel(r'$x$')
ax.set_xlim([low,high])
#ax.set_ylim([0,0.7])
ax.set_ylim([0,0.5])

#plt.savefig("normal_density.pdf", bbox_inches='tight')
plt.savefig("lognormal_density.pdf", bbox_inches='tight')

