"""Density estimation for the algorithms in 1D."""

# Guilherme Franca <guifranca@gmail.com>
# Johns Hopkins University

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from prettytable import PrettyTable
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
import scipy.stats

import sys

import wrapper
import data
import eclust
import metric


### parameters ################################################################

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
#ymax = 1.1
low = -2
high = 20 
ymax = 0.5

methods = ['k-means', 'GMM', 'kernel k-groups']
colors = ["#54278f", "#5F8793", "#F41711"]

#fname = "normal_density2.pdf"
fname = "lognormal_density2.pdf"

###############################################################################

t = PrettyTable(['Method', 'Accuracy'])

km = KMeans(k, n_init=5)
km.fit(Y)
zh_kmeans = km.labels_
x1_kmeans = X[np.where(zh_kmeans==0)][:, np.newaxis]
x2_kmeans = X[np.where(zh_kmeans==1)][:, np.newaxis]
x1_mu_kmeans, x2_mu_kmeans = km.cluster_centers_
x1_mu_kmeans, x2_mu_kmeans = x1_mu_kmeans[0], x2_mu_kmeans[0]
x1_var_kmeans, x2_var_kmeans = np.var(x1_kmeans), np.var(x2_kmeans)
acc_kmeans = metric.accuracy(z, zh_kmeans)
t.add_row(['k-means', acc_kmeans])

gm = GMM(k, n_init=5, init_params="kmeans")
gm.fit(Y)
zh_gmm = gm.predict(Y)
#x1_gmm = X[np.where(zh_gmm==0)][:, np.newaxis]
#x2_gmm = X[np.where(zh_gmm==1)][:, np.newaxis]
x1_mu_gmm, x2_mu_gmm = gm.means_
x1_mu_gmm, x2_mu_gmm = x1_mu_gmm[0], x2_mu_gmm[0]
x1_var_gmm, x2_var_gmm = gm.covariances_
x1_var_gmm, x2_var_gmm = x1_var_gmm[0][0], x2_var_gmm[0][0]
acc_gmm = metric.accuracy(z, zh_gmm)
t.add_row(['gmm', acc_gmm])

G = eclust.kernel_matrix(Y, lambda x, y: np.linalg.norm(x-y))
zh_kgroups = wrapper.kernel_kgroups(k, Y, G)
x1_kgroups = X[np.where(zh_kgroups==0)][:, np.newaxis]
x2_kgroups = X[np.where(zh_kgroups==1)][:, np.newaxis]
acc_kgroups = metric.accuracy(z, zh_kgroups)
t.add_row(['kernel k-groups', acc_kgroups])

print t

### kernel density estimation for truth
X_plot = np.linspace(low,high,num_points)[:,np.newaxis]
x1_true = X[np.where(z==0)][:, np.newaxis]
x2_true = X[np.where(z==1)][:, np.newaxis]

fig = plt.figure()
ax = fig.add_subplot(111)

kde1 = KernelDensity(kernel='gaussian', bandwidth=bw).fit(x1_true)
log_dens1 = kde1.score_samples(X_plot)
kde2 = KernelDensity(kernel='gaussian', bandwidth=bw).fit(x2_true)
log_dens2 = kde2.score_samples(X_plot)
ax.fill_between(X_plot[:,0], np.exp(log_dens1), alpha=.3, color='k')
ax.plot(X_plot[:,0], np.exp(log_dens1), color='k', label='truth')
ax.fill_between(X_plot[:,0], np.exp(log_dens2), alpha=.3, color='k')
ax.plot(X_plot[:,0], np.exp(log_dens2), color='k')

xs = np.linspace(low,high,num_points)
ax.plot(xs, scipy.stats.norm.pdf(xs, x1_mu_kmeans, np.sqrt(x1_var_kmeans)),
        label="%s"%(methods[0]), color=colors[0])
ax.plot(xs, scipy.stats.norm.pdf(xs, x2_mu_kmeans, np.sqrt(x2_var_kmeans)),
        color=colors[0])
ax.plot(xs, scipy.stats.norm.pdf(xs, x1_mu_gmm, np.sqrt(x1_var_gmm)),
        label="%s"%(methods[1]), color=colors[1])
ax.plot(xs, scipy.stats.norm.pdf(xs, x2_mu_gmm, np.sqrt(x2_var_gmm)),
        color=colors[1])

kde1 = KernelDensity(kernel='gaussian', bandwidth=bw).fit(x1_kgroups)
log_dens1 = kde1.score_samples(X_plot)
kde2 = KernelDensity(kernel='gaussian', bandwidth=bw).fit(x2_kgroups)
log_dens2 = kde2.score_samples(X_plot)
ax.plot(X_plot[:,0], np.exp(log_dens1), 
        label="%s"%(methods[2]), color=colors[2])
ax.plot(X_plot[:,0], np.exp(log_dens2), color=colors[2])



"""
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


for i, (d, c, m) in enumerate(zip(r, colors, methods)):
    kde1 = KernelDensity(kernel='gaussian', bandwidth=bw).fit(d[0])
    log_dens1 = kde1.score_samples(X_plot)
    kde2 = KernelDensity(kernel='gaussian', bandwidth=bw).fit(d[1])
    log_dens2 = kde2.score_samples(X_plot)
    ax.plot(X_plot[:,0], np.exp(log_dens1), label=m, color=c)
    ax.plot(X_plot[:,0], np.exp(log_dens2), color=c)
"""

ax.legend()
ax.set_ylabel(r'density')
ax.set_xlabel(r'$x$')
ax.set_xlim([low,high])
ax.set_ylim([0,ymax])

plt.savefig(fname, bbox_inches='tight')

