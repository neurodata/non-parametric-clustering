
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GMM 
import matplotlib.pyplot as plt

import gap

import sys

#f = open('data.txt')
#X = np.array([[float(l.strip())] for l in f.readlines()])

X = np.concatenate((
        np.random.normal(0, 1, 10),
        np.random.normal(5, .5, 10),
        np.random.normal(10, 1, 10)
    ))
X = np.array([[x] for x in X])

ks = range(1, 10)
kmeans_scores = []
for k in ks:
    km = KMeans(n_clusters=k)
    km.fit(X)
    kmeans_scores.append(-km.score(X))

gmm_scores = []
bic_scores = []
for k in ks:
    gmm = GMM(n_components=k)
    gmm.fit(X)
    gmm_scores.append(-gmm.score(X).sum())
    bic_scores.append(gmm.bic(X))

_, kmeans_gaps = gap.kmeans_gap(X, nrefs=50, max_clusters=10)
_, gmm_gaps = gap.gmm_gap(X, nrefs=50, max_clusters=10)

fig = plt.figure(figsize=(3*6, 2*4))
ax = fig.add_subplot(231)
ax.plot(ks, kmeans_scores, 'b-o', lw=2, ms=8)
ax.set_xlabel('number of clusters')
ax.set_ylabel('k-means objective')

ax = fig.add_subplot(232)
ax.plot(ks, kmeans_gaps, 'b-o', lw=2, ms=8)
ax.set_xlabel('number of clusters')
ax.set_ylabel('k-means gap statistic')

ax = fig.add_subplot(234)
ax.plot(ks, gmm_scores, 'r-o', lw=2, ms=8)
ax.set_xlabel('number of clusters')
ax.set_ylabel('gmm log likelihood')

ax = fig.add_subplot(235)
ax.plot(ks, bic_scores, 'r-o', lw=2, ms=8)
ax.set_xlabel('number of clusters')
ax.set_ylabel('gmm BIC')


ax = fig.add_subplot(236)
ax.plot(ks, kmeans_gaps, 'r-o', lw=2, ms=8)
ax.set_xlabel('number of clusters')
ax.set_ylabel('gmm gap statistic')

fig.savefig('gaussians.pdf')
#fig.savefig('areas.pdf')
