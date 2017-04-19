"""Optimize Energy Clustering through iterative procedure. This idea
is the same as graph partitioning. This algorithm is extremelly slow.

"""

# Guilherme Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata

from __future__ import division

import numpy as np
from sklearn.cluster import KMeans
import numbers

from energy import energy
import kmeanspp


class EClust:

    def __init__(self, n_clusters, max_iter=50, init='kmeans++',
                    labels=None):
        self.n_clusters = n_clusters
        self.max_iter= max_iter
        self.init = init
        self.labels_ = labels

    def initial_labels(self):
        if self.labels_ is not None:
            pass
        elif self.init == 'kmeans++':
            if isinstance(self.X[0], numbers.Number):
                Y = np.array([[x] for x in self.X])
            else:
                Y = self.X
            self.labels_ = kmeanspp.kpp(self.n_clusters, Y)
        elif self.init == 'random':
            self.labels_ = np.random.choice(range(self.n_clusters), len(self.X))
        else:
            self.labels_ = np.random.choice(range(self.n_clusters), len(X))

    def fit(self, X):
        """Fit data X against our algorithm. It will set self.labels_"""
        # main algorithm
        # go through each point and compute the cost with its current
        # partition and minimal cost with other partitions
        # move label in case there is a smaller one
        # repeat until convergence
        self.X = X
        self.initial_labels()
        count = 0
        converged = False
        while not converged and count < self.max_iter:
            converged = True 
            for i, k in enumerate(self.labels_):
                point = self.X[i]
                curr_cost = self.cost_current_partition(point, k)
                min_cost, j = self.min_cost_other_partitions(point, k)
                if not (curr_cost < min_cost):
                    self.labels_[i] = j
                    converged = False
            count += 1
        self.count = count
        if count >= self.max_iter:
            print "Warning: didn't converge in %i iterations." % self.max_iter

    def fit_predict(self, X):
        """Fit and return predicted labels."""
        self.fit(X)
        return self.labels_

    def cost_current_partition(self, point, label):
        """Compute the cost of point with its current partition."""
        points_in_partition = self.X[np.where(self.labels_ == label)]
        n = len(points_in_partition)
        cost = energy([point], points_in_partition)*(n/(n-1))
        return cost

    def min_cost_other_partitions(self, point, label):
        """Compute the minimal cost between 'point' with all the other
        partitions.
        
        """
        costs = []
        for j in range(self.n_clusters):
            if j == label:
                costs.append(np.inf)
            else:
                points_in_partition = self.X[np.where(self.labels_ == j)]
                n = len(points_in_partition)
                cost = energy([point], points_in_partition)*(n/(n+1))
                costs.append(cost)
        costs = np.array(costs)
        min_index = costs.argmin()
        min_cost = costs[min_index]
        return min_cost, min_index


###############################################################################
if __name__ == '__main__':
    import data
    from metric import accuracy

    m1 = np.array([0,0])
    s1 = np.array([[1,0],[0,1]])
    n1 = 100

    m2 = np.array([3,0])
    s2 = np.array([[1,0],[0,10]])
    n2 = 100

    X, true_labels = data.multivariate_normal([m1,m2], [s1,s2], [n1,n2])
    
    ec = EClust(n_clusters=2, max_iter=10, init='kmeans++')
    labels = ec.fit_predict(X)
    print accuracy(labels, true_labels)

    km = KMeans(2)
    labels2 = km.fit_predict(X)
    print accuracy(labels2, true_labels)

