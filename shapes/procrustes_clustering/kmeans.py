
"""K-means, K-means++, and classification error."""

from __future__ import division

import numpy as np
import itertools
import scipy.optimize


def discrete_rv(p):
    """Return an integer according to probability function p."""
    u = np.random.uniform()
    cdf = np.cumsum(p)
    j = np.searchsorted(cdf, u)
    return j

def kpp(K, X, distance):
    """This is the k-means++ initialization proposed by Arthur and
    Vassilvitskii (2007).
    
    """
    N = X.shape[0]
    C = [X[np.random.randint(0, N)]] # centers
    D = np.zeros(N)                  # distances
    for k in range(1, K):
        for n in range(N):
            D[n] = np.array([distance(X[n], c) for c in C]).min()
        p = D/D.sum()
        j = discrete_rv(p)
        C.append(X[j])
    return np.array(C)
    
def kmeans_(K, X, distance, max_iter=50, collapsed=0, max_collapse=5):
    """K is the number of clusters. X is the dataset, which is a matrix
    with one object (another matrix) per row. distance is a function
    which accepts two data points and returns a real number.
    
    """
    N = X.shape[0] # number of points
    mus = kpp(K, X, distance)
    labels = np.empty(N, dtype=np.int8)
    
    converged = False
    count = 0
    while not converged and count < max_iter:
        
        converged = True
        
        # for each vector label it according to the closest centroid
        for n in range(N):
            for k in range(K):
                D = np.array([distance(X[n], mus[k]) for k in range(K)])
            labels[n] = np.argmin(D)

        # sanity test to make sure we don't collapse clusters
        # in very rare cases, due to poor initialization or "bad data"
        # some clusters can collapse and disapear and the algorithm
        # does not converge, this is not part of the standard K-means
        # algorithm
        if np.unique(labels).shape[0] != K:
            collapsed += 1
            if collapsed == max_collapse:
                print "    Warning: clusters collapsed %d times. Aborting ..."\
                    %max_collapse
                J = sum([0.5*distance(x, mus[k]) for k in np.unique(labels)
                                    for x in X[np.where(labels==k)]])
                return labels, mus, J, count
            else: # recursion
                print "    Warning: clusters collapsed. Starting again ..."
                return kmeans_(K, X, distance, max_iter, collapsed,
                                max_collapse)
        
        # update the centroids based on the vectors in the cluster
        for k in range(K):
            old_mu = mus[k]
            new_mu = np.mean(X[np.where(labels==k)], axis=0)
            if not np.array_equal(new_mu, old_mu):
                mus[k] = new_mu
                converged = False
        
        count += 1

    # compute objective
    J = sum([0.5*distance(x, mus[k]) for k in range(K) 
                                     for x in X[np.where(labels==k)]])

    if count == max_iter:
        print "Warning: K-means didn't converge after %i iterations." % count
    
    return labels, mus, J, count

def kmeans(K, X, distance, num_times=5, max_iter=50):
    """Run the algorithm few times and pick the best answer from the objective
    function.
    
    """
    for i in range(num_times):
        cZ, cM, cJ, cN = kmeans_(K, X, distance, max_iter)
        if i == 0 or cJ < J:
            J, Z, M = cJ, cZ, cM
    return Z, M

def accuracy(z, zh):
    """Compute misclassification error, or better the accuracy which is
    1 - error. Use Hungarian algorithm for that, which is O(n^3) instead
    of O(n!). z and zh are vectors with the dimension
    being the number of points, and each entry is the cluster label assigned
    to that point.

    """
    n, k = len(z), len(np.unique(z))
    m, q = len(zh), len(np.unique(zh))

    assert m == n
    
    Q = np.zeros((n,k))
    for i in range(n):
        Q[i, z[i]] = 1
    Qh = np.zeros((n,k))
    for i in range(m):
        Qh[i, zh[i]] = 1
    
    cost_matrix = Qh.T.dot(Q)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-cost_matrix)
    return cost_matrix[row_ind, col_ind].sum()/n

def class_error(true_labels, pred_labels):
    """Clustering misclassification error."""
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(pred_labels)
    min_wrong = np.inf
    for permutation in itertools.permutations(unique_pred):
        f = {a:b for a, b in zip(unique_true, permutation)}
        wrong = 0
        for i in range(len(true_labels)):
            if f[true_labels[i]] != pred_labels[i]:
                wrong += 1
        if wrong < min_wrong:
            min_wrong = wrong
    return min_wrong/len(true_labels)

