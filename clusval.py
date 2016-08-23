#!/usr/bin/env python

"""
Clustering with K-Medoids and an analogous of K-Means++ initialization.

Guilherme S. Franca <guifranca@gmail.com>
08/18/2016

"""


from __future__ import division

import numpy as np

from sklearn import datasets
from sklearn.cluster import KMeans

import itertools

from kmeans import kmeans
from kmedoids import kmedoids


def misclassification_error(true_labels, pred_labels):
    """Clustering misclassification error. Gives the percentage
    of wrongly clustered points.
    
    """
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

def MNIST_eval(distance_func, metric_func, 
               numbers=[1,2,3], nrange=[10,100,10], num_avg=10):
    """Return metric evaluation on MNIST dataset.

    distance_func - to be used in k-medoids
    metric_func - metric being evaluated
    numbers - digits chosen in MNIST data set
    nrange - range of N's to be tested, number of data points
    num_avg - number of times we cluster the same points and take
              the average, min, and max 
    
    """
    digits = datasets.load_digits()
    images = digits.images
    kmedoids_metric = []
    kmeans_metric = []
    kmeans_sklearn_metric = []
    
    for n in nrange:
        # generate true labels
        labels = np.concatenate([[m]*n for m in numbers])

        data = np.concatenate([
                (images[np.where(digits.target==i)][np.random.choice(
                    range(173), n)]).reshape((n, 64)) 
                for i in numbers
        ])

        m1 = []; m2 = []; m3 = [];
        for i in range(num_avg):
            j1, _ = kmedoids(len(numbers), distance_func(data))
            j2, _ = kmeans(len(numbers), data)
            km = KMeans(len(numbers))
            j3 = km.fit(data).labels_
            a = metric_func(labels, j1)
            b = metric_func(labels, j2)
            c = metric_func(labels, j3)
            m1.append(a)
            m2.append(b)
            m3.append(c)
        
        kmedoids_metric.append([np.mean(m1), np.min(m1), np.max(m1)])
        kmeans_metric.append([np.mean(m2), np.min(m2), np.max(m2)])
        kmeans_sklearn_metric.append([np.mean(m3), np.min(m3), np.max(m3)])
    
    return kmedoids_metric, kmeans_metric, kmeans_sklearn_metric

def gauss_eval(distance_func, metric_func, nrange=[10,100,10], num_avg=50):
    """Return metric evaluation on gaussian dataset against N.

    Uses the distance_func for K-medoids, and metric_func is the index
    to be evaluated. This is a function that has two arguments, the true
    labels and the clustered labels.
    
    """
    kmedoids_metric = []
    kmeans_metric = []
    kmeans_sklearn_metric = []
    
    # we generate data with n points in each cluster and evaluate 
    # the algorithm
    for n in nrange:
        data = np.concatenate((
            np.random.multivariate_normal([0, 0], [[4,0], [0,1]], n),
            np.random.multivariate_normal([3, 5], [[1,0.8], [0.8,2]], n),
            np.random.multivariate_normal([-2, 3], [[0.5,0], [0,0.5]], n))
        )
        labels = np.concatenate([[m]*n for m in range(3)])

        m1 = []
        m2 = []
        m3 = []
        k = 3
        for i in range(num_avg):
            j1, _ = kmedoids(k, distance_func(data))
            j2, _ = kmeans(k, data)
            
            km = KMeans(k)
            r = km.fit(data)
            j3 = r.labels_

            a = metric_func(labels, j1)
            b = metric_func(labels, j2)
            c = metric_func(labels, j3)
            m1.append(a)
            m2.append(b)
            m3.append(c)

        kmedoids_metric.append([np.mean(m1), np.min(m1), np.max(m1)])
        kmeans_metric.append([np.mean(m2), np.min(m2), np.max(m2)])
        kmeans_sklearn_metric.append([np.mean(m3), np.min(m3), np.max(m3)])
    
    return kmedoids_metric, kmeans_metric, kmeans_sklearn_metric


###############################################################################
if __name__ == '__main__':
    pass
    #numbers = [1, 2, 3]
    #nrange = range(10,100,10)
    #dfunc = euclidean
    #mfunc = metrics.normalized_mutual_info_score
    #m1, m2 = mnist_eval(dfunc, mfunc, numbers, nrange)
    #m1, m2 = gauss_eval(dfunc, mfunc, nrange=[10, 30, 10], num_avg=20)
