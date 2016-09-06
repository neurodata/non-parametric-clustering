#!/usr/bin/env python

"""
Evaluation and comparison of K-Means++ and K-Medoids++ on different data sets.

"""


from __future__ import division

import numpy as np

from sklearn import datasets
from sklearn.cluster import KMeans

import itertools

import kmeans
import kmedoids
import distance


def class_error(true_labels, pred_labels):
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

def MNIST_eval_euclidean(metric_func, numbers=[1,2,3], 
                         nrange=range(10,100,10), num_avg=10):
    """Return metric evaluation on MNIST dataset using Euclidean distance
    on all the algorithms.

    Input
    -----
    metric_func - metric being evaluated
    numbers - digits chosen in MNIST data set
    nrange - range of N's to be tested, number of data points
    num_avg - number of times we cluster the same points and take
              the average, min, and max 
    
    Output
    ------
    kmedoids_metric - metric computed with K-medoids
    kmeans_metric - metric computed with K-means
    kmeans_sklearn_metric - metric with kmeans from sklearn
    
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
          images[np.where(digits.target==i)][np.random.choice(range(173), n)] 
          for i in numbers
        ])
        data2 = data.reshape(len(data), 64)

        m1 = []; m2 = []; m3 = [];
        for i in range(num_avg):
            # our algorithms
            j1, _ = kmedoids.kmedoids(len(numbers),
                                      distance.euclidean_matrix(data2))
            j2, _ = kmeans.kmeans(len(numbers), data2, distance.euclidean)
            
            # sklearn k-means
            km = KMeans(len(numbers))
            j3 = km.fit(data2).labels_
            
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

def MNIST_eval_procrustes(metric_func, numbers=[1,2,3],
                          nrange=range(10,100,10), num_avg=5):
    """Evaluate metric and compare with different algorithms. In our
    implementation of K-medoids and K-means we use procrustes distance. We 
    compare with kmeans/sklearn.

    Input
    -----
    metric_func - metric being evaluated
    numbers - digits chosen in MNIST data set
    nrange - range of N's to be tested, number of data points
    num_avg - number of times we cluster the same points and take
              the average, min, and max 
    
    Output
    ------
    kmedoids_procrustes_metric - metric computed with K-medoids
    kmeans_procrustes_metric - metric computed with K-means
    kmeans_sklearn_metric - metric with kmeans from sklearn
    
    """
    digits = datasets.load_digits()
    images = digits.images
    
    m1 = []; m2 = []; m3 = [];
    for n in nrange:
        # generate true labels
        labels = np.concatenate([[m]*n for m in numbers])
        
        data = np.concatenate([
            images[np.where(digits.target==i)][np.random.choice(range(173), n)] 
            for i in numbers
        ])
        data2 = data.reshape(len(data), 64)
        
        j1, _ = kmedoids.kmedoids(len(numbers),
                                        distance.procrustes_matrix(data))
        j2, _ = kmeans.kmeans(len(numbers), data, distance.procrustes)
            
        km = KMeans(len(numbers))
        j3 = km.fit(data2).labels_
            
        m1.append(metric_func(labels, j1))
        m2.append(metric_func(labels, j2))
        m3.append(metric_func(labels, j3))
        
    return m1, m2, m3

def gauss_eval(dist_matrix_kmedoids, dist_func_kmeans, metric_func, 
               nrange=range(10,100,10), num_avg=5):
    """Return metric evaluation on gaussian dataset against N.
    Compare K-medoids and K-means.
    
    Input
    -----
    dist_matrix_kmedoids - function to generate the distance matrix for
                            kmedoids
    dist_func_kmeans - distance function to be used in kmeans
    metric_func - metric function being evaluated
    nrange - range of N's to be tested, number of data points
    num_avg - number of times we cluster the same points and take
              the average, min, and max 
    
    Output
    ------
    kmedoids_metric - metric computed with K-medoids
    kmeans_metric - metric computed with K-means
    kmeans_sklearn_metric - metric with kmeans from sklearn
    
    """
    kmedoids_metric = []
    kmeans_metric = []
    kmeans_sklearn_metric = []
    
    # we generate data with n points in each cluster and evaluate 
    # the algorithms
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
            j1, _ = kmedoids.kmedoids(k, dist_matrix_kmedoids(data))
            j2, _ = kmeans.kmeans(k, data, dist_func_kmeans)
            
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


if __name__ == '__main__':
    #a, b, c = MNIST_eval_euclidean(class_error, numbers=[3,4,5], 
    #                     nrange=range(10,100,10), num_avg=5)
    #print a, b, c
    
    a, b, c = MNIST_eval_procrustes(class_error, numbers=[3,4,5],
                          nrange=range(10,50,10), num_avg=5)
    print a, b, c
    
