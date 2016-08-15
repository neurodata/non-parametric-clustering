#!/usr/bin/env python

"""Image recognition using clustering."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import datasets

from kmedoids import kmedoids
from kmeans import kmeans

def plot_sample(images):
    # a sample from the the data set
    fig = plt.figure()
    for i, img in enumerate(images[0:80]):
        ax = fig.add_subplot(8, 10, i+1)
        ax.imshow(img, cmap=plt.cm.gray_r)
        ax.set_title("%i" % (digits.target[i]))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


###############################################################################
if __name__ == '__main__':
    digits = datasets.load_digits()
    images = digits.images

    # getting samples from 3 different digits
    n = 15 
    A = images[np.where(digits.target==2)][:n]
    B = images[np.where(digits.target==3)][:n]
    C = images[np.where(digits.target==9)][:n]

    # transforming them into a vector
    A = A.reshape((n, 64))
    B = B.reshape((n, 64))
    C = C.reshape((n, 64))

    data = np.concatenate((A, B, C))

    K = 3
    J, M = kmedoids(K, data)

    # visualizing
    xcoords = np.array([i for i in range(len(data))])
    trueJ = np.concatenate(([0]*len(A), [1]*len(B), [2]*len(C)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    colors = getattr(cm, 'brg')(np.linspace(0.2, 1, K))
    for k in range(K):
        idx = np.where(trueJ==k)[0]
        xs = xcoords[idx]
        ys = [1]*len(xs)
        ax.scatter(ys, xs, color=colors[k], marker='s', s=15)
        
    
    colors = getattr(cm, 'Paired')(np.linspace(0.2, 1, K))
    for k in range(K):
        idx = np.where(J==k)[0]
        xs = xcoords[idx]
        ys = [1.5]*len(xs)
        ax.scatter(ys, xs, color=colors[k], marker='s', s=15)
    
    J, M = kmeans(K, data)
    colors = getattr(cm, 'Paired')(np.linspace(0.2, 1, K))
    for k in range(K):
        idx = np.where(J==k)[0]
        xs = xcoords[idx]
        ys = [2]*len(xs)
        ax.scatter(ys, xs, color=colors[k], marker='s', s=15)

    plt.show()


