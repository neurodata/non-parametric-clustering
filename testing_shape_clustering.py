#/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans

import kmeans
import clusval
import distance
import mnistshape

import cPickle, gzip


def pick_data(ns, digits):
    """Choose bunch of digits from MNIST dataset and return originals,
    shapes, and true labels. ns must be an array with the number of elements
    for each class, and digits is the corresponding array with the digit
    labels to choose from.
    
    """
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    images, labels = train_set

    originals = []
    shapes = []
    true_labels = []
    i = 0
    for n, d in zip(ns, digits):
        x = np.where(labels==d)[0]
        idx = np.random.choice(x, n, replace=False)
        imgs = images[idx]
        originals.append(imgs)
        shapes.append([mnistshape.get_shape(im.reshape((28,28)), n=30, s=4)
                        for im in imgs])
        true_labels.append([i]*n)
        i += 1
    originals = np.concatenate(originals)
    shapes = np.concatenate(shapes)
    true_labels = np.concatenate(true_labels)

    idx = range(len(originals))
    np.random.shuffle(idx)
    return originals[idx], shapes[idx], true_labels[idx]

def kmeans_procrustes(k, data, maxiter=5):
    for i in range(maxiter):
        z, m, f, n = kmeans.kmeans_(k, data, distance.procrustes,
                                        max_iter=maxiter)
        if n == 30:
            print "WARNING: K-means didn't converge after %i iterations"%n
        if i == 0 or f < ff:
            zz = z
            mm = m
            ff = f
    return z

def scikmeans(k, data):
    km = KMeans(k)
    r = km.fit(data)
    return r.labels_


###############################################################################
if __name__ == '__main__':
    
    ns = range(10, 250, 50)
    ds = [1,2,3]
    e1 = []
    e2 = []
    for n in ns:
        print "Doing %i of %i"%(n, ns[-1])
        data, shapes, z = pick_data([n,n,n], ds)
        z1 = kmeans_procrustes(3, shapes, maxiter=3)
        z2 = scikmeans(3, data)
        e1.append((1-clusval.class_error(z1, z))*100)
        e2.append((1-clusval.class_error(z2, z))*100)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ns, e1, 'o-b', label='procrustes')
    ax.plot(ns, e2, 'o-r', label='k-means')
    ax.set_xlabel('cluster size')
    ax.set_ylabel('accuracy (%)')
    ax.set_xlim([0, 215])
    leg = ax.legend()
    leg.get_frame().set_alpha(0.7)
    ax.set_title('digits = %s'%str(ds))
    fig.savefig('proc_kmeans_123.pdf')
