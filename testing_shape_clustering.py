#/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans

import kmeans
import clusval
import distance
import mnistshape
import shapes

import cPickle, gzip


def pick_data(ns, digits):
    """Pick a bunch of digits from MNIST dataset.
    
    Input
    -----

    ns = [n1, n2, n3, ...]
    digits = [1,2,3, ....]

    both must be the same length. ns contais the number of items for
    each class determined by digits.

    Output
    ------
    
    original images (vector)
    extracted shapes
    true labels 
    
    """
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    images, labels = train_set

    originals = []; 
    shapes = []; 
    true_labels = [];
    i = 0
    for n, d in zip(ns, digits):
        x = np.where(labels==d)[0]
        idx = np.random.choice(x, n, replace=False)
        imgs = images[idx]
        originals.append(imgs)
        shapes.append([mnistshape.get_shape(im.reshape((28,28)), n=30, s=5) 
                        for im in imgs])
        true_labels.append([i]*n)
        i += 1
    originals = np.concatenate(originals)
    shapes = np.concatenate(shapes)
    true_labels = np.concatenate(true_labels)

    # return shuffled data
    idx = range(len(originals))
    np.random.shuffle(idx)
    return originals[idx], shapes[idx], true_labels[idx]

def kmeans_procrustes(k, data, true_labels):
    def dist_func(X, Y):
        return distance.procrustes(X, Y, cycle=False)
    labels, mus, obj, count = kmeans.kmeans_(k, data, dist_func, 30)
    error = clusval.class_error(true_labels, labels)
    return error

def kmeans_euclidean(k, data, true_labels):
    dist_func = distance.euclidean
    labels, mus, obj, count = kmeans.kmeans_(k, data, dist_func, 30)
    error = clusval.class_error(true_labels, labels)
    return error


###############################################################################
if __name__ == '__main__':

    ###########################################################################
    # testing with MNIST digit dataset
    ###########################################################################
    
    nrange = range(10, 50, 10) # range of n
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # classes to cluster
    times_sample = 2 # number of times to sample for each n
    k = len(digits)
    
    proc = []
    eucl = []
    for n in nrange:
        
        print "Doing %i of %i"%(n, nrange[-1])

        ns = [n for d in digits]
        
        for m in range(times_sample):
            data, shapes, true_labels = pick_data(ns, digits)
            error_proc = kmeans_procrustes(k, shapes, true_labels)
            error_eucl = kmeans_euclidean(k, data, true_labels)
            proc.append([n, error_proc])
            eucl.append([n, error_eucl])
            
            print '    ', error_proc, error_eucl 
            
    proc = np.array(proc)
    eucl = np.array(eucl)

    # ploting results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(proc[:,0], proc[:,1], 'o', color='b', alpha=.7, label='procrustes')
    ax.plot(eucl[:,0], eucl[:,1], 's', color='r', alpha=.7, label='euclidean')
   
    # extracting average per n
    proc_avg = []
    eucl_avg = []
    for n in nrange:
        mu = proc[np.where(proc[:,0]==n)][:,1].mean()
        proc_avg.append([n, mu])
        mu = eucl[np.where(eucl[:,0]==n)][:,1].mean()
        eucl_avg.append([n, mu])
    proc_avg = np.array(proc_avg)
    eucl_avg = np.array(eucl_avg)

    ax.plot(proc_avg[:,0], proc_avg[:,1], '-', color='b')
    ax.plot(eucl_avg[:,0], eucl_avg[:,1], '-', color='r')

    ax.set_xlabel('cluster size')
    ax.set_ylabel('error')
    ax.set_xlim([nrange[0]-5, nrange[-1]+5])
    leg = ax.legend(loc=0)
    leg.get_frame().set_alpha(0.7)
    ax.set_title('digits = %s'%str(digits))
    fig.savefig('mnist_procrustes_euclidean_%s.pdf'%\
                    (''.join(['%i'%d for d in digits])))


    ###########################################################################
    # testing with ellipses - add noise
    ###########################################################################
    
#    params = [[1, 3], [2, 5], [4, 4]]
#    K = 3
#    
#    # generate data first
#    data = []
#    labels = []
#    N = 1000
#    z = 0
#    for a, b in params:
#        for i in range(N):
#            a = a + .007*np.random.uniform(-1,1)
#            b = b + .007*np.random.uniform(-1,1)
#            x0, y0 = 2*np.random.randn(2)
#            alpha = np.random.uniform(0, np.pi/2.)
#            s = 2*np.random.randn(1)
#            m = 15
#            data.append(shapes.ellipse(a, b, x0, y0, alpha, s, m))
#            labels.append(z)
#        z += 1
#    data = np.array(data)
#    labels = np.array(labels)
#    idx = range(len(labels))
#    np.random.shuffle(idx)
#    data = data[idx]
#    labels = labels[idx]
#    
#    # clustering
#    nrange = range(10, 110, 10)
#    times_sample = 10 # number of times to sample for each n
#    
#    proc = []
#    eucl = []
#    for n in nrange:
#        
#        print "Doing %i of %i"%(n, nrange[-1])
#
#        count = 0
#        while count  < times_sample:
#            # pick n elements at random from data for each class and cluster
#            sample = []
#            true_labels = []
#            for k in range(len(params)):
#                x = np.where(labels==k)[0]
#                idx = np.random.choice(x, n, replace=False)
#                sample.append(data[idx])
#                true_labels.append([k for i in range(n)])
#            sample = np.concatenate(sample)
#            true_labels = np.concatenate(true_labels)
#            idx = range(len(sample))
#            np.random.shuffle(idx)
#            sample = sample[idx]
#            true_labels = true_labels[idx]
#
#            try:
#                error_proc = kmeans_procrustes(K, sample, true_labels)
#            except:
#                count -= 1
#                continue
#            error_eucl = kmeans_euclidean(K, sample, true_labels)
#            proc.append([n, error_proc])
#            eucl.append([n, error_eucl])
#            print '    ', error_proc, error_eucl
#            count += 1
#    
#    proc = np.array(proc)
#    eucl = np.array(eucl)
#
#    # ploting results
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    
#    ax.plot(proc[:,0], proc[:,1], 'o', color='b', alpha=.7, label='procrustes')
#    ax.plot(eucl[:,0], eucl[:,1], 's', color='r', alpha=.7, label='euclidean')
#   
#    # extracting average per n
#    proc_avg = []
#    eucl_avg = []
#    for n in nrange:
#        mu = proc[np.where(proc[:,0]==n)][:,1].mean()
#        proc_avg.append([n, mu])
#        mu = eucl[np.where(eucl[:,0]==n)][:,1].mean()
#        eucl_avg.append([n, mu])
#    proc_avg = np.array(proc_avg)
#    eucl_avg = np.array(eucl_avg)
#
#    ax.plot(proc_avg[:,0], proc_avg[:,1], '-', color='b')
#    ax.plot(eucl_avg[:,0], eucl_avg[:,1], '-', color='r')
#
#    ax.set_xlabel('cluster size')
#    ax.set_ylabel('error')
#    ax.set_xlim([nrange[0]-5, nrange[-1]+5])
#    ax.set_ylim([-0.1, eucl[:,1].max()+0.1])
#    leg = ax.legend(loc=0)
#    leg.get_frame().set_alpha(0.7)
#    ax.set_title('ellipses = %s'%str(params))
#    fig.savefig('ellipse_noise_procrustes_euclidean.pdf')

"""

    ###########################################################################
    # Testing with random perturbation on the points of ellipse
    ###########################################################################
    
    params = [[1, 3], [2, 5], [4, 4]]
    K = 3
    
    # generate data first
    data = []
    labels = []
    N = 1000
    z = 0
    for a, b in params:
        for i in range(N):
            x0, y0 = 2*np.random.randn(2)
            alpha = np.random.uniform(0, np.pi/2.)
            s = 2*np.random.randn(1)
            m = 10
            data.append(shapes.ellipse_random(a, b, x0, y0, alpha, s, m))
            labels.append(z)
        z += 1
    data = np.array(data)
    labels = np.array(labels)
    idx = range(len(labels))
    np.random.shuffle(idx)
    data = data[idx]
    labels = labels[idx]
    
    # clustering
    nrange = range(10, 110, 10)
    times_sample = 10 # number of times to sample for each n
    
    proc = []
    eucl = []
    for n in nrange:
        
        print "Doing %i of %i"%(n, nrange[-1])

        count = 0
        while count  < times_sample:
            # pick n elements at random from data for each class and cluster
            sample = []
            true_labels = []
            for k in range(len(params)):
                x = np.where(labels==k)[0]
                idx = np.random.choice(x, n, replace=False)
                sample.append(data[idx])
                true_labels.append([k for i in range(n)])
            sample = np.concatenate(sample)
            true_labels = np.concatenate(true_labels)
            idx = range(len(sample))
            np.random.shuffle(idx)
            sample = sample[idx]
            true_labels = true_labels[idx]

            try:
                error_proc = kmeans_procrustes(K, sample, true_labels)
            except:
                count -= 1
                continue
            error_eucl = kmeans_euclidean(K, sample, true_labels)
            proc.append([n, error_proc])
            eucl.append([n, error_eucl])
            print '    ', error_proc, error_eucl
            count += 1
    
    proc = np.array(proc)
    eucl = np.array(eucl)

    # ploting results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(proc[:,0], proc[:,1], 'o', color='b', alpha=.7, label='procrustes')
    ax.plot(eucl[:,0], eucl[:,1], 's', color='r', alpha=.7, label='euclidean')
   
    # extracting average per n
    proc_avg = []
    eucl_avg = []
    for n in nrange:
        mu = proc[np.where(proc[:,0]==n)][:,1].mean()
        proc_avg.append([n, mu])
        mu = eucl[np.where(eucl[:,0]==n)][:,1].mean()
        eucl_avg.append([n, mu])
    proc_avg = np.array(proc_avg)
    eucl_avg = np.array(eucl_avg)

    ax.plot(proc_avg[:,0], proc_avg[:,1], '-', color='b')
    ax.plot(eucl_avg[:,0], eucl_avg[:,1], '-', color='r')

    ax.set_xlabel('cluster size')
    ax.set_ylabel('error')
    ax.set_xlim([nrange[0]-5, nrange[-1]+5])
    ax.set_ylim([-0.1, eucl[:,1].max()+0.1])
    leg = ax.legend(loc=0)
    leg.get_frame().set_alpha(0.7)
    ax.set_title('ellipses = %s'%str(params))
    fig.savefig('ellipse_random_procrustes_euclidean.pdf')

"""
