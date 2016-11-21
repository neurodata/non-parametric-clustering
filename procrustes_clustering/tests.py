
"""Here we implement all the experiments and tests."""

from __future__ import division

import numpy as np
import shape
import matplotlib.pyplot as plt
import cPickle, gzip, sys

import procrustes
import kmeans

def strip_zeros(X):
    """Return matrix X without the rows [0,0,0,...]."""
    XX = X.sum(axis=1)
    return X[np.where(XX!=0)]

def procrustes_alignment_example(a, b, outputfile):
    """Choose two digits from MNIST, namelly "a" and "b".
    Extract the contour and apply procrustes alignment. 
    Show a figure comparing the pairs.
    
    """
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    images, labels = train_set
    
    id1 = np.where(labels==a)[0]
    im11 = images[np.random.choice(id1)].reshape((28,28))
    im12 = images[np.random.choice(id1)].reshape((28,28))
    
    id2 = np.where(labels==b)[0]
    im21 = images[np.random.choice(id2)].reshape((28,28))
    im22 = images[np.random.choice(id2)].reshape((28,28))

    pairs = [[im11,im12], [im21,im22], 
             [im11,im21], [im11,im22], 
             [im12,im21], [im12,im22]]

    fig, axes = plt.subplots(nrows=len(pairs), ncols=7, 
                             figsize=(2*7, 2*len(pairs)))
    for (im1, im2), row in zip(pairs, axes):
        
        ax1, ax2, ax3, ax4, ax5, ax6, ax7 = row

        X1 = shape.get_all_contours(im1, 100, 5, 50)
        X2 = shape.get_all_contours(im2, 100, 5, 50)
        X1, X2 = procrustes.fix_dimensions(X1, X2)
        Y1 = shape.get_external_contour(im1, 100, 5)
        Y2 = shape.get_external_contour(im2, 100, 5)
    
        ax1.imshow(im1, cmap=plt.cm.gray)
        ax1.axis('off')
        ax2.imshow(im2, cmap=plt.cm.gray)
        ax2.axis('off')

        # purelly Euclidean distance between landmark points
        XX1, XX2 = strip_zeros(X1), strip_zeros(X2)
        ax3.plot(XX1[:,0], XX1[:,1], 'ob', alpha=.7)
        ax3.plot(XX2[:,0], XX2[:,1], 'or', alpha=.7)
        ax3.set_title(r'$d_{E}=%.2f/%.2f$'%(np.linalg.norm(X1-X2),
                                            np.linalg.norm(im1-im2)))
        ax3.set_xlim([0,30])
        ax3.set_ylim([0,30])
        ax3.set_aspect('equal')
        ax3.axis('off')

        # just the outside contour
        A, B, d = procrustes.procrustes(Y1, Y2, fullout=True)
        ax4.plot(A[:,0], A[:,1], 'ob', alpha=.7)
        ax4.plot(B[:,0], B[:,1], 'or', alpha=.7)
        ax4.set_title(r'$d_{P_0}=%f$'%d)
        ax4.set_xlim([-.18,.18])
        ax4.set_ylim([-.18,.18])
        ax4.set_aspect('equal')
        ax4.axis('off')

        # outside contour for alignment only
        A, B, d = procrustes.procrustes3(X1, X2, 100, fullout=True)
        AA, BB = strip_zeros(A), strip_zeros(B)
        ax5.plot(AA[:,0], AA[:,1], 'ob', alpha=.7)
        ax5.plot(BB[:,0], BB[:,1], 'or', alpha=.7)
        ax5.set_title(r'$d_{P_3}=%f$'%d)
        ax5.set_xlim([-.18,.18])
        ax5.set_ylim([-.18,.18])
        ax5.set_aspect('equal')
        ax5.axis('off')

        # regular procrustes
        A, B, d = procrustes.procrustes(X1, X2, fullout=True)
        AA, BB = strip_zeros(A), strip_zeros(B)
        ax6.plot(AA[:,0], AA[:,1], 'ob', alpha=.7)
        ax6.plot(BB[:,0], BB[:,1], 'or', alpha=.7)
        ax6.set_title(r'$d_{P}=%f$'%d)
        ax6.set_xlim([-.18,.18])
        ax6.set_ylim([-.18,.18])
        ax6.set_aspect('equal')
        ax6.axis('off')

        # procrustes from library
        A, B, d = procrustes.procrustes2(X1, X2, fullout=True)
        AA, BB = strip_zeros(A), strip_zeros(B)
        ax7.plot(AA[:,0], AA[:,1], 'ob', alpha=.7)
        ax7.plot(BB[:,0], BB[:,1], 'or', alpha=.7)
        ax7.set_title(r'$d_{P_l}=%f$'%d)
        ax7.set_xlim([-.18,.18])
        ax7.set_ylim([-.18,.18])
        ax7.set_aspect('equal')
        ax7.axis('off')
    
    fig.tight_layout()
    fig.savefig(outputfile)

def expand_matrix(X, n):
    """Fill rows of X with zero entries to have n rows."""
    m, k = X.shape
    if n <= m:
        return X
    newX = np.zeros((n, k))
    newX[:m,:k] = X
    return newX

def pick_data(ns, digits):
    """Pick digits to cluster. 
    Example of parameters: ns=[30, 30, 30], digits=[1, 2, 3]
    This will pick 30 elements for each class at random.
    
    """
    
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    images, labels = train_set

    originals = []; shapes = []; ext_shapes = []; true_labels = []; 
    max_dim = 0
    i = 0
    for n, d in zip(ns, digits):
        x = np.where(labels==d)[0]
        js = np.random.choice(x, n, replace=False)
        for j in js:
            im = images[j]
            originals.append(im)
            s = shape.get_all_contours(im.reshape((28,28)),50,5,25)
            dim = s.shape[0]
            if dim > max_dim:
                max_dim = dim
            shapes.append(s)
            ext_s = shape.get_external_contour(im.reshape((28,28)),50,5)
            ext_shapes.append(ext_s)
            true_labels.append(i)
        i += 1
    originals = np.array(originals)
    
    for i in range(len(shapes)):
        shapes[i] = expand_matrix(shapes[i], max_dim)
    shapes = np.array(shapes)
    
    ext_shapes = np.array(ext_shapes)
    true_labels = np.array(true_labels)
    
    idx = range(len(originals))
    np.random.shuffle(idx)
    
    return originals[idx], shapes[idx], ext_shapes[idx], true_labels[idx]

def mnist_standard_vs_procrustes(nrange, digits, num_sample, outfile):
    """Plot accuracy when clustering MNIST digits, using procrustes
    and Euclidean distance.
    
    """
    
    eucl_dist = lambda a, b: np.linalg.norm(a-b)
    proc_dist1 = lambda a, b: procrustes.procrustes(a, b)
    proc_dist2 = lambda a, b: procrustes.procrustes2(a, b)
    proc_dist3 = lambda a, b: procrustes.procrustes3(a, b, 50)
    
    k = len(digits)
    a1, a2, a3, a4, a5 = [], [], [], [], [] 
    for n in nrange:
        
        print "Doing %i of %i"%(n, nrange[-1])
        
        ns = [n]*k
        for m in range(num_sample):
            
            originals, shapes, ext_shapes, labels = pick_data(ns, digits)
            
            l1, _, _, _ = kmeans.kmeans_(k, originals, eucl_dist)
            l2, _, _, _ = kmeans.kmeans_(k, ext_shapes, proc_dist1)
            l3, _, _, _ = kmeans.kmeans_(k, shapes, proc_dist3)
            l4, _, _, _ = kmeans.kmeans_(k, shapes, proc_dist1)
            l5, _, _, _ = kmeans.kmeans_(k, shapes, proc_dist2)

            ac1 = kmeans.accuracy(labels, l1)
            ac2 = kmeans.accuracy(labels, l2)
            ac3 = kmeans.accuracy(labels, l3)
            ac4 = kmeans.accuracy(labels, l4)
            ac5 = kmeans.accuracy(labels, l5)
            
            a1.append([n, ac1])
            a2.append([n, ac2])
            a3.append([n, ac3])
            a4.append([n, ac4])
            a5.append([n, ac5])
            
            print '    ', ac1, ac2, ac3, ac4, ac5

    a1 = np.array(a1)
    a2 = np.array(a2)
    a3 = np.array(a3)
    a4 = np.array(a4)
    a5 = np.array(a5)

    # plotting results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(a1[:,0], a1[:,1], 'o', color='b', alpha=.5, label=r'$d_E$')
    ax.plot(a2[:,0], a2[:,1], 'o', color='r', alpha=.5, label=r'$d_{P_0}$')
    ax.plot(a3[:,0], a3[:,1], 'o', color='g', alpha=.5, label=r'$d_{P_3}$')
    ax.plot(a4[:,0], a4[:,1], 'o', color='c', alpha=.5, label=r'$d_{P}$')
    ax.plot(a5[:,0], a5[:,1], 'o', color='m', alpha=.5, label=r'$d_{P_l}$')
   
    a1_avg, a2_avg, a3_avg, a4_avg, a5_avg = [], [], [], [], []
    for n in nrange:
        mu1 = a1[np.where(a1[:,0]==n)][:,1].mean()
        mu2 = a2[np.where(a2[:,0]==n)][:,1].mean()
        mu3 = a3[np.where(a3[:,0]==n)][:,1].mean()
        mu4 = a4[np.where(a4[:,0]==n)][:,1].mean()
        mu5 = a5[np.where(a5[:,0]==n)][:,1].mean()

        a1_avg.append([n, mu1])
        a2_avg.append([n, mu2])
        a3_avg.append([n, mu3])
        a4_avg.append([n, mu4])
        a5_avg.append([n, mu5])
    a1_avg = np.array(a1_avg)
    a2_avg = np.array(a2_avg)
    a3_avg = np.array(a3_avg)
    a4_avg = np.array(a4_avg)
    a5_avg = np.array(a5_avg)

    ax.plot(a1_avg[:,0], a1_avg[:,1], '-', color='b')
    ax.plot(a2_avg[:,0], a2_avg[:,1], '-', color='r')
    ax.plot(a3_avg[:,0], a3_avg[:,1], '-', color='g')
    ax.plot(a4_avg[:,0], a4_avg[:,1], '-', color='c')
    ax.plot(a5_avg[:,0], a5_avg[:,1], '-', color='m')
    
    ax.set_xlabel(r'$N_i$')
    ax.set_ylabel(r'$A$')
    leg = ax.legend(loc=0)
    leg.get_frame().set_alpha(0.6)
    ax.set_title(r'$\{%s\}$'%(','.join([str(d) for d in digits])))
    fig.savefig(outfile)
    
def mnist_eucl_proc(digits, num_points, num_avg):
    """Evaluate kmeans accuracy """

    eucl_dist = lambda a, b: np.linalg.norm(a-b)
    proc_dist1 = lambda a, b: procrustes.procrustes(a, b)
    proc_dist2 = lambda a, b: procrustes.procrustes2(a, b)
    proc_dist3 = lambda a, b: procrustes.procrustes3(a, b, 50)
    
    k = len(digits)
    a1, a2, a3, a4, a5 = [], [], [], [], [] 
    
    for i in range(num_avg):
        originals, shapes, ext_shapes, labels = pick_data([num_points]*k, 
                                                            digits)
        
        l1, _, _, _ = kmeans.kmeans_(k, originals, eucl_dist)
        l2, _, _, _ = kmeans.kmeans_(k, ext_shapes, proc_dist1)
        l3, _, _, _ = kmeans.kmeans_(k, shapes, proc_dist3)
        l4, _, _, _ = kmeans.kmeans_(k, shapes, proc_dist1)
        l5, _, _, _ = kmeans.kmeans_(k, shapes, proc_dist2)
        
        a1.append(kmeans.accuracy(labels, l1))
        a2.append(kmeans.accuracy(labels, l2))
        a3.append(kmeans.accuracy(labels, l3))
        a4.append(kmeans.accuracy(labels, l4))
        a5.append(kmeans.accuracy(labels, l5))
    
    print "d_E = %f" % np.mean(a1)
    print "d_{P_0} = %f" % np.mean(a2)
    print "d_{P_3} = %f" % np.mean(a3)
    print "d_{P} = %f" % np.mean(a4)
    print "d_{P_l} = %f" % np.mean(a5)

def mnist_euclidean(digits, num_points, num_avg):
    eucl_dist = lambda a, b: np.linalg.norm(a-b)
    k = len(digits)
    a = [] 
    for i in range(num_avg):
        originals, shapes, ext_shapes, labels = pick_data([num_points]*k, 
                                                            digits)
        l, _, _, _ = kmeans.kmeans_(k, originals, eucl_dist)
        accu = kmeans.accuracy(labels, l)
        a.append(accu)
        print accu
    print
    print "d_E = %f" % np.mean(a)

def mnist_procrustes0(digits, num_points, num_avg):
    proc_dist = lambda a, b: procrustes.procrustes(a, b)
    k = len(digits)
    a = [] 
    for i in range(num_avg):
        originals, shapes, ext_shapes, labels = pick_data([num_points]*k, 
                                                            digits)
        l, _, _, _ = kmeans.kmeans_(k, ext_shapes, proc_dist)
        accu = kmeans.accuracy(labels, l)
        a.append(accu)
        print accu
    print
    print "d_{P_0} = %f" % np.mean(a)

def mnist_procrustes(digits, num_points, num_avg):
    proc_dist = lambda a, b: procrustes.procrustes(a, b)
    k = len(digits)
    a = [] 
    for i in range(num_avg):
        originals, shapes, ext_shapes, labels = pick_data([num_points]*k, 
                                                            digits)
        l, _, _, _ = kmeans.kmeans_(k, shapes, proc_dist)
        accu = kmeans.accuracy(labels, l)
        a.append(accu)
        print accu
    print
    print "d_{P} = %f" % np.mean(a)
    
def pick_orig_binary(ns, digits):
    """Return original and binary version of images, with labels."""
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    images, labels = train_set
    
    originals = []; bin_imgs = []; true_labels = [];
    i = 0
    for n, d in zip(ns, digits):
        x = np.where(labels==d)[0]
        js = np.random.choice(x, n, replace=False)
        for j in js:
            im = images[j].reshape((28,28))
            originals.append(im)
            bin_imgs.append(shape.im_ones(im, im.mean()))
            true_labels.append(i)
        i += 1
    originals = np.array(originals)
    bin_imgs = np.array(bin_imgs)
    true_labels = np.array(true_labels)
    idx = range(len(originals))
    np.random.shuffle(idx)
    return originals[idx], bin_imgs[idx], true_labels[idx]
   
def clustering_orig_binary(nrange, digits, num_sample, outfile):
    """Cluster originals and binaries with K-means/Euclidean."""
    
    eucl_dist = lambda a, b: np.linalg.norm(a-b)
    
    k = len(digits)
    a1, a2 = [], []
    for n in nrange:
        
        print "Doing %i of %i"%(n, nrange[-1])
        
        ns = [n]*k
        for m in range(num_sample):
            
            originals, binaries, labels = pick_orig_binary(ns, digits)
            
            l1, _, _, _ = kmeans.kmeans_(k, originals, eucl_dist)
            l2, _, _, _ = kmeans.kmeans_(k, binaries, eucl_dist)

            ac1 = kmeans.accuracy(labels, l1)
            ac2 = kmeans.accuracy(labels, l2)
            
            a1.append([n, ac1])
            a2.append([n, ac2])
            
            print '    ', ac1, ac2

    a1 = np.array(a1)
    a2 = np.array(a2)

    # plotting results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(a1[:,0], a1[:,1], 'o', color='b', alpha=.5, label=r'$d_E$')
    ax.plot(a2[:,0], a2[:,1], 'o', color='r', alpha=.5, label=r'$d_{E_b}$')
   
    a1_avg, a2_avg = [], []
    for n in nrange:
        mu1 = a1[np.where(a1[:,0]==n)][:,1].mean()
        mu2 = a2[np.where(a2[:,0]==n)][:,1].mean()

        a1_avg.append([n, mu1])
        a2_avg.append([n, mu2])
    a1_avg = np.array(a1_avg)
    a2_avg = np.array(a2_avg)

    ax.plot(a1_avg[:,0], a1_avg[:,1], '-', color='b')
    ax.plot(a2_avg[:,0], a2_avg[:,1], '-', color='r')
    
    ax.set_xlabel(r'$N_i$')
    ax.set_ylabel(r'$A$')
    leg = ax.legend(loc=0)
    leg.get_frame().set_alpha(0.6)
    ax.set_title(r'$\{%s\}$'%(','.join([str(d) for d in digits])))
    fig.savefig(outfile)

def clustering_eucl(nrange, digits, num_sample, outfile):
    """Cluster originals and binaries with K-means/Euclidean."""
    
    eucl_dist = lambda a, b: np.linalg.norm(a-b)
    
    k = len(digits)
    a1, a2 = [], []
    for n in nrange:
        
        print "Doing %i of %i"%(n, nrange[-1])
        
        ns = [n]*k
        for m in range(num_sample):
            
            originals, shapes, ext_shapes, labels = pick_data(ns, digits)
            
            l1, _, _, _ = kmeans.kmeans_(k, originals, eucl_dist)
            l2, _, _, _ = kmeans.kmeans_(k, shapes, eucl_dist)

            ac1 = kmeans.accuracy(labels, l1)
            ac2 = kmeans.accuracy(labels, l2)
            
            a1.append([n, ac1])
            a2.append([n, ac2])
            
            print '    ', ac1, ac2

    a1 = np.array(a1)
    a2 = np.array(a2)

    # plotting results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(a1[:,0], a1[:,1], 'o', color='b', alpha=.5, label=r'$d_E$')
    ax.plot(a2[:,0], a2[:,1], 'o', color='r', alpha=.5, label=r'$d_{E_b}$')
   
    a1_avg, a2_avg = [], []
    for n in nrange:
        mu1 = a1[np.where(a1[:,0]==n)][:,1].mean()
        mu2 = a2[np.where(a2[:,0]==n)][:,1].mean()

        a1_avg.append([n, mu1])
        a2_avg.append([n, mu2])
    a1_avg = np.array(a1_avg)
    a2_avg = np.array(a2_avg)

    ax.plot(a1_avg[:,0], a1_avg[:,1], '-', color='b')
    ax.plot(a2_avg[:,0], a2_avg[:,1], '-', color='r')
    
    ax.set_xlabel(r'$N_i$')
    ax.set_ylabel(r'$A$')
    leg = ax.legend(loc=0)
    leg.get_frame().set_alpha(0.6)
    ax.set_title(r'$\{%s\}$'%(','.join([str(d) for d in digits])))
    fig.savefig(outfile)

def shape_to_image(d, outfile):
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    images, labels = train_set
    x = np.where(labels==d)[0]
    im = images[np.random.choice(x)].reshape((28,28))
    s = shape.get_all_contours(im, 300, 5, 150)
    im2 = shape.shape_to_image(s, scale=10)

    fig = plt.figure(figsize=(9,3))
    ax = fig.add_subplot(141)
    ax.imshow(im, cmap=plt.cm.gray)
    ax.set_aspect('equal')
    ax.axis('off')

    ax = fig.add_subplot(142)
    ax.plot(s[:,0], s[:,1], 'ob', alpha=.6)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax = fig.add_subplot(143)
    ax.imshow(im2, cmap=plt.cm.gray)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax = fig.add_subplot(144)
    shape.fill(im2, 0, 0)
    ax.imshow(im2, cmap=plt.cm.gray)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.savefig(outfile)


if __name__ == '__main__':
    # generating figures
    #procrustes_alignment_example(1, 2, 'figs/alignment_12.pdf')
    #procrustes_alignment_example(1, 7, 'figs/alignment_17.pdf')
    #procrustes_alignment_example(1, 3, 'figs/alignment_13.pdf')
    #procrustes_alignment_example(2, 3, 'figs/alignment_23.pdf')
    #procrustes_alignment_example(2, 4, 'figs/alignment_24.pdf')
    #procrustes_alignment_example(2, 8, 'figs/alignment_28.pdf')
    #procrustes_alignment_example(4, 8, 'figs/alignment_48.pdf')
    #procrustes_alignment_example(3, 5, 'figs/alignment_35.pdf')

    #ns = range(20,250,20)
    #ds = [1,3,5]
    #mnist_standard_vs_procrustes(ns, ds, 5, 'figs/clustering_135.pdf')
    
    #mnist_eucl_proc([2,4,8], 200, 6)

    #mnist_euclidean([0,1,2,3,4,5,6,7,8,9], 400, 5)
    #mnist_procrustes0([0,1,2,3,4,5,6,7,8,9], 400, 5)
    #mnist_procrustes([0,1,2,3,4,5,6,7,8,9], 400, 5)
    
    ns = range(20,250,20)
    ds = [1,7]
    #clustering_orig_binary(ns, ds, 5, 'figs/clustering_binary_012345.pdf')
    clustering_eucl(ns, ds, 5, 'figs/clustering_eucl_17.pdf')

    #shape_to_image(4, 'figs/shape_image_4.pdf')
    #shape.test(1)

