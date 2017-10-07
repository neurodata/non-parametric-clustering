"""
Tests and comparison between Energy, K-means, and GMM.

We run Energy, K-means, and GMM a few times, and show average accuracy 
and variability for several settings.

"""

# Guilhere Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata

from __future__ import division

import numpy as np
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import cPickle, gzip
from sklearn.decomposition import PCA

import eclust.data as data
import eclust.kmeans as km
import eclust.gmm as gm
import eclust.qcqp as ke
import eclust.kmeanspp as kpp
import eclust.energy1d as e1d

from eclust import metric 
from eclust import objectives

from pplotcust import *

import sys


###############################################################################
# clustering functions

def kernel_energy(k, X, kernel_matrix, z, run_times=5):
    """Run few times and pick the best objective function value."""
    G = kernel_matrix
    best_score = -np.inf
    for rt in range(run_times):
        
        z0 = kpp.kpp(k, X, ret='labels')
        Z0 = ke.ztoZ(z0)
        
        zh = ke.kernel_kmeans(k, G, Z0, max_iter=300)
        Zh = ke.ztoZ(zh)
        score = ke.objective(Zh, G)
        
        if score > best_score:
            best_score = score
            best_z = zh

    return metric.accuracy(z, best_z)

def cost_energy(k, X, kernel_matrix, z, run_times=5):
    """Run few times and pick the best objective function value."""
    G = kernel_matrix
    best_score = -np.inf
    for rt in range(run_times):
        
        z0 = kpp.kpp(k, X, ret='labels')
        Z0 = ke.ztoZ(z0)
        
        zh = ke.minw(k, G, Z0, max_iter=300)
        Zh = ke.ztoZ(zh)
        score = ke.objective(Zh, G)
        if score > best_score:
            best_score = score
            best_z = zh
    
    return metric.accuracy(z, best_z)

def kmeans(k, X, z, run_times=5):
    """Run k-means couple times and pick the best answer."""
    best_score = np.inf
    for rt in range(run_times):
        
        mu0, z0 = kpp.kpp(k, X, ret='both')
        
        zh = km.kmeans(k, X, labels_=z0, mus_=mu0, max_iter=300)
        score = objectives.kmeans(data.from_label_to_sets(X, zh))
        
        if score < best_score:
            best_score = score
            best_z = zh
    
    return metric.accuracy(best_z, z)

def gmm(k, X, z, run_times=5):
    """Run gmm couple times and pick the best answer.
    GMM crashes badly for few datapoint in high dimension, so this
    exceptions are ugly hacks.
    """
    best_score = -np.inf
    best_z = None
    for rt in range(run_times):
        try:
            zh = gm.em_gmm_vect(k, X)
            score = objectives.loglikelihood(data.from_label_to_sets(X, zh))
            if score > best_score:
                best_score = score
                best_z = zh
        #except np.linalg.LinAlgError:
        except:
            pass
    if best_z is not None:
        return metric.accuracy(best_z, z)
    else:
        return 0

def two_clusters1D(X, z):
    """Clustering in 1 dimension."""
    zh, cost = e1d.two_clusters1D(X)
    return metric.accuracy(zh, z)


###############################################################################
# functions to test different settings

def gauss_dimensions_mean(dimensions=range(2,100,20), num_points=[200, 200],
                          delta=0.7, run_times=5, num_experiments=10, d=None):
    """Here we keep signal in one dimension and increase the ambient dimension.
    The covariances are kept fixed.
    
    """
    k = 2
    if not d:
        d = dimensions[0]
    n1, n2 = num_points
    table = np.zeros((num_experiments*len(dimensions), 5))
    count = 0
    
    for D in dimensions:

        for i in range(num_experiments):
        
            # generate data
            m1 = np.zeros(D)
            s1 = np.eye(D)
            m2 = np.concatenate((delta*np.ones(d), np.zeros(D-d)))
            s2 = np.eye(D)
        
            X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])
            rho = lambda x, y: np.linalg.norm(x-y)
            G = ke.kernel_matrix(X, rho)
        
            table[count, 0] = D
            table[count, 1] = cost_energy(k, X, G, z, run_times=run_times)
            table[count, 2] = kernel_energy(k, X, G, z, run_times=run_times)
            table[count, 3] = kmeans(k, X, z, run_times=run_times)
            table[count, 4] = gmm(k, X, z, run_times=run_times)

            count += 1

    return table

def gauss_dimensions_cov(dimensions=range(2,100,20), num_points=[200, 200],
                         run_times=3, num_experiments=10, d=None):
    """High dimensions but with nontrivial covariance."""
    k = 2
    if not d:
        d = dimensions[0]
    n1, n2 = num_points
    table = np.zeros((num_experiments*len(dimensions), 5))
    count = 0
    
    for D in dimensions:

        for l in range(num_experiments):
        
            #m1 = np.zeros(D)
            #s1 = np.eye(D)
            #m2 = np.concatenate((np.ones(d), np.zeros(D-d)))
            #s2 = np.eye(D)
            #for a in range(int(d/2)):
            #    s2[a,a] = a+1

            m1 = np.zeros(D)
            m2 = np.concatenate((np.ones(d), np.zeros(D-d)))
            s1 = np.eye(D)
            s2 = np.eye(D)
            for a in range(d):
                s1[a,a] = np.power(1/(a+1), 0.5)
            for a in range(d):
                s2[a,a] = np.power(a+1, 0.5)
            
            X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])
        
            rho = lambda x, y: np.linalg.norm(x-y)
            G = ke.kernel_matrix(X, rho)
        
            table[count, 0] = D
            table[count, 1] = cost_energy(k, X, G, z, run_times=run_times)
            table[count, 2] = kernel_energy(k, X, G, z, run_times=run_times)
            table[count, 3] = kmeans(k, X, z, run_times=run_times)
            table[count, 4] = gmm(k, X, z, run_times=run_times)

            count += 1

    return table

def circles_or_spirals(num_points=[200,200], num_times=10, run_times=10):
    
    k = 2
    n1, n2 = num_points
    
    X, z = data.circles([1, 3], [0.1, 0.1], [n1, n2])
    #X, z = data.spirals([1, -1], [n1, n2], noise=.2)
    #rho = lambda x, y: ke.euclidean_rho(x, y, alpha=0.5)
    #rho = lambda x, y: np.exp(-np.linalg.norm(x-y)/2/(1**2))
    
    # this one works for circles
    #rho = lambda x, y: np.power(np.linalg.norm(x-y), 2)*\
    #                    np.exp(-0.5*np.linalg.norm(x-y))
    
    # this one is working decently for spirals
    rho = lambda x, y: np.power(np.linalg.norm(x-y), 2)*\
        .9*np.sin(np.linalg.norm(x-y)/0.9)

    #rho = lambda x, y: np.power(np.linalg.norm(x-y), 2)*\
    #            delta(np.linalg.norm(x-y)/0.1)
    G = ke.kernel_matrix(X, rho)

    table = np.zeros((num_times, 4))
    for nt in range(num_times):
        table[nt, 0] = cost_energy(k, X, G, z, run_times=run_times)
        table[nt, 1] = kernel_energy(k, X, G, z, run_times=run_times)
        table[nt, 2] = kmeans(k, X, z, run_times=run_times)
        table[nt, 3] = gmm(k, X, z, run_times=run_times)
    
    return table

def gauss_dimensions_pi(num_points=range(0, 180, 10), N=200, D=4, d=2,
                        run_times=3, num_experiments=10):
    """Test unbalanced clusters."""
    k = 2
    table = np.zeros((num_experiments*len(num_points), 5))
    count = 0
    
    for p in num_points:

        for i in range(num_experiments):
        
            # generate data
            m1 = np.zeros(D)
            s1 = np.eye(D)
            m2 = np.concatenate((1.5*np.ones(d), np.zeros(D-d)))
            s2 = np.diag(np.concatenate((.5*np.ones(d), np.ones(D-d))))
            n1 = N-p
            n2 = N+p
        
            X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])
            rho = lambda x, y: np.linalg.norm(x-y)
            G = ke.kernel_matrix(X, rho)
        
            table[count, 0] = p
            table[count, 1] = cost_energy(k, X, G, z, run_times=run_times)
            table[count, 2] = kernel_energy(k, X, G, z, run_times=run_times)
            table[count, 3] = kmeans(k, X, z, run_times=run_times)
            table[count, 4] = gmm(k, X, z, run_times=run_times)

            count += 1

    return table

def normal_or_lognormal(numpoints=range(10,100,10), 
                        run_times=10, num_experiments=10, kind='normal'):
    """Testing lognormal distributions."""
    table = np.zeros((num_experiments*len(numpoints), 9))
    count = 0
    k = 2
    
    for n in numpoints:

        for i in range(num_experiments):
        
            # generate data
            m1 = np.zeros(20)
            s1 = 0.5*np.eye(20)
            m2 = 0.5*np.concatenate((np.ones(5), np.zeros(15)))
            s2 = np.eye(20)
        
            if kind == 'normal':
                X, z = data.multivariate_normal([m1, m2], [s1, s2], [n, n])
            else:
                X, z = data.multivariate_lognormal([m1, m2], [s1, s2], [n, n])
            
            rho = lambda x, y: np.linalg.norm(x-y)
            G = ke.kernel_matrix(X, rho)
            
            rho2 = lambda x, y: np.power(np.linalg.norm(x-y), 0.5)
            G2 = ke.kernel_matrix(X, rho2)
            
            rho3 = lambda x, y: 2-2*np.exp(-np.linalg.norm(x-y)/2)
            G3 = ke.kernel_matrix(X, rho3)
        
            table[count, 0] = n
            table[count, 1] = cost_energy(k, X, G, z, run_times=run_times)
            table[count, 2] = cost_energy(k, X, G2, z, run_times=run_times)
            table[count, 3] = cost_energy(k, X, G3, z, run_times=run_times)
            table[count, 4] = kernel_energy(k, X, G, z, run_times=run_times)
            table[count, 5] = kernel_energy(k, X, G2, z, run_times=run_times)
            table[count, 6] = kernel_energy(k, X, G3, z, run_times=run_times)
            table[count, 7] = kmeans(k, X, z, run_times=run_times)
            table[count, 8] = gmm(k, X, z, run_times=run_times)

            count += 1

    return table

def testing1d(num_points_range=range(50, 100, 20), run_times=4,
                num_experiments=100):
    """Comparing clustering methods in 1 dimension."""
    k = 2
    table = np.zeros((num_experiments*len(num_points_range), 4))
    count = 0
    
    for ne in range(num_experiments):

        for i, ne in enumerate(num_points_range):
        
            #m1 = 0
            #s1 = 0.3
            #m2 = -1.5
            #s2 = 1.5
            #n1 = n2 = ne
            #X, z = data.univariate_lognormal([m1, m2], [s1, s2], [n1, n2])
        
            m1 = 0
            s1 = 1
            m2 = 5
            s2 = 2
            n1 = n2 = ne
            X, z = data.univariate_normal([m1, m2], [s1, s2], [n1, n2])
        
            Y = np.array([[x] for x in X])
        
            table[count, 0] = ne
            table[count, 1] = two_clusters1D(X, z)
            table[count, 2] = kmeans(k, Y, z, run_times=run_times)
            table[count, 3] = gmm(k, Y, z, run_times=run_times)

            count += 1

    return table

# some kernels/semmimetric function
rho_standard = lambda x, y: np.power(np.linalg.norm(x-y), 1)
rho_half = lambda x, y: np.power(np.linalg.norm(x-y), 1/2)
rho_gauss = lambda x, y: 2-2*np.exp(-np.linalg.norm(x-y)**2/4)
rho_exp = lambda x, y, sigma: 2-2*np.exp(-np.linalg.norm(x-y)/(2*sigma))
rho_rbf = lambda x, y, sigma: 2-2*np.exp(-np.linalg.norm(x-y)**2/(2*sigma**2))

def other_examples(num_experiments=10, run_times=4):
    """Some other examples."""

    rhos = [rho_standard, rho_half]
    rho_names = ['standard', 'half', 'exp', 'rbf']

    table = []
    for i in range(num_experiments):
        
        ### generate data ###
        
        # cigars
        #m1 = [0,0]
        #m2 = [6.5,0]
        #s1 = np.array([[1,0],[0,20]])
        #s2 = np.array([[1,0],[0,20]])
        #X, z = data.multivariate_normal([m1, m2], [s1, s2], [200, 200])
        #rho_exp2= lambda x, y: rho_exp(x, y, 2)
        #rho_rbf2 = lambda x, y: rho_rbf(x, y, 2)
        
        # 2 circles
        #X, z = data.circles([1, 3], [[0,0], [0,0]], [0.2, 0.2], [400, 400])
        #rho_exp2= lambda x, y: rho_exp(x, y, 1)
        #rho_rbf2 = lambda x, y: rho_rbf(x, y, 1)
        
        # 3 circles
        X, z = data.circles([1, 3, 5], [[0,0], [0,0], [0,0]], [0.2, 0.2, 0.2], 
                             [400, 400, 400])
        rho_exp2= lambda x, y: rho_exp(x, y, 2)
        rho_rbf2 = lambda x, y: rho_rbf(x, y, 2)

        # 2 spirals
        #X, z = data.spirals([1,-1], [[0,0], [0,0]], [200,200], noise=0.1)

        #####################
        
        k = 3
        
        n = len(X)

        # build several kernels
        Gs = [ke.kernel_matrix(X, rho) for rho in rhos]
        Gs.append(ke.kernel_matrix(X, rho_rbf2))
        Gs.append(ke.kernel_matrix(X, rho_exp2))
        
        for G in Gs:
            table.append(cost_energy(k, X, G, z, run_times=run_times))
        for G in Gs:
            table.append(kernel_energy(k, X, G, z, run_times=run_times))
        table.append(kmeans(k, X, z, run_times=run_times))
        table.append(gmm(k, X, z, run_times=run_times))
    
    table = np.array(table)
    num_kernels = len(Gs)
    table = table.reshape((num_experiments, 2*num_kernels+2))
    
    num_cols = num_kernels*2
    for j in range(num_kernels):
        vals = table[:, j]
        print "Energy %-10s:"%rho_names[j], vals.mean(), scipy.stats.sem(vals) 
    for i, j in enumerate(range(num_kernels, num_cols)):
        vals = table[:, j]
        print "Kernel %-10s:"%rho_names[i], vals.mean(), scipy.stats.sem(vals) 
    vals = table[:, -2]
    print "k-means          :", vals.mean(), scipy.stats.sem(vals) 
    vals = table[:, -1]
    print "gmm              :", vals.mean(), scipy.stats.sem(vals) 

def mnist(num_experiments=10, digits=[0,1,2], num_points=100, run_times=4):
    """MNIST clustering. We use Hartigan's and Lloyd's with different
    kernels and compare to k-means.
    
    """
    
    k = len(digits)

    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    images, labels = train_set

    rhos = [rho_standard, rho_half]
    rho_names = ['standard', 'half', 'rbf']

    table = []
    for i in range(num_experiments):
        
        # sample digits
        data = []
        true_labels = []
        for l, d in enumerate(digits):
            x = np.where(labels==d)[0]
            js = np.random.choice(x, num_points, replace=False)
            for j in js:
                im = images[j]
                label = l
                data.append(im)
                true_labels.append(label)
        idx = range(len(data))
        data = np.array(data)
        true_labels = np.array(true_labels)
        np.random.shuffle(idx)
        X = data[idx]
        z = true_labels[idx]
        ################

        n = len(X)
        sigma = np.sqrt(sum([np.linalg.norm(X[i]-X[j])**2 
                     for i in range(n) for j in range(n)])/(n**2))
        #rho_exp2= lambda x, y: rho_exp(x, y, sigma)
        rho_rbf2 = lambda x, y: rho_rbf(x, y, sigma)
        
        #rho_exp2 = lambda x, y: rho_exp(x, y, 1)
        #rho_rbf2 = lambda x, y: rho_rbf(x, y, 1)

        # build several kernels
        Gs = [ke.kernel_matrix(X, rho) for rho in rhos]
        #Gs.append(ke.kernel_matrix(X, rho_exp2))
        Gs.append(ke.kernel_matrix(X, rho_rbf2))
        
        for G in Gs:
            table.append(cost_energy(k, X, G, z, run_times=run_times))
        for G in Gs:
            table.append(kernel_energy(k, X, G, z, run_times=run_times))
        table.append(kmeans(k, X, z, run_times=run_times))
    table = np.array(table)
    num_kernels = len(Gs)
    table = table.reshape((num_experiments, 2*num_kernels+1))
    
    num_cols = num_kernels*2
    for j in range(num_kernels):
        vals = table[:, j]
        print "Energy %-10s:"%rho_names[j], vals.mean(), scipy.stats.sem(vals) 
    for i, j in enumerate(range(num_kernels, num_cols)):
        vals = table[:, j]
        print "Kernel %-10s:"%rho_names[i], vals.mean(), scipy.stats.sem(vals) 
    vals = table[:, -1]
    print "k-means          :", vals.mean(), scipy.stats.sem(vals) 

def mnist_pca(num_experiments=10, digits=[0,1,2], num_points=100,
                n_components=20, run_times=4):
    """MNIST clustering. We use Hartigan's and Lloyd's with different
    kernels and compare to k-means. We project the data into PCA
    with 20 or other components.
    
    """
    
    k = len(digits)

    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    images, labels = train_set

    rhos = [rho_standard, rho_half]
    rho_names = ['standard', 'half', 'rbf']

    table = []
    for i in range(num_experiments):
        
        # sample digits
        data = []
        true_labels = []
        for l, d in enumerate(digits):
            x = np.where(labels==d)[0]
            js = np.random.choice(x, num_points, replace=False)
            for j in js:
                im = images[j]
                label = l
                data.append(im)
                true_labels.append(label)
        idx = range(len(data))
        data = np.array(data)
        true_labels = np.array(true_labels)
        np.random.shuffle(idx)
        X = data[idx]
        z = true_labels[idx]
        ################
        
        # do PCA
        pca = PCA(n_components=n_components)
        pca.fit(X)
        X_new = pca.transform(X)
        ########

        n = len(X)
        sigma = np.sqrt(sum([np.linalg.norm(X[i]-X[j])**2 
                     for i in range(n) for j in range(n)])/(n**2))
        #rho_exp2= lambda x, y: rho_exp(x, y, sigma)
        rho_rbf2 = lambda x, y: rho_rbf(x, y, sigma)
        #rho_exp2 = lambda x, y: rho_exp(x, y, 1)
        #rho_rbf2 = lambda x, y: rho_rbf(x, y, 1)

        # build several kernels
        Gs = [ke.kernel_matrix(X, rho) for rho in rhos]
        #Gs.append(ke.kernel_matrix(X, rho_exp2))
        Gs.append(ke.kernel_matrix(X, rho_rbf2))
        
        for G in Gs:
            table.append(cost_energy(k, X, G, z, run_times=run_times))
        for G in Gs:
            table.append(kernel_energy(k, X, G, z, run_times=run_times))
        table.append(kmeans(k, X, z, run_times=run_times))
        
        X = X_new
        n = len(X)
        sigma = np.sqrt(sum([np.linalg.norm(X[i]-X[j])**2 
                     for i in range(n) for j in range(n)])/(n**2))
        #rho_exp2= lambda x, y: rho_exp(x, y, sigma)
        rho_rbf2 = lambda x, y: rho_rbf(x, y, sigma)
        #rho_exp2 = lambda x, y: rho_exp(x, y, 1)
        #rho_rbf2 = lambda x, y: rho_rbf(x, y, 1)

        # build several kernels
        Gs = [ke.kernel_matrix(X, rho) for rho in rhos]
        #Gs.append(ke.kernel_matrix(X, rho_exp2))
        Gs.append(ke.kernel_matrix(X, rho_rbf2))
        
        for G in Gs:
            table.append(cost_energy(k, X, G, z, run_times=run_times))
        for G in Gs:
            table.append(kernel_energy(k, X, G, z, run_times=run_times))
        table.append(kmeans(k, X, z, run_times=run_times))
    table = np.array(table)
    num_kernels = len(Gs)
    table = table.reshape((num_experiments, 2*(2*num_kernels+1)))
    
    num_cols = 2*num_kernels
    for j in range(num_kernels):
        vals = table[:, j]
        print "Energy %-10s:"%rho_names[j], vals.mean(), scipy.stats.sem(vals) 
    for i, j in enumerate(range(num_kernels, num_cols)):
        vals = table[:, j]
        print "Kernel %-10s:"%rho_names[i], vals.mean(), scipy.stats.sem(vals) 
    vals = table[:, num_cols+1]
    print "k-means          :", vals.mean(), scipy.stats.sem(vals) 
    
    start = num_cols+1
    for i, j in enumerate(range(start,start+num_kernels)):
        vals = table[:, j]
        print "Energy PCA %-10s:"%rho_names[i], vals.mean(), scipy.stats.sem(vals) 
    for i, j in enumerate(range(start+num_kernels,start+2*num_kernels)):
        vals = table[:, j]
        print "Kernel PCA %-10s:"%rho_names[i], vals.mean(), scipy.stats.sem(vals) 
    vals = table[:, -1]
    print "k-means PCA          :", vals.mean(), scipy.stats.sem(vals) 
    

###############################################################################
# ploting functions

def plot_accuracy_errorbar(table, xlabel='dimension', ylabel='accuracy',
        output='plot.pdf', symbols=['o', 's', 'D', 'v'], 
        legends=['energy', r'$k$-means', 'GMM', r'kernel $k$-means'],
        xlim=None, ylim=None, bayes=None, loc=None, doublex=False):
    """Plot average accuracy of several algorithms with errobars being
    standard error.
    
    """
    
    col0 = np.array([int(x) for x in table[:,0]])
    xs = np.unique(col0)
    n = len(table[0][1:])
    
    colors = iter(plt.cm.brg(np.linspace(0,1,n+1)))
    legends = iter(legends)
    symbols = iter(symbols)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if doublex:
        xs2 = 2*xs
    else:
        xs2 = xs
    for i in range(1, n+1):

        mean_ = []
        stderr_ = []
        for x in xs:
            a = table[np.where(col0==x)[0],i]
            a = a[np.where(a!=0)]
            mean_.append(a.mean())
            stderr_.append(scipy.stats.sem(a))
        mean_ = np.array(mean_)
        stderr_ = np.array(stderr_)
        #mean_ = np.array([table[np.where(col0==x)[0],i][].mean() 
        #                        for x in xs])
        #stderr_ = np.array([scipy.stats.sem(table[np.where(col0==x)[0],i]) 
        #                        for x in xs])
        c = next(colors)
        l = next(legends)
        m = next(symbols)
        
        if bayes:
            if not isinstance(bayes, list):
                bayes = [bayes]*len(xs)
            ax.plot(xs2, bayes, '--', color='black', 
                    linewidth=1, zorder=0)
        
        ax.errorbar(xs2, mean_, yerr=stderr_, 
            linestyle='-', marker=m, color=c, markersize=4, elinewidth=.5, 
            capthick=0.4, label=l, linewidth=1, barsabove=False)
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    if not loc:
        loc = 0
    leg = plt.legend()
    ax.legend(loc=loc, framealpha=.5)
    #ax.legend(loc=loc, framealpha=.5, ncol=2)
    fig.savefig(output, bbox_inches='tight')


###############################################################################
# functions to write data to file and load back to numpy

def array_to_csv(table, fname):
    np.savetxt(fname, table, delimiter=',')

def csv_to_array(fname):
    table = np.loadtxt(fname, delimiter=',')
    table[:,0] = np.array([int(D) for D in table[:,0]])
    return table

def gen_data(fname):
    #X, z = data.univariate_normal([0, 5], [1, 2], [6000, 6000])
    X, z = data.univariate_lognormal([0, -1.5], [0.3, 1.5], [6000, 6000])
    data.histogram(X, z, colors=['#FF7500', '#0072FF'], fname=fname)
    

###############################################################################
if __name__ == '__main__':

    from timeit import default_timer as timer
    import multiprocessing as mp
    import sys

#    ### first experiment ###
#    #
#    def worker(dimensions, d, i):
#        table = gauss_dimensions_mean(dimensions=dimensions,
#                                  num_points=[100, 100],
#                                  delta=0.7, 
#                                  run_times=10,
#                                  num_experiments=100,
#                                  d=d)
#        np.savetxt("./data/gauss_means%i.csv"%i, table, delimiter=',')
#        
#    dim_array = [range(10,50,10), 
#                 range(50,100,10), 
#                 range(100, 150, 10),
#                 range(150, 210, 10)]
#    jobs = []
#    for i, dim in enumerate(dim_array):
#        p = mp.Process(target=worker, args=(dim, 10, i))
#        jobs.append(p)
#        p.start()
#
#    ### second experiment ###
#    #
    """
    def worker(dimensions, d, i):
        table = gauss_dimensions_cov(dimensions=dimensions,
                                  num_points=[100, 100],
                                  run_times=10,
                                  num_experiments=100,
                                  d=d)
        np.savetxt("./data/gauss_cov_v5_%i.csv"%i, table, delimiter=',')
        
    dim_array = [range(10,50,10), 
                 range(50,100,10), 
                 range(100, 150, 10),
                 range(150, 210, 10)]
    jobs = []
    for i, dim in enumerate(dim_array):
        p = mp.Process(target=worker, args=(dim, 10, i))
        jobs.append(p)
        p.start()
    sys.exit()
    """
#    
#    #dimensions = range(10,210,20) 
#    #table = gauss_dimensions_cov(dimensions=dimensions,
#    #                              num_points=[100, 100],
#    #                              run_times=3,
#    #                              num_experiments=5,
#    #                              )
#    #np.savetxt("./data/gauss_cov_v2.csv", table, delimiter=',')
#
#    ### third experiment ###
#    #
#    def worker(num_points, i):
#        table = gauss_dimensions_pi(num_points=num_points,
#                                    N=210, 
#                                    D=4,
#                                    d=2, 
#                                    run_times=10,
#                                    num_experiments=100)
#        np.savetxt("./data/gauss_pis%i.csv"%i, table, delimiter=',')
#        
#    m_array = [range(0,50,10), 
#               range(50,100,10), 
#               range(100, 150, 10),
#               range(150, 210, 10)]
#    jobs = []
#    for i, m in enumerate(m_array):
#        p = mp.Process(target=worker, args=(m, i))
#        jobs.append(p)
#        p.start()
#
#    ### fourth experiment ###
#    #
#    def worker(num_points, i):
#        table = normal_or_lognormal(numpoints=num_points,
#                                    run_times=4, num_experiments=100, 
#                                    kind='lognormal')
#        #np.savetxt("./data/normal2%i.csv"%i, table, delimiter=',')
#        np.savetxt("./data/lognormal2%i.csv"%i, table, delimiter=',')
#        
#    n_array = [range(20,220,20),
#               range(220,320,20),
#               range(320,420,20),
#               range(420,520,20)]
#    jobs = []
#    for i, n in enumerate(n_array):
#        p = mp.Process(target=worker, args=(n, i))
#        jobs.append(p)
#        p.start()

#    ### 1D clustering
#    #
#    def worker(num_points, i):
#        table = testing1d(num_points_range=num_points, run_times=4,
#                            num_experiments=100)
#        np.savetxt("./data/normal1d%i.csv"%i, table, delimiter=',')
#        #np.savetxt("./data/lognormal1d%i.csv"%i, table, delimiter=',')
#        
#    n_array = [range(10,500,50),
#               range(510,1000,50),
#               range(1010,1300,50),
#               range(1310, 1550,50)]
#    jobs = []
#    for i, n in enumerate(n_array):
#        p = mp.Process(target=worker, args=(n, i))
#        jobs.append(p)
#        p.start()
#    
#    ### make the plot
#    #
    table = csv_to_array("./data/gauss_means.csv")
#    table = csv_to_array("./data/gauss_cov_v2.csv")
#    table = csv_to_array("./data/gauss_cov_v4.csv") # linear
#    table = csv_to_array("./data/gauss_cov_v3.csv") # square
#    table = csv_to_array("./data/gauss_cov_v5.csv") # square root
#    table = csv_to_array("./data/gauss_pis.csv")
#    table = csv_to_array("./data/normal2.csv")
#    table = csv_to_array("./data/lognormal2.csv")
#    #table = csv_to_array("./data/loggauss_means.csv")
#    #table = csv_to_array("./data/cigars.csv")
#    #table = csv_to_array("./data/circles.csv")
#    #table = csv_to_array("./data/spirals.csv")
#    #table = csv_to_array("./data/lognormal.csv")
#    #table = csv_to_array("./data/normal.csv")
#    table = csv_to_array("./data/lognormal1d.csv")
#    table = csv_to_array("./data/normal1d.csv")
    plot_accuracy_errorbar(table, 
                    #xlabel='number of points', 
                    xlabel='number of dimensions', 
                    #xlabel='number of unbalanced points', 
                    ylabel='accuracy', 
                    output='./gauss_dim.pdf', 
                    #output='./gauss_cov.pdf', 
                    #output='./gauss_cov_linear.pdf', 
                    #output='./gauss_cov_square.pdf', 
                    #output='./gauss_cov_squareroot.pdf', 
                    #output='./gauss_pi.pdf', 
                    #output='./gauss.pdf', 
                    #output='./loggauss.pdf', 
                    #output='./loggauss1d.pdf', 
                    #output='./gauss1d.pdf', 
                    #legends=[
                    #    r'Alg.3--$\rho$', 
                    #    r'Alg.3--$\rho_{1/2}$', 
                    #    r'Alg.3--$\rho_{e}$', 
                    #    r'Alg.2--$\rho$', 
                    #    r'Alg.2--$\rho_{1/2}$', 
                    #    r'Alg.2--$\rho_{e}$', 
                    #    r'$k$-means', 
                    #    r'GMM'],
                    legends=['Algorithm 3', r'Algorithm 2', '$k$-means', 'GMM'],
                    #legends=['Algorithm 1', '$k$-means', 'GMM'],
                    #symbols=['o','o','o','s','s','s','D','D','D','v','v','v'],
                    #xlim=[10,3020],
                    xlim=[10,200],
                    #xlim=[0,200],
                    #xlim=[0,2000],
                    #xlim=[40,1000],
                    #ylim=[0.5,1.01],
                    #ylim=[0.5,0.915],
                    ylim=[0.4,0.87],
                    #ylim=[0.4,0.9],
                    #ylim=[0.5,0.91],
                    #ylim=[0.0,0.91],
                    #bayes=0.956,
                    #bayes=0.852,
                    bayes=0.86,
                    #bayes=0.95,
                    #bayes=1.0,
                    #bayes=0.9,
                    #loc=[.23,.2],
                    loc=0,
                    #doublex=True
    )
#    #gen_data('../draft/figs/two_lognormal_hist.pdf')
    
    #other_examples()
    
    #mnist(num_experiments=10, digits=[0,1], num_points=100, run_times=4)
    #mnist(num_experiments=10, digits=[0,1,2], num_points=100, run_times=4)
    #mnist(num_experiments=10, digits=[0,1,2,3], num_points=100, run_times=4)
    #mnist(num_experiments=10, digits=[0,1,2,3,4], num_points=100, run_times=4)
    #mnist(num_experiments=10, digits=[0,1,2,3,4,5], num_points=100, run_times=4)
    #mnist(num_experiments=10, digits=[0,1,2,3,4,5,6], num_points=100, run_times=4)
    #mnist(num_experiments=10, digits=[0,1,2,3,4,5,6,7], num_points=100, run_times=4)
    #mnist(num_experiments=10, digits=[0,1,2,3,4,5,6,7,8], num_points=100, run_times=4)
    #mnist(num_experiments=10, digits=[0,1,2,3,4,5,6,7,8,9], num_points=100, run_times=4)

    #mnist_pca(num_experiments=10, digits=[0,1,2,3,4,5,6], num_points=100,
    #            n_components=20, run_times=4)
