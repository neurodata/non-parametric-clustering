"""
Tests and comparison between Energy, K-means, and GMM.

We run Energy, K-means, and GMM a few times, and show average accuracy 
and variability for several settings.

"""

# Guilhere Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata

import numpy as np
import matplotlib.pyplot as plt

import eclust.data as data
import eclust.kmeans as km
import eclust.gmm as gm
import eclust.qcqp as ke
import eclust.kmeanspp as kpp
import eclust.energy1d as e1d

from eclust import metric 
from eclust import objectives

from pplotcust import *


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

def circles_or_spirals(num_points=[200,200], num_times=5, run_times=3):
    
    k = 2
    n1, n2 = num_points
    
    #X, z = data.circles([1, 3], [0.1, 0.1], [n1, n2])
    X, z = data.spirals([1, -1], [n1, n2], noise=.2)
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

def loggauss_dimensions_mean(dimensions=range(2,100,20), num_points=[200, 200],
                          delta=0.5, run_times=5, num_experiments=10, d=None):
    """Same as above but use lognormal distribution."""
    k = 2
    if not d:
        d = dimensions[0]
    n1, n2 = num_points
    table = np.zeros((num_experiments*len(dimensions), 6))
    count = 0
    
    for D in dimensions:

        for i in range(num_experiments):
        
            # generate data
            m1 = np.zeros(D)
            s1 = 0.3*np.eye(D)
            m2 = np.concatenate((delta*np.ones(d), np.zeros(D-d)))
            s2 = 1.0*np.eye(D)
        
            X, z = data.multivariate_lognormal([m1, m2], [s1, s2], [n1, n2])
            
            rho = lambda x, y: np.power(np.linalg.norm(x-y), 1)
            G = ke.kernel_matrix(X, rho)
            rho2 = lambda x, y: np.power(np.linalg.norm(x-y), 0.5)
            G2 = ke.kernel_matrix(X, rho2)
        
            table[count, 0] = D
            table[count, 1] = cost_energy(k, X, G2, z, run_times=run_times)
            table[count, 2] = cost_energy(k, X, G, z, run_times=run_times)
            table[count, 3] = kernel_energy(k, X, G, z, run_times=run_times)
            table[count, 4] = kmeans(k, X, z, run_times=run_times)
            table[count, 5] = gmm(k, X, z, run_times=run_times)

            count += 1

    return table

def gauss_dimensions_cov(dimensions=range(2,100,20), num_points=[200, 200],
                         delta=.5, sigma=.5, run_times=3, num_experiments=10,
                        d=None):
    """High dimensions but in spite of the mean we also change the 
    covariance.
    
    """
    k = 2
    if not d:
        d = dimensions[0]
    n1, n2 = num_points
    table = np.zeros((num_experiments*len(dimensions), 5))
    count = 0
    
    for D in dimensions:

        for l in range(num_experiments):
        
            # generate data
            m1 = np.zeros(D)
            s1 = np.eye(D)
            m2 = np.concatenate((delta*np.ones(d), np.zeros(D-d)))
            s2 = np.eye(D)
            for i in range(d): 
                for j in range(i+1,d):
                    if i == j:
                        value = sigma
                    else:
                        value = 0
                s2[i,j] = value
                s2[j,i] = value
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

def trunk(dimensions=range(2,100,20), num_points=[200, 200], num_times=5):
    """We test Trunk's problem for clustering."""
    
    k = 2
    n1, n2 = num_points
    table = np.zeros((num_times*len(dimensions), 4))
    count = 0
    
    for D in dimensions:
        
        # generate data
        m1 = np.array([np.sqrt(d) for d in range(1, D+1)])
        s1 = np.eye(D)
        
        m2 = np.array([-np.sqrt(d) for d in range(1, D+1)])
        s2 = np.eye(D)
        
        X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])
        
        # for each D, run a few times, cluster, and pick accuracy
        for i in range(num_times):

            table[count, 0] = D
            table[count, 1] = kernel_energy(k, X, z, alpha=1, run_times=3)
            table[count, 2] = kmeans(k, X, z, run_times=3)
            table[count, 3] = gmm(k, X, z, run_times=3)

            count += 1

    return table

def cigars(num_experiments=range(10), num_points=[100, 100], 
            num_times=5, run_times=3):
    """2D Parallel Cigars."""
    k = 2
    n1, n2 = num_points
    table = np.zeros((num_times*len(num_experiments), 5))
    
    m1 = np.zeros(2)
    s1 = np.array([[1,0], [0,20]])
    
    m2 = 6.5*np.array([1,0])
    s2 = np.array([[1,0], [0,20]])
    
    count = 0
    
    for ne in num_experiments:
        
        X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])
        
        rho = lambda x, y: ke.euclidean_rho(x, y, alpha=0.5)
        G = ke.kernel_matrix(X, rho)
        
        for i in range(num_times):

            table[count, 0] = ne
            table[count, 1] = cost_energy(k, X, G, z, run_times=run_times)
            table[count, 2] = kernel_energy(k, X, G, z, run_times=run_times)
            table[count, 3] = kmeans(k, X, z, run_times=run_times)
            table[count, 4] = gmm(k, X, z, run_times=run_times)

            count += 1

    return table

def circles_or_spirals(num_experiments=range(10), num_points=[100, 100], 
            num_times=5, run_times=3, which='circles'):
    """2D Parallel Cigars."""
    k = 2
    n1, n2 = num_points
    table = np.zeros((num_times*len(num_experiments), 5))
    
    m1 = np.zeros(2)
    s1 = np.array([[1,0], [0,20]])
    
    m2 = 6.5*np.array([1,0])
    s2 = np.array([[1,0], [0,20]])
    
    count = 0
    
    for ne in num_experiments:
        
        if which == 'circles':
            X, z = data.circles([1, 3], [0.1, 0.1], [n1, n2])
            # this one works for circles
            rho = lambda x, y: np.power(np.linalg.norm(x-y), 2)*\
                            np.exp(-0.5*np.linalg.norm(x-y))
        else:
            X, z = data.spirals([1, -1], [n1, n2], noise=.15)
            # this one is working decently for spirals
            rho = lambda x, y: np.power(np.linalg.norm(x-y), 2)*\
                        .9*np.sin(np.linalg.norm(x-y)/0.9)

        G = ke.kernel_matrix(X, rho)
        
        for i in range(num_times):

            table[count, 0] = ne
            table[count, 1] = cost_energy(k, X, G, z, run_times=run_times)
            table[count, 2] = kernel_energy(k, X, G, z, run_times=run_times)
            table[count, 3] = kmeans(k, X, z, run_times=run_times)
            table[count, 4] = gmm(k, X, z, run_times=run_times)

            count += 1

    return table

def testing1d(num_points_range=range(50, 100, 20), run_times=3):
    """Comparing clustering methods in 1 dimension."""
    k = 2
    table = np.zeros((len(num_points_range), 4))
    
    for i, ne in enumerate(num_points_range):
        
        m1 = 0
        s1 = 0.3
        m2 = -1.5
        s2 = 1.5
        n1 = n2 = ne
        X, z = data.univariate_lognormal([m1, m2], [s1, s2], [n1, n2])
        
        #m1 = 0
        #s1 = 1
        #m2 = 5
        #s2 = 2
        #n1 = n2 = ne
        #X, z = data.univariate_normal([m1, m2], [s1, s2], [n1, n2])
        
        Y = np.array([[x] for x in X])
        
        table[i, 0] = ne
        table[i, 1] = two_clusters1D(X, z)
        table[i, 2] = kmeans(k, Y, z, run_times=run_times)
        table[i, 3] = gmm(k, Y, z, run_times=run_times)

    return table



###############################################################################
# ploting functions

def plot_accuracy_errorbar(table, 
                    xlabel='dimension', 
                    ylabel='accuracy',
                    output='plot.pdf', 
                    colors=['green', 'red', 'blue'], 
                    symbols=['o', 'o', 'o'],
                    legends=['GMM', 'k-means', 'Energy'],
                    loc=(0, 0.1)):
    
    col0 = np.array([int(x) for x in table[:,0]])
    dimensions = np.unique(col0)
    
    colors = iter(colors)
    legends = iter(legends)
    symbols = iter(symbols)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(table[0][1:]), 0, -1):

        mean_ = np.array([table[np.where(col0==D)[0],i].mean() 
                                for D in dimensions])
        max_ = np.array([table[np.where(col0==D)[0],i].max() 
                                for D in dimensions])
        min_ = np.array([table[np.where(col0==D)[0],i].min() 
                                for D in dimensions])
    
        c = next(colors)
        l = next(legends)
        m = next(symbols)
        
        #ax.plot(dimensions_names, mean_, linestyle='-', marker=m, color=c, 
        #        alpha=.7, linewidth=1, markersize=4, label=l)
        #ax.fill_between(dimensions_names, min_, max_, color=c,
        #                alpha=.4, linewidth=1)

        ax.errorbar(dimensions, mean_, yerr=[mean_-min_, max_-mean_], 
                    linestyle='-', marker=m, markersize=3,
                    ecolor=c, color=c, capthick=0.5, label=l)

    #ax.set_xlabel('Dimension', fontsize=12)
    #ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.set_xticks([int(D) for D in range(0, 1600, 300)])
    #ax.set_ylim([0, 1])
    #ax.set_xlim([min(dimensions)-2, max(dimensions)+2])
    #leg = plt.legend()
    #leg.get_frame().set_linewidth(0.5)
    ax.legend(loc=loc, framealpha=.5)
    fig.tight_layout()
    fig.savefig(output)

def plot_accuracy(table, 
                  xlabel='dimension', 
                  ylabel='accuracy',
                  output='compare_gauss_dim.pdf', 
                  colors=['green', 'red', 'blue'], 
                  symbols=['v', '^', 'o'],
                  legends=['GMM', 'k-means', 'Energy'],
                  loc=(0, 0.1)):
    """Just plot results. Use first column as x axis."""
    
    x_vals = np.array([int(a) for a in table[:,0]])
    colors = iter(colors)
    legends = iter(legends)
    symbols = iter(symbols)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    num_plots = len(table[0][1:])
    
    for i in range(num_plots, 0, -1):
        c = next(colors)
        l = next(legends)
        m = next(symbols)
        ax.plot(x_vals, table[:,i], linestyle='-', marker=m, color=c, 
                alpha=.8, linewidth=0.5, label=l)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.set_xlim([10,200])
    ax.legend(loc=loc, framealpha=.5)
    fig.tight_layout()
    fig.savefig(output)


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

    ### use this to test different settings
    #
    #table = gauss(d=2, D=4, num_points=[20+120,350-120], 
    #              num_times=2, run_times=5)
    #table = gauss(d=2, D=2, num_points=[200,200], num_times=10, run_times=10)
    #table = circles_or_spirals(num_points=[200,200], num_times=5, run_times=5)
    #print table

    ### first experiment ###
    #
    start = timer()
    table = gauss_dimensions_mean(dimensions=range(10,210,10), 
                                  num_points=[100, 100],
                                  delta=0.7, 
                                  run_times=10,
                                  num_experiments=1)
    np.savetxt("./data/gauss_means.csv", table, delimiter=',')
    end = timer()
    print end - start
    
    
    ### second experiment ###
    #
    #table = gauss_dimensions_cov(dimensions=range(10,210,10), 
    #                             num_points=[100, 100],
    #                             delta=.7, sigma=.5, 
    #                             run_times=10)
    #np.savetxt("./data/gauss_cov.csv", table, delimiter=',')

    ### third experiment ###
    #
    #table = gauss_dimensions_pi(num_points=range(0, 210, 10), N=210, D=4,
    #                            d=2, run_times=10)
    #np.savetxt("./data/gauss_pis.csv", table, delimiter=',')
    
    ### fourth experiment ###
    #
    #table = loggauss_dimensions_mean(dimensions=range(10,210,10), 
    #                              num_points=[100, 100],
    #                              run_times=10)
    #np.savetxt("./data/loggauss_means.csv", table, delimiter=',')

    ### cigars ###
    #
    #table = cigars(num_experiments=range(10), num_points=[100, 100], 
    #               num_times=10, run_times=10)
    #np.savetxt("./data/cigars.csv", table, delimiter=',')

    ### circles and spirals ###
    #
    #table = circles_or_spirals(num_experiments=range(10), 
    #                           num_points=[150, 150], 
    #                           num_times=10, 
    #                           run_times=10, 
    #                           which='spirals')
    #np.savetxt("./data/circles.csv", table, delimiter=',')
    #np.savetxt("./data/spirals.csv", table, delimiter=',')

    ### trunk experiment ###
    #table = compare_trunk(dimensions=range(2,212,10), 
    #                      num_points=[40, 40],
    #                      num_times=5)
    #np.savetxt("data_trunk.csv", table, delimiter=',')

    ### 1D clustering
    #
    #table = testing1d(num_points_range=range(20, 2000, 20), 
    #                  run_times=3)
    #np.savetxt("./data/normal.csv", table, delimiter=',')
    #table = testing1d(num_points_range=range(20, 2000, 20), 
    #                  run_times=3)
    #np.savetxt("./data/lognormal.csv", table, delimiter=',')
    
    """
    ### make the plot
    #
    table = csv_to_array("./data/gauss_means.csv")
    #table = csv_to_array("./data/gauss_cov.csv")
    #table = csv_to_array("./data/gauss_pis.csv")
    #table = csv_to_array("./data/loggauss_means.csv")
    #table = csv_to_array("./data/cigars.csv")
    #table = csv_to_array("./data/circles.csv")
    #table = csv_to_array("./data/spirals.csv")
    #table = csv_to_array("./data/lognormal.csv")
    #table = csv_to_array("./data/normal.csv")
    #table = csv_to_array("./data/lognormal.csv")
    plot_accuracy_errorbar(table, 
                    #xlabel='number of points', 
                    xlabel='dimension', 
                    #xlabel='number of unbalanced points', 
                    ylabel='accuracy', 
                    #output='../draft/figs/cigars.pdf', 
                    #output='../draft/figs/circles.pdf', 
                    #output='../draft/figs/spirals.pdf', 
                    #output='../draft/figs/normal.pdf', 
                    #output='../draft/figs/lognormal.pdf', 
                    output='./gauss_dim.pdf', 
                    #output='./gauss_cov.pdf', 
                    #output='./gauss_pi.pdf', 
                    #output='./loggauss_dim.pdf', 
                    #output='./lognormal.pdf', 
                    #colors=['#4FC894', '#1F77B4', '#FF7F0E', '#9E65A5'], 
                    #colors=['g', 'y', 'b', 'r'], 
                    colors=['c', 'r', 'm', 'b', 'g'], 
                    symbols=['o', 'o', 'o', 'o', 'o'], 
                    legends=['GMM', r'$k$-means', r'kernel $k$-means', 
                                r'energy', r'energy2'],
                    #loc=(.51,0.1)
                    #loc=(.51,0.1)
                    #loc=(.05,0.1)
                    #loc=(.2,0.6)
                    loc=(0.3,0.05)
    )
    """
    #gen_data('../draft/figs/two_lognormal_hist.pdf')
