"""
Tests and comparison between Energy, K-means, and GMM.

We run Energy, K-means, and GMM a few times, and show average accuracy 
and variability for several settings.

"""

# Guilhere Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata

import numpy as np
import matplotlib.pyplot as plt
import pylab

import eclust.data as data
import eclust.kmeans as km
import eclust.gmm as gm
import eclust.kenergy as ke
import eclust.kmeanspp as kpp

from eclust import metric 
from eclust import objectives


# pyplot customization
pylab.rc('lines', linewidth=.5, antialiased=True, markeredgewidth=0.1)
pylab.rc('font', family='computer modern roman', style='normal',
         weight='normal', serif='computer modern sans serif', size=14)
pylab.rc('text', usetex=True)
pylab.rc('text.latex', preamble=[
        '\usepackage{amsmath,amsfonts,amssymb,relsize,cancel}'])
pylab.rc('axes', linewidth=0.5, labelsize=14)
pylab.rc('xtick', labelsize=14)
pylab.rc('ytick', labelsize=14)
#pylab.rc('legend', numpoints=1, fontsize=10, handlelength=0.5)
pylab.rc('legend', numpoints=1, fontsize=14)
fig_width_pt = 455.0 / 1.5 # take this from LaTeX \textwidth in points
inches_per_pt = 1.0/72.27
golden_mean = (np.sqrt(5.0)-1.0)/2.0
fig_width = fig_width_pt*inches_per_pt
fig_height = fig_width
#fig_height = fig_width*golden_mean
pylab.rc('figure', figsize=(fig_width, fig_height))


###############################################################################
# clustering functions

def kernel_energy(k, X, kernel_matrix, z, run_times=3):
    """Run few times and pick the best objective function value."""
    G = kernel_matrix
    best_score = -np.inf
    for rt in range(run_times):
        
        z0 = kpp.kpp(k, X, ret='labels')
        Z0 = ke.ztoZ(z0)
        
        kernel_energy = ke.KernelEnergy(k, G, z0, max_iter=300)
        zh = kernel_energy.fit_predict(X)
        Zh = ke.ztoZ(zh)
        score = ke.objective(Zh, G)
        
        if score > best_score:
            best_score = score
            best_z = zh

    return metric.accuracy(z, best_z)

def cost_energy(k, X, kernel_matrix, z, run_times=3):
    """Run few times and pick the best objective function value."""
    G = kernel_matrix
    best_score = -np.inf
    for rt in range(run_times):
        
        z0 = kpp.kpp(k, X, ret='labels')
        Z0 = ke.ztoZ(z0)
        
        zh = ke.kenergy_brute(k, G, Z0, max_iter=300)
        Zh = ke.ztoZ(zh)
        score = ke.objective(Zh, G)
        if score > best_score:
            best_score = score
            best_z = zh
    return metric.accuracy(z, best_z)

def kmeans(k, X, z, run_times=3):
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

def gmm(k, X, z, run_times=3):
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


###############################################################################
# functions to test different settings

def gauss(d=2, D=10, num_points=[200,200], num_times=5, run_times=3):
    """Compare algorithms for gaussians in D dimensions."""
    
    k = 2
    n1, n2 = num_points
    
    m1 = np.zeros(D)
    #s1 = np.eye(D)
    #s1 = np.array([[1,0], [0,20]])
    s1 = np.array([[4,8], [8,20]])
    
    #m2 = np.concatenate((2*np.ones(d), np.zeros(D-d)))
    #m2 = 4*np.ones(d)
    #m2 = 6.5*np.array([1,0])
    #m2 = 8*np.array([1,0])
    m2 = np.array([0,0])
    #s2 = np.eye(d)
    #s2 = np.array([[1,0], [0,20]])
    s2 = np.array([[4,-8], [-8,20]])
    #for i in range(d): 
    #    for j in range(i+1,d):
    #        if i == j:
    #            value = .5
    #        else:
    #            value =  0  
    #        s2[i,j] = value
    #        s2[j,i] = value
                
    X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])
    rho = lambda x, y: ke.euclidean_rho(x, y, alpha=0.5)
    G = ke.kernel_matrix(X, rho, x0=np.array([20,20]))

    table = np.zeros((num_times, 4))
    for nt in range(num_times):
        table[nt, 0] = cost_energy(k, X, G, z, run_times=run_times)
        table[nt, 1] = kernel_energy(k, X, G, z, run_times=run_times)
        table[nt, 2] = kmeans(k, X, z, run_times=run_times)
        table[nt, 3] = gmm(k, X, z, run_times=run_times)
    
    return table

def delta(x):
    if x > 1:
        return x
    else:
        return 0

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
                          delta=0.5, num_times=5, run_times=3, d=None):
    """Do the same as above but we keep increasing the dimension.
    We only change the mean and keep unit covariance. This should
    keep Bayes error fixed.

    """
    k = 2
    if not d:
        d = dimensions[0]
    n1, n2 = num_points
    table = np.zeros((num_times*len(dimensions), 5))
    count = 0
    
    for D in dimensions:
        
        # generate data
        m1 = np.zeros(D)
        s1 = np.eye(D)
        m2 = np.concatenate((delta*np.ones(d), np.zeros(D-d)))
        s2 = np.eye(D)
        
        X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])
        rho = lambda x, y: ke.euclidean_rho(x, y, alpha=1.5)
        G = ke.kernel_matrix(X, rho)
        
        # for each D, run a few times, cluster, and pick accuracy
        for i in range(num_times):

            table[count, 0] = D
            table[count, 1] = cost_energy(k, X, G, z, run_times=run_times)
            table[count, 2] = kernel_energy(k, X, G, z, run_times=run_times)
            table[count, 3] = kmeans(k, X, z, run_times=run_times)
            table[count, 4] = gmm(k, X, z, run_times=run_times)

            count += 1

    return table

def gauss_dimensions_cov(dimensions=range(2,100,20), num_points=[200, 200],
                         delta=.5, sigma=.5, num_times=5, run_times=3, d=None):
    """Do the same as above but we keep increasing the dimension.
    We change the covariance of one gaussian and keep unit covariance in
    the other one. We also change the mean as above.
    This should keep Bayes error fixed.

    """
    
    k = 2
    if not d:
        d = dimensions[0]
    n1, n2 = num_points
    table = np.zeros((num_times*len(dimensions), 5))
    count = 0
    
    for D in dimensions:
        
        # generate data
        m1 = np.zeros(D)
        s1 = np.eye(D)
        m2 = np.concatenate((delta*np.ones(d), np.zeros(D-d)))
        #s2 = np.diag(np.concatenate((sigma*np.ones(d), np.ones(D-d))))
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
        rho = lambda x, y: ke.euclidean_rho(x, y, alpha=1.5)
        G = ke.kernel_matrix(X, rho)
        
        # for each D, run a few times, cluster, and pick accuracy
        for i in range(num_times):

            table[count, 0] = D
            table[count, 1] = cost_energy(k, X, G, z, run_times=run_times)
            table[count, 2] = kernel_energy(k, X, G, z, run_times=run_times)
            table[count, 3] = kmeans(k, X, z, run_times=run_times)
            table[count, 4] = gmm(k, X, z, run_times=run_times)

            count += 1

    return table

def gauss_dimensions_pi(num_points=range(0, 180, 10), N=200, D=4, d=2,
                        num_times=5, run_times=3):
    """Vary the number of points in each cluster."""
    k = 2
    table = np.zeros((num_times*len(num_points), 5))
    count = 0
    
    for p in num_points:
        
        # generate data
        m1 = np.zeros(D)
        s1 = np.eye(D)
        m2 = np.concatenate((1.5*np.ones(d), np.zeros(D-d)))
        s2 = np.diag(np.concatenate((.5*np.ones(d), np.ones(D-d))))
        n1 = N-p
        n2 = N+p
        
        X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])
        rho = lambda x, y: ke.euclidean_rho(x, y, alpha=1.5)
        G = ke.kernel_matrix(X, rho)
        
        # for each D, run a few times, cluster, and pick accuracy
        for i in range(num_times):
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


###############################################################################
# ploting functions

def plot_mean_range(table, 
                    xlabel='dimension', 
                    ylabel='accuracy',
                    output='compare_gauss_dim.pdf', 
                    colors=['green', 'red', 'blue'], 
                    symbols=['v', '^', 'o'],
                    legends=['GMM', 'k-means', 'Energy'],
                    loc=(0, 0.1)):
    """Given a table of results, where table[0] contains a value of
    several runs of an experiment, and the remaining columns contains
    the accuracy when clustering with different algorithms, we plot
    table[0] versus the mean accuracy of each method, and include a shaded
    region illustrating the variability.
    
    """
    
    col0 = np.array([int(x) for x in table[:,0]])
    dimensions = np.unique(col0)
    dimensions_names = np.unique(col0)+1
    
    #colors = iter(plt.cm.rainbow(np.linspace(0, 1, 3)))
    colors = iter(colors)
    legends = iter(legends)
    symbols = iter(symbols)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(dimensions_names, [.5]*len(dimensions), '--', color='black')
    
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
        
        ax.plot(dimensions_names, mean_, linestyle='-', marker=m, color=c, 
                alpha=.7, linewidth=2, markersize=7, label=l)
        ax.fill_between(dimensions_names, min_, max_, color=c,
                        alpha=.4, linewidth=1)
    

    #ax.set_xlabel('Dimension', fontsize=12)
    #ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.set_xticks([int(D) for D in dimensions])
    #ax.set_ylim([0, 1])
    #ax.set_xlim([min(dimensions)-2, max(dimensions)+2])
    ax.set_ylim([0.48, 1.02])
    ax.set_xlim([0.84, 10.16])
    #leg = plt.legend()
    #leg.get_frame().set_linewidth(0.5)
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
    

###############################################################################
if __name__ == '__main__':

    ### use this to test different settings
    #
    #table = gauss(d=2, D=4, num_points=[20+120,350-120], 
    #              num_times=2, run_times=5)
    #table = gauss(d=2, D=2, num_points=[200,200], num_times=10, run_times=10)
    #table = circles_or_spirals(num_points=[200,200], num_times=5, run_times=5)
    #print table

    ### first experiment ###
    #
    #table = gauss_dimensions_mean(dimensions=range(10,100,10), 
    #                              num_points=[100, 100],
    #                              delta=0.7, 
    #                              num_times=10,
    #                              run_times=4)
    #np.savetxt("./data/gauss_means.csv", table, delimiter=',')
    #table = gauss_dimensions_mean(dimensions=range(100,200,10), 
    #                              num_points=[100, 100],
    #                              delta=0.7, 
    #                              num_times=10, d=10)
    #np.savetxt("./data/gauss_means2.csv", table, delimiter=',')
    #
    
    
    ### second experiment ###
    #
    #table = gauss_dimensions_cov(dimensions=range(10,100,10), 
    #                             num_points=[100, 100],
    #                             delta=.7, sigma=.5, 
    #                             num_times=10, 
    #                             run_times=4)
    #np.savetxt("./data/gauss_cov.csv", table, delimiter=',')
    #table = gauss_dimensions_cov(dimensions=range(100,200,10), 
    #                             num_points=[100, 100],
    #                             delta=.7, sigma=.5, 
    #                             num_times=10, 
    #                             run_times=4,
    #                             d=10)
    #np.savetxt("./data/gauss_cov2.csv", table, delimiter=',')
    #table = gauss_dimensions_cov(dimensions=range(330,430,10), 
    #                             num_points=[200, 200],
    #                             delta=.7, sigma=.5, num_times=10, d=10)
    #np.savetxt("gauss_cov3.csv", table, delimiter=',')
    #table = csv_to_array("gauss_cov.csv")
    #plot_mean_range(table, xlabel='dimension', ylabel='accuracy',
    #                output='gauss_covs.pdf', 
    #                colors=['#4FC894', '#1F77B4', '#FF7F0E'], 
    #                symbols=['v', '^', 'o'], 
    #                legends=['GMM', r'$k$-means', 'Energy'],
    #                loc=(.65,0.1))

    ### third experiment ###
    #
    #table = gauss_dimensions_pi(num_points=range(0, 200, 10), N=200, D=4,
    #                            d=2, num_times=10, run_times=4)
    #np.savetxt("./data/gauss_pis.csv", table, delimiter=',')
    #table = gauss_dimensions_mean(dimensions=range(100,200,10), 
    #                              num_points=[100, 100],
    #                              delta=0.7, 
    #                              num_times=10, d=10)
    #np.savetxt("./data/gauss_means2.csv", table, delimiter=',')
    #

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
    
    ### parallel cigars ###
    #table = parallel_cigars()
    #np.savetxt("data_cigar.csv", table, delimiter=',')
    
    #table = csv_to_array("data_trunk.csv")
    #plot_accuracy_vs_dimension(table, 'compare_gauss_trunk.pdf')
    

    ### make the plot
    #
    #table = csv_to_array("./data/gauss_means.csv")
    #table = csv_to_array("./data/gauss_cov.csv")
    #table = csv_to_array("./data/gauss_pis.csv")
    #table = csv_to_array("./data/cigars.csv")
    #table = csv_to_array("./data/circles.csv")
    table = csv_to_array("./data/spirals.csv")
    plot_mean_range(table, 
                    xlabel='number of experiments', 
                    #xlabel='dimension', 
                    #xlabel='number of unbalanced points', 
                    ylabel='accuracy', 
                    #output='../draft/figs/gauss_means.pdf', 
                    #output='../draft/figs/gauss_covs.pdf', 
                    #output='../draft/figs/gauss_pis.pdf', 
                    #output='../draft/figs/cigars.pdf', 
                    #output='../draft/figs/circles.pdf', 
                    output='../draft/figs/spirals.pdf', 
                    #colors=['#4FC894', '#1F77B4', '#FF7F0E', '#9E65A5'], 
                    colors=['g', 'y', 'b', 'r'], 
                    symbols=['v', 'h', 'o', 'd'], 
                    legends=['GMM', r'$k$-means', 'K-Energy', 'C-Energy'],
                    #loc=(.51,0.1)
                    #loc=(.51,0.1)
                    #loc=(.05,0.1)
                    loc=(.2,0.6)
    )


