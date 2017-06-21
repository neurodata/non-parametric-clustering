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
import eclust.kernel as kernel
import eclust.kmeans as km
import eclust.gmm as gm
from eclust.metric import accuracy
from eclust import objectives


# pyplot customization
pylab.rc('lines', linewidth=.5, antialiased=True, markeredgewidth=0.1)
pylab.rc('font', family='computer modern roman', style='normal',
         weight='normal', serif='computer modern sans serif', size=10)
pylab.rc('text', usetex=True)
pylab.rc('text.latex', preamble=[
        '\usepackage{amsmath,amsfonts,amssymb,relsize,cancel}'])
pylab.rc('axes', linewidth=0.5, labelsize=10)
pylab.rc('xtick', labelsize=10)
pylab.rc('ytick', labelsize=10)
#pylab.rc('legend', numpoints=1, fontsize=10, handlelength=0.5)
pylab.rc('legend', numpoints=1, fontsize=10)
fig_width_pt = 455.0 / 1.5 # take this from LaTeX \textwidth in points
inches_per_pt = 1.0/72.27
golden_mean = (np.sqrt(5.0)-1.0)/2.0
fig_width = fig_width_pt*inches_per_pt
fig_height = fig_width*golden_mean
pylab.rc('figure', figsize=(fig_width, fig_height))


###############################################################################
# clustering functions

def kernel_energy(k, X, z, alpha=1, run_times=3):
    """Run few times and pick the best objective function value. This
    stabilizes a little bit the algorithm.
    
    """
    best_score = 0
    for i in range(run_times):
        ke = kernel.KernelEnergy(n_clusters=k, max_iter=300,
                                 kernel_params={'alpha':1})
        zh = ke.fit_predict(X)
        score = objectives.kernel_score(ke.kernel_matrix_, zh)
        if score > best_score:
            best_score = score
            best_z = zh
    return accuracy(best_z, z)

def kmeans(k, X, z, run_times=3):
    """Run k-means couple times and pick the best answer."""
    best_score = np.inf
    for i in range(run_times):
        zh = km.kmeans(k, X)
        score = objectives.kmeans(data.from_label_to_sets(X, zh))
        if score < best_score:
            best_score = score
            best_z = zh
    return accuracy(best_z, z)

def gmm(k, X, z, run_times=3):
    """Run gmm couple times and pick the best answer."""
    best_score = -np.inf
    best_z = None
    for i in range(run_times):
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
        return accuracy(best_z, z)
    else:
        return 0


###############################################################################
# functions to test different settings

def gauss(d=2, D=10, num_points=[200,200], num_times=5):
    """We compare algorithms for two gaussians in D dimensions.
    The gaussians have identity covariance and we only change the mean
    of the second gaussian.
    
    """
    
    k = 2
    n1, n2 = num_points
    
    m1 = np.zeros(D)
    s1 = np.eye(D)
    
    m2 = np.concatenate((0.6*np.ones(d), np.zeros(D-d)))
    s21 = np.zeros((d,d))
    for i in range(d) for j in range(d):
            s21 = 
            
    s2 = np.diag(np.concatenate((np.array([2*i for i in range(1,d+1)]), 
                 np.ones(D-d))))
    
    X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])

    table = np.zeros((num_times, 3))
    for i in range(num_times):
        table[i, 0] = kernel_energy(k, X, z, alpha=1, run_times=5)
        table[i, 1] = kmeans(k, X, z, run_times=5)
        table[i, 2] = gmm(k, X, z, run_times=5)
    
    return table

def gauss_dimensions_mean(dimensions=range(2,100,20), num_points=[200, 200],
                          delta=0.5, num_times=5, d=2):
    """Do the same as above but we keep increasing the dimension.
    We only change the mean and keep unit covariance. This should
    keep Bayes error fixed.

    """
    k = 2
    n1, n2 = num_points
    table = np.zeros((num_times*len(dimensions), 4))
    count = 0
    
    for D in dimensions:
        
        # generate data
        m1 = np.zeros(D)
        s1 = np.eye(D)
        
        #d = dimensions[0]
        m2 = np.concatenate((delta*np.ones(d), np.zeros(D-d)))
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

def gauss_dimensions_cov(dimensions=range(2,100,20), num_points=[200, 200],
                         delta=.5, sigma=.5, num_times=5, d=2):
    """Do the same as above but we keep increasing the dimension.
    We change the covariance of one gaussian and keep unit covariance in
    the other one. We also change the mean as above.
    This should keep Bayes error fixed.

    """
    
    k = 2
    n1, n2 = num_points
    table = np.zeros((num_times*len(dimensions), 4))
    count = 0
    
    for D in dimensions:
        
        # generate data
        m1 = np.zeros(D)
        s1 = np.eye(D)
        
        #d = dimensions[0]
        m2 = np.concatenate((delta*np.ones(d), np.zeros(D-d)))
        s2 = np.diag(np.concatenate((sigma*np.ones(d), np.ones(D-d))))
        
        X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])
        
        # for each D, run a few times, cluster, and pick accuracy
        for i in range(num_times):

            table[count, 0] = D
            table[count, 1] = kernel_energy(k, X, z, alpha=1, run_times=3)
            table[count, 2] = kmeans(k, X, z, run_times=3)
            table[count, 3] = gmm(k, X, z, run_times=3)

            count += 1

    return table

def gauss_dimensions_pis(dimensions=range(2,100,20), num_points=[100, 300],
                          delta=0.5, num_times=5, d=2):
    """Do the same as above but we keep increasing the dimension.
    We only change the mean and keep unit covariance. This should
    keep Bayes error fixed.

    """
    k = 2
    n1, n2 = num_points
    table = np.zeros((num_times*len(dimensions), 4))
    count = 0
    
    for D in dimensions:
        
        # generate data
        m1 = np.zeros(D)
        s1 = np.eye(D)
        
        #d = dimensions[0]
        m2 = np.concatenate((delta*np.ones(d), np.zeros(D-d)))
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

def cigars(num_points=[200, 200], mean=np.array([6,0]), cov=np.diag([1,20]),
           num_experiments=5, run_times=5):
    """We cluster parallel cigars. We generate data 'num_experiments' times,
    and for each experiment we run the algorithms 'run_times' times, on the
    same data.
    
    """
    k = 2
    n1, n2 = num_points
    table = np.zeros((num_experiments*run_times, 4))
    count = 0
    
    for j in range(num_experiments):
        
        m1 = np.array([0,0])
        s1 = cov
        
        m2 = mean
        s2 = cov
        
        X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])
        
        for i in range(run_times):

            table[count, 0] = j
            table[count, 1] = kernel_energy(k, X, z, alpha=.5, run_times=5)
            table[count, 2] = kmeans(k, X, z, run_times=5)
            table[count, 3] = gmm(k, X, z, run_times=5)

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
    
    #colors = iter(plt.cm.rainbow(np.linspace(0, 1, 3)))
    colors = iter(colors)
    legends = iter(legends)
    symbols = iter(symbols)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(dimensions, [.5]*len(dimensions), '--', color='black')
    
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
        
        ax.plot(dimensions, mean_, linestyle='-', marker=m, color=c, 
                alpha=.9, linewidth=2, markersize=8, label=l)
        ax.fill_between(dimensions, min_, max_, color=c,
                        alpha=.6, linewidth=1)
    

    #ax.set_xlabel('Dimension', fontsize=12)
    #ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.set_xticks([int(D) for D in dimensions])
    ax.set_ylim([0, 1])
    ax.set_xlim([min(dimensions)-2, max(dimensions)+2])
    ax.legend().get_frame().set_linewidth(0.5)
    ax.legend(loc=loc, shadow=False, framealpha=.5)
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

    table = gauss(d=10, D=11, num_points=[200,200], num_times=3)
    print table
    

    ### first experiment ###
    #
    #table = gauss_dimensions_mean(dimensions=range(10,230,10), 
    #                              num_points=[200, 200],
    #                              delta=0.7, 
    #                              num_times=10)
    #np.savetxt("gauss_means.csv", table, delimiter=',')
    #table = gauss_dimensions_mean(dimensions=range(330,430,10), 
    #                              num_points=[200, 200],
    #                              delta=0.7, 
    #                              num_times=10, d=10)
    #np.savetxt("gauss_means3.csv", table, delimiter=',')
    #
    #table = csv_to_array("gauss_means.csv")
    #plot_mean_range(table, xlabel='dimension', ylabel='accuracy',
    #                output='gauss_means.pdf', 
    #                colors=['#4FC894', '#1F77B4', '#FF7F0E'], 
    #                symbols=['v', '^', 'o'], 
    #                legends=['GMM', r'$k$-means', 'Energy'],
    #                loc=(.65,0.1))
    
    
    ### second experiment ###
    #
    #table = gauss_dimensions_cov(dimensions=range(10,230,10), 
    #                             num_points=[200, 200],
    #                             delta=.7, sigma=.5, num_times=10, d=10)
    #np.savetxt("gauss_cov1.csv", table, delimiter=',')
    #table = gauss_dimensions_cov(dimensions=range(230,330,10), 
    #                             num_points=[200, 200],
    #                             delta=.7, sigma=.5, num_times=10, d=10)
    #np.savetxt("gauss_cov2.csv", table, delimiter=',')
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
    #table = gauss_dimensions_pis(dimensions=range(10,230,10), 
    #                             num_points=[100, 300],
    #                             delta=0.7, num_times=10, d=10)
    #np.savetxt("gauss_pis1.csv", table, delimiter=',')
    #table = gauss_dimensions_pis(dimensions=range(230,330,10), 
    #                             num_points=[100, 300],
    #                             delta=0.7, num_times=10, d=10)
    #np.savetxt("gauss_pis2.csv", table, delimiter=',')
    #table = gauss_dimensions_pis(dimensions=range(330,430,10), 
    #                             num_points=[100, 300],
    #                             delta=0.7, num_times=10, d=10)
    #np.savetxt("gauss_pis3.csv", table, delimiter=',')
    #table = csv_to_array("gauss_pis.csv")
    #plot_mean_range(table, xlabel='dimension', ylabel='accuracy',
    #                output='gauss_pis.pdf', 
    #                colors=['#4FC894', '#1F77B4', '#FF7F0E'], 
    #                symbols=['v', '^', 'o'], 
    #                legends=['GMM', r'$k$-means', 'Energy'],
    #                loc=(.65,0.1))
    
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


