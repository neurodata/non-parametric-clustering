"""Second experiment. Normal distributions in high dimensions.
The difference here is only that we use bernoulli coins to get
number of points for each cluster, and also we are using spectral
clustering from scikit-learn.

"""

# Guilhere Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata

import numpy as np
import multiprocessing as mp

import energy.data as data
import energy.eclust as eclust
import energy.initialization as initialization
import run_clustering
import plot

def gauss_dimensions_mean(dimensions=range(2,100,20), total_points=200,
                          num_experiments=100, d=None):
    # data distribution
    k = 2
    delta = 0.7
    if not d:
        d = dimensions[0]
    table = np.zeros((num_experiments*len(dimensions), 6))
    count = 0

    for D in dimensions:
        for i in range(num_experiments):

            ### generate data
            m1 = np.zeros(D)
            s1 = np.eye(D)
            m2 = np.concatenate((delta*np.ones(d), np.zeros(D-d)))
            s2 = np.eye(D)
            # flip Bernoulli coins to get number of points in each cluster
            n1, n2 = np.random.multinomial(total_points, [0.5,0.5])
            # get data, construct gram Matrix
            X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])
            rho = lambda x, y: np.linalg.norm(x-y)
            G = eclust.kernel_matrix(X, rho)
            ##################

            ### cluster with different algorithms
            # can change number of times we execute each experiment
            # and initialization as well
            table[count, 0] = D
            table[count, 1] = run_clustering.energy_hartigan(k, X, G, z, 
                                                init="k-means++", run_times=5)
            table[count, 2] = run_clustering.energy_lloyd(k, X, G, z, 
                                                init="k-means++", run_times=5)
            table[count, 3] = run_clustering.spectral(k, X, G, z, 
                                                run_times=5)
            table[count, 4] = run_clustering.kmeans(k, X, z, 
                                                init="k-means++", run_times=5)
            table[count, 5] = run_clustering.gmm(k, X, z, 
                                                init="kmeans", run_times=5)
            ####################
            
            count += 1

    return table

def gauss_dimensions_cov(dimensions=range(2,100,20), total_points=200,
                         num_experiments=100, d=10):
    """High dimensions but with nontrivial covariance."""
    k = 2
    if not d:
        d = dimensions[0]
    table = np.zeros((num_experiments*len(dimensions), 6))
    count = 0

    for D in dimensions:
        for l in range(num_experiments):

            # generate data
            m1 = np.zeros(D)
            m2 = np.concatenate((np.ones(d), np.zeros(D-d)))
            s1 = np.eye(D)
            # from uniform 1, 5
            s2_1 = np.array([1.367,  3.175,  3.247,  4.403,  1.249,                                             1.969, 4.035,   4.237,  2.813,  3.637])
            s2 = np.diag(np.concatenate((s2_1, np.ones(D-d))))
            n1, n2 = np.random.multinomial(total_points, [0.5,0.5])
            X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])

            rho = lambda x, y: np.linalg.norm(x-y)
            G = eclust.kernel_matrix(X, rho)
            
            # can change the number of times we execute each experiment
            # and initialization method as well
            table[count, 0] = D
            table[count, 1] = run_clustering.energy_hartigan(k, X, G, z, 
                                                init="k-means++", run_times=5)
            table[count, 2] = run_clustering.energy_lloyd(k, X, G, z, 
                                                init="k-means++", run_times=5)
            table[count, 3] = run_clustering.spectral(k, X, G, z, 
                                                      run_times=5)
            table[count, 4] = run_clustering.kmeans(k, X, z, 
                                                init="k-means++", run_times=5)
            table[count, 5] = run_clustering.gmm(k, X, z, 
                                                init="kmeans", run_times=5)
            count += 1

    return table
    
def make_plot(*data_files):
    table = []
    for f in data_files:
        t = np.loadtxt(f, delimiter=',')
        t[:,0] = np.array([int(D) for D in t[:,0]])
        table.append(t)
    table = np.concatenate(table)

    ## customize plot below ##
    p = plot.ErrorBar()
    p.xlabel = r'$\#$ dimensions'
    p.ylabel = 'accuracy'
    p.legends = [r'$\mathcal{E}^{H}$', 
                 #r'kernel', 
                 r'$\mathcal{E}^L$', 
                 r'spectral', 
                 r'$k$-means', 
                 r'GMM']
    p.loc = 3
    p.colors = ['b', 'r', 'g', 'm', 'c']
    p.symbols = ['o', 's', 'D', '^', 'v']
    p.lines = ['-', '-', '-', '-', '-']
    #p.output = './experiments_figs2/normal_highdim_mean.pdf'
    p.output = './experiments_figs/normal_highdim_cov.pdf'
    #p.bayes = 0.86
    p.bayes = 0.9537075
    #p.xlim = [10, 200]
    p.xlim = [10, 700]
    #p.ylim = [0.55, 0.87]
    p.ylim = [0.55, 0.965]
    p.make_plot(table)

def gen_data(fname):
    ## choose the range for each worker ##
    #n_array = [range(10,50,10),
    #           range(50,100,10),
    #           range(100,150,10),
    #           range(150, 210,10)]
    n_array = [range(10,200,25),
               range(200,350,25),
               range(350,500,25),
               range(500,725,25)]
    jobs = []
    for i, n in enumerate(n_array):
        p = mp.Process(target=worker, args=(n, fname%i))
        jobs.append(p)
        p.start()

def worker(dimensions, fname):
    """Used for multiprocessing. i is the index of the file, each process
    will generate its own output file.
    
    """
    #table = gauss_dimensions_mean(dimensions, d=10, num_experiments=100)
    table = gauss_dimensions_cov(dimensions, d=10, num_experiments=100)
    np.savetxt(fname, table, delimiter=',')
    

###############################################################################
if __name__ == '__main__':
    #fname = './experiments_data2/experiment_highdim_mean_%i.csv'
    fname = './experiments_data2/experiment_highdim_cov1_%i.csv'
    #gen_data(fname)
    make_plot(fname%0, fname%1, fname%2, fname%3)
