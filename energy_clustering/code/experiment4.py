"""Fourth experiment. Normal and lognormal distributions in high dimensions,
using different kernels.

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

def normal_or_lognormal(numpoints=range(10,100,10), num_experiments=100,
                        kind='normal'):
    table = np.zeros((num_experiments*len(numpoints), 6))
    count = 0
    k = 2

    for n in numpoints:
        for i in range(num_experiments):

            # generate data
            D = 20
            d = 5
            m1 = np.zeros(D)
            s1 = 0.5*np.eye(D)
            m2 = 0.5*np.concatenate((np.ones(d), np.zeros(D-d)))
            s2 = np.eye(D)

            if kind == 'normal':
                X, z = data.multivariate_normal([m1, m2], [s1, s2], [n, n])
            else:
                X, z = data.multivariate_lognormal([m1, m2], [s1, s2], [n, n])

            rho = lambda x, y: np.linalg.norm(x-y)
            G = eclust.kernel_matrix(X, rho)

            rho2 = lambda x, y: np.power(np.linalg.norm(x-y), 0.5)
            G2 = eclust.kernel_matrix(X, rho2)

            rho3 = lambda x, y: 2-2*np.exp(-np.linalg.norm(x-y)/2)
            G3 = eclust.kernel_matrix(X, rho3)

            table[count, 0] = n
            table[count, 1] = run_clustering.energy_hartigan(k, X, G, z, 
                                                init="k-means++", run_times=5)
            table[count, 2] = run_clustering.energy_hartigan(k, X, G2, z, 
                                                init="k-means++", run_times=5)
            table[count, 3] = run_clustering.energy_hartigan(k, X, G3, z, 
                                                init="k-means++", run_times=5)
            table[count, 4] = run_clustering.kmeans(k, X, z, 
                                                init="k-means++", run_times=5)
            table[count, 5] = run_clustering.gmm(k, X, z, 
                                                init="kmeans", run_times=5)
            
            count += 1

    return table

def normal_or_lognormal_difference(numpoints=range(10,100,10), 
                                   num_experiments=100,
                                   kind='normal'):
    k = 2
    table = []
    for n in numpoints:
        for i in range(num_experiments):
            this_res = [n]

            # generate data
            D = 20
            d = 5
            m1 = np.zeros(D)
            s1 = 0.5*np.eye(D)
            m2 = 0.5*np.concatenate((np.ones(d), np.zeros(D-d)))
            s2 = np.eye(D)

            if kind == 'normal':
                X, z = data.multivariate_normal([m1, m2], [s1, s2], [n, n])
            else:
                X, z = data.multivariate_lognormal([m1, m2], [s1, s2], [n, n])

            rho = lambda x, y: 2-2*np.exp(-np.linalg.norm(x-y)/2)
            G = eclust.kernel_matrix(X, rho)

            hart = run_clustering.energy_hartigan(k, X, G, z, 
                                        init="k-means++", run_times=5)
            lloyd = run_clustering.energy_lloyd(k, X, G, z, 
                                        init="k-means++", run_times=5)
            spectral = run_clustering.energy_spectral(k, X, G, z, 
                                            init="k-means++", run_times=5)
            this_res.append(hart-lloyd)
            this_res.append(hart-spectral)
            
            table.append(this_res)
    table = np.array(table)
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
    p.xlabel = 'number of points'
    p.legends = [r'$\mathcal{E}^{H}$-clustering $\rho$', 
                 r'$\mathcal{E}^{H}$-clustering $\rho_{1/2}$', 
                 r'$\mathcal{E}^{H}$-clustering $\rho_{e}$', 
                 r'$k$-means', 
                 r'GMM']
    p.colors = ['b', 'r', 'g', 'm', 'c']
    p.symbols = ['o', 's', 'D', '^', 'v']
    p.lines = ['-', '-', '-', '-', '-']
    #p.output = './experiments_figs/normal_kernels.pdf'
    p.output = './experiments_figs/lognormal_kernels.pdf'
    p.doublex = True
    p.bayes = 0.9
    p.xlim = [5, 400]
    p.make_plot(table)

def make_plot_difference(*data_files):
    table = []
    for f in data_files:
        t = np.loadtxt(f, delimiter=',')
        t[:,0] = np.array([int(D) for D in t[:,0]])
        table.append(t)
    table = np.concatenate(table)

    ## customize plot below ##
    p = plot.ErrorBar()
    p.xlabel = 'number of points'
    p.ylabel = 'difference in accuracy'
    p.legends = [
        r'$\mathcal{E}^{H} - \mathcal{E}^{L}$', 
        r'$\mathcal{E}^{H} - \mathcal{E}^{S}$', 
    ]
    p.colors = ['b', 'r']
    p.symbols = ['o', 's']
    #p.output = './experiments_figs/normal_kernels_difference.pdf'
    p.output = './experiments_figs/lognormal_kernels_difference.pdf'
    p.doublex = True
    p.legcols = 1
    #p.bayes = 0.0
    p.xlim = [10, 400]
    p.make_plot(table)

def gen_data(fname):
    ## choose the range for each worker ##
    n_array = [range(5,50,5),
               range(50,100,5),
               range(100,150,5),
               range(150,205,5)]
    jobs = []
    for i, n in enumerate(n_array):
        p = mp.Process(target=worker, args=(n, fname%i))
        jobs.append(p)
        p.start()

def worker(numpoints, fname):
    """Used for multiprocessing. i is the index of the file, each process
    will generate its own output file.
    
    """
    #kind = 'normal'
    kind = 'lognormal'
    #table = normal_or_lognormal(numpoints, kind=kind, num_experiments=100)
    table = normal_or_lognormal_difference(numpoints, kind=kind,
                                           num_experiments=100)
    np.savetxt(fname, table, delimiter=',')
    

###############################################################################
if __name__ == '__main__':
    #fname = './experiments_data/experiment_normal_kernels_%i.csv'
    #fname = './experiments_data/experiment_lognormal_kernels_%i.csv'
    #fname = './experiments_data/experiment_normal_kernels_difference_%i.csv'
    fname='./experiments_data/experiment_lognormal_kernels_difference_%i.csv'
    #gen_data(fname)
    #make_plot(fname%0, fname%1, fname%2, fname%3)
    make_plot_difference(fname%0, fname%1, fname%2, fname%3)
