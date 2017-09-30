"""Second experiment. Normal distributions in high dimensions."""

# Guilhere Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata

import numpy as np
import multiprocessing as mp

import energy.data as data
import energy.eclust as eclust
import energy.initialization as initialization
import run_clustering
import plot

def gauss_dimensions_mean(dimensions=range(2,100,20), num_points=[100, 100],
                          num_experiments=100, d=None):
    """Here we keep signal in one dimension and increase the ambient dimension.
    The covariances are kept fixed.
    
    """
    k = 2
    delta = 0.7
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
            G = eclust.kernel_matrix(X, rho)

            table[count, 0] = D
            table[count, 1] = run_clustering.energy_hartigan(k, X, G, z, 
                                    init="spectral", run_times=1)
            table[count, 2] = run_clustering.energy_lloyd(k, X, G, z, 
                                    init="spectral", run_times=1)
            table[count, 3] = run_clustering.kmeans(k, X, z)
            table[count, 4] = run_clustering.gmm(k, X, z)

            count += 1

    return table

def gauss_dimensions_cov(dimensions=range(2,100,20), num_points=[100, 100],
                         num_experiments=100, d=None):
    """High dimensions but with nontrivial covariance."""
    k = 2
    q = 0.5
    if not d:
        d = dimensions[0]
    n1, n2 = num_points
    table = np.zeros((num_experiments*len(dimensions), 5))
    count = 0

    for D in dimensions:
        for l in range(num_experiments):

            # generate data
            m1 = np.zeros(D)
            m2 = np.concatenate((np.ones(d), np.zeros(D-d)))
            s1 = np.eye(D)
            s2 = np.eye(D)
            for a in range(d):
                s1[a,a] = np.power(1/(a+1), q)
            for a in range(d):
                s2[a,a] = np.power(a+1, q)
            X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])

            rho = lambda x, y: np.linalg.norm(x-y)
            G = ke.kernel_matrix(X, rho)
            
            table[count, 0] = D
            table[count, 1] = run_clustering.energy_hartigan(k, X, G, z, 
                                    init="spectral", run_times=1)
            table[count, 2] = run_clustering.energy_lloyd(k, X, G, z, 
                                    init="spectral", run_times=1)
            table[count, 3] = run_clustering.kmeans(k, X, z)
            table[count, 4] = run_clustering.gmm(k, X, z)

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
    p.xlabel = 'number of dimensions'
    p.legends = [r'$\mathcal{E}^{H}$-clustering', 
                 r'$\mathcal{E}^{L}$-clustering', 
                 r'$k$-means', 
                 'GMM']
    p.colors = ['b', 'r', 'g', 'm']
    p.symbols = ['o', 's', '^', 'v']
    p.output = './experiments_figs/normal_highdim_mean.pdf'
    p.bayes = 0.86
    p.make_plot(table)

def gen_data(fname):
    ## choose the range for each worker ##
    n_array = [range(10,50,10),
               range(50,100,10),
               range(100,150,10),
               range(150, 210,10)]
    jobs = []
    for i, n in enumerate(n_array):
        p = mp.Process(target=worker, args=(n, fname%i))
        jobs.append(p)
        p.start()

def worker(dimensions, fname):
    """Used for multiprocessing. i is the index of the file, each process
    will generate its own output file.
    
    """
    table = gauss_dimensions_mean(dimensions, d=10)
    np.savetxt(fname, table, delimiter=',')
    

###############################################################################
if __name__ == '__main__':
    fname = './experiments_data/experiment_highdim_mean_%i.csv'
    gen_data(fname)
    #make_plot(fname%0, fname%1)
