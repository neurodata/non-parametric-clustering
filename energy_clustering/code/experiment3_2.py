"""Third experiment. Normal distributions for unbalanced clusters.
Using mixtures now.

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

def gauss_dimensions_pi(num_points=range(0, 180, 10), num_experiments=10):
    """Test unbalanced clusters."""
    k = 2
    D = 4
    d = 2
    N = 250
    table = np.zeros((num_experiments*len(num_points), 6))
    count = 0

    for p in num_points:
        for i in range(num_experiments):

            # generate data
            m1 = np.zeros(D)
            s1 = np.eye(D)
            m2 = np.concatenate((1.5*np.ones(d), np.zeros(D-d)))
            s2 = np.diag(np.concatenate((.5*np.ones(d), np.ones(D-d))))
            pi1 = (N-p)/N
            pi2 = (N+p)/N
            print pi1, pi2
            n1, n2 = np.random.multinomial(N, [pi1,pi2])
            X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])
            rho = lambda x, y: np.linalg.norm(x-y)
            G = eclust.kernel_matrix(X, rho)

            table[count, 0] = p
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
    p.xlabel = 'number of unbalanced points'
    p.legends = [r'$\mathcal{E}^{H}$', 
                 r'$\mathcal{E}^{L}$', 
                 r'$\mathcal{E}^{S}$', 
                 r'$k$-means', 
                 r'GMM']
    p.colors = ['b', 'r', 'g', 'm', 'c']
    p.symbols = ['o', 's', 'D', '^', 'v']
    p.xlim = [0,240]
    p.output = './experiments_figs/normal_unbalanced.pdf'
    p.make_plot(table)

def gen_data(fname):
    ## choose the range for each worker ##
    n_array = [range(0,50,10),
               range(50,100,10),
               range(100,150,10),
               range(150, 250,10)]
    jobs = []
    for i, n in enumerate(n_array):
        p = mp.Process(target=worker, args=(n, fname%i))
        jobs.append(p)
        p.start()

def worker(dimensions, fname):
    """Used for multiprocessing. i is the index of the file, each process
    will generate its own output file.
    
    """
    table = gauss_dimensions_pi(dimensions, num_experiments=100)
    np.savetxt(fname, table, delimiter=',')
    

###############################################################################
if __name__ == '__main__':
    fname = './experiments_data2/experiment_unbalanced_%i.csv'
    gen_data(fname)
    #make_plot(fname%0, fname%1, fname%2, fname%3)
