"""Other experiments such as cigars, circles and MNIST."""

# Guilhere Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata

from __future__ import division

import numpy as np
import multiprocessing as mp
import cPickle, gzip
import scipy.stats

import energy.data as data
import energy.eclust as eclust
import run_clustering
import energy.gmm
import energy.metric


# some kernels/semmimetric function
rho_standard = lambda x, y: np.power(np.linalg.norm(x-y), 1)
rho_half = lambda x, y: np.power(np.linalg.norm(x-y), 1/2)
rho_gauss = lambda x, y: 2-2*np.exp(-np.linalg.norm(x-y)**2/4)
rho_exp = lambda x, y, sigma: 2-2*np.exp(-np.linalg.norm(x-y)/(2*sigma))
rho_rbf = lambda x, y, sigma: 2-2*np.exp(-np.linalg.norm(x-y)**2/(2*sigma**2))

def cigars_circles(num_experiments=10, run_times=5, kind='cigars'):
    table = []
    for i in range(num_experiments):
        this_experiment = []
        
        if kind == 'cigars':
            m1 = [0,0]
            m2 = [6.5,0]
            s1 = np.array([[1,0],[0,20]])
            s2 = np.array([[1,0],[0,20]])
            X, z = data.multivariate_normal([m1, m2], [s1, s2], [200, 200])
            k = 2
            init = 'k-means++'
        elif kind == '2circles':
            X, z = data.circles([1, 3], [0.2, 0.2], [400, 400])
            k = 2
            init = 'random'
        elif kind == '3circles':
            X, z = data.circles([1, 3, 5], [0.2, 0.2, 0.2], [400, 400, 400])
            init = 'random'
            k = 3
        else:
            raise ValueError("Don't know which example to sample.")

        #sigma = 2
        sigma = 2.2
        G = eclust.kernel_matrix(X, rho_standard)
        G_half = eclust.kernel_matrix(X, rho_half)
        G_exp = eclust.kernel_matrix(X, lambda x,y: rho_exp(x, y, sigma))
        G_rbf = eclust.kernel_matrix(X, lambda x,y: rho_rbf(x, y, sigma))
        #G_exp = eclust.kernel_matrix(X, lambda x,y: rho_exp(x, y, 1))
        #G_rbf = eclust.kernel_matrix(X, lambda x,y: rho_rbf(x, y, 1))

        this_experiment.append(
            run_clustering.energy_hartigan(k,X,G,z,init=init,
                                           run_times=run_times))
        this_experiment.append(
            run_clustering.energy_hartigan(k,X,G_half,z,
                                           init=init,run_times=run_times))
        this_experiment.append(
            run_clustering.energy_hartigan(k,X,G_exp,z,
                                           init=init,run_times=run_times))
        this_experiment.append(
            run_clustering.energy_hartigan(k,X,G_rbf,z,
                                           init=init,run_times=run_times))
        this_experiment.append(
            run_clustering.energy_spectral(k,X,G_exp,z,
                                        init=init,run_times=run_times))
        
        this_experiment.append(
            run_clustering.kmeans(k,X,z,init="random",run_times=run_times))
        this_experiment.append(
            run_clustering.gmm(k,X,z,init="random",run_times=run_times))
        this_experiment.append(energy.metric.accuracy(z, energy.gmm.gmm(k,X)))
        
        table.append(this_experiment)
    
    table = np.array(table)
    for i in range(8):
        print table[:,i].mean(), scipy.stats.sem(table[:,i])
        
def sample_digits(digits, images, labels, num_points): 
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
    return X, z

def mnist(num_experiments=10, digits=[0,1,2], num_points=100, run_times=5):
    
    k = len(digits)
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    images_test, labels_test = train_set
    images, labels = valid_set

    # using training test to compute sigma
    X_train, z_train = sample_digits(digits, images_test, labels_test, 
                                     num_points)
    n, _ = X_train.shape
    sigma = np.sqrt(sum([np.linalg.norm(X_train[i]-X_train[j])**2 
                         for i in range(n) for j in range(n)])/(n**2))
    print sigma
    print
    
    table = []
    init = 'k-means++'
    for i in range(num_experiments):
        this_experiment = []
        
        # now cluster on validation set
        X, z = sample_digits(digits, images, labels, num_points)
        
        G = eclust.kernel_matrix(X, rho_standard)
        G_half = eclust.kernel_matrix(X, rho_half)
        G_exp = eclust.kernel_matrix(X, lambda x,y: rho_exp(x, y, sigma))
        G_rbf = eclust.kernel_matrix(X, lambda x,y: rho_rbf(x, y, sigma))
        
        this_experiment.append(
            run_clustering.energy_hartigan(k,X,G,z,init=init,
                                           run_times=run_times))
        this_experiment.append(
            run_clustering.energy_hartigan(k,X,G_half,z,
                                           init=init,run_times=run_times))
        this_experiment.append(
            run_clustering.energy_hartigan(k,X,G_exp,z,
                                           init=init,run_times=run_times))
        this_experiment.append(
            run_clustering.energy_hartigan(k,X,G_rbf,z,
                                           init=init,run_times=run_times))
        this_experiment.append(
            run_clustering.energy_spectral(k,X,G_rbf,z,
                                        init=init,run_times=run_times))
        
        this_experiment.append(
            run_clustering.kmeans(k,X,z,init="k-means++",run_times=run_times))
        this_experiment.append(
            run_clustering.gmm(k,X,z,init="kmeans",run_times=run_times))
        # my gmm was breaking for some unknown reason
        #this_experiment.append(energy.metric.accuracy(z, energy.gmm.gmm(k,X)))

        table.append(this_experiment)

    table = np.array(table)
    for i in range(table[0,:].shape[0]):
        print table[:,i].mean(), scipy.stats.sem(table[:,i])


###############################################################################
if __name__ == "__main__":
    #cigars_circles(num_experiments=10, run_times=5, kind='cigars')
    #cigars_circles(num_experiments=10, run_times=5, kind='2circles')
    #cigars_circles(num_experiments=10, run_times=5, kind='3circles')
    cigars_circles(num_experiments=2, run_times=1, kind='3circles')
    #mnist(num_experiments=10,digits=[0,1,2],num_points=100,run_times=5)
    #mnist(num_experiments=10,digits=[0,1,2,3,4], num_points=100,run_times=5)
    #mnist(num_experiments=10,digits=[0,1,2,3,4,5,6],num_points=100,
    #      run_times=5)
    #mnist(num_experiments=10,digits=[0,1,2,3,4,5,6,7,8],num_points=100,
    #      run_times=5)
    
