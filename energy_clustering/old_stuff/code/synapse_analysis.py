"""Synapse data analsys. Second dataset with labelled data.
We have k=2 here.

"""

# author: Guilherme S. Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata


from __future__ import division

import numpy as np
import pandas as pd
from sets import Set

import run_clustering
import energy.eclust as eclust
from energy.metric import accuracy
from energy.data import shuffle_data
    


def get_data(fname="./synapse_data2/rorb_gaussianAvg_at_GoogleDoc.csv"):
    df = pd.read_csv(fname, dtype=float)
    return df.values

def get_labels(fname="./synapse_data2/gaba_labels.csv"):
    df = pd.read_csv(fname, dtype=int)
    return df.values.flatten()

def type_errors(z, zh):
    n1 = len(Set(np.where(zh==0)[0]).intersection(Set(np.where(z==1)[0])))
    n2 = len(Set(np.where(zh==1)[0]).intersection(Set(np.where(z==0)[0])))
    return n1, n2

if __name__ == "__main__":

    data = get_data()
    labels = get_labels()
    print "# Class 0:", len(labels[np.where(labels==0)])
    print "# Class 1:", len(labels[np.where(labels==1)])
    print

    n0 = 500
    n1 = 500
    data_class0 = data[np.where(labels==0)]
    data_class1 = data[np.where(labels==1)]
    idx0 = np.random.choice(range(len(data_class0)), n0, replace=True)
    idx1 = np.random.choice(range(len(data_class1)), n1, replace=True)
    data, labels = shuffle_data([data_class0[idx0], data_class1[idx1]])

    #data = (data - data.mean(axis=0))/data.std(axis=0)

    rho = lambda x, y: np.power(np.linalg.norm(x-y), 1)
    #rho = lambda x, y: 2-2*np.exp(-np.power(np.linalg.norm(x-y),1)/(2*1**2))
    G = eclust.kernel_matrix(data, rho)
    
    labels_hat = run_clustering.kmeans(2, data)
    print accuracy(labels, labels_hat)
    print type_errors(labels, labels_hat)
    print
    
    labels_hat = run_clustering.gmm(2, data)
    print accuracy(labels, labels_hat)
    print type_errors(labels, labels_hat)
    print
    
    labels_hat = run_clustering.energy_hartigan(2, data, G, run_times=5,
                                                    init="gmm")
    print accuracy(labels, labels_hat)
    print type_errors(labels, labels_hat)
    print


