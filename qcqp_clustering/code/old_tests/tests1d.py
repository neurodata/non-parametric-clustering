
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GMM

from eclust.kkmeans import KernelKMeans
from eclust.metric import accuracy
import eclust.data as data
from eclust.withingraph import EClust
from eclust.energy1d import two_clusters1D
from eclust.energy import energy_kernel
from eclust.objectives import kernel_score
from eclust.energy1d import within_sample
import eclust.data as data


def kernel_energy(k, X, alpha=1, cutoff=0, num_times=10):
    best_score = 0
    for i in range(num_times):
        km = KernelKMeans(n_clusters=k, max_iter=300, kernel=energy_kernel,
                      kernel_params={'alpha':alpha, 'cutoff':cutoff}) 
        zh = km.fit_predict(Xv)
        score = kernel_score(km.kernel_matrix_, zh)
        if score > best_score:
            best_score = score
            best_z = zh
    return best_z

def within_energy(X, z, num_times=10):
    best_score = np.inf
    for i in range(num_times):
        ec = EClust(n_clusters=2, max_iter=100)
        zh = ec.fit_predict(X)
        score = within_sample(data.from_label_to_sets(X, zh))
        if score < best_score:
            best_score = score
            best_z = zh
    return best_z
    

###############################################################################
if __name__ == '__main__':
    
    m1 = 0
    s1 = 0.3
    n1 = 100
    
    m2 = 1
    s2 = 2
    n2 = 100
    
    X, z = data.univariate_normal([m1, m2], [s1, s2], [n1, n2])
    #X, z = data.univariate_normal([m1, m2], [s1, s2], [n1, n2])
    Xv = np.array([[x] for x in X])
    k = 2
    
    data.hist(Xv, z, 'hist1d.pdf')
    
    ### clustering with different algorithms ###
    
    zh = kernel_energy(k, X, alpha=.4, cutoff=0., num_times=10)
    print "Kernel Energy:", accuracy(z, zh)
    #print accuracy(z, zh)

    zh, score = two_clusters1D(X)
    print "Energy 1D:", accuracy(z, zh)
    #print accuracy(z, zh)
    

    #zh = within_energy(X, z, num_times=15)
    #print "Within Graph Energy:", accuracy(z, zh)
    #print accuracy(z, zh)
    
    gmm = GMM(n_components=k)
    gmm.fit(X)
    zh = gmm.predict(X)
    print "GMM:", accuracy(z, zh)
    #print accuracy(z, zh)

    km = KMeans(n_clusters=k)
    zh = km.fit_predict(np.array([[x] for x in X]))
    print "k-means:", accuracy(z, zh)
    #print accuracy(z, zh)

