import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GMM

from eclust.kernel import KernelEnergy
from eclust.metric import accuracy
import eclust.data as data
from eclust.objectives import kernel_score


def kernel_energy(k, X, alpha=.5, num_times=15):
    best_score = 0
    for i in range(num_times):
        km = KernelEnergy(n_clusters=k, max_iter=300, 
                          kernel_params={'alpha':alpha})
        zh = km.fit_predict(X)
        score = kernel_score(km.kernel_matrix_, zh)
        if score > best_score:
            best_score = score
            best_z = zh
    return best_z

def kmeans(k, X):
    km = KMeans(n_clusters=k)
    zh = km.fit_predict(X)
    return zh

def gmm(k, X):
    gmm = GMM(n_components=k)
    gmm.fit(X)
    zh = gmm.predict(X)
    return zh


###############################################################################
if __name__ == '__main__':
    
    k = 2
    D = 2
    
    m1 = np.zeros(D)
    #s1 = np.eye(D)
    s1 = np.array([[1,0], [0,25]])
    n1 = 200

    m2 = np.zeros(D)
    #m2 = .5*np.ones(D)
    m2[0] = 7
    #m2[np.random.choice(range(0,D), D/2)] = .15*1.
    #s2 = .6*np.eye(D)
    s2 = np.array([[1,0], [0,25]])
    n2 = 200
    
    X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])

    data.plot(X, z, 'cigar.pdf')

    for i in range(10):

        zh = kernel_energy(k, X, 1, 5)
        print accuracy(zh, z)
    
        zh = kmeans(k, X)
        print accuracy(zh, z)
    
        zh = gmm(k, X)
        print accuracy(zh, z)

        print
