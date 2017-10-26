
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import energy.data as data

sns.set_style({"font.size": 16, "axes.labelsize": 30})

#d = 10
#D = 50
#N = 200
#delta = 0.7
#m1 = np.zeros(D)
#s1 = np.eye(D)
#m2 = np.concatenate((delta*np.ones(d), np.zeros(D-d)))
#s2 = np.eye(D)
#n1, n2 = np.random.multinomial(N, [0.5,0.5])
#X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])

d = 10
D = 200
N = 200
m1 = np.zeros(D)
m2 = np.concatenate((np.ones(d), np.zeros(D-d)))
s1 = np.eye(D)
s2_1 = np.array([1.367,  3.175,  3.247,  4.403,  1.249,                                             1.969, 4.035,   4.237,  2.813,  3.637])
s2 = np.diag(np.concatenate((s2_1, np.ones(D-d))))
n1, n2 = np.random.multinomial(N, [0.5,0.5])
X, z = data.multivariate_normal([m1, m2], [s1, s2], [n1, n2])

numcols=5
Y = np.zeros(shape=(N,numcols+1))
Y[:,:numcols] = X[:,:numcols]
idx0 = np.where(z==0)
idx1 = np.where(z==1)
Y[idx0,numcols] = 0
Y[idx1,numcols] = 1
df = pd.DataFrame(Y, 
        columns=[r"$x_%i$"%i for i in range(1, numcols+1)]+["class"])

g = sns.PairGrid(df, hue="class", #palette="hls",
                    vars=[r"$x_%i$"%i for i in range(1, numcols+1)])

def scatter_fake_diag(x, y, *a, **kw):
    if x.equals(y):
        kw["color"] = (0, 0, 0, 0)
    plt.scatter(x, y, *a, **kw)

g.map(scatter_fake_diag)
g.map_diag(plt.hist)

#g.map_offdiag(plt.scatter)
#g.map_diag(plt.hist)

#g.savefig("pairsplot1.pdf")
g.savefig("pairsplot2.pdf")
