# author: Guilherme S. Franca
# Johns Hopkins University

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import sys

parser = argparse.ArgumentParser(description="Plot the high dimensional"\
                    "Gaussian experiment which changes in the covariance.")
parser.add_argument('-o', dest='output', required=True,
                    type=str, action='store', help="file name for output")
parser.add_argument('-i', dest='input', required=True,
                    type=str, action='store', nargs='*', 
                    help="file name for output")

args = parser.parse_args()

fnames = args.input
output = args.output

df = pd.concat(pd.read_csv(f) for f in fnames)

methods = ['k-means', 'gmm', 'spectral clustering', 'kernel k-means', 
           'kernel k-groups']
markers = iter(['^', 'v', 'p', 's', 'o'])
colors = iter(["#54278f", "#5F8793", "#99B4C6", "#318dde", "#F41711"])

dimensions = np.unique(df['dimension'].values)

fig = plt.figure()
ax = fig.add_subplot(111)
for method in methods:
    r = []
    for d in dimensions:
        df2 = df[(df['method']==method) & (df['dimension']==d)]
        r.append([d, df2['accuracy'].mean(), df2['accuracy'].sem()])
    r = np.array(r)
    ax.plot(r[:,0], [0.95]*len(r[:,0]), '--', linewidth=1, color='k')
    mk = next(markers)
    cl = next(colors)
    #ax.errorbar(r[:,0], r[:,1], yerr=r[:,2], marker=mk, elinewidth=1,
    #            label=method)
    ax.errorbar(r[:,0], r[:,1], marker=mk, color=cl, label=method)
ax.set_xlabel(r'$D$')
ax.set_ylabel(r'accuracy')
ax.set_xlim([10,700])
ax.legend()
fig.savefig(output, bbox_inches='tight')

