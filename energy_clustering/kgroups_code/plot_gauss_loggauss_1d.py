# author: Guilherme S. Franca
# Johns Hopkins University

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import sys

parser = argparse.ArgumentParser(description="Plot 1D experiments.")
parser.add_argument('-o', dest='output', required=True,
                    type=str, action='store', help="file name for output")
parser.add_argument('-i', dest='input', required=True,
                    type=str, action='store', nargs='*', 
                    help="file name for output")

args = parser.parse_args()

fnames = args.input
output = args.output

df = pd.concat(pd.read_csv(f) for f in fnames)

methods = ['k-means', 'gmm', 'kernel k-groups']
markers = iter(['^', 'v', 'o'])

num_points = np.unique(df['num_points'].values)

fig = plt.figure()
ax = fig.add_subplot(111)
for method in methods:
    r = []
    for n in num_points:
        df2 = df[(df['method']==method) & (df['num_points']==n)]
        r.append([n, df2['accuracy'].mean(), df2['accuracy'].sem()])
    r = np.array(r)
    ax.plot(r[:,0], [0.88]*len(r[:,0]), '--', linewidth=1, color='k', alpha=0.5)
    mk = next(markers)
    ax.errorbar(r[:,0], r[:,1], yerr=r[:,2], marker=mk, elinewidth=1,
                label=method)
ax.set_xlabel(r'$n$')
ax.set_ylabel(r'accuracy')
ax.set_xlim([10,800])
ax.legend()
fig.savefig(output, bbox_inches='tight')

