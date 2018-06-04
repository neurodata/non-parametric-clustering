# author: Guilherme S. Franca
# Johns Hopkins University

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import sys

parser = argparse.ArgumentParser(description="Plot unbalanced Gaussians.")
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

points = np.unique(df['points'].values)

fig = plt.figure()
ax = fig.add_subplot(111)
for method in methods:
    r = []
    for p in points:
        df2 = df[(df['method']==method) & (df['points']==p)]
        r.append([p, df2['accuracy'].mean(), df2['accuracy'].sem()])
    r = np.array(r)
    mk = next(markers)
    ax.errorbar(r[:,0], r[:,1], yerr=r[:,2], marker=mk, elinewidth=1,
                label=method)
ax.set_xlabel(r'$\#$ unbalanced points')
ax.set_ylabel(r'accuracy')
ax.set_xlim([0,240])
ax.legend()
fig.savefig(output, bbox_inches='tight')

