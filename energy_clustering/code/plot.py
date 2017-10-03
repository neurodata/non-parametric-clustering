"""Plot accuracy results."""

# Guilhere Franca <guifranca@gmail.com>
# Johns Hopkins University, Neurodata

from __future__ import division

import numpy as np
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt

from customize_plots import *


class ErrorBar:

    def __init__(self):
        self.xlabel = 'dimension'
        self.ylabel = 'accuracy'
        self.symbols = ['o', 's', 'D', 'v']
        self.legends = ['energy', r'$k$-means', 'GMM', r'kernel $k$-means']
        self.colors = ['b', 'r', 'g', 'c']
        self.lines = ['-', '-', '-', '-', '-', '-']
        self.xlim = None
        self.ylim = None
        self.bayes = None
        self.loc = 0
        self.doublex = False
        self.legcols = 1
        self.output = 'plot.pdf'

    def _set_params(self, table):
        self.col0 = col0 = np.array([int(x) for x in table[:,0]])
        self.xs = np.unique(col0)
        
        self.n = len(table[0][1:])
        #self.colors = iter(plt.cm.brg(np.linspace(0,1,self.n+1)))
        self.colors = iter(self.colors)
        self.legends = iter(self.legends)
        self.symbols = iter(self.symbols)
        self.lines = iter(self.lines)
    
        if self.doublex:
            self.xs2 = 2*self.xs
        else:
            self.xs2 = self.xs
            
        if self.bayes is not None:
            if not isinstance(self.bayes, list):
                self.bayes = [self.bayes]*len(self.xs)

    def make_plot(self, table):
        self._set_params(table)

        fig = plt.figure()
        ax = fig.add_subplot(111)
    
        for i in range(1, self.n+1):
            
            c = next(self.colors)
            l = next(self.legends)
            m = next(self.symbols)
            t = next(self.lines)

            mean_ = np.array([table[np.where(self.col0==x)[0],i].mean() 
                              for x in self.xs])
            stderr_ = np.array([scipy.stats.sem(
                table[np.where(self.col0==x)[0],i]) for x in self.xs])
            if self.bayes:
                ax.plot(self.xs2, self.bayes, '--', color='black', 
                        linewidth=1, zorder=0)
            ax.errorbar(self.xs2, mean_, yerr=stderr_, 
                        linestyle=t, marker=m, color=c, markersize=4, 
                        elinewidth=.5,  capthick=0.4, label=l, 
                        linewidth=1, barsabove=False)
        
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        if self.ylim:
            ax.set_ylim(self.ylim)
        if self.xlim:
            ax.set_xlim(self.xlim)
        leg = plt.legend()
        ax.legend(loc=self.loc, framealpha=.5, ncol=self.legcols)
        
        fig.savefig(self.output, bbox_inches='tight')

