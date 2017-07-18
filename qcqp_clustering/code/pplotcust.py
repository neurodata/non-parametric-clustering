
import pylab
import numpy as np

# pyplot customization
pylab.rc('lines', linewidth=.5, antialiased=True, markersize=4,
            markeredgewidth=0.00)
pylab.rc('font', family='computer modern roman', style='normal',
         weight='normal', serif='computer modern sans serif', size=10)
pylab.rc('text', usetex=True)
pylab.rc('text.latex', preamble=[
        '\usepackage{amsmath,amsfonts,amssymb,relsize,cancel}'])
pylab.rc('axes', linewidth=0.5, labelsize=14)
pylab.rc('xtick', labelsize=9)
pylab.rc('ytick', labelsize=9)
#pylab.rc('legend', numpoints=1, fontsize=10, handlelength=0.5)
pylab.rc('legend', numpoints=1, fontsize=10)
fig_width_pt = 455.0 / 1.5 # take this from LaTeX \textwidth in points
#fig_width_pt = 430.0 / 3 # take this from LaTeX \textwidth in points
inches_per_pt = 1.0/72.27
golden_mean = (np.sqrt(5.0)-1.0)/2.0
fig_width = fig_width_pt*inches_per_pt
#fig_height = fig_width
fig_height = fig_width*golden_mean
#fig_height = fig_width
pylab.rc('figure', figsize=(fig_width, fig_height))


