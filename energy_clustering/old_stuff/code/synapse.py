
import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt

#from customize_plots import *
#matplotlib.rcParams.update({'font.size':4, 'axes.labelsize':16})


def scatter_positions():
    positions_file = 'synapse_data/locations_k15F0_20170616.csv'
    positions = pd.read_csv(positions_file)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions['xm1'], positions['ym1'], positions['zm1'], c='b')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')
    fig.savefig('scatter_original.pdf')



################################################################################
if __name__ == "__main__":
    scatter_positions()
