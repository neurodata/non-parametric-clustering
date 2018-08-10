#!/bin/bash

#colors = iter(["#54278f", "#5F8793", "#F41711"])

# first plot, 1d data
python plot_gauss_loggauss_1d.py -o gauss_1d.pdf -i data/gauss_1d_*
python plot_gauss_loggauss_1d.py -o loggauss_1d.pdf -i data/logggauss_1d_*

# fig 3
python plot_gauss1.py -o gauss1.pdf -i data/gauss1_*
python plot_gauss2.py -o gauss2.pdf -i data/gauss2_*
python plot_gauss_loggaus_n.py -o gauss_n.pdf -i data/gauss_n_*
python plot_gauss_loggaus_n.py -o loggauss_n.pdf -i data/loggauss_n_*
