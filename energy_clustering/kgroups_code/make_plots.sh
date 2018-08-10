#!/bin/bash

#colors = iter(["#54278f", "#5F8793", "#F41711"])

# first plot, 1d data
python plot_gauss_loggauss_1d.py -o gauss_1d.pdf -i data/gauss_1d_*
python plot_gauss_loggauss_1d.py -o loggauss_1d.pdf -i data/logggauss_1d_*

# fig 3

