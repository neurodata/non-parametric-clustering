from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import pylab
from PIL import Image
from sklearn import datasets

digits = datasets.load_digits()
images = digits.images

X = images[0]

fig = plt.figure()
plt.imshow(X)
plt.show()
