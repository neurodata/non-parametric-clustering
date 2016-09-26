#!/usr/bin/env python

"""
Testing different methods to extract contour of images.

"""

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

from skimage import measure, feature
from scipy import ndimage as ndi


def zeroone(X, val):
    """Put all elements bigger than val to 1, otherwise 0."""
    n, m = X.shape
    idx = np.where(X >= val)
    X = np.zeros((n, m))
    X[idx] = 1
    return X

def plot_image_shape(X, ax, contours=[], title=''):
    """Plot image X together with its contour."""
    ax.imshow(X, cmap=plt.cm.gray)
    if contours:
        for n, contour in enumerate(contours):
            ax.plot(contour[:,1], contour[:,0], 'o', ms=4)
    ax.set_title(title) 
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])

def plot_image(X, ax, title=''):
    ax.imshow(X, cmap=plt.cm.gray)
    ax.set_title(title) 
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])

def plot_contours(contours, ax, title=''):
    # need to reflect the points because the origin was on top left
    refl = np.array([[-1, 0], [0, 1]])
    for c in contours:
        c = c.dot(refl) + np.array([[28, 0]])
        ax.plot(c[:,1], c[:,0], '-o', ms=4)
    ax.set_title(title) 
    ax.set_xlim([0,28])
    ax.set_ylim([0, 28])

def square():
    """Draw a square."""
    im = np.zeros((128, 128))
    im[32:-32, 32:-32] = 1
    im = ndi.rotate(im, 15, mode='constant')
    im = ndi.gaussian_filter(im, 4)
    im += 0.2 * np.random.random(im.shape)
    return im

def digit(n):
    """Get digit n from MNIST dataset."""
    digits = datasets.load_digits()
    images = digits.images
    return images[n]

def range_values(X):
    """Range of non-zero values of matrix X."""
    vals = X[np.where(X > 0)]
    return vals.min(), vals.max()

def contours_zeroone(X, v):
    """Apply zero-one transformation then get contour."""
    Y = zeroone(X, v)
    return measure.find_contours(Y, 0.3)

def contours_raw(X, v):
    """Get contours directly."""
    return measure.find_contours(X, v)

def canny(X, sigma=1):
    """Use canny filter"""
    return feature.canny(X, sigma)

def get_contours(X, v):
    contours = measure.find_contours(X, v)
    return contours

    

###############################################################################
if __name__ == '__main__':

    import cPickle, gzip, sys

    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    n = 10
    images, labels = train_set
    images = images[np.random.randint(0, len(images), n)].reshape((n, 28, 28))

    fig = plt.figure(figsize=(40,8))
    i = 1
    for X in images:
        ax = fig.add_subplot(2, n, i)
        v = X[np.where( (X > 0.1) & (X < 0.8) )].mean()
        contours = contours_raw(X, v)
        plot_image_shape(X, ax, contours)
        i += 1
    for X in images:
        ax = fig.add_subplot(2, n, i)
        v = X[np.where( (X > 0.1) & (X < 0.8) )].mean()
        contours = contours_raw(X, v)
        plot_contours(contours, ax)
        i += 1
    fig.savefig('digits_contour.png')
    
