#!/usr/bin/env python

"""Extract shapes from digits."""

import numpy as np
from skimage import measure
from scipy.interpolate import splprep, splev


def get_contours(X, v):
    """Get contours of image X (matrix). v is the value of an intensity."""
    return measure.find_contours(X, v)

def interpolate(X, n, s=0.0):
    """Given a 2D array X, where each row represents a 2D vector, interpolate
    between these points using B-spline. Return x and y coords with n points 
    each.

    """
    tck, u = splprep(X.T, u=None, s=s, per=1) 
    u_new = np.linspace(u.min(), u.max(), n)
    x_new, y_new = splev(u_new, tck, der=0)
    return x_new, y_new

def get_shape(X, n=50, s=5):
    cs = get_contours(X, X[np.where( (X > 0.1) & (X < 0.6) )].mean())
    c = sorted(cs, key=len, reverse=True)[0]
    R = np.array([[0, -1], [1, 0]])
    c = c.dot(R) + np.array([[0, 28]])
    x, y = interpolate(c, n=n, s=s)
    return np.array([[x[i], y[i]] for i in range(len(x))])
    


###############################################################################
if __name__ == '__main__':

    import cPickle, gzip, sys
    import matplotlib.pyplot as plt

    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    images, labels = train_set

    idx = np.where(labels==4)
    im1 = images[idx][0].reshape((28,28))
    im2 = images[idx][1].reshape((28,28))

    X = get_shape(im1)
    Y = get_shape(im2)
    
    n = 15 
    images = images[np.random.randint(0, len(images), n)].reshape((n, 28, 28))

    fig = plt.figure(figsize=(15, 3))
    i = 1
    for X in images:
        
        ax = fig.add_subplot(3, n, i)
        ax.imshow(X, cmap=plt.cm.gray)
        ax.axis('off')
    
        # get contours of X
        cs = get_contours(X, X[np.where( (X > 0.1) & (X < 0.6) )].mean())
    
        ax = fig.add_subplot(3, n, n+i)
        for contour in cs:
            R = np.array([[0, -1], [1, 0]])
            contour = contour.dot(R) + np.array([[0, 28]])
            ax.plot(contour[:,0], contour[:,1], 'o', ms=3)
        ax.axis('off')
        ax.set_xlim([0, 28])
        ax.set_ylim([0, 28])

        # pick only the largest contour and fix orientation
        c = sorted(cs, key=len, reverse=True)[0]
        R = np.array([[0, -1], [1, 0]])
        c = c.dot(R) + np.array([[0, 28]])
    
        # interpolate largest contour with n points
        x, y = interpolate(c, n=100, s=5)

        ax = fig.add_subplot(3, n, 2*n+i)
        ax.plot(x, y, '-')
        ax.set_xlim([0, 28])
        ax.set_ylim([0, 28])
        ax.axis('off')

        i += 1
    
    fig.savefig('digits_shape.png')


    
