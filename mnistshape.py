#!/usr/bin/env python

"""Extract shapes from digits."""

import numpy as np
from skimage import measure
from scipy.interpolate import splprep, splev

import matplotlib.pyplot as plt


def get_contours(X, v):
    """Get contours of image X (2D matrix). 
    v is the value of an intensity threshold.
    
    """
    return measure.find_contours(X, v)

def interpolate(X, n, s=2.0):
    """Given a 2D array X, where each row represents a 2D vector, interpolate
    between these points using B-spline. 
    Return x and y coords with n points each.

    """
    tck, u = splprep(X.T, u=None, s=s, per=0) 
    u_new = np.linspace(u.min(), u.max(), n)
    x_new, y_new = splev(u_new, tck, der=0)
    return x_new, y_new

def get_shape(X, n=50, s=5):
    """Given an image X (2D matrix), extract a contour consisting of
    n points. s controls the smoothness of the contour, where s=0
    is a sharp interpolation and higher s makes it smoother.
    
    """
    v = X.mean() # use mean value of all entries
    cs = get_contours(X, v)
    if len(cs) == 0:
        raise ValueError('Unable to extract contour.')
    # get only outside contour
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

    # these were problematic images, where extracting the contour
    # was giving errors
    bad = [
    49268,
    13650,
    46104,
    33522
    ]
    #X = images[bad[0]].reshape((28,28))
    X = images[np.random.randint(0,len(images))].reshape((28,28))
    
    # using just the mean instead of picking a range solved the problem
    # with bad images
    v = X.mean()
    #v = X[np.where((X > 0) & (X < 1))].mean()
    
    cs = get_contours(X, v)
    shape = get_shape(X)
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121)
    ax.imshow(X)
    for c in cs:
        ax.plot(c[:,1], c[:,0], '-o', color='y', ms=5)
    ax = fig.add_subplot(122)
    ax.plot(shape[:,0], shape[:,1], '-o', color='k', ms=5)
    plt.show()
    

    
    """
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
    """
    
