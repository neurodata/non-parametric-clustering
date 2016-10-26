#!/usr/bin/env python

"""Extract shapes from digits."""

from __future__ import division

import numpy as np
from skimage import measure
from scipy.interpolate import splprep, splev

import matplotlib.pyplot as plt

import cv2


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

def get_shape2(X, n=50, s=5):
    """Given an image X (2D matrix), extract contours consisting of
    n points. s controls the smoothness of the contour, where s=0
    is a sharp interpolation and higher s makes it smoother.
    
    """
    v = X.mean() # use mean value of all entries
    cs = get_contours(X, v)
    if len(cs) == 0:
        raise ValueError('Unable to extract contour.')
    
    # now we get all contours that are inside the largest one
    cs = sorted(cs, key=len, reverse=True)
    out_x, out_y = cs[0][:,0], cs[0][:,1]
    min_x, max_x = out_x.min(), out_x.max()
    min_y, max_y = out_y.min(), out_y.max()
    R = np.array([[0, -1], [1, 0]])
    new_cs = [cs[0].dot(R) + np.array([[0, 28]])]
    for c in cs[1:]:
        mu_x, mu_y = c[:,0].mean(), c[:,1].mean()
        if mu_x >= min_x and mu_x <= max_x and \
           mu_y >= min_y and mu_y <= max_y and \
           len(c) > 10:
            c = c.dot(R) + np.array([[0, 28]])
            new_cs.append(c)
    shapes = []
    for c in new_cs:
        x, y = interpolate(c, n, s=s)
        shape = [[x[i], y[i]] for i in range(len(x))]
        shapes.append(shape)
    return np.array(shapes)
    
def shape_to_array(contour):
    x = np.round(10*contour[:,0]).astype(int)
    y = np.round(10*contour[:,1]).astype(int)
    max_x, max_y = x.max(), y.max()
    im = np.zeros((max_y + 1, max_x + 1))
    for i in range(contour.shape[0]):
        im[max_y - y[i], int(x[i])] = 1
    return im

def fillstuff(im_in):
 
    # Threshold.
    # Set values equal to or above 220 to 0.
    # Set values below 220 to 255.
    #th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV)
    th, im_th = cv2.threshold(im_in, 0, 1, cv2.THRESH_BINARY_INV)
   
    # Copy the thresholded image.
    im_floodfill = im_th.copy()
    
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
     
    # Floodfill from point (0, 0)
    #cv2.floodFill(im_floodfill, mask, (0,0), 255)
    cv2.floodFill(im_floodfill, mask, (0,0), 1)
    im_floodfill = im_floodfill.astype(int)
    im_th = im_th.astype(int)
    
    # Invert floodfilled image
    #im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_floodfill_inv = np.invert(im_floodfill)
    
    # Combine the two images to get the foreground.
    #im_out = im_th | im_floodfill_inv
    im_out = np.bitwise_or(im_th, im_floodfill_inv)
        
    return im_out

###############################################################################
if __name__ == '__main__':

    import cPickle, gzip, sys
    import matplotlib.pyplot as plt

    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    images, labels = train_set

    """
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
    
    """
    idx = np.where(labels==8)
    im1 = images[idx][0].reshape((28,28))
    im2 = images[idx][1].reshape((28,28))

    X = get_shape(im1)
    Y = get_shape(im2)

    n = 12 
    images = images[np.random.randint(0, len(images), n)].reshape((n, 28, 28))

    fig = plt.figure(figsize=(40, 15))
    i = 1
    for X in images:
        
        ax = fig.add_subplot(3, n, i)
        ax.imshow(X, cmap=plt.cm.gray)
        ax.axis('off')
    
        # get contours of X
        cs = get_contours(X, X.mean())
    
        ax = fig.add_subplot(3, n, n+i)
        for contour in cs:
            R = np.array([[0, -1], [1, 0]])
            contour = contour.dot(R) + np.array([[0, 28]])
            ax.plot(contour[:,0], contour[:,1], 'o', ms=6)
        ax.axis('off')
        ax.set_xlim([0, 28])
        ax.set_ylim([0, 28])

        # pick only the largest contour and fix orientation
        c = sorted(cs, key=len, reverse=True)[0]
        R = np.array([[0, -1], [1, 0]])
        c = c.dot(R) + np.array([[0, 28]])
    
        # interpolate largest contour with n points
        x, y = interpolate(c, n=30, s=5)

        ax = fig.add_subplot(3, n, 2*n+i)
        ax.plot(x, y, '-', lw=2)
        ax.set_xlim([0, 28])
        ax.set_ylim([0, 28])
        ax.axis('off')

        i += 1
    
    fig.savefig('figs/digits_shape.pdf')

    """

    
    ###########################################################################
    # Keeping multiple contours
    ###########################################################################
    n = 12 
    images = images[np.random.randint(0, len(images), n)].reshape((n, 28, 28))

    fig = plt.figure(figsize=(40, 15))
    i = 1
    for X in images:
        
        ax = fig.add_subplot(3, n, i)
        ax.imshow(X, cmap=plt.cm.gray)
        ax.axis('off')
    
        # get contours of X
        cs = get_contours(X, X.mean())
        ax = fig.add_subplot(3, n, n+i)
        for contour in cs:
            R = np.array([[0, -1], [1, 0]])
            contour = contour.dot(R) + np.array([[0, 28]])
            ax.plot(contour[:,0], contour[:,1], 'o', ms=6)
        ax.axis('off')
        ax.set_xlim([0, 28])
        ax.set_ylim([0, 28])

        # get the shape
        ss = get_shape2(X, n=30, s=5)
        ax = fig.add_subplot(3, n, 2*n+i)
        for s in ss:
            x, y = s[:,0], s[:,1]
            ax.plot(x, y, '-', lw=2)
        ax.set_xlim([0, 28])
        ax.set_ylim([0, 28])
        ax.axis('off')

        i += 1
    
    fig.savefig('figs/digits_multi_shapes.pdf')

