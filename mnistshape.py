#!/usr/bin/env python

"""Extract shapes from digits."""

from __future__ import division

import numpy as np
from skimage import measure
from scipy.interpolate import splprep, splev
from scipy.ndimage.interpolation import zoom

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

def get_shape2(X, n=50, s=5, ir=2):
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
        #mu_x, mu_y = c[:,0].mean(), c[:,1].mean()
        x1, x2 = c[:,0].min(), c[:,0].max()
        y1, y2 = c[:,1].min(), c[:,1].max()
        #if mu_x >= min_x and mu_x <= max_x and \
        #   mu_y >= min_y and mu_y <= max_y and \
        #   len(c) > 10:
        if len(c) > 10 and \
                x1 >= min_x and x2 <= max_x and \
                y1 >= min_y and y2 <= max_y:
            c = c.dot(R) + np.array([[0, 28]])
            new_cs.append(c)
    shapes = []
    for count, c in enumerate(new_cs):
        if count != 0:
            x, y = interpolate(c, int(n/ir), s=s)
        else:
            x, y = interpolate(c, n, s=s)
        shape = np.array([[x[i], y[i]] for i in range(len(x))])
        shapes.append(shape)
    return np.array(shapes)
    
def shape_to_image(shapes, scale=10):
    """Assume that shapes is a list of shapes, 
    ordered by the outside most contour.
    
    """
    larger = shapes[0]
    x = np.round(scale*larger[:,0]).astype(int)
    y = np.round(scale*larger[:,1]).astype(int)
    max_x, max_y = max(x.max(), scale*27), max(y.max(), scale*27)
    im = np.zeros((max_y + 1, max_x + 1))
    for i in range(larger.shape[0]):
        im[max_y - y[i], x[i]] = 1
    for shape in shapes[1:]:
        x = np.round(scale*shape[:,0]).astype(int)
        y = np.round(scale*shape[:,1]).astype(int)
        for i in range(shape.shape[0]):
            im[max_y - y[i], x[i]] = 1
    return im


def fill(data, start_coords, fill_value):
    """
    Flood fill algorithm
    
    Parameters
    ----------
    data : (M, N) ndarray of uint8 type
        Image with flood to be filled. Modified inplace.
    start_coords : tuple
        Length-2 tuple of ints defining (row, col) start coordinates.
    fill_value : int
        Value the flooded area will take after the fill.
        
    Returns
    -------
    None, ``data`` is modified inplace.
    """
    xsize, ysize = data.shape
    orig_value = data[start_coords[0], start_coords[1]]
    
    stack = set(((start_coords[0], start_coords[1]),))
    if fill_value == orig_value:
        raise ValueError("Filling region with same value "
                     "already present is unsupported. "
                     "Did you already fill this region?")

    while stack:
        x, y = stack.pop()

        if data[x, y] == orig_value:
            data[x, y] = fill_value
            if x > 0:
                stack.add((x - 1, y))
            if x < (xsize - 1):
                stack.add((x + 1, y))
            if y > 0:
                stack.add((x, y - 1))
            if y < (ysize - 1):
                stack.add((x, y + 1))


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
    """

    """
    ###########################################################################
    # Writing shapes to an image
    ###########################################################################

    ls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    fig, axes = plt.subplots(nrows=len(ls), ncols=3, figsize=(2.5*3,2.5*10))
    for l, row in zip(ls, axes):
        ax1, ax2, ax3 = row
        idx = np.where(labels==l)[0]
        im = images[np.random.choice(idx)].reshape((28,28))
        
        ax1.imshow(im, cmap=plt.cm.gray)
        ax1.axis('off')

        shapes = get_shape2(im, 200, s=5.0, ir=2)
        for c in shapes:
            ax2.plot(c[:,0], c[:,1], 'o-')
        ax2.set_aspect('equal')
        ax2.set_xlim([0,30])
        ax2.set_ylim([0,30])
        
        im2 = shape_to_image(shapes, scale=2)
        ax3.imshow(im2, cmap=plt.cm.gray)
        ax3.axis('off')
    
    fig.savefig('figs/im_shape_im.pdf')
    """
    
    idx = np.where(labels==0)[0]
    im = images[np.random.choice(idx)].reshape((28,28))
    shapes = get_shape2(im, 200, s=5.0, ir=2)
    im2 = shape_to_image(shapes, scale=2)

    im3 = fill(im2, [10, 10], 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(im, cmap=plt.cm.gray)
    ax2 = fig.add_subplot(122)
    print im3
    ax2.imshow(im3, cmap=plt.cm.gray)
    
    plt.show()
