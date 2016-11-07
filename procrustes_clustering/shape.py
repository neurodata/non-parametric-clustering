
"""Functions to deal with shapes and extract contours from images, etc."""

from __future__ import division

import numpy as np
from skimage import measure
from scipy.interpolate import splprep, splev
from scipy.ndimage.interpolation import zoom

import matplotlib.pyplot as plt

import cv2


def get_contours(im_array, intensity):
    return measure.find_contours(im_array, intensity)

def interpolate(array2Dpoints, numpoints, smooth=2.0):
    tck, u = splprep(array2Dpoints.T, u=None, s=smooth, per=0) 
    u_new = np.linspace(u.min(), u.max(), numpoints)
    x_new, y_new = splev(u_new, tck, der=0)
    return np.array([x_new, y_new]).T

def get_external_contour(im_array, numpoints=50, smooth=5,
            rotate=np.array([[0,-1],[1,0]]), translate=np.array([[0,28]])):
    """Get numpoints from the outside most contour."""
    contours = get_contours(im_array, im_array.mean())
    outside = sorted(contours, key=len, reverse=True)[0]
    outside = outside.dot(rotate) + translate 
    return interpolate(outside, numpoints=numpoints, smooth=smooth)

def get_all_contours(im_array, numpoints=50, smooth=5, numpoints_internal=20, 
            neglect=10, rotate=np.array([[0,-1],[1,0]]), 
            translate=np.array([[0,28]])):
    """Get all contours from image."""
    contours = get_contours(im_array, im_array.mean())
    contours = sorted(contours, key=len, reverse=True)
    
    out_x, out_y = contours[0][:,0], contours[0][:,1]
    min_x, max_x = out_x.min(), out_x.max()
    min_y, max_y = out_y.min(), out_y.max()
    new_contours = [contours[0].dot(rotate) + translate]
    for contour in contours[1:]:
        x1, x2 = contour[:,0].min(), contour[:,0].max()
        y1, y2 = contour[:,1].min(), contour[:,1].max()
        if len(contour) > 10 and \
                x1 >= min_x and x2 <= max_x and \
                y1 >= min_y and y2 <= max_y:
            contour = contour.dot(rotate) + translate
            new_contours.append(contour)
    
    shapes = []
    for i, contour in enumerate(new_contours):
        if i != 0: # internal contour
            X = interpolate(contour, numpoints_internal, smooth)
        else:
            X = interpolate(contour, numpoints, smooth)
        shapes.append(X)
    shapes = np.array(shapes)
    return np.concatenate(shapes)
    
def shape_to_image(X, scale=10):
    x = np.round(scale*X[:,0]).astype(int)
    y = np.round(scale*X[:,1]).astype(int)
    max_x, max_y = x.max(), y.max()
    im = np.zeros((max_y + 1, max_x + 1))
    for i in range(X.shape[0]):
        im[max_y - y[i], x[i]] = 1
    return im

def im_ones(im, v):
    n, k = im.shape
    bin_im = np.zeros((n,k))
    bin_im[np.where(im > v)] = 1
    return bin_im


def test(d):
    import cv2
    import gzip, cPickle
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    images, labels = train_set
    x = np.where(labels==d)[0]
    #im_in = images[np.random.choice(x)].reshape((28,28))
    im_in = images[np.random.choice(x)]
    
    #Read image
    #im_in = cv2.imread("nickel.jpg", cv2.IMREAD_GRAYSCALE);
    
    # Threshold.
    # Set values equal to or above 220 to 0.
    # Set values below 220 to 255.
    
    #v = 220
    v = im_in.mean()
    th, im_th = cv2.threshold(im_in, v, 255, cv2.THRESH_BINARY_INV);
    
    # Copy the thresholded image.
    im_floodfill = im_th.copy()
         
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
           
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
            
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
             
    # Display images.
    cv2.imshow("Thresholded Image", im_th)
    cv2.imshow("Floodfilled Image", im_floodfill)
    cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
    cv2.imshow("Foreground", im_out)
    cv2.waitKey(0)
