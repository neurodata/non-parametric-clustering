
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
    first = outside[0]
    outside[-1] = first
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
    
def shape_to_image(points, scale=10, N=None):
    """Given a set of points, create a binary matrix image.
    scale controls how large the final image should be compared to
    the coordinate points, and N creates a larger matrix.

    """
    x = np.round(scale*points[:,0]).astype(int)
    y = np.round(scale*points[:,1]).astype(int)
    max_x, max_y = x.max(), y.max()
    min_x, min_y = x.min(), y.min()
    im = np.zeros((max_x-min_x+4, max_y-min_y+4))
    for i in range(points.shape[0]):
        im[x[i]-min_x+2, y[i]-min_y+2] = 1
    return im.astype(int)

def im_ones(im, v):
    n, k = im.shape
    bin_im = np.zeros((n,k))
    bin_im[np.where(im > v)] = 1
    return bin_im

def test(fname):
    img = cv2.imread(fname, 0)
    cv2.bilateralFilter(img, 9, 90,16)
    img = cv2.GaussianBlur(img,(5,5),0)
    #binImg = np.zeros((img.shape[0], img.shape[1]), np.uint8)   
    binImg = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 55, -3)
    #cv2.bilateralFilter(binImg, 9, 90,16)
    #binImg = cv2.GaussianBlur(binImg, (3,3), 0)
    #ret, binImg = cv2.threshold(img, 35000, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    plt.imshow(binImg, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


if __name__ == '__main__':
    test('data/orig_img.png')
