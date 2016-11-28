
import gzip, cPickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

import shape
import procrustes


def pick_digit(d, i=None):
    """Pick an MNIST digit. If i is not set, it will pick at random."""
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    images, labels = train_set
    x = np.where(labels==d)[0]
    if i == None:
        i = np.random.choice(x)
    return images[i].reshape((28,28))

def fill(im_in):
    """Fill image."""
    v, u = im_in.mean(), 1
    th, im_th = cv2.threshold(im_in, v, u, cv2.THRESH_BINARY_INV)
    im_th = im_th.astype(np.uint8)
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), u)
    im_floodfill_inv = np.invert(im_floodfill)
    im_out = np.bitwise_or(im_th, im_floodfill_inv)
    return im_out

def flood_fill(data, start_coords, fill_value=1):
    """Flood fill algorithm. start_coords is a list of x,y indexes of
    initial seed. fill_value is the value to be filled, and data is
    a 2D array. Have to be sure start_coords is inside contour.
    
    """
    xsize, ysize = data.shape
    orig_value = data[start_coords[0], start_coords[1]]
    stack = set(((start_coords[0], start_coords[1]),))
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

def get_inner_point(im):
    """Find a point inside the contour."""
    xsize, ysize = im.shape
    found = False
    while not found:
        x = np.random.randint(0, xsize)
        y = np.random.randint(0, ysize)
        if im[x,y] == 0:
            count = sum([1 for i in range(xsize - x) if im[x+i,y] == 1])
            if count%2 != 0:
                found = True
    return (x,y)

def img_to_filled(digit):
    """Testing filling an image from the extracted contour of landmark
    points.
    
    """
    
    n = 10 
    numpoints = 200
    
    fig, rows = plt.subplots(nrows=n, ncols=3, figsize=(2*3, 2*n))
    for row in rows:
        
        ax1, ax2, ax3 = row
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        ax3.set_aspect('equal')
        
        im = pick_digit(digit)
        land_points = shape.get_external_contour(im, numpoints, smooth=5,
                rotate=np.eye(2), translate=np.array([[0,0]]))
        im_contour = shape.shape_to_image(land_points, scale=1)
        im_filled = im_contour.copy()
        im_filled = np.invert(im_filled)
        flood_fill(im_filled, (0,0), fill_value=0)
        im_filled[np.where(im_filled != 0)] = 1

        ax1.imshow(im, cmap=plt.cm.gray)
        R = np.array([[0,1],[-1,0]]).dot(np.array([[-1,0],[0,1]]))
        land_points = land_points.dot(R)
        ax1.plot(land_points[:,0], land_points[:,1], 'or', ms=4)
        ax2.imshow(im_contour, cmap=plt.cm.gray)
        ax3.imshow(im_filled, cmap=plt.cm.gray)
    
    plt.tight_layout()
    fig.savefig('figs/image_to_filled_%d_2.pdf'%digit)

def shape_to_filled_image(landmark_points, N=20, scale=80):
    """Given a set of 'landmark_points', we create a binary 
    image with its interior filled. The image matrix will be
    of size NxN. Ideally, N should be larger than the scalled image.

    """
    im_contour = shape.shape_to_image(landmark_points, scale=scale)
    im_filled = im_contour.copy()
    im_filled = np.invert(im_filled)
    flood_fill(im_filled, (0,0), fill_value=0)
    im_filled[np.where(im_filled != 0)] = 1
    # now creating a larger matrix and puting the previous image in the center
    n, k = im_filled.shape
    N = max(N, max(n, k))
    new_im = np.zeros((N, N))
    i = (N-n)//2
    j = (N-k)//2
    new_im[i:i+n,j:j+k] = im_filled
    return new_im

def procrustes_filling_test(im1, im2, fname, numpoints=300, N=20, scale=80):
    """Comparing pure procrustes versus "image filling" distance."""
    fig = plt.figure(figsize=(12, 3))
    
    ax = fig.add_subplot(151)
    ax.imshow(im1, cmap=plt.cm.gray)
    ax.axis('off')
    ax.set_aspect('equal')
    
    ax = fig.add_subplot(152)
    ax.imshow(im2, cmap=plt.cm.gray)
    ax.axis('off')
    ax.set_aspect('equal')

    s1 = shape.get_external_contour(im1, numpoints, smooth=1,
                rotate=np.eye(2), translate=np.array([[0,0]]))
    s2 = shape.get_external_contour(im2, numpoints, smooth=1,
                rotate=np.eye(2), translate=np.array([[0,0]]))
    im2_res, im2_hat, proc_dist = procrustes.procrustes(s1, s2, fullout=True)
    
    ax = fig.add_subplot(153)
    ax.plot(im2_res[:,0], im2_res[:,1], 'or', alpha=.7)
    ax.plot(im2_hat[:,0], im2_hat[:,1], 'ob', alpha=.7)
    ax.set_aspect('equal')
    ax.set_title(r'$d_P=%f$'%proc_dist)
    ax.set_xticks([])
    ax.set_yticks([])
        
    im2_matrix = shape_to_filled_image(im2_res, N=N, scale=scale)
    im2hat_matrix = shape_to_filled_image(im2_hat, N=N, scale=scale)
    fill_dist = np.linalg.norm(im2_matrix - im2hat_matrix)
    fill_dist = fill_dist/N

    ax = fig.add_subplot(154)
    ax.imshow(im2_matrix, cmap=plt.cm.gray)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(r'$d_{P_f}=%f$'%fill_dist)

    ax = fig.add_subplot(155)
    ax.imshow(im2hat_matrix, cmap=plt.cm.gray)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.savefig(fname)

def procrustes_filling(s1, s2, N=50, scale=250):
    im2_res, im2_hat, proc_dist = procrustes.procrustes(s1, s2, fullout=True)
    im2_matrix = shape_to_filled_image(im2_res, N=N, scale=scale)
    im2hat_matrix = shape_to_filled_image(im2_hat, N=N, scale=scale)
    fill_dist = np.linalg.norm(im2_matrix - im2hat_matrix)
    fill_dist = fill_dist/N
    return fill_dist


###############################################################################
if __name__ == '__main__':
    im1 = pick_digit(5)
    im2 = pick_digit(5)
    procrustes_filling_test(im1, im2, 
                            'shape_to_fill_comparison_57.pdf',
                            numpoints=800, N=50, scale=250)
    #for a, b in [(7,7), (5,5), (5,7)]:
    #    im1 = pick_digit(a)
    #    im2 = pick_digit(b)
    #    procrustes_filling_test(im1, im2, 
    #                        'shape_to_fill_comparison_%d%d.pdf'%(a,b),
    #                        numpoints=300, N=20, scale=80)


