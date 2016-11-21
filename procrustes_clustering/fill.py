
import cv2
import numpy as np
import matplotlib.pyplot as plt

import shape


def pick_digit(d):
    """Pick an MNIST digit."""
    import gzip, cPickle
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    images, labels = train_set
    x = np.where(labels==d)[0]
    return images[np.random.choice(x)].reshape((28,28))

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

def flood_fill(data, start_coords, fill_value):
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

def show_fill(d):
    im = pick_digit(d)
    s = shape.get_external_contour(im, 200, 5)
    im_in = shape.shape_to_image(s, 1)
    #im_in = shape.shape_to_image2(s)
    im_out = im_in.copy()
    #im_in = cv2.imread("nickel.jpg", cv2.IMREAD_GRAYSCALE)
    #im_out = fill2(im_in)
    idx = np.where(im_out > 0)
    xs = idx[0]
    ys = idx[1]
    maxx = xs.max()
    maxy = ys.max()
    minx = xs.min()
    miny = ys.min()
    x = (minx + maxx)/2
    y = (miny + maxy)/2
    print x, y
    flood_fill(im_out, (x, y), 1)


    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(im_in, cmap=plt.cm.gray)
    #ax.axis('off')
    ax.set_aspect('equal')

    ax = fig.add_subplot(122)
    ax.imshow(im_out, cmap=plt.cm.gray)
    #ax.axis('off')
    ax.set_aspect('equal')

    fig.savefig('figs/fill_%d.pdf'%d)



###############################################################################
if __name__ == '__main__':
    show_fill(7)
