import matplotlib.pyplot as plt
import cPickle, gzip
import matplotlib.pyplot as plt
import numpy as np

from pplotcust import *


f = gzip.open('data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
images, labels = train_set

X = []
for d in range(0,10):
    x = np.where(labels==d)[0]
    js = np.random.choice(x, 2, replace=False)
    for j in js:
        im = np.array(images[j])
        X.append(im)
X = np.array(X)
idx = range(20)
np.random.shuffle(idx)
X = X[idx]
fig, axes = plt.subplots(nrows=5, ncols=4,
                        #figsize=(3.85,4.69))
                        figsize=(4,5))
i = 0
for row in axes:
    for ax in row:
        ax.imshow(X[i].reshape((28,28)), cmap=plt.cm.gray)
        ax.axis('off')
        ax.set_aspect('equal')
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
        i += 1
fig.subplots_adjust(wspace=-0.1, hspace=-0.1)
#fig.tight_layout()
fig.savefig('mnist.pdf')

