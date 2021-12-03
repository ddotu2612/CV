from scipy.cluster.vq import *
import numpy as np
from scipy.spatial import distance

def cluster(S, k, ndim):
    """ Spectral clustering from a similarity matrix. """

    # check for symmetry
    if np.sum(np.abs(S - S.T)) > 1e-10:
        print('not symmetric')
    
    # create Laplacian matrix
    rowsum = np.sum(np.abs(S), axis=0)
    D = np.diag(1 / np.sqrt(rowsum + 1e-6))
    L = np.dot(D, np.dot(S, D))

    # compute eigenvectors of L
    U, sigma, V = np.linalg.svd(L)

    # create feature vector from ndim first eigenvectors
    # by stacking eigenvectors as columns
    features = np.array(V[:ndim]).T

    # k-means
    features = whiten(features)
    centroids, distortion = kmeans(features, k)
    code, distance = vq(features, centroids)

    return code, V 

import ncut
import cv2 
from PIL import Image
from pylab import *

im = np.array(Image.open('C-uniform03.ppm'))
m, n = im.shape[:2]

# resize image to (wid, wid)
wid = 50
rim = cv2.resize(im, (wid, wid), interpolation=cv2.INTER_LINEAR)
rim = np.array(rim, 'f')

# create normalized cut matrix
A = ncut.ncut_graph_matrix(rim, sigma_d=1, sigma_g=1e-2)

# cluster
code, V = cluster(A, k=3, ndim=3)

# reshape to original image size
codeim = cv2.resize(code.reshape(wid, wid), (m, n), interpolation=cv2.INTER_NEAREST)

# plot result
fig, ax = subplots(1, 4)
# imshow(codeim)
for i in range(4):
    ax[i].imshow(cv2.resize(V[i].reshape(wid, wid), (m, n), interpolation=cv2.INTER_LINEAR))
gray()
show()
