import sys
from numpy.core.numeric import identity
sys.path.insert(0, r'G:/dvtu/ThucTap/Chapter1')
import imtools as it
import numpy as np
from PIL import Image
from scipy.cluster.vq import *
import math
from pylab import *

# get list images
imlist = it.get_imlist(r'G:\dvtu\ThucTap\img\data\selectedfontimages\a_selected_thumbs')
imnbr = len(imlist)

# create metrix to store all flattened images
immatrix = np.array([np.array(Image.open(im)).flatten() for im in imlist], 'f')
# perform PCA
V, S, immean = it.pca(immatrix)

# project on the 40 first Principal components
immean = immean.flatten()
projected = np.array([np.dot(V[:7], immatrix[i] - immean) for i in range(imnbr)])

n = len(projected)

# compute distance matrix
S = np.array([[math.sqrt(np.sum((projected[i]-projected[j])**2)) for i in range(n)] for j in range(n)], 'f')
# create Laplacian matrix
rowsum = np.sum(S, axis=0)
# print(rowsum)
D = np.diag(1.0/sqrt(rowsum))
I = identity(n)
L = I - np.dot(D, np.dot(S, D))

# compute eigenvectors of L
U, sigma, V = np.linalg.svd(L)

k = 5
# create feature vector from k first eigenvectors
# by stacking eigenvectors as columns
features = np.array(V[:k]).T

# k-means
features = whiten(features)
centroids, distortion = kmeans(features, k)
code, distance = vq(features, centroids)

# plot clusters
for c in range(k):
    ind = np.where(code == c)[0]
    figure()
    for i in range(min(len(ind), 39)):
        im = Image.open(imlist[ind[i]])
        subplot(4, 10, i + 1)
        imshow(np.array(im))
        axis('equal')
        axis('off')
show()
