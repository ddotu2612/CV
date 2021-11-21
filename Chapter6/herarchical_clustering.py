import sys
sys.path.insert(0, r'G:/dvtu/ThucTap/Chapter1')
import imtools as it
from numpy.core.shape_base import vstack
from numpy.lib.histograms import histogram, histogramdd
import hcluster
from numpy.random import randn
import numpy as np

# # Test clustering 2D data
# class1 = 1.5 * randn(100, 2) # Create 100 points 2D with normalize distribute
# class2 = randn(100, 2) + np.array([5, 5])

# features = vstack((class1, class2))
# # print(features)

# tree = hcluster.hcluster(features)

# clusters = tree.extract_clusters(5)

# print('Number of clusters:', len(clusters))
# for c in clusters:
#     print(c.get_cluster_elements())

# Clustering images sunsets
import os
from PIL import Image

# create a list of images
path = r'G:\dvtu\ThucTap\img\data\sunsets\flickr-sunsets-small'
imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

# extract feature vector (8 bins per color channel)
features = np.zeros([len(imlist), 512])
for i, f in enumerate(imlist):
    im = np.array(Image.open(f))

    # multi-dimensional histogram
    h, edges = histogramdd(im.reshape(-1, 3), 8, normed=True, range=[(0, 255), (0, 255), (0, 255)])
    features[i] = h.flatten()

# tree = hcluster.hcluster(features)
# # hcluster.draw_dendrogram(tree, imlist, filename='sunset.pdf')

# # visualize clusters with some (arbitrary) threshold
# clusters = tree.extract_clusters(0.23*tree.distance)

# # plot images for clusters with more than 3 elements
# from pylab import *

# for c in clusters:
#     elements = c.get_cluster_elements()
#     nbr_elements = len(elements)
#     if nbr_elements > 3:
#         figure()
#         for p in range(min(nbr_elements, 20)):
#             subplot(4, 5, p +1)
#             im = np.array(Image.open(imlist[elements[p]]))
#             imshow(im)
#             axis('off')
# show()

# Data font images

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

tree = hcluster.hcluster(projected)
hcluster.draw_dendrogram(tree, imlist, filename='font.jpg')