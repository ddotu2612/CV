from numpy.core.fromnumeric import mean
from scipy.cluster.vq import *
import cv2
import  numpy as np
from PIL import Image

steps = 100 # image is devided in steps*steps region
im = np.array(Image.open(r'G:\DepTrai\ThucTap\img\data\empire.jpg'))

dx = im.shape[0] // steps
dy = im.shape[1] // steps
print(im.shape)

# Compute color features for each region
features = []
for x in range(steps):
    for y in range(steps):
        R = mean(im[x*dx:(x+1)*dx, y*dy: (y+1)*dy, 0])
        G = mean(im[x*dx:(x+1)*dx, y*dy: (y+1)*dy, 1])
        B = mean(im[x*dx:(x+1)*dx, y*dy: (y+1)*dy, 2])
        features.append([R, G, B])

features = np.array(features, 'f')

# cluster
centroids, variance = kmeans(features, 3)
code, distance = vq(features, centroids)

# create image with cluster labels
codeim = code.reshape(steps, steps)
codeim = cv2.resize(codeim, im.shape[:2], interpolation=cv2.INTER_NEAREST)

from pylab import *

fig, ax = subplots(1, 2, figsize=(5,5))
ax[0].imshow(im)
ax[1].imshow(codeim)
show()

