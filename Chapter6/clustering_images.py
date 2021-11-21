import sys
sys.path.insert(0, r'G:/DepTrai/ThucTap/Chapter1')
import imtools
import numpy as np
from scipy.cluster.vq import *
from PIL import Image

# get list images
imlist = imtools.get_imlist(r'G:\DepTrai\ThucTap\img\data\selectedfontimages\a_selected_thumbs')
imnbr = len(imlist)
# print(imnbr)

# create metrix to store all flattened images
immatrix = np.array([np.array(Image.open(im)).flatten() for im in imlist], 'f')
# perform PCA
V, S, immean = imtools.pca(immatrix)

# project on the 40 first Principal components
immean = immean.flatten()
# projected = np.array([np.dot(V[:7], immatrix[i] - immean) for i in range(imnbr)])
projected = np.array([np.dot(V[[0,2]], immatrix[i]-immean) for i in range(imnbr)])
# # k-means
# projected = whiten(projected) # normalization
# centroids, distortion = kmeans(projected, 4)

# code, distance = vq(projected, centroids)

from pylab import *

# # Hiển thị các phân cụm
# for k in range(4):
#     ind = np.where(code==k)[0]
#     figure()
#     gray()
#     for i in range(np.minimum(len(ind), 40)):
#         subplot(4, 10, i +1)
#         imshow(immatrix[ind[i]].reshape((25, 25)))
#         axis('off')

# show()

# Hiển thị trên các cặp của principal components
from PIL import Image, ImageDraw

# height and width
h, w = 1200, 1200

# create a new image with a white background
img = Image.new('RGB', (w, h), (255, 255, 255))
draw = ImageDraw.Draw(img)

# draw axis
draw.line((0, h/2, w, h/2), fill=(255, 0, 0))
draw.line((w/2, 0, w/2, h), fill=(255, 0, 0))

# scale coordinates to fit
scale = abs(projected).max(0)
scaled = floor(array([(p/scale) * (w/2-20,h/2-20) +
(w/2,h/2) for p in projected]))
print(scaled)
# paste thumbnail of each image
for i in range(imnbr):
    nodeim = Image.open(imlist[i])
    nodeim.thumbnail((25, 25))
    ns = nodeim.size
    img.paste(nodeim, ((int)(scaled[i][0] - ns[0]//2), (int)(scaled[i][1]-ns[1]//2), 
                        (int)(scaled[i][0] + ns[0]//2 + 1), (int)(scaled[i][1] + ns[1]//2+1)))

img.save('pca_font.jpg')
