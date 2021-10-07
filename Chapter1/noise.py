from numpy import *
import numpy as np
from scipy.ndimage import filters
import rof
from pylab import *
from PIL import Image

# create synthetic image with noise
# im = zeros((500,500))
# im[100:400,100:400] = 128
# im[200:300,200:300] = 255
# im = im + 30 * (np.random.standard_normal((500, 500)))

# U, T = rof.denoise(im, im)
# G = filters.gaussian_filter(im, 10)

# Với ảnh thật
im = array(Image.open('cat 2.jpg').convert('L'))
U, T = rof.denoise(im,im)
G = filters.gaussian_filter(im, 10)

fig, axs = subplots(1, 3, figsize=(10,4))
gray()
axis('equal')
axis('off')
axs[0].imshow(im)
axs[1].imshow(G)
axs[2].imshow(U)
show()