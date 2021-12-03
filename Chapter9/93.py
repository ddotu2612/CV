import sys
sys.path.insert(0, r'G:/dvtu/ThucTap/Chapter1')
import rof
import numpy as np
from PIL import Image
from pylab import *

im = np.array(Image.open('ceramic-houses_t0.png').convert("L"))
U, T = rof.denoise(im, im, tolerance=0.001)
t = 0.4 #threshold

U_ = (U < t*U.max())*1.0

fig, ax = subplots(1, 3)
ax[0].imshow(im)
ax[1].imshow(U)
ax[2].imshow(U_)
show()


