# import stereo
# from numpy import *
# from PIL import Image

# im_l = array(Image.open('scene1.row3.col3.ppm').convert('L'), 'f')
# im_r = array(Image.open('scene1.row3.col4.ppm').convert('L'), 'f')

# # starting displacement and steps
# steps = 12
# start = 4

# # width for ncc
# wid = 9
# res = stereo.phane_sweep_ncc(im_l,im_r,start,steps,wid)

# import cv2
# cv2.imwrite('depth.png', res)

# Stereo images in openCV
import numpy as np
import cv2
from pylab import *

imgL = cv2.imread('tsukuba_l.png', 0)
imgR = cv2.imread('tsukuba_r.png', 0)

stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)
disparity = stereo.compute(imgL,imgR)
imshow(disparity,'gray')
show()
