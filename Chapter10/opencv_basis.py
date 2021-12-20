import cv2
import numpy as np

# # read image
# im = cv2.imread('empire.jpg')
# h, w = im.shape[:2]
# print(h, w)

# # save image
# cv2.imwrite('result.png', im)

# # create a grayscale version
# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# displaying images and results
# # read images 
# im = cv2.imread('fisherman.jpg')
# gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# # compute intergral image
# intim = cv2.integral(gray)

# # normalize and save 
# intim = (255.0 * intim) / intim.max()
# cv2.imwrite('result_integral.jpg', intim)

# # flood filling 
# filename = 'fisherman.jpg'
# im = cv2.imread(filename)
# h, w = im.shape[:2]

# # flood fill example
# diff = (40, 40, 40)
# mask = np.zeros((h + 2, w + 2), np.uint8)
# cv2.floodFill(im, mask, (10, 10), (255, 255, 0), diff, diff)

# # show the result in an OpenCV window
# cv2.imshow('flood fill', im)
# cv2.waitKey()

# # save the result
# cv2.imwrite('result_floodfill.jpg', im)

# Extract SURF features
# read image
im = cv2.imread('empire.jpg')

# down sample
im_lowres = cv2.pyrDown(im)

# convert to grayscale 
gray = cv2.cvtColor(im_lowres, cv2.COLOR_RGB2GRAY)

# detect feature points
s = cv2.ORB_create()
mask = np.uint8(np.ones(gray.shape))
keypoints, des = s.detectAndCompute(gray, mask)
# print(keypoints[1])
# show image and points
vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

for k in keypoints[::3]:
    # cv2.circle(vis,(int(k.pt[0]), int(k.pt[1])), 2, (0,255,0), -1)
    cv2.circle(vis, (int(k.pt[0]), int(k.pt[1])), int(k.size), (0,255,0), 2)

cv2.imshow('local descriptors', vis)
cv2.waitKey()
