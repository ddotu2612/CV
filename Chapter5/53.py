import sys
sys.path.insert(0, 'G:\DepTrai\ThucTap\Chapter3')
sys.path.insert(0, 'G:\DepTrai\ThucTap\Chapter2')
sys.path.insert(0, 'G:\DepTrai\ThucTap\Chapter4')
import homogeneous as homography
import class_camera as camera
import sift
import sfm
import cv2
import numpy as np

im1 = cv2.imread('alcatraz1.jpg', 0)
im2 = cv2.imread('alcatraz2.jpg', 0)

detector = cv2.xfeatures2d.SIFT_create()
kp1, des1 = detector.detectAndCompute(im1, None)
kp2, des2 = detector.detectAndCompute(im2, None)

# # create feature matcher
# bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
# # match descriptors of both images
# matches = bf.match(des1, des2)
# # sort matches by distance
# matches = sorted(matches, key = lambda x : x.distance)
kp1 = cv2.KeyPoint_convert(kp1)
kp2 = cv2.KeyPoint_convert(kp2)
matches = sift.match_twosided(des1, des2)
matches = matches.reshape(-1)
ndx = matches.nonzero()[0]
# kp1 = np.int32([kp.pt for kp in kp1])
# kp2 = np.int32([kp.pt for kp in kp2])
# idx1 = np.int32([m.queryIdx for m in matches])
# idx2 = np.int32([m.trainIdx for m in matches])

# x1 = homography.make_homog(kp1[idx1, :2].T)
# x2 = homography.make_homog(kp2[idx2, :2].T)
# make homogeneous and normalize with inv(K)
x1 = homography.make_homog(kp1[ndx, :2].T)
ndx2 = [int(matches[i]) for i in ndx]
x2 = homography.make_homog(kp2[ndx2, :2].T)

# calibration
K = np.array([[2394,0,932],[0,2398,628],[0,0,1]])

x1n = np.dot(np.linalg.inv(K), x1)
x2n = np.dot(np.linalg.inv(K), x2)
# estimate E with RANSAC
model = sfm.RansacModel()
E, inliers = sfm.F_from_ransac(x1n, x2n, model)

# compute camera matrices (P2 will be list of four solutions)
P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
P2 = sfm.compute_P_from_essential(E)

# pick the solution with points in front of cameras
ind = 0
maxres = 0
for i in range(4):
    # triangular inliers and computer depth for each camera
    X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2[i])
    d1 = np.dot(P1,X)[2]
    d2 = np.dot(P2[i],X)[2]
    if sum(d1>0)+sum(d2>0) > maxres:
        maxres = sum(d1>0) + sum(d2>0)
        ind = i
        infront = (d1>0) & (d2>0)
    
# triangulate inliers and remove points not in front of both cameras
X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2[ind])
X = X[:,infront]

from mpl_toolkits.mplot3d import axes3d
from pylab import *

# fig = figure()
# ax = fig.gca(projection='3d')
# ax.plot(-X[0], X[1], X[2], 'k.')
# axis('off')

# plot the projection of X
# project 3D points
cam1 = camera.Camera(P1)
cam2 = camera.Camera(P2[ind])
x1p = cam1.project(X)
x2p = cam2.project(X)

# reverse K normalization
x1p = dot(K, x1p)
x2p = dot(K, x2p)

figure()
imshow(im1)
gray()
plot(x1p[0], x1p[1], 'o')
plot(x1[0], x1[1], 'r.')
axis('off')

figure()
imshow(im2)
gray()
plot(x2p[0], x2p[1], 'o')
plot(x2[0], x2[1], 'r.')
axis('off')

show()