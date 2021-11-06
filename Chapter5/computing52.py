exec(open("load_vggdata.py", encoding='utf-8').read())
import sys
sys.path.insert(0, r'G:/DepTrai/ThucTap/Chapter4')
import class_camera as camera
import numpy as np
import sfm

# corr = coor
# points2D = points2D
# points3D = points3D

# # index for points in first two views
# ndx = (corr[:,0]>=0) & (corr[:,1]>=0)

# # get coordinates and make homogeneous
# x1 = points2D[0][:, corr[ndx,0]]
# x1 = np.vstack((x1, np.ones(x1.shape[1])))
# x2 = points2D[1][:, corr[ndx,1]]
# x2 = np.vstack((x2, np.ones(x2.shape[1])))

# Xtrue = points3D[:, ndx]
# Xtrue = np.vstack((Xtrue, np.ones(Xtrue.shape[1])))

# # check first 3 points
# Xest = sfm.triangulate(x1, x2, P[0].P, P[1].P)
# print(Xest[:, :3])
# print(Xtrue[:, :3])

# # plotting
# from mpl_toolkits.mplot3d import axes3d

# fig = figure()
# ax = fig.gca(projection='3d')
# ax.plot(Xest[0], Xest[1], Xest[2], 'ko')
# ax.plot(Xtrue[0], Xtrue[1], Xtrue[2], 'r.')
# axis('auto')

# show()

# Computing P from 2D, 3D points
corr = corr[:, 0] # image1
ndx3D = np.where(corr>=0)[0] # missing values are -1
ndx2D = corr[ndx3D]

# select visibel points and make homogeneous
x = points2D[0][:, ndx2D] # image1
x = np.vstack((x, np.ones(x.shape[1])))
X = points3D[:, ndx3D]
X = np.vstack((X, np.ones(X.shape[1])))

# estimate P
Pest = camera.Camera(sfm.computer_P(x, X))

# compare
print(Pest.P / Pest.P[2, 3])
print(P[0].P / P[0].P[2, 3])
xest = Pest.project(X)

# plotting
from mpl_toolkits.mplot3d import axes3d
from pylab import *

figure()
imshow(im1)
plot(x[0], x[1], 'b.')
plot(xest[0], xest[1], 'r.')
axis('off')

show()
