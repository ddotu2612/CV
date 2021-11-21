import sys
sys.path.insert(0, r'G:/DepTrai/ThucTap/Chapter4')
import class_camera as camera
import numpy as np
from PIL import Image
from pylab import *

im1 = np.array(Image.open(r'G:\DepTrai\ThucTap\img\Metron\001.jpg')) 
im2 = np.array(Image.open(r'G:\DepTrai\ThucTap\img\Metron\002.jpg'))

# Load 2D points
points2D = [np.loadtxt(r'G:\DepTrai\ThucTap\img\Metron\2D/00' + str(i+1) + '.corners').T for i in range(3)]

# Load 3D points
points3D = np.loadtxt(r'G:\DepTrai\ThucTap\img\Metron\3D\p3d').T

# Load correspondences
corr = np.genfromtxt(r'G:\DepTrai\ThucTap\img\Metron\2D\nview-corners', dtype='int', missing_values='*')

# load cameras to a list of Camera objects
P = [camera.Camera(np.loadtxt(r'G:\DepTrai\ThucTap\img\Metron\2D/00' + str(i+1) + '.P')) for i in range(3)]

# make 3D points homogeneous and project
X = np.vstack((points3D, np.ones(points3D.shape[1])))
x = P[0].project(X)

## plotting the points in view 1
# figure()
# imshow(im1)
# plot(points2D[0][0], points2D[0][1], '*')
# axis('off')

# figure()
# imshow(im1)
# plot(x[0], x[1], 'r.')
# axis('off')
# show()

## Plotting 3D data with Matplotlib
from mpl_toolkits.mplot3d import axes3d

# fig = figure()
# ax = fig.gca(projection="3d")

# # # generate 3D sample data
# # X, Y, Z = axes3d.get_test_data(0.5)

# # # plot the points in 3D
# # ax.plot(X.flatten(), Y.flatten(), Z.flatten(), 'o')
# ax.plot(points3D[1], points3D[2], points3D[0], 'k.')

# show()

## Thử tìm epipole
# import sfm

# # index for points in first two views
# # trả về một mảng các giá trị true tại vị trí
# # cả hai toạ độ dương và false nếu ngược lại
# ndx = (corr[:, 0] >= 0) & (corr[:, 1] >= 0)

# # get coordinates and make homogeneous
# x1 = points2D[0][:, corr[ndx, 0]]
# x1 = np.vstack( (x1, np.ones(x1.shape[1])) )
# x2 = points2D[1][:, corr[ndx, 1]]
# x2 = np.vstack( (x2, np.ones(x2.shape[1])) )

# # compute F
# F = sfm.compute_fundamental(x1, x2)

# # compute the epipole
# e = sfm.compute_epipole(F)

# # plotting
# figure()
# imshow(im1)
# # plot each line individually, this gives nice colors
# for i in range(5):
#     sfm.plot_epipolar_line(im1, F, x2[:, i], e, False)
#     axis('off')

# figure()
# imshow(im2)
# # plot each point individually, this gives same colors as the lines
# for i in range(5):
#     plot(x2[0, i], x2[1, i], 'o')
# axis('off')
# show()


