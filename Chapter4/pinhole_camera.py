from numpy.core.shape_base import hstack
from class_camera import Camera, rotation_matrix
from numpy import loadtxt, eye, array
from numpy.random import rand
from pylab import *

points = loadtxt('housemodel\house.p3d').T
points = vstack((points, ones(points.shape[1])))
print(points)

# setup camera
P = hstack((eye(3), array([[0], [0], [-10]])))
cam = Camera(P)
x = cam.project(points)

# # plot projection
# figure()
# plot(x[0], x[1], 'k.')
# show()

# Thực hiện một phép xoay máy ảnh xem hình chiếu thay đổi
# Tạo một phép dịch chuyển
r = 0.05*rand(3)
rot = rotation_matrix(r)

# xoay camera và chiếu
# figure()
# for t in range(20):
#     cam.P = dot(cam.P, rot)
#     x = cam.project(points)
#     plot(x[0], x[1], 'k.')
# show()

# Factorization
K = array([[1000, 0, 500], [0, 1000, 300], [0, 0, 1]])
tmp = rotation_matrix([0, 0, 1])[:3, :3]
# Ghép R và t
Rt = hstack((tmp, array([[50], [40], [30]])))
cam = Camera(dot(K, Rt))

print(K, Rt)
print(cam.factor())