from scipy import ndimage
from PIL import Image
from numpy import *
from numpy import array, meshgrid, loadtxt
from pylab import *
import warp
import homogeneous as homography

# im = array(Image.open('cat 2.jpg').convert('L'))
# H = array([[1.4, 0.05, -100],[0.05, 1.5, -100],[0, 0, 1]])
# im2 = ndimage.affine_transform(im, H[:2,:2], (H[0,2], H[1,2]))

# fig, ax = subplots(1, 2, figsize=(10, 2))
# ax[0].imshow(im)
# ax[1].imshow(im2)
# gray()
# show()

# 2. Image to image
# import warp
# # example of affine warp of im1 onto im2
# im1 = array(Image.open('cat 2.jpg').convert('L'))
# im2 = array(Image.open("house1.jpg").convert('L'))

# # Image to image
# # set to points
# tp = array([[264,538,540,264], [40,36,605,605], [1,1,1,1]])
# # tp = array([[675, 826, 826, 677], [55, 52, 281, 277], [1, 1, 1, 1]])
# im3 = warp.image_in_image(im1, im2, tp)
# figure()
# gray()
# imshow(im3)
# axis('equal')
# axis('off')
# show()

# # Maching hoàn toàn using affine transformations
# # set from points to corners of im1
# m,n = im1.shape[:2]
# fp = array([[0, m, m, 0], [0, 0, n, n], [1, 1, 1, 1]])
# tp = array([[264,538,540,264], [40,36,605,605], [1,1,1,1]])

# # first triangle
# tp2 = tp[:,:3]
# fp2 = fp[:,:3]

# # compute H
# H = homography.Haffine_from_points(tp2, fp2)
# im1_t = ndimage.affine_transform(im1, H[:2, :2], (H[0, 2], H[1, 2]), im2.shape[:2])

# # alpha for triangle
# alpha = warp.alpha_for_triangle(tp2, im2.shape[0], im2.shape[1])
# im3 = (1-alpha)*im2 + alpha*im1_t

# # second triangle
# tp2 = tp[:, [0,2,3]]
# fp2 = fp[:, [0,2,3]]

# # compute H
# H = homography.Haffine_from_points(tp2, fp2)
# im1_t = ndimage.affine_transform(im1, H[:2,:2], (H[0,2], H[1,2]), im2.shape[:2])

# # alpha for triangle
# alpha = warp.alpha_for_triangle(tp2,im2.shape[0], im2.shape[1])
# im4 = (1-alpha)*im3 + alpha*im1_t

# figure()
# gray()
# imshow(im4)
# axis('equal')
# axis('off')
# show()

# # Delaunay triangulations
# from scipy.spatial import Delaunay
# from numpy import random, concatenate

# x, y = array(random.standard_normal((2,100))) # Tạo ra hai mảng 100 phần tử theo phân phối chuẩn
# a = concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
# tri = Delaunay(a)

# figure()
# # # Cách 1
# # for t in tri.simplices:
# #     t_ext = [t[0], t[1], t[2], t[0]] # thêm điểm đầu tiên tới cuối cùng 
# #     plot(x[t_ext], y[t_ext], 'r')

# # # Cách 2
# triplot(a[:, 0], a[:, 1], tri.simplices)
# plot(x, y, '*')
# axis('off')
# show()

# # piecewise affine image warping
# fromim = array(Image.open('cat 2.jpg'))
# x, y = meshgrid(range(5), range(6))
# # print(x)
# # print(y)
# # plot(x, y, '*')
# # show()

# x = (fromim.shape[1]/4) * x.flatten()
# y = (fromim.shape[0]/5) * y.flatten()
# print(x)

# # triangulate
# tri = warp.triangulate_points(x, y)

# # open image and destination points
# im = array(Image.open('turningtorso1.jpg'))
# tp = loadtxt('turningtorso1_points.txt') # destination points




