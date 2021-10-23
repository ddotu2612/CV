from numpy import *
from numpy import array, zeros, linalg, hstack, dot
import homogeneous as homography
from scipy import ndimage
from scipy.spatial import Delaunay
from numpy import random, concatenate
from pylab import *

def image_in_image(im1, im2, tp):
    """ Put im1 in im2 with an affine transformation
    such that corners are as close to tp as possible.
    tp are homogeneous and counter-clockwise from top left. """

    # points to warp from
    m,n = im1.shape[:2]
    fp = array([[0, m, m, 0], [0, 0, n, n], [1, 1, 1, 1]])

    H = homography.Haffine_from_points(tp, fp)
    im1_t = ndimage.affine_transform(im1, H[:2, :2], (H[0, 2], H[1, 2]), im2.shape[:2])
    alpha = (im1_t > 0)

    return (1 - alpha)*im2 + alpha*im1_t

def alpha_for_triangle(points, m, n):
    """ Creates alpha map of size (m,n)
    for a triangle with corners defined by points
    (given in normalized homogeneous coordinates). """

    alpha = zeros((m,n))

    for i in range(min(points[0]), max(points[0])):
        for j in range(min(points[1]), max(points[1])):
            x = linalg.solve(points, [i, j, 1])
            if min(x) > 0: #all coefficients positive
                alpha[i,j] = 1

    return alpha

def triangulate_points(x, y):
    """ Delaunay triangulation of 2D points. """

    a = concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    tri = Delaunay(a)

    return tri.simplices

def pw_affine(fromim, toim, fp, tp, tri):
    """ Warp triangular patches from an image.
    fromim = image to warp
    toim = destination image
    fp = from points in hom. coordinates
    tp = to points in hom. coordinates
    tri = triangulation. """

    im = toim.copy()

    # check if image is grayscale or color
    is_color = len(fromim.shape) == 3

    # create image to warp to (needed if iterate colors)
    im_t = zeros(im.shape, 'uint8')

    for t in tri:
        # compute affine transformation
        H = homography.Haffine_from_points(tp[:,t], fp[:,t])

        if is_color:
            for col in range(fromim.shape[2]):
                im_t[:, :, col] = ndimage.affine_transform(fromim[:, :, col], H[:2, :2], 
                (H[0,2], H[1,2]), im.shape[:2])
        else:
            im_t = ndimage.affine_transform(fromim, H[:2,:2], (H[0,2],H[1,2]), im.shape[:2])

        # alpha for triangle
        alpha = alpha_for_triangle(tp[:, t], im.shape[0], im.shape[1])

        # add triangle to image
        im[alpha > 0] = im_t[alpha > 0]

    return im

# Vẽ các tam giác
def plot_mesh(x, y, tri):
    """ Plot triangles. """

    for t in tri:
        t_ext = [t[0], t[1], t[2], t[0]] # add first point to end
        plot(x[t_ext],y[t_ext], 'r')


### Panorama Ảnh toàn cảnh
def panorama(H, fromim, toim, padding=2400, delta=2400):
    """ Tạo một ảnh toàn cảnh bằng cách kết hợp hai ảnh
    sử dụng một phép đồng nhất H ( đã được ước tính bằng
    cách sử dụng RANSAC) Kết quả là một bức ảnh với chiều
    cao bằng chiều cao của toim. Padding chỉ định số pixel
    lấp đầy và delta dịch chuyển bổ sung """

    # Kiểm tra ảnh có là ảnh xám hay ảnh màu
    is_color = len(fromim.shape) == 3

    # Phép đồng nhất cho phép biến đổi hình  học
    def transf(p):
        p2 = dot(H, [p[0], p[1], 1])
        return (p2[0] / p2[2], p2[1] / p2[2])

    # Nếu fromim là bên phải
    if H[1, 2] < 0:
        print('warp - right')
        # Biến đổi fromim
        if is_color:
            # padding tới ảnh gốc với 0 về bên phải
            toim_t = hstack((toim, zeros((toim.shape[0], padding, 3))))
            fromim_t =  zeros((toim.shape[0], toim.shape[1] + padding, toim.shape[2]))
            for col in range(3):
                fromim_t[:, :, col] = ndimage.geometric_transform(fromim[:, :, col], transf, (toim.shape[0], toim.shape[1] + padding))
        else:
            # padding ảnh đích với không về phía bên phải
            toim_t = hstack((toim, zeros((toim.shape[0], padding))))
            fromim_t = ndimage.geometric_transform(fromim, transf, ((toim.shape[0], toim.shape[1] + padding)))
    else: # Bên trái
        print('warp - left')
        # Thêm một dịch chuyển để bù cho padding ở bên trái
        H_delta = array([[1, 0, 0], [0, 1, -delta], [0, 0, 1]])
        H = dot(H, H_delta)

        # Chuyển đổi fromim 
        if is_color:
            # thêm vào ảnh đích với không tới bên trái
            toim_t = hstack((zeros((toim.shape[0], padding, 3))), toim)
            fromim_t = zeros((toim.shape[0], toim.shape[1] + padding, 3))
            for col in range(3):
                fromim_t[:, :, col] = ndimage.geometric_transform(fromim_t, transf, (toim.shape[0], toim.shape[1]) + padding)
        else:
            # pad the destination image with zeros to the left
            toim_t = hstack((zeros((toim.shape[0], padding)), toim))
            fromim_t = ndimage.geometric_transform(fromim,
            transf, (toim.shape[0], toim.shape[1] + padding))

    # Kết hợp và trả lại kết quả ( đặt fromim trên toim)
    if is_color:
        # Lấy pixel không phải màu đen:
        alpha = ((fromim_t[:, :, 0] * fromim_t[:, :, 1] * fromim_t[:, :, 2]) > 0)
        for col in range(3):
            toim_t[:, :, col] = fromim_t[:, :, col]*alpha + toim_t[:, :, col]*(1 - alpha)
    else:
        alpha = (fromim_t > 0)
        toim_t = fromim_t*alpha + toim_t*(1 - alpha)
    
    return toim_t
    
