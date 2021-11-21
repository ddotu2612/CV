from PIL import Image
from numpy import uint8, sqrt
from numpy import *
from scipy.ndimage import filters
from pylab import *

# 1. Làm mờ ảnh
# im = array(Image.open('cat 2.jpg').convert('L'))
# im2 = filters.gaussian_filter(im, 2)
# # Làm mờ ảnh theo các kênh khác nhau
# im = array(Image.open('cat 2.jpg'))
# im2 = zeros(im.shape)
# for i in range(3):
#     im2[:,:,i] = filters.gaussian_filter(im[:,:,i],5)
#     im2 = uint8(im2)
#     imshow(im2)
# show()

# 2. Image derivatives
#Sobel derivative filters
# im = array(Image.open('cat 2.jpg').convert('L'))
# imx = zeros(im.shape)
# filters.sobel(im,1,imx)
# imy = zeros(im.shape)
# filters.sobel(im,0,imy)
# magnitude = sqrt(imx**2+imy**2)

# # Gaussian derivative filters
im = array(Image.open('cat 2.jpg').convert('L'))
sigma = 5 #standard deviation
imx = zeros(im.shape)
filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)
imy = zeros(im.shape)
filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)
magnitude = sqrt(imx**2+imy**2)
# show image
figure(figsize=(10, 3))
gray()
subplot(1, 4, 1)
imshow(im) # Chuyển lại hình ảnh từ dạng flatten
subplot(1, 4, 2)
imshow(imx)
subplot(1, 4, 3)
imshow(imy)
subplot(1, 4, 4)
imshow(magnitude)
show()

# 3. Morphology
# from scipy.ndimage import measurements,morphology

# # load image and threshold to make sure it is binary
# im = array(Image.open('houses.png').convert('L'))
# # Chuyển sang ảnh nhị phân
# im = 1*(im<128) # Sử dụng threshold
# # morphology - opening to separate objects better
# im_open = morphology.binary_opening(im, ones((9,5)), iterations=2)

# labels, nbr_objects = measurements.label(im_open)
# print("Number of objects: ", nbr_objects)

# 4. Useful SciPy modules

# Đọc file .mat trong Matlab
# import scipy
# data = scipy.io.loadmat('test.mat')

# # Lưu dữ liệu vào file .mat
# x = 'Hello World'
# data = {}
# data['x'] = x
# scipy.io.savemat('test.mat', data)

# import scipy.misc
# im = ones((300, 300))
# # Lưu một mảng thành một image
# scipy.misc.imsave('test.jpg', im)

# Lấy hình ảnh Lena cho test image, hiện tại đã bị xoá
# lena1 = scipy.misc.lena()
# imshow(lena1)
# show()

