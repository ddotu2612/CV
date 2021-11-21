import cv2
from sift import  plot_features, process_image, appendimages, plot_matches
from numpy import *
from pylab import *
from PIL import Image
from math import *
import sift

# imname = 'cat 2.jpg'
# im1 = array(Image.open(imname).convert('L'))
# KeyPoint, Descriptors = process_image(im1)
# plot_features(im1, KeyPoint)

im1 = array(Image.open('cat 2.jpg').convert('L'))
im2 = array(Image.open('cat 2.jpg').convert('L'))

KeyPoint1, Descriptors1 = process_image(im1)
KeyPoint2, Descriptors2 = process_image(im2)
# pts = cv2.KeyPoint_convert(KeyPoint1)
print(Descriptors2)


# # Vẽ Keypoint trên hai ảnh
# sift_image1 = cv2.drawKeypoints(im1, KeyPoint1, None)
# sift_image2 = cv2.drawKeypoints(im1, KeyPoint2, None)
# # Nối hai ảnh lại theo chiều ngang
# concat12 = appendimages(sift_image1, sift_image2)

# # create feature matcher
# bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
# # match descriptors of both images
# matches = bf.match(Descriptors1, Descriptors2)
# # sort matches by distance
# matches = sorted(matches, key = lambda x : x.distance)
# # draw first 50 matches
# matched_img = cv2.drawMatches(im1, KeyPoint1, im2, KeyPoint2, matches[:200], None, flags=2)

# matched_img = vstack((concat12, matched_img))
# # show the image
# cv2.imshow('image', matched_img)
# # save the image
# cv2.imwrite("matched_images.jpg", matched_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Trích xuất toạ độ từ Keypoint
# def draw_circle(c, r):

#     # Tao một array các phần tử trong khoảng từ 0-2pi
#     t = arange(0, 1.01, .01) * 2 * pi

#     # Toạ độ các điểm ấy tính từ tâm
#     x = r*cos(t) + c[0]
#     y = r*sin(t) + c[1]

#     # vẽ các đường nối qua các điểm ấy tạo thành một hình tròn
#     plot(x, y, 'b', linewidth=2)

# Lấy ra toạ độ x, y của các điểm keypoint
# pts = cv2.KeyPoint_convert(KeyPoint)
# print(pts)
# circle = False
# imshow(im1)
# if circle:
#     for p in pts:
#         draw_circle(p[:2], p[2])
# else:
#     plot(pts[:, 0], pts[:, 1], 'ob')
# axis('off')
# show()

# Vẽ match sử dụng hàm tự code
# locs1 = cv2.KeyPoint_convert(KeyPoint1)
# locs2 = cv2.KeyPoint_convert(KeyPoint2)
# print(locs2[0][1])
# matchscores = sift.match_twosided(Descriptors1, Descriptors2)
# plot_matches(im1, im2, locs1, locs2, matchscores.reshape(-1)[0:50])

