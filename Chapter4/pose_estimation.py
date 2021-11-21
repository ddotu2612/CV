import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import imageio
from class_camera import my_calibration, Camera
from numpy import *
import sys
sys.path.insert(0, r'G:/DepTrai/ThucTap/Chapter3')
import homogeneous as homography

# Hàm vẽ một ảnh
def plot_img(img, size=(7, 7), title=""):
    cmap = "gray" if len(img.shape) == 2 else None
    plt.figure(figsize=size)
    plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
    plt.show()

# Hàm vẽ nhiều ảnh
def plot_imgs(imgs, cols=5, size=7, title=""):
    rows = len(imgs) // cols + (0 if len(imgs) % cols == 0 else 1)
    fig = plt.figure(figsize=(cols*size, rows*size))
    for i, img in enumerate(imgs):
        cmap = "gray" if len(img.shape) == 2 else None
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
    plt.show()

# Đọc ảnh và chuyển sang ảnh xám
src_img = imageio.imread(r'G:\DepTrai\ThucTap\img\data\book_frontal.JPG')
tar_img = imageio.imread(r'G:\DepTrai\ThucTap\img\data\book_perspective.JPG')
src_gray = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
tar_gray = cv2.cvtColor(tar_img, cv2.COLOR_RGB2GRAY)
# plot_imgs([src_img, tar_img], cols=2, size=8)

# Lấy ra các điểm Keypoints và Detectors
SIFT_detector = cv2.xfeatures2d.SIFT_create()
kp1, des1 = SIFT_detector.detectAndCompute(src_gray, None)
kp2, des2 = SIFT_detector.detectAndCompute(tar_gray, None)

# Maching keypoint
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Bruce Force KNN trả về list k ứng viên cho mỗi keypoint
rawMatches = bf.knnMatch(des1, des2, 2)
matches = []
ratio = 0.75

for m, n in rawMatches:
    # Giữ lại các cặp keypoint sao cho với kp1 thì khoảng cách giữa kp1 và ứng viên 1 là rất gần so với khoảng cách giữa ứng viên 2 và kp1
    if m.distance < n.distance * ratio:
        matches.append(m)
# Lấy 200 matches tốt nhất 
matches = sorted(matches, key=lambda x: x.distance, reverse=True)
# matches = matches[:200]

# img3 = cv2.drawMatches(src_img, kp1, tar_img, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plot_img(img3, size=(15, 10))

# Estimate Homography matrix and transform image
kp1 = np.float32([kp.pt for kp in kp1])
kp2 = np.float32([kp.pt for kp in kp2])
pts1 = np.float32([kp1[m.queryIdx] for m in matches])
pts2 = np.float32([kp2[m.trainIdx] for m in matches])

# estimate the homography between các tập điểm
(H, status) = cv2.findHomography(pts1, pts2, cv2.RANSAC)

def cube_points(c, wid):
    """ Tạo một danh sách các điểm để vẽ một khối lập phương. """
    p = []
    #bottom
    p.append([c[0]-wid, c[1]-wid, c[2]-wid])
    p.append([c[0]-wid, c[1]+wid, c[2]-wid])
    p.append([c[0]+wid, c[1]+wid, c[2]-wid])
    p.append([c[0]+wid, c[1]-wid, c[2]-wid])
    p.append([c[0]-wid, c[1]-wid, c[2]-wid]) #same as first to close plot
    
    # top
    p.append([c[0]-wid, c[1]-wid, c[2]+wid])
    p.append([c[0]-wid, c[1]+wid, c[2]+wid])
    p.append([c[0]+wid, c[1]+wid, c[2]+wid])
    p.append([c[0]+wid, c[1]-wid, c[2]+wid])
    p.append([c[0]-wid, c[1]-wid, c[2]+wid]) #same as first to close plot vertical 
    
    # sides
    p.append([c[0]-wid, c[1]-wid, c[2]+wid])
    p.append([c[0]-wid, c[1]+wid, c[2]+wid])
    p.append([c[0]-wid, c[1]+wid, c[2]-wid])
    p.append([c[0]+wid, c[1]+wid, c[2]-wid])
    p.append([c[0]+wid, c[1]+wid, c[2]+wid])
    p.append([c[0]+wid, c[1]-wid, c[2]+wid])
    p.append([c[0]+wid, c[1]-wid, c[2]-wid])

    return np.array(p).T

# camera calibration
K = my_calibration((747,1000))

# Tập điển khối 3D lập phương có độ dài cạnh bằng 0.2
box = cube_points([0, 0, 0.1], 0.1)

# Chiếu hình vuông trong bức ảnh đầu tiên
cam1 = Camera(hstack((K, dot(K, array([[0], [0], [-1]])))))
# 5 Điểm đầu là hình vuông tại đáy
box_cam1 = cam1.project(homography.make_homog(box[:, :5]))

# use H to transfer points to the second image
box_trans = homography.normalize(dot(H, box_cam1))

# Tính toán ma trận camera từ cam1 và H
cam2 = Camera(dot(H, cam1.P))
A = dot(np.linalg.inv(K), cam2.P[:, :3])
A = np.array([A[:, 0], A[:, 1], np.cross(A[:, 0], A[:, 1])]).T
cam2.P[:, :3] = dot(K, A)

# Chiếu với camera thứ 2
box_cam2 = cam2.project(homography.make_homog(box))

# Test, phép chiếu trên z=0 nên cho cùng một kết quả
point = array([1, 1, 0, 1]).T
print(homography.normalize(dot(dot(H, cam1.P), point)))
print(cam2.project(point))

## Lấy toạ độ để cắt ảnh lấy dữ liệu
# from pylab import *
# import cv2

# imshow(tar_img)
# x = ginput(2)
# print(x)
# img2 = tar_img[int(x[0][1]):int(x[1][1]), int(x[0][0]):int(x[1][0])] 
# cv2.imshow("Haha", img2)
# cv2.imwrite('img2.jpg', img2)
# show()

from pylab import *
im0 = array(imageio.imread(r'G:\DepTrai\ThucTap\img\data\book_frontal.JPG'))
im1 = array(imageio.imread(r'G:\DepTrai\ThucTap\img\data\book_perspective.JPG'))

# 2D projection of bottom square
figure()
imshow(im0)
plot(box_cam1[0, :],box_cam1[1, :], linewidth=3)

# 2D projection transferred with H
figure()
imshow(im1)
plot(box_trans[0, :], box_trans[1, :],linewidth=3)

# 3D cube
figure()
imshow(im1)
plot(box_cam2[0, :],box_cam2[1, :],linewidth=3)

show()

# Lưu lại K và R2
import pickle

with open('ar_camera.pkl', 'wb') as f:
    pickle.dump(K, f)
    pickle.dump(dot(linalg.inv(K), cam2.P), f)
