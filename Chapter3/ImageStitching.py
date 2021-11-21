### Using OpenCV Stitching Images
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import imageio

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
src_img = imageio.imread('foto1A.jpg')
tar_img = imageio.imread('foto1B.jpg')
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
matches = matches[:200]

img3 = cv2.drawMatches(src_img, kp1, tar_img, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plot_img(img3, size=(15, 10))

# Estimate Homography matrix and transform image
kp1 = np.float32([kp.pt for kp in kp1])
kp2 = np.float32([kp.pt for kp in kp2])
pts1 = np.float32([kp1[m.queryIdx] for m in matches])
pts2 = np.float32([kp2[m.trainIdx] for m in matches])

# estimate the homography between các tập điểm
(H, status) = cv2.findHomography(pts1, pts2, cv2.RANSAC)
print(H)

# Transform image
h1, w1 = src_img.shape[:2]
h2, w2 = tar_img.shape[:2]
result = cv2.warpPerspective(src_img, H, (w1 + w2, h1))
result[0:h2, 0:w2] = tar_img
plot_img(result, size=(20, 10))