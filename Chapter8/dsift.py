import cv2 
import matplotlib.pyplot as plt
import numpy as np

def process_image_dsift(imname, step_size=5, resize=(50, 50), scale=10):
    #reading image
    img = np.array(cv2.imread(imname))
    img = cv2.resize(img, resize, cv2.INTER_AREA)
    gray= cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    step_size = 5
    kp = [cv2.KeyPoint(float(x), float(y), scale) for y in range(0, gray.shape[0], step_size) 
                                            for x in range(0, gray.shape[1], step_size)]

    # img=cv2.drawKeypoints(gray, kp, img)

    _, dense_feat = sift.compute(img, kp)
    return dense_feat.flatten()

# print(process_image_dsift(r'G:\dvtu\ThucTap\img\train\A\A-uniform01.ppm'))