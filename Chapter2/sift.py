from numpy import *
from PIL import Image
from pylab import *
import os
import cv2
from numpy import arccos

# def process_image(imagename, resultname, params="--edge-thresh 10 --peak-thresh 5"):
#     """ Process an image and save the results in a file. """

#     if imagename[-3:] != 'pgm':
#         # create a pgm file
#         im = Image.open(imagename).convert('L')
#         im.save('tmp.pgm')
#         imagename = 'tmp.pgm'
    
#     cmmd = str("sift " + imagename + " --output=" + resultname + " " + params)
#     print(cmmd)
#     os.system(cmmd)

#     print('processed ', imagename, ' to ', resultname)

def process_image(image_gray):
    # create SIFT feature extractor
    sift = cv2.xfeatures2d.SIFT_create()
    # detect features from the image
    keypoints, descriptors = sift.detectAndCompute(image_gray, None)
    return keypoints, descriptors

def read_features_from_file(filename):
    """ Read feature properties and return in matrix form. """
    f = loadtxt(filename)

    return f[:,:4],f[:,4:] # feature locations, descriptors

def write_features_to_file(filename, locs, desc):
    """ Save feature location and descriptor to file. """
    savetxt(filename, hstack((locs, desc)))

from numpy import pi, cos, sin
def draw_circle(c, r):

    # Tao một array các phần tử trong khoảng từ 0-2pi
    t = arange(0, 1.01, .01) * 2 * pi

    # Toạ độ các điểm ấy tính từ tâm
    x = r*cos(t) + c[0]
    y = r*sin(t) + c[1]

    # vẽ các đường nối qua các điểm ấy tạo thành một hình tròn
    plot(x, y, 'b', linewidth=2)

# def plot_features(im, locs, circle=False):
#     """ Show image with features. input: im (image as array),
#     locs (row, col, scale, orientation of each feature). """

#     imshow(im)
#     if circle:
#         for p in locs:
#             draw_circle(p[:2], p[2])
#     else:
#         plot(locs[:, 0], locs[:, 1], 'ob')
#     axis('off')

def plot_features(image_gray, keypoints):
    #draw keypoints
    sift_image = cv2.drawKeypoints(image_gray, keypoints, None)

    cv2.imshow("Features Image", sift_image)
    cv2.waitKey(0)

def match(desc1, desc2):
    """ For each descriptor in the first image,
    select its match in the second image.
    input: desc1 (descriptors for the first image),
    desc2 (same for second image). """

    desc1 = array([d / linalg.norm(d) for d in desc1])
    desc2 = array([d / linalg.norm(d) for d in desc2])

    dist_ratio = 0.6
    desc1_size = desc1.shape    

    matchscores = zeros((desc1_size[0],1), 'uint8')
    desc2t = desc2.T # Chuyển vị ma trận

    for i in range(desc1_size[0]):
        dotprods = dot(desc1[i,:],desc2t) # vector of dot products
        dotprods = 0.9999 * dotprods
        # inverse cosine and sort, return index for features in second image
        indx = argsort(arccos(dotprods)) # Góc càng nhỏ càng gần

        # check if nearest neighbor has angle less than dist_ratio times 2nd
        if arccos(dotprods)[indx[0]] < dist_ratio * arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])

    return matchscores

def match_twosided(desc1, desc2):
    """ Two-sided symmetric version of match(). """

    matches_12 = match(desc1,desc2)
    matches_21 = match(desc2,desc1)

    ndx_12 = matches_12.nonzero()[0]

    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0

    return matches_12

# Nối hai ảnh
def appendimages(im1, im2):
    """ Return a new image that appends the two images side-by-side. """

    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0] # với ảnh thì shape[0] là chiều dọc, shape[1] là chiều ngang
    rows2 = im2.shape[0]

    # Nối vào cho đủ kích thước im1 = im2
    if rows1 < rows2:
        im1 = concatenate((im1, zeros((rows2 - rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = concatenate((im2, zeros((rows1 - rows2, im2.shape[1]))), axis=0)

    # if none of these cases they are equal, no filling needed.
    return concatenate((im1, im2), axis=1)

# Vẽ matches
def plot_matches(im1, im2, locs1, locs2, matchscores, show_below=True):
    """ Show a figure with lines joining the accepted matches
    input: im1,im2 (images as arrays), locs1, locs2 (feature locations),
    matchscores (as output from ’match()’),
    show_below (if images should be shown below matches). """

    im3 = appendimages(im1,im2)

    # Nếu muốn show ảnh gốc bên dưới ảnh match
    if show_below:
        im3 = vstack((im3,im3))

    imshow(im3)
    cols1 = im1.shape[1]
    for i,m in enumerate(matchscores):
        if m > 0: # Khác -1
            plot([locs1[i][1], locs2[m][1] + cols1], [locs1[i][0], locs2[m][0]], 'c')
            axis('off')
    show()









