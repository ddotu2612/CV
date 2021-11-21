import os
from PIL import Image
from numpy import  uint8, sqrt
from numpy import *

# Hàm trả lại một list các link file
def get_imlist(path):
    return [os.path.join(path, file) for file in os.listdir(path) if (file.endswith('.jpg') or file.endswith('.png'))]

# Hàm thay đổi kích thước ảnh
def imresize(im,sz):
    """ Resize an image array using PIL. """
    pil_im = Image.fromarray(uint8(im))
    return array(pil_im.resize(sz))

# Hàm cân bằng histogram, sử dụng để tăng độ tương phản,
# chuẩn hoá ảnh, camulative distribution function để chuẩn hoá
# giá trị pixel về một khoảng mong muốn
def histeq(im, nbr_bins=256):
    # get image histogram, trả về số lượng giá trị trong khoảng tính histogram
    imhist, bins = histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize
    # use linear interpolation of cdf to find new pixel values
    im2 = interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf
# im = Image.open('cat 2.jpg').convert('L')
# im.show()

# im_hist, cdf = histeq(array(im))
# im_hist = Image.fromarray(im_hist)
# im_hist.show()

# Trung bình ảnh => dùng để giảm nhiễu
def compute_average(imlist):
    """ Compute the average of a list of images. """
    # open first image and make into array of type float
    averageim = array(Image.open(imlist[0]), 'f')
    for imname in imlist[1:]:
        try:
            averageim += array(Image.open(imname))
        except:
            print(imname + '...skipped')
    averageim /= len(imlist)
    # return average as uint8
    return array(averageim, 'uint8')
    
# Tính PCA
def pca(X):
    """ Principal Component Analysis
    input: X, matrix with training data stored as flattened arrays in rows
    return: projection matrix (with important dimensions first), variance and mean.
    """
    # get dimensions
    num_data, dim = X.shape
    # print(num_data, dim)
    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        # PCA - compact trick used
        M = dot(X, X.T) # covariance matrix
        e, EV = linalg.eigh(M) # eigenvalues and eigenvectors
        tmp = dot(X.T, EV).T # this is the compact trick
        V = tmp[: :-1] # reverse since last eigenvectors are the ones we want
        print('V', V.shape[1])
        S = sqrt(e)[: :-1] # reverse since eigenvalues are in increasing order
  
        for i in range(V.shape[1]):
            V[:, i] /= S
    else:
        # PCA - SVD used
        U, S, V = linalg.svd(X)
        V = V[:num_data] # only makes sense to return the first num_data
    # return the projection matrix, the variance and the mean
    return V, S, mean_X