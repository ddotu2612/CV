import os
from PIL import Image
from numpy import  uint8, sqrt
from numpy import *
from pylab import *

# Hàm trả lại một list các link file
def get_imlist(path):
    return [os.path.join(path, file) for file in os.listdir(path) if (file.endswith('.jpg') or file.endswith('.png'))]

# Hàm thay đổi kích thước ảnh
def imresize(im, sz):
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

def plot_2D_boundary(plot_range, points, decisionfcn, labels, values=[0]):
    """ Plot_range is (xmin,xmax,ymin,ymax), points is a list
    of class points, decisionfcn is a funtion to evaluate,
    labels is a list of labels that decisionfcn returns for each class,
    values is a list of decision contours to show. """
    
    clist = ['b', 'r', 'g', 'k', 'm', 'y'] # colors for the classes
    
    # evaluate on a grid and plot contour of decision function
    x = np.arange(plot_range[0], plot_range[1], .1)
    y = np.arange(plot_range[2], plot_range[3], .1)
    xx, yy = np.meshgrid(x, y)
    xxx, yyy = xx.flatten(), yy.flatten() # lists of x,y in grid
    zz = array(decisionfcn(xxx, yyy))
    zz = zz.reshape(xx.shape)
    # plot contour(s) at values
    contour(xx, yy, zz, values)

    # for each class, plot the points with '*' for correct, ’o’ for incorrect
    for i in range(len(points)): # Have length = 2
        d = decisionfcn(points[i][:, 0], points[i][:, 1])
        correct_ndx = (labels[i] == d)
        incorrect_ndx = (labels[i] != d)
        plot(points[i][correct_ndx, 0], points[i][correct_ndx, 1], '*', color = clist[i])
        plot(points[i][incorrect_ndx, 0], points[i][incorrect_ndx, 1], 'o', color = clist[i])
    
    axis('equal')
