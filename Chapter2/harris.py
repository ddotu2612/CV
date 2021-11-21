from scipy.ndimage import filters
from numpy import *
from numpy import zeros
from pylab import *
from PIL import Image

def compute_harris_response(im, sigma=3):
    """ Compute the Harris corner detector response function
    for each pixel in a graylevel image. """
    # Derivatives
    imx = zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    imy = zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)

    # compute components of the Harris matrix
    Wxx = filters.gaussian_filter(imx*imx, sigma) # Tinhs W*Wi
    Wxy = filters.gaussian_filter(imx*imy, sigma)
    Wyy = filters.gaussian_filter(imy*imy, sigma)

    # determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy

    return Wdet / Wtr

# Lấy danh sách các điểm thoã mãn các điều kiện
def get_harris_points(harrisim, min_dist=10, threshold=0.1):
    """ Return corners from a Harris response image
    min_dist is the minimum number of pixels separating
    corners and image boundary. """

    # find top corner candidates above a threshold
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1 # Chuyển sang ảnh nhị phân

    # get coordinates of candidates
    coords = array(harrisim_t.nonzero()).T # nonzero() trả ra một tuple các chỉ số theo từng chiều

    # ...and their values
    candidate_values = [harrisim[c[0],c[1]] for c in coords]

    # sort candidates
    index = argsort(candidate_values)

    # store allowed point locations in array
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1

    # select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
                (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0 # Các điểm lân cận nằm trong khoảng min_dist thì k được phép chọn
            
    return filtered_coords

def plot_harris_points(image, filtered_coords, harrisim):
    """ Plots corners found in image with 3 threshold differents """
    fig, ax = subplots(2, 4, figsize=(15, 4))
    gray()
    size_title = 10
    ax[0][0].imshow(harrisim)
    ax[0][0].set_title('Harris response function - min distance = 10', fontsize=size_title)
    ax[0][1].imshow(image)
    ax[0][1].set_title('Threshold = 0.01 - min distance = 10', fontsize=size_title)
    ax[0][1].plot([p[1] for p in filtered_coords[0]], [p[0] for p in filtered_coords[0]], '*')
    ax[0][2].imshow(image)
    ax[0][2].set_title('Threshold = 0.05 - min distance = 10', fontsize=size_title)
    ax[0][2].plot([p[1] for p in filtered_coords[1]], [p[0] for p in filtered_coords[1]], '*')
    ax[0][3].imshow(image)
    ax[0][3].set_title('Threshold = 0.1 - min distance = 10', fontsize=size_title)
    ax[0][3].plot([p[1] for p in filtered_coords[2]], [p[0] for p in filtered_coords[2]], '*')
    
    ax[1][0].imshow(harrisim)
    ax[1][0].set_title('Harris response function - min distance = 10', fontsize=size_title)
    ax[1][1].imshow(image)
    ax[1][1].set_title('Threshold = 0.01 - min distance = 30', fontsize=size_title)
    ax[1][1].plot([p[1] for p in filtered_coords[3]], [p[0] for p in filtered_coords[3]], '*')
    ax[1][2].imshow(image)
    ax[1][2].set_title('Threshold = 0.05 - min distance = 30', fontsize=size_title)
    ax[1][2].plot([p[1] for p in filtered_coords[4]], [p[0] for p in filtered_coords[4]], '*')
    ax[1][3].imshow(image)
    ax[1][3].set_title('Threshold = 0.1 - min distance = 30', fontsize=size_title)
    ax[1][3].plot([p[1] for p in filtered_coords[5]], [p[0] for p in filtered_coords[5]], '*')
    show()

# extract image patches 
def get_descriptors(image, filtered_coords, wid=5):
    """ For each point return pixel values around the point
    using a neighbourhood of width 2*wid+1. (Assume points are
    extracted with min_distance > wid). """
    desc = []
    for coords in filtered_coords:
        patch = image[coords[0] - wid : coords[0] + wid + 1,
        coords[1] - wid : coords[1] + wid + 1].flatten()
        desc.append(patch)
    return desc

def match(desc1, desc2, threshold=0.5):
    """ For each corner point descriptor in the first image,
    select its match to second image using
    normalized cross correlation. """

    n = len(desc1[0])

    # pair-wise distances
    d = -ones((len(desc1), len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - mean(desc1[i])) / std(desc1[i])
            d2 = (desc2[j] - mean(desc2[j])) / std(desc2[j])
            ncc_value = sum(d1 * d2) / (n-1)
            if ncc_value > threshold:
                d[i,j] = ncc_value

    ndx = argsort(-d) # Sap xep lay ra phan tu lon nhat moi row
    matchscores = ndx[:,0] # Lay ra len(desc1) phan tu lon nhat
    
    return matchscores

def match_twosided(desc1, desc2, threshold=0.5):
    """ Two-sided symmetric version of match(). """

    matches_12 = match(desc1, desc2, threshold)
    matches_21 = match(desc2, desc1, threshold)

    ndx_12 = where(matches_12 >= 0)[0]

    # remove matches that are not symmetric # xoá những cái match mà không đối xứng
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1
    return matches_12

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

def plot_matches(im1, im2, locs1, locs2, matchscores, show_below=True):
    """ Show a figure with lines joining the accepted matches
    input: im1,im2 (images as arrays), locs1,locs2 (feature locations),
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

im = array(Image.open('cat 2.jpg').convert('L'))
harrisim = compute_harris_response(im)
filtered_coords_1 = get_harris_points(harrisim, 10, 0.01)
filtered_coords_2 = get_harris_points(harrisim, 10, 0.05)
filtered_coords_3 = get_harris_points(harrisim, 10, 0.1)
filtered_coords_4 = get_harris_points(harrisim, 30, 0.01)
filtered_coords_5 = get_harris_points(harrisim, 30, 0.05)
filtered_coords_6 = get_harris_points(harrisim, 30, 0.1)
list_filtered = [filtered_coords_1, filtered_coords_2, filtered_coords_3, filtered_coords_4,
filtered_coords_5, filtered_coords_6]
plot_harris_points(im, list_filtered, harrisim)

# gray()
# title('Threshold = 0.05 - min distance = 10', fontsize=10)
# plot([p[1] for p in filtered_coords_1], [p[0] for p in filtered_coords_1], '*')
# imshow(im)
show()
# im1 = array(Image.open('cat 2.jpg').convert('L'))
# im2 = array(Image.open('cat 2.jpg').convert('L'))
# wid = 20
# harrisim = compute_harris_response(im1, 5)
# filtered_coords1 = get_harris_points(harrisim, wid+1)
# d1 = get_descriptors(im1, filtered_coords1, wid)

# harrisim = compute_harris_response(im2, 5)
# filtered_coords2 = get_harris_points(harrisim, wid + 1)
# d2 = get_descriptors(im2, filtered_coords2, wid)

# print('starting matching')
# print(filtered_coords1[0:3][0][1])
# matches = match_twosided(d1, d2)
# figure()
# gray()
# plot_matches(im1, im2, filtered_coords1, filtered_coords2, matches[0:200])

    











