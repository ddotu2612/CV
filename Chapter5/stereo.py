import numpy as np
from scipy.ndimage import uniform_filter

def phane_sweep_ncc(im_l, im_r, start, steps, wid):
    """ Find disparity image using normalized cross-correlation. """

    m, n = im_l.shape

    # arrays to hold the different sums
    mean_l = np.zeros((m, n))
    mean_r = np.zeros((m, n))
    s = np.zeros((m, n))
    s_l = np.zeros((m, n))
    s_r = np.zeros((m, n))

    # array to hold depth planes
    dmaps = np.zeros((m, n, steps))

    # compute mean of patch
    uniform_filter(im_l,wid,mean_l)
    uniform_filter(im_r,wid,mean_r)

    # normalized images
    norm_l = im_l - mean_l
    norm_r = im_r - mean_r

    # try different disparities
    for displ in range(steps):
        # move left image to the right, compute sums
        uniform_filter(np.roll(norm_l, -displ-start)*norm_r, wid, s) # sum nominator
        uniform_filter(np.roll(norm_l, -displ-start)*np.roll(norm_l,-displ-start), wid, s_l)
        uniform_filter(norm_r*norm_r,wid,s_r) # sum denominator

        # store ncc scores
        dmaps[:,:,displ] = s / np.sqrt(s_l*s_r)
    
    # pick best depth for each pixel
    return np.argmax(dmaps,axis=2)

