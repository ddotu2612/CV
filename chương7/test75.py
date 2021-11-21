import pickle
import imagesearch
import cv2
import numpy as np
from PIL import Image

def process_image(imname):
    im = np.array(Image.open(imname).convert('L'))
    
    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    kp, d = orb_detector.detectAndCompute(im, None)

    return kp, d


with open('imname.pkl', 'rb') as f:
    imlist = pickle.load(f)

nbr_images = len(imlist)

with open('vocabulary.pkl', 'rb') as f:
    voc = pickle.load(f)

src = imagesearch.Searcher('test.db', voc)

# index of query and number of results to return
q_ind = 50
nbr_results = 20

# regular query
res_reg = [w[1] for w in src.query(imlist[q_ind])[:nbr_results]]
print('top matches (regular):', res_reg)

# load image features for query image
q_kp, q_descr = process_image(imlist[q_ind])

rank = {}
# load image features for result
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
for ndx in res_reg[1:]:
    kp, descr = process_image(imlist[ndx])

    # get maches
    matches = matcher.match(q_descr, descr)
    matches = sorted(matches, key = lambda x: x.distance)
    nbr_matches = len(matches)
 
    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((nbr_matches, 2))
    p2 = np.zeros((nbr_matches, 2))
    
    for i in range(len(matches)):
        p1[i, :] = q_kp[matches[i].queryIdx].pt
        p2[i, :] = kp[matches[i].trainIdx].pt
    
    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    
    # store inlier count
    rank[ndx] = len(list(filter(lambda x: (x[0] == 1), mask)))

# sort dictionary to get the most inliers first
sorted_rank = sorted(rank.items(), key=lambda t: t[1], reverse=True)
res_geom = [res_reg[0]] + [s[0] for s in sorted_rank]
print('top matches (homography):', res_geom)

# plot the top results
imagesearch.plot_results(src, res_reg[:8])
imagesearch.plot_results(src, res_geom[:8])
