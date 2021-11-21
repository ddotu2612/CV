import pickle
import imagesearch
import cv2
from PIL import Image
import numpy as np


def process_image(imname):
    im = np.array(Image.open(imname).convert('L'))

    # create SIFT feature extractor
    sift = cv2.xfeatures2d.SIFT_create()
    
    # detect features from the image
    keypoints, descriptors = sift.detectAndCompute(im, None)
    
    return keypoints, descriptors



with open('imname.pkl', 'rb') as f:
    imlist = pickle.load(f)

nbr_images = len(imlist)
# print(imlist)

# load vocabulary
with open('vocabulary.pkl', 'rb') as f:
    voc = pickle.load(f)

# # create indexer
# indx = imagesearch.Indexer('test.db', voc)
# indx.create_tables()

# # go through all images, project features on vocalulary and insert
# for i in range(nbr_images)[:100]:
#     locs, descr = process_image(imlist[i])
#     indx.add_to_index(imlist[i], descr)

# # commit to database
# indx.db_commit()

# from sqlite3 import dbapi2 as sqlite

# con = sqlite.connect('test.db')
# print(con.execute('select count (filename) from imlist').fetchone())
# print(con.execute('select * from imlist').fetchone())

# test using index to get candidates
src = imagesearch.Searcher('test.db', voc)
kp, descr = process_image(imlist[0])
iw = voc.project(descr)

# print('Ask using a histogram...')
# print(src.candidates_from_histogram(iw)[:10])

# # test query by image
# print('try a query...')
# print(src.query(imlist[0])[:10])

## search image
nbr_results = 6
res = [w[1] for w in src.query(imlist[6])[:nbr_results]]
imagesearch.plot_results(src, res)
