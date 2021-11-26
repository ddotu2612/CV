import sys
from numpy.lib.function_base import diff
sys.path.insert(0, r'G:/dvtu/ThucTap/Chapter1')
import imtools
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from scipy.ndimage import measurements
from pylab import *
import cv2

def compute_feature(im):
    """ Returns a feature vector for an 
        ocr image patch. """
    # resize and remove border
    norm_im = imtools.imresize(im, (30, 30))
    norm_im = norm_im[3:-3, 3:-3]

    return norm_im.flatten()

def load_ocr_data(path):
    """ Return labels and ocr features for all images
        in path. """
    
    # create list of all files ending in .jpg
    imlist = [os.path.join(path, f) for f in tqdm(os.listdir(path), desc="Loading images...") if f.endswith('.jpg')]
    # create labels
    labels = [int(imfile.split('\\')[-1][0]) for imfile in tqdm(imlist, desc="Loading labels...")]

    # create feature from the images
    features = []
    for imname in tqdm(imlist, desc="Loading extract features..."):
        im = np.array(Image.open(imname).convert('L'))
        features.append(compute_feature(im))
    
    return np.array(features), labels

features, labels = load_ocr_data(r'G:\dvtu\ThucTap\img\data\sudoku_images\sudoku_images\ocr_data\training')
test_features, test_labels = load_ocr_data(r'G:\dvtu\ThucTap\img\data\sudoku_images\sudoku_images\ocr_data\testing')

# svm
clf = make_pipeline(StandardScaler(), SVC(kernel='linear',gamma='auto'))
clf.fit(features, labels)

y_pred = clf.predict(test_features)
print('Accuracy:', accuracy_score(test_labels, y_pred))

# crop images
def find_sudoku_edges(im, axis=0):
    """ Finds the cell edges for an aligned sudoku image. """

    # theshold and sum rows and columns
    trim = 1 * (im < 128)
    s = trim.sum(axis=axis)

    # find center of strongest lines
    s_labels, s_nbr = measurements.label(s > (0.5 * np.max(s)))
    m = measurements.center_of_mass(s, s_labels, range(1, s_nbr + 1))
    x = [int(x[0]) for x in m]

    # if only the strong lines are detected ad lines in between
    if len(x) == 4:
        dx = np.diff(x)
        x = [x[0], int(x[0] + dx[0]/3), int(x[0] + 2*dx[0]/3),
             x[1], int(x[1] + dx[1]/3), int(x[1] + 2*dx[1]/3),
             x[2], int(x[2] + dx[2]/3), int(x[2] + 2*dx[2]/3), x[3]]
    
    if len(x) == 10:
        return x
    else:
        raise RuntimeError('Edges not detected.')

imname = r'G:\dvtu\ThucTap\img\data\sudoku_images\sudoku_images\sudokus\sudoku18.JPG'
vername = r'G:\dvtu\ThucTap\img\data\sudoku_images\sudoku_images\sudokus\sudoku18.sud'

# im = cv2.imread(imname)
# im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# # print(im.shape)
# # cv2.imshow('', im)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

im = np.array(Image.open(imname).convert('L'))
# find the cell edges
x = find_sudoku_edges(im, axis=0)
y = find_sudoku_edges(im, axis=1)

# crop cells and classify
crops = []
fig, ax = subplots(9, 9, figsize=(10, 20))
for col in range(9):
    for row in range(9):
        crop = im[y[col]:y[col + 1], x[row]:x[row + 1]]
        crops.append(compute_feature(crop))
        ax[col][row].imshow(crop)

y_pred = clf.predict(np.array(crops))
y_pred_im = np.array(y_pred).reshape(9, 9)

print('Result:')
print(y_pred_im)
