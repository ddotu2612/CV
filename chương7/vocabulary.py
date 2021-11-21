from numpy.core.shape_base import vstack
from scipy.cluster.vq import *
import cv2
import os
from PIL import Image
import numpy as np
import pickle

def process_image(image_gray):
    # create SIFT feature extractor
    sift = cv2.xfeatures2d.SIFT_create()
    # detect features from the image
    keypoints, descriptors = sift.detectAndCompute(image_gray, None)
    return keypoints, descriptors

def get_imlist(path):
    with open(path, 'rb') as f:
        imlist = pickle.load(f)
    
    return imlist

class Vocabulary(object):

    def __init__(self, name) -> None:
        super().__init__()
        self.name =name
        self.voc = []
        self.idf = []
        self.trainingdata = []
        self.nbr_words = 0
    
    def train(self, path, k=100, subsampling=10):
        imlist = get_imlist(path)
        descr = []
        descr.append(process_image(imlist[0])[1])
        descriptors = descr[0]

        for i in np.arange(1, len(imlist)):
            print('train file thu ', i)
            descr.append(process_image(imlist[i])[1])
            descriptors = vstack((descriptors, descr[i]))
        print('Descriptors: ', descriptors.shape)
        # k-means: lasr number determines number of runs
        self.voc, distortion = kmeans(descriptors[::subsampling, :], k, 1)
        self.nbr_words = self.voc.shape[0]

        # go through all training images and project on vocabulary
        nbr_images = len(imlist)
        imwords = np.zeros((nbr_images, self.nbr_words))
        for i in range(len(imlist)):
            imwords[i] = self.project(descr[i])
        
        nbr_occurences = np.sum((imwords > 0) * 1, axis=0)
        
        self.idf = np.log((1.0 * nbr_images) / (1.0*nbr_occurences + 1))
        self.trainingdata = descriptors
    
    def project(self, descriptors):
        """ Project descriptors on the vocabulary
            to create a histogram of words. """
        # histogram of image words
        imhist = np.zeros((self.nbr_words))
        words, distance = vq(descriptors, self.voc)
        for w in words:
            imhist[w] += 1
        
        return imhist