import sys
sys.path.insert(0, r'G:/dvtu/ThucTap/Chapter1')
import imtools
import pickle
import bayes
import numpy as np
from pylab import *
import os
import dsift

# load 2D example points using Pickle
with open('points_normal.pkl', 'rb') as f:
    class_1 = pickle.load(f)
    class_2 = pickle.load(f)
    labels = pickle.load(f)

# train Bayes classifier
bc = bayes.BayesClassifier()
bc.train([class_1, class_2], [-1, 1])

# load test data using Pickle
with open('points_normal_test.pkl', 'rb') as f:
    class_1 = pickle.load(f)
    class_2 = pickle.load(f)
    labels = pickle.load(f)

# test on some points
# print(bc.classify(class_1[:10])[0])

# plot points and dicision boundary
def classify(x, y, bc=bc):
    points = np.vstack((x, y))
    return bc.classify(points.T)[0]

features = np.vstack((class_1, class_2))
# classnames = [ c for c in np.unique(labels)]
# blist = [features[where(labels==c)[0]] for c in classnames]


res = bc.classify(features)[0]
acc = np.sum(1.0 * (res==labels)) / len(labels)
print('Accuracy:', acc)

imtools.plot_2D_boundary([-6, 6, -6, 6], [class_1, class_2], classify, [1, -1])
show()



# # Gesture recognition problem
# def read_gesture_features_labels(path):
#     features = []
#     labels = []
#     for folder in os.listdir(path):
#         folder_path = os.path.join(path, folder)
#         for file in os.listdir(folder_path):
#             print(f'{file} in {folder}')
#             d = dsift.process_image_dsift(os.path.join(folder_path, file))
#             features.append(d)
#             labels.append(folder)
    
#     return np.array(features), np.array(labels)

# # read features, labels
# features, labels = read_gesture_features_labels(r'G:\dvtu\ThucTap\img\train')
# test_features, test_labels = read_gesture_features_labels(r'G:\dvtu\ThucTap\img\test')

# V, S, m = imtools.pca(features)

# # keep most important dimentions
# V = V[:50]
# features = np.array([np.dot(V, f-m) for f in features])
# test_features = np.array([np.dot(V, f-m) for f in test_features])

# classnames = [ c for c in np.unique(labels)]

# print(classnames)

# # test Bayes
# bc = bayes.BayesClassifier()
# blist = [features[where(labels==c)[0]] for c in classnames]

# bc.train(blist, classnames)
# res = bc.classify(test_features)[0]

# acc = np.sum(1.0 * (res==test_labels)) / len(test_labels)
# print('Accuracy:', acc)
