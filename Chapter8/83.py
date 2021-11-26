import sys
sys.path.insert(0, r'G:/dvtu/ThucTap/Chapter1')
import imtools
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import numpy as np
from pylab import *
import os
import dsift



# # load 2D example points using Pickle
# with open('points_normal.pkl', 'rb') as f:
#     class_1 = pickle.load(f)
#     class_2 = pickle.load(f)
#     labels = pickle.load(f)

# samples = np.vstack((class_1, class_2))

# # create SVM
# clf = make_pipeline(SVC(gamma='auto'))
# clf.fit(samples, labels)

# # load test data using Pickle
# with open('points_normal_test.pkl', 'rb') as f:
#     class_1 = pickle.load(f)
#     class_2 = pickle.load(f)
#     labels = pickle.load(f)

# samples = np.vstack((class_1, class_2))
# y_pred = clf.predict(samples)
# print('Accuracy:', accuracy_score(labels, y_pred))

# # define function for plotting
# def predict(x, y, model=clf):
#     x = x.reshape(-1, 1)
#     y = y.reshape(-1, 1)
#     return model.predict(np.hstack((x, y)))

# # plot the classification boundary
# imtools.plot_2D_boundary([-6, 6, -6, 6], [class_1, class_2], predict, [1, -1])
# show()

# Hand gesture recognition again
# create SVM
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

def read_gesture_features_labels(path):
    features = []
    labels = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        for file in os.listdir(folder_path):
            print(f'{file} in {folder}')
            d = dsift.process_image_dsift(os.path.join(folder_path, file))
            features.append(d)
            labels.append(folder)
    
    return np.array(features), np.array(labels)

# read features, labels
features, labels = read_gesture_features_labels(r'G:\dvtu\ThucTap\img\train')
test_features, test_labels = read_gesture_features_labels(r'G:\dvtu\ThucTap\img\test')

# class_name = np.unique(labels)
# print(class_name)

# test SVM
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(features, labels)

y_pred = clf.predict(test_features)
print('Accuracy:', accuracy_score(test_labels, y_pred))