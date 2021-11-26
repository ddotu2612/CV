import sys
sys.path.insert(0, r'G:/dvtu/ThucTap/Chapter1')
from sklearn.preprocessing import StandardScaler
import imtools
import pickle
import knn
import numpy as np
from pylab import *
import os
import dsift

# # load 2D points using pickle
# with open('points_ring.pkl', 'rb') as f:
#     class_1 = pickle.load(f)
#     class_2 = pickle.load(f)
#     labels = pickle.load(f)

# model = knn.KnnClassifier(labels, np.vstack((class_1, class_2)))

# # load test data using pickle
# with open('points_ring_test.pkl', 'rb') as f:
#     class_1 = pickle.load(f)
#     class_2 = pickle.load(f)
#     labels = pickle.load(f)

# # test on the first point
# # print(model.classify(class_1[0]))

# # define function for plotting
# def classify(x, y, model = model):
#     return np.array([model.classify([xx, yy]) for (xx, yy) in zip(x, y)])

# # plot the classificaton boundary
# imtools.plot_2D_boundary([-6, 6, -6, 6], [class_1, class_2], classify, [1, -1])
# show()

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
scaler = StandardScaler()

features, labels = read_gesture_features_labels(r'G:\dvtu\ThucTap\img\train')
test_features, test_labels = read_gesture_features_labels(r'G:\dvtu\ThucTap\img\test')

# standard scaler
scaler.fit(features)
features = scaler.transform(features)
test_features = scaler.transform(test_features)

class_name = np.unique(labels)
print(class_name)

# test KNN
k = 1
knn_classifier = knn.KnnClassifier(labels, features)
res = np.array([knn_classifier.classify(test_features[i], k) for i in range(len(test_labels))])

# accuracy
acc = np.sum(1.0 * (res==test_labels)) / len(test_labels)
print('Accuracy', acc)


